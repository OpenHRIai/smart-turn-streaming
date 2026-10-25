"""Stage B1: train the tau-aware heads on cached frozen-encoder features and evaluate the gates.

    .venv/bin/python scripts/b1_train.py --feats data/b1/feats.npz --items data/b1/items.parquet \
        --backbone v32 --sources std,ami --epochs 6 --out runs/b1/v32_std_ami

Ablations: --backbone mm, --no-mel, --no-tau, --sources std | ami. The report (``report.md`` in
--out) compares the trained heads with the frozen backbone's own probability (``p_base``) on
AMI dev/test and smart-turn-data test: AUC per tau, eager AUC per delta, the re-query decision
rules fitted on dev and scored on test, false alarms by pause length, and the std regression
guard (accuracy at tau 0.2, FAR at every tau).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import rules_eval as RE  # noqa: E402
from eot.model import Heads, masked_bce, tau_features  # noqa: E402
from eot.windows import DELTAS, TAUS  # noqa: E402


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Store:
    """Items + cached features aligned by item_id."""

    def __init__(self, feats: str, items: str, backbone: str, extra_feats: str | None = None, extra_items: str | None = None):
        self.items = pd.read_parquet(items)
        z = np.load(feats)
        ids = z["item_id"]
        assert (ids == np.arange(len(ids))).all() and len(ids) == len(self.items), "feats/items misaligned"
        pooled, mel, pb = z[f"pooled_{backbone}"], z["mel20"], z[f"p_{backbone}"].astype(np.float32)
        if extra_feats:
            ex = pd.read_parquet(extra_items)
            ze = np.load(extra_feats)
            assert (ze["item_id"] == ex.item_id.values).all() and ex.item_id.min() == len(ids), "extra feats/items misaligned"
            self.items = pd.concat([self.items, ex], ignore_index=True)
            pooled = np.concatenate([pooled, ze[f"pooled_{backbone}"]]); mel = np.concatenate([mel, ze["mel20"]])
            pb = np.concatenate([pb, ze[f"p_{backbone}"].astype(np.float32)])
        self.pooled, self.mel, self.p_base = pooled, mel, pb
        self.tau = tau_features(self.items.kind.values, self.items.value.values)
        self.y_end = self.items.p_end.values.astype(np.int64)
        self.y_eager = self.items.p_eager.values.astype(np.int64)

    def batch(self, idx: np.ndarray, dev: torch.device):
        return (torch.from_numpy(self.pooled[idx]).to(dev), torch.from_numpy(self.mel[idx]).to(dev),
                torch.from_numpy(self.tau[idx]).to(dev), torch.from_numpy(self.y_end[idx]).to(dev),
                torch.from_numpy(self.y_eager[idx]).to(dev))


@torch.no_grad()
def predict(model: Heads, st: Store, idx: np.ndarray, dev: torch.device, bs: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        pooled, mel, tau, _, _ = st.batch(b, dev)
        out.append(torch.sigmoid(model(pooled, mel, tau)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)


# --- evaluation helpers ----------------------------------------------------------------

def event_matrix(it: pd.DataFrame, p: np.ndarray, offsets=RE.OFFSETS):
    """AMI tau items of one split -> per-event (p[N,K], avail[N,K], label[N], gap[N])."""
    d = it.assign(p=p)
    d = d[(d.kind == "tau") & d.value.isin(offsets)]
    key = ["meeting", "speaker", "t_off"]
    piv = d.pivot_table(index=key, columns="value", values="p", aggfunc="first").reindex(columns=list(offsets))
    meta = d.groupby(key).agg(label=("p_end", "first"), gap=("gap_s", "first"))
    P = piv.values.astype(np.float64)
    return P, ~np.isnan(P), meta.label.values.astype(int), meta.gap.values.astype(float)


def auc_by_tau(it: pd.DataFrame, p_model: np.ndarray, p_base: np.ndarray) -> str:
    rows = ["| tau | n complete / incomplete | AUC model | AUC base | mean P incomplete: model / base |", "|---|---|---|---|---|"]
    d = it.assign(pm=p_model, pb=p_base)
    for t in TAUS:
        g = d[(d.kind == "tau") & (d.value == t)]
        if g.empty:
            continue
        inc = g[g.p_end == 0]
        rows.append(f"| {t:.1f} | {int((g.p_end == 1).sum())} / {len(inc)} | {auc(g.p_end, g.pm):.3f} | {auc(g.p_end, g.pb):.3f} | "
                    f"{inc.pm.mean():.3f} / {inc.pb.mean():.3f} |")
    return "\n".join(rows)


def eager_by_delta(it: pd.DataFrame, pe_model: np.ndarray, p_base: np.ndarray) -> str:
    """Pre-offset windows: positives = complete clips/events (label of the window's turn)."""
    rows = ["| delta | n | eager AUC (model p_eager) | AUC of frozen p (base) | mean p_eager: complete / incomplete |", "|---|---|---|---|---|"]
    d = it.assign(pe=pe_model, pb=p_base)
    lab = d.turn_label.values
    for dl in DELTAS:
        g = d[(d.kind == "pre") & (d.value == dl)]
        if g.empty:
            continue
        y = g.turn_label.values
        rows.append(f"| {dl:.1f} | {len(g)} | {auc(y, g.pe):.3f} | {auc(y, g.pb):.3f} | "
                    f"{g[g.turn_label == 1].pe.mean():.3f} / {g[g.turn_label == 0].pe.mean():.3f} |")
    return "\n".join(rows)


def std_guard(it: pd.DataFrame, p_model: np.ndarray, p_base: np.ndarray) -> str:
    d = it.assign(pm=p_model, pb=p_base)
    rows = ["| tau | AUC model / base | acc@0.5 model / base | recall@0.5 model / base | FAR@0.5 model / base | recall@FAR5% model / base |",
            "|---|---|---|---|---|---|"]
    for t in TAUS:
        g = d[(d.kind == "tau") & (d.value == t)]
        if g.empty:
            continue
        c, i = g[g.p_end == 1], g[g.p_end == 0]
        accm = ((g.pm >= 0.5) == (g.p_end == 1)).mean(); accb = ((g.pb >= 0.5) == (g.p_end == 1)).mean()
        thm, thb = np.quantile(i.pm, 0.95), np.quantile(i.pb, 0.95)
        rows.append(f"| {t:.1f} | {auc(g.p_end, g.pm):.3f} / {auc(g.p_end, g.pb):.3f} | {accm:.3f} / {accb:.3f} | "
                    f"{(c.pm >= .5).mean():.2f} / {(c.pb >= .5).mean():.2f} | {(i.pm >= .5).mean():.3f} / {(i.pb >= .5).mean():.3f} | "
                    f"{(c.pm >= thm).mean():.2f} / {(c.pb >= thb).mean():.2f} |")
    return "\n".join(rows)


def rules_report(dev_it, dev_p, dev_pb, test_it, test_p, test_pb) -> str:
    out = []
    Pd, Ad, Ld, Gd = event_matrix(dev_it, dev_p); Pdb, _, _, _ = event_matrix(dev_it, dev_pb)
    Pt, At, Lt, Gt = event_matrix(test_it, test_p); Ptb, _, _, _ = event_matrix(test_it, test_pb)
    for name, (P_d, P_t) in {"model": (Pd, Pt), "base (frozen p)": (Pdb, Ptb)}.items():
        rows = [("one-shot theta 0.5 at 0.2 s", RE.one_shot(P_t, At, Lt, Gt))]
        for far in (0.05, 0.10, 0.15):
            fitted = RE.fit(P_d, Ad, Ld, Gd, far, "min_latency")
            rows.append((f"re-query, FAR<={int(far * 100)}% (fit on dev), min latency", RE.simulate(P_t, At, Lt, Gt, fitted.theta)))
        out.append(f"\n**{name}**, scored on test ({int((Lt == 1).sum())} complete / {int((Lt == 0).sum())} incomplete):\n\n" + RE.table(rows))
    rows = [("silence only, commit at 1.2 s", RE.simulate(np.ones_like(Pt), At, Lt, Gt, (1.01, 1.01, 0.0)))]
    out.append("\n**no model**:\n\n" + RE.table(rows))
    # sweep of FAR budgets: recall / mean latency / p90 on test with thresholds fitted on dev
    sweep = ["| FAR target | model: recall / FAR / mean / p90 | base: recall / FAR / mean / p90 |", "|---|---|---|"]
    for far in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
        rm = RE.simulate(Pt, At, Lt, Gt, RE.fit(Pd, Ad, Ld, Gd, far, "min_latency").theta)
        rb = RE.simulate(Ptb, At, Lt, Gt, RE.fit(Pdb, Ad, Ld, Gd, far, "min_latency").theta)
        sweep.append(f"| {far:.3f} | {rm.recall:.2f} / {rm.far:.3f} / {rm.mean:.2f} s / {rm.p90:.1f} s | "
                     f"{rb.recall:.2f} / {rb.far:.3f} / {rb.mean:.2f} s / {rb.p90:.1f} s |")
    out.append("\n**FAR sweep** (re-query rule, min-latency objective, fit on dev, scored on test):\n\n" + "\n".join(sweep))
    return "\n".join(out)


def plot_curves(h: pd.DataFrame, e: pd.DataFrame, path, title: str, best_epoch: int) -> None:
    """Training loss and lr per step, dev metrics per epoch -> one PNG per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    if len(h):
        ax[0].plot(h.epoch, h.loss, alpha=0.15, color="C0")
        ax[0].plot(h.epoch, h.loss.rolling(15, min_periods=1).mean(), color="C0", label="total")
        ax[0].plot(h.epoch, h.loss_end.rolling(15, min_periods=1).mean(), color="C1", label="p_end")
        ax[0].plot(h.epoch, h.loss_eager.rolling(15, min_periods=1).mean(), color="C2", label="p_eager")
        ax[1].plot(h.epoch, h.lr)
    ax[0].set_title("training loss per step (15-step mean)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
    ax[1].set_title("learning rate"); ax[1].set_xlabel("epoch")
    ax[2].plot(e.epoch, e.dev_auc_tau02, "-o", ms=3, label="dev AUC tau 0.2")
    ax[2].plot(e.epoch, e.dev_auc_tau12, "-o", ms=3, label="dev AUC tau 1.2")
    ax[2].plot(e.epoch, e.dev_eager_auc, "-o", ms=3, label="dev eager AUC")
    ax[2].axvline(best_epoch, color="k", ls=":", lw=1, label=f"selected epoch {best_epoch}")
    ax[2].set_title("AMI dev ranking per epoch"); ax[2].set_xlabel("epoch"); ax[2].legend(fontsize=8)
    ax[3].plot(e.epoch, e.dev_recall_far10, "-o", ms=3, label="recall @ FAR 10 %")
    ax[3].plot(e.epoch, e.dev_far_ge15, "-o", ms=3, label="FAR on pauses >= 1.5 s")
    ax[3].plot(e.epoch, e.dev_mean_lat_far10 / 3, "-o", ms=3, label="mean latency / 3 s")
    ax[3].axvline(best_epoch, color="k", ls=":", lw=1)
    ax[3].set_title("AMI dev rule metrics per epoch"); ax[3].set_xlabel("epoch"); ax[3].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# --- main --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="data/b1/feats.npz")
    ap.add_argument("--items", default="data/b1/items.parquet")
    ap.add_argument("--backbone", default="v32", choices=["v32", "mm"])
    ap.add_argument("--sources", default="std,ami")
    ap.add_argument("--no-mel", action="store_true")
    ap.add_argument("--no-tau", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-eager", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-2, help="AdamW weight decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", default="score", choices=["score", "auc"],
                    help="checkpoint selection on dev: 'score' = recall@FAR10 - mean latency/10; 'auc' = mean AUC at tau 0.2 and 1.2")
    ap.add_argument("--ami-weight", type=float, default=0.5, help="fraction of each epoch drawn from AMI when both sources train")
    ap.add_argument("--extra-feats", default=None, help="additional feats npz (e.g. data/b1/feats_extra.npz)")
    ap.add_argument("--extra-items", default=None, help="items parquet matching --extra-feats")
    ap.add_argument("--wandb", default=None, help="W&B project name; logs step loss/lr and epoch dev metrics (needs `wandb login`)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project=a.wandb, name=Path(a.out).name, config=vars(a))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = device_auto()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = Store(a.feats, a.items, a.backbone, a.extra_feats, a.extra_items)
    it = st.items
    # label of the turn each window belongs to (for eager AUC): from the tau items of the same clip/event
    key = np.where(it.source == "std", it.clip_id.astype(str), it.meeting.astype(str) + "/" + it.speaker.astype(str) + "/" + it.t_off.round(3).astype(str))
    it["ekey"] = key
    lab = it[it.kind == "tau"].groupby("ekey").p_end.first()
    it["turn_label"] = it.ekey.map(lab).fillna(0).astype(int)

    sources = a.sources.split(",")
    # extra items (e.g. long-pause negatives) train only; dev/test stay the original sets so that
    # every run is scored on the same silences
    is_extra = it["extra"].notna().values if "extra" in it.columns else np.zeros(len(it), bool)
    tr = np.where((it.split == "train") & it.source.isin(sources))[0]
    dev_idx = np.where((it.split == "dev") & (it.source == "ami") & ~is_extra)[0]
    test_ami = np.where((it.split == "test") & (it.source == "ami") & ~is_extra)[0]
    test_std = np.where((it.split == "test") & (it.source == "std"))[0]
    print(f"train {len(tr)} items from {sources}; dev {len(dev_idx)}; test ami {len(test_ami)}, std {len(test_std)}; device {dev}")

    # per-source balancing: equal expected mass per source per epoch
    src = it.source.values[tr]
    w = np.ones(len(tr))
    share = {"ami": a.ami_weight, "std": 1.0 - a.ami_weight} if len(sources) == 2 else {sources[0]: 1.0}
    for s in sources:
        m = src == s
        w[m] = share[s] / max(m.sum(), 1)
    w = w / w.sum()

    model = Heads(use_mel=not a.no_mel, use_tau=not a.no_tau, dropout=a.dropout).to(dev)
    print(f"model params {model.n_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    best, best_state, best_epoch, log = -1.0, None, 0, []
    history = []          # per-step samples: step, lr, loss, loss_end, loss_eager
    gstep = 0
    for ep in range(a.epochs):
        model.train()
        order = rng.choice(tr, size=len(tr), replace=True, p=w)
        tl, n = 0.0, 0
        for i in range(0, len(order) - a.bs + 1, a.bs):
            pooled, mel, tau, ye, yg = st.batch(order[i:i + a.bs], dev)
            loss, parts = masked_bce(model(pooled, mel, tau), ye, yg, a.w_eager)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += float(loss); n += 1; gstep += 1
            if gstep % 20 == 0:
                rec = dict(step=gstep, epoch=ep + (i + a.bs) / len(order), lr=sched.get_last_lr()[0], loss=float(loss), **{f"loss_{k}": v for k, v in parts.items()})
                history.append(rec)
                if wb:
                    wb.log(rec, step=gstep)
        pd_ = predict(model, st, dev_idx, dev)
        d_it = it.iloc[dev_idx]
        P, A, L, G = event_matrix(d_it, pd_[:, 0])
        r10 = RE.fit(P, A, L, G, 0.10, "min_latency")
        a12 = auc(d_it[(d_it.kind == "tau") & (d_it.value == 1.2)].p_end, pd_[:, 0][(d_it.kind == "tau").values & (d_it.value == 1.2).values])
        a02 = auc(d_it[(d_it.kind == "tau") & (d_it.value == 0.2)].p_end, pd_[:, 0][(d_it.kind == "tau").values & (d_it.value == 0.2).values])
        pre = (d_it.kind == "pre").values
        ae = auc(d_it.turn_label.values[pre], pd_[:, 1][pre])
        score = r10.recall - r10.mean / 10 if a.select == "score" else (a02 + a12) / 2
        row = dict(epoch=ep + 1, loss=tl / max(n, 1), dev_auc_tau02=a02, dev_auc_tau12=a12, dev_eager_auc=ae,
                   dev_recall_far10=r10.recall, dev_mean_lat_far10=r10.mean, dev_far_ge15=r10.far_by_bin[">=1.5"])
        log.append(row)
        if wb:
            wb.log({f"dev/{k}": v for k, v in row.items() if k != "epoch"}, step=gstep)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if score > best:
            best, best_state, best_epoch = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "config": vars(a)}, out / "heads.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(log).to_csv(out / "epochs.csv", index=False)
    plot_curves(pd.DataFrame(history), pd.DataFrame(log), out / "curves.png", Path(a.out).name, best_epoch)

    # --- final report ---
    p_dev = predict(model, st, dev_idx, dev); p_ta = predict(model, st, test_ami, dev); p_ts = predict(model, st, test_std, dev)
    d_it, ta_it, ts_it = it.iloc[dev_idx], it.iloc[test_ami], it.iloc[test_std]
    rep = [f"# B1 report: backbone {a.backbone}, sources {a.sources}, mel {not a.no_mel}, tau {not a.no_tau}",
           f"\n{len(tr)} training items, {a.epochs} epochs, {model.n_params():,} trainable parameters, "
           f"{time.time() - t0:.0f} s on {dev}. Selection: {a.select}; best epoch {best_epoch}. lr {a.lr}, AMI weight {a.ami_weight}, extra items {len(it) - 1137921 if a.extra_feats else 0}.",
           "\n## Training log\n", pd.DataFrame(log).round(4).to_markdown(index=False),
           "\n## AMI dev: AUC of P(complete) per tau (complete vs incomplete silences)\n", auc_by_tau(d_it, p_dev[:, 0], st.p_base[dev_idx]),
           "\n## AMI test: AUC per tau\n", auc_by_tau(ta_it, p_ta[:, 0], st.p_base[test_ami]),
           "\n## AMI dev: eager head on pre-offset windows (positive = the turn is complete)\n", eager_by_delta(d_it, p_dev[:, 1], st.p_base[dev_idx]),
           "\n## AMI test: eager head\n", eager_by_delta(ta_it, p_ta[:, 1], st.p_base[test_ami]),
           "\n## Decision rules (thresholds fitted on AMI dev, scored on AMI test)\n", rules_report(d_it, p_dev[:, 0], st.p_base[dev_idx], ta_it, p_ta[:, 0], st.p_base[test_ami]),
           "\n## smart-turn-data test: regression guard (threshold 0.5; model / frozen base)\n", std_guard(ts_it, p_ts[:, 0], st.p_base[test_std]),
           "\n## smart-turn-data test: eager head\n", eager_by_delta(ts_it, p_ts[:, 1], st.p_base[test_std])]
    (out / "report.md").write_text("\n".join(rep))
    np.savez(out / "predictions.npz", dev_idx=dev_idx, p_dev=p_dev, test_ami=test_ami, p_test_ami=p_ta, test_std=test_std, p_test_std=p_ts)
    print("\n".join(rep[6:]))
    print(f"\nwrote {out}/report.md")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
