"""Stage C1: the two-stage eager-then-commit rule, simulated offline on the B1 predictions.

    .venv/bin/python scripts/c1_eager_sweep.py --run runs/b1/knob_ami07_auc --out runs/c1/sweep.md

Pipecat's speculation path lets a stop strategy start the LLM on a *provisional* end of turn
(``UserTurnSpeculation``), hold the response in the LLM service's ``SpeculationGate`` and either
release it when the turn commits or discard it when the user resumes. For a turn detector that
means a two-stage rule on the same decision points as the B1/B3 re-query rule:

    eager   p_end(tau) >= theta_e(tau)   -> start generating (speculative inference)
    commit  p_end(tau) >= theta_c(tau)   -> end the turn, release the response
    cancel  the VAD reports speech       -> EagerEndOfTurnCancelFrame, response discarded

The commit table theta_c is the dev-fitted re-query table (false-alarm budget on turn splits);
the eager table theta_e <= theta_c is fitted here on AMI dev under a budget on *generations spent
on pauses* (the cost of a wrong eager is an LLM call, not an interruption), minimising the
perceived response time on turn ends. With a response latency L (inference trigger -> first bot
audio) the user hears the bot at

    max(t_commit, t_eager + L)   if an eager preceded the commit, else   t_commit + L

and a false alarm on a pause is *audible* only if that time comes before the speaker resumes.
Scored on AMI test with the misses charged the 3 s timeout, as the B1 tables were. The eager
head (p_eager during speech, pre-offset windows) is evaluated the same way as a variant.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.pipecat_analyzer import RULES  # noqa: E402
from eot.windows import DELTAS  # noqa: E402

OFFSETS = (0.2, 0.5, 1.2)
TIMEOUT_S = 3.0
GRID = np.concatenate([np.round(np.arange(0.0, 1.0001, 0.02), 2), [1.01]])   # 1.01 = never


# --- data --------------------------------------------------------------------------------------

def load_split(it: pd.DataFrame, idx: np.ndarray, p: np.ndarray):
    """One AMI split -> per-event arrays: P_end[N,K] at OFFSETS, avail[N,K], label[N], gap[N],
    P_eager[N,D] at DELTAS before the offset (nan where no window), p_eager of mid-speech windows."""
    d = it.iloc[idx].copy()
    d["p_end_hat"], d["p_eager_hat"] = p[:, 0], p[:, 1]
    key = ["meeting", "speaker", "t_off"]
    tau = d[(d.kind == "tau") & d.value.isin(OFFSETS)]
    piv = tau.pivot_table(index=key, columns="value", values="p_end_hat", aggfunc="first").reindex(columns=list(OFFSETS))
    meta = tau.groupby(key).agg(label=("p_end", "first"), gap=("gap_s", "first"))
    P = piv.values.astype(np.float64)
    label = meta.label.values.astype(int)
    gap = meta.gap.values.astype(float)
    gap = np.where(label == 1, np.inf, gap)
    avail = ~np.isnan(P) & (np.asarray(OFFSETS)[None, :] < gap[:, None])
    pre = d[d.kind == "pre"]
    pe = pre.pivot_table(index=key, columns="value", values="p_eager_hat", aggfunc="first").reindex(index=piv.index, columns=list(DELTAS))
    mid = d[d.kind == "mid"].p_eager_hat.values
    return dict(P=P, avail=avail, label=label, gap=gap, PE=pe.values.astype(np.float64), mid=mid, n=len(P))


# --- simulation ------------------------------------------------------------------------------

def first_fire(P, avail, theta, offsets=OFFSETS):
    """Time of the first offset with p >= theta (and the silence still going on); inf if none."""
    hit = avail & (np.nan_to_num(P, nan=-1.0) >= np.asarray(theta)[None, :])
    t = np.where(hit.any(1), np.asarray(offsets)[hit.argmax(1)], np.inf)
    return t


def simulate(S, theta_c, theta_e=None, L=0.8, t_eager=None):
    """Two-stage rule on one split. ``theta_e`` None = one stage (no speculation). ``t_eager`` lets a
    caller supply eager times from another signal (the eager head). Returns a dict of metrics."""
    label, gap = S["label"], S["gap"]
    comp, inc = label == 1, label == 0
    t_c = first_fire(S["P"], S["avail"], theta_c)
    t_c_eff = np.where(np.isinf(t_c), TIMEOUT_S, t_c)              # completes: timeout; pauses: no commit
    if t_eager is None:
        t_e = first_fire(S["P"], S["avail"], theta_e) if theta_e is not None else np.full(len(label), np.inf)
    else:
        t_e = np.asarray(t_eager, float)
    eager_before = t_e < t_c_eff                                     # an eager that precedes the commit / timeout
    resp = np.where(eager_before, np.maximum(t_c_eff, t_e + L), t_c_eff + L)
    committed = ~np.isinf(t_c)
    # completes
    rec = committed[comp].mean()
    lat = t_c_eff[comp]
    r = resp[comp]
    # pauses: a generation is spent when the eager fires or the commit fires (whichever first)
    spent = (np.minimum(t_e, t_c_eff)[inc] < gap[inc])
    fa = committed[inc]                                              # commit before the speaker resumed (avail already enforces it)
    audible = committed[inc] & (resp[inc] < gap[inc])
    held = eager_before[comp] & committed[comp]
    return dict(theta_c=tuple(theta_c), theta_e=tuple(theta_e) if theta_e is not None else None, L=L,
                recall=float(rec), lat_p50=float(np.percentile(lat, 50)), lat_p90=float(np.percentile(lat, 90)), lat_mean=float(lat.mean()),
                resp_p50=float(np.percentile(r, 50)), resp_p90=float(np.percentile(r, 90)), resp_mean=float(r.mean()),
                eager_rate_complete=float((t_e[comp] < np.inf).mean()), held=float(held.mean()),
                gen_pause=float(spent.mean()), far=float(fa.mean()), audible=float(audible.mean()),
                far_ge15=float(fa[gap[inc] >= 1.5].mean()) if (gap[inc] >= 1.5).any() else float("nan"))


def fit_eager(S, theta_c, budget, L=0.8, offsets=OFFSETS):
    """theta_e per offset, each <= theta_c there, minimising the mean perceived response time on
    completes subject to generations spent on pauses <= budget. Exhaustive grid."""
    label, gap = S["label"], S["gap"]
    comp, inc = label == 1, label == 0
    t_c = first_fire(S["P"], S["avail"], theta_c)
    t_c_eff = np.where(np.isinf(t_c), TIMEOUT_S, t_c)
    Pn = np.nan_to_num(S["P"], nan=-1.0)
    grids = [GRID[GRID <= theta_c[k] + 1e-9] if theta_c[k] <= 1 else GRID for k in range(len(offsets))]
    grids = [np.append(g, 1.01) if 1.01 not in g else g for g in grids]
    hits = [np.stack([S["avail"][:, k] & (Pn[:, k] >= t) for t in grids[k]]) for k in range(len(offsets))]
    best = None
    for combo in itertools.product(*[range(len(g)) for g in grids]):
        h = np.stack([hits[k][combo[k]] for k in range(len(offsets))], axis=1)
        t_e = np.where(h.any(1), np.asarray(offsets)[h.argmax(1)], np.inf)
        spent = (np.minimum(t_e, t_c_eff)[inc] < gap[inc]).mean()
        if spent > budget:
            continue
        resp = np.where(t_e < t_c_eff, np.maximum(t_c_eff, t_e + L), t_c_eff + L)[comp].mean()
        key = (resp, spent)
        if best is None or key < best[0]:
            best = (key, tuple(float(grids[k][combo[k]]) for k in range(len(offsets))))
    return best[1] if best else tuple([1.01] * len(offsets))


# --- the eager head during speech -----------------------------------------------------------

def eager_head_times(S, theta_h, deltas=DELTAS):
    """Eager times from p_eager on pre-offset windows: the earliest delta (largest, scanned from 0.8 s
    down) with p_eager >= theta -> t = -delta. inf if none fires. Oracle-timed: the head is evaluated
    at the trained deltas only, not at every hop of the speech."""
    PE = np.nan_to_num(S["PE"], nan=-1.0)
    order = np.argsort(-np.asarray(deltas))                 # 0.8, 0.6, ... 0.1
    hit = PE[:, order] >= theta_h
    t = np.where(hit.any(1), -np.asarray(deltas)[order][hit.argmax(1)], np.inf)
    return t


def fit_eager_head(S, theta_c, budget, L=0.8):
    label, gap = S["label"], S["gap"]
    comp, inc = label == 1, label == 0
    t_c = first_fire(S["P"], S["avail"], theta_c)
    t_c_eff = np.where(np.isinf(t_c), TIMEOUT_S, t_c)
    best = None
    for th in GRID:
        t_e = eager_head_times(S, th)
        spent = (np.minimum(t_e, t_c_eff)[inc] < gap[inc]).mean()
        if spent > budget:
            continue
        resp = np.where(t_e < t_c_eff, np.maximum(t_c_eff, t_e + L), t_c_eff + L)[comp].mean()
        if best is None or (resp, spent) < best[0]:
            best = ((resp, spent), float(th))
    return best[1] if best else 1.01


# --- report -----------------------------------------------------------------------------------

def fmt_theta(t):
    return "–" if t is None else " / ".join("off" if x > 1 else f"{x:.2f}" for x in t)


HEAD = ("| rule | θ_commit | θ_eager | generations on pauses | turn splits (strict FAR) | audible FAR | "
        "recall by 3 s | eager on turn ends / held | commit latency p50 / p90 / mean | response time p50 / p90 / mean |")
SEP = "|---|---|---|---|---|---|---|---|---|---|"


def row(name, m):
    return (f"| {name} | {fmt_theta(m['theta_c'])} | {fmt_theta(m['theta_e'])} | {m['gen_pause']:.1%} | {m['far']:.1%} | {m['audible']:.1%} | "
            f"{m['recall']:.2f} | {m['eager_rate_complete']:.2f} / {m['held']:.2f} | {m['lat_p50']:.2f} / {m['lat_p90']:.2f} / {m['lat_mean']:.2f} s | "
            f"{m['resp_p50']:.2f} / {m['resp_p90']:.2f} / {m['resp_mean']:.2f} s |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/b1/knob_ami07_auc")
    ap.add_argument("--items", default="data/b1/items.parquet")
    ap.add_argument("--L", type=float, default=0.8, help="response latency: inference trigger -> first bot audio, s")
    ap.add_argument("--budgets", default="0.2,0.3,0.4,0.5")
    ap.add_argument("--out", default="runs/c1/sweep.md")
    a = ap.parse_args()

    it = pd.read_parquet(a.items)
    z = np.load(Path(a.run) / "predictions.npz")
    dev = load_split(it, z["dev_idx"], z["p_dev"])
    test = load_split(it, z["test_ami"], z["p_test_ami"])
    L = a.L
    budgets = [float(b) for b in a.budgets.split(",")]
    out = [f"# C1 offline sweep: two-stage eager-then-commit rule ({a.run}, L = {L:g} s)", "",
           f"AMI dev {dev['n']} events ({(dev['label'] == 1).sum()} complete), AMI test {test['n']} events "
           f"({(test['label'] == 1).sum()} complete); offsets {OFFSETS}; misses charged {TIMEOUT_S:g} s. "
           "Generations on pauses: an eager or a commit before the speaker resumed. Audible FAR: the bot's "
           f"first audio (L = {L:g} s after the inference trigger) before the speaker resumed. Response time: completes, "
           "max(commit, eager + L) when a speculation was held, else commit + L. Eager tables fitted on dev; "
           "every number below is test.", ""]

    # baselines
    stock = simulate(test, (0.5, 1.01, 1.01), None, L)
    out += ["## Baselines and one-stage rules (no speculation)", "", HEAD, SEP, row("stock one-shot θ 0.5", stock)]
    for name in ("far10", "far15"):
        out.append(row(f"streaming {name}, one stage", simulate(test, RULES[name].thresholds, None, L)))
    out.append("")

    out += ["## Two stages: eager from p_end at the same offsets, θ_eager fitted on dev per generation budget", "", HEAD, SEP]
    tables = {}
    for name in ("far10", "far15"):
        th_c = RULES[name].thresholds
        for b in budgets:
            th_e = fit_eager(dev, th_c, b, L)
            m = simulate(test, th_c, th_e, L)
            tables[(name, b)] = th_e
            out.append(row(f"{name} + eager, budget {b:.0%}", m))
    out.append("")

    # eager at the VAD stop only (0.2 s) vs the full table, at one budget
    out += ["## Where the eager point matters: budget 30 %, far15 commit", "", HEAD, SEP]
    th_c = RULES["far15"].thresholds
    for label_, offs in (("eager at 0.2 s only", (0.2,)), ("eager at 0.2 / 0.5 s", (0.2, 0.5)), ("eager at 0.2 / 0.5 / 1.2 s", OFFSETS)):
        # restrict by forcing 'off' on the other offsets
        S_dev = dict(dev); S_test = dict(test)
        mask = np.array([o in offs for o in OFFSETS])
        S_dev["avail"] = dev["avail"] & mask[None, :]; S_test["avail"] = test["avail"] & mask[None, :]
        th_e = fit_eager(S_dev, th_c, 0.3, L)
        # eager times from the restricted set, commit from the full table
        t_e = first_fire(S_test["P"], S_test["avail"], th_e)
        m = simulate(test, th_c, None, L, t_eager=t_e); m["theta_e"] = th_e
        out.append(row(label_, m))
    out.append("")

    # the eager head during speech
    out += ["## Variant: the eager head (p_eager on pre-offset windows, oracle-timed at the trained δ), far15 commit", "",
            "Fires at the earliest δ ∈ {0.8 … 0.1} s before the acoustic offset with p_eager ≥ θ_h; the mid-speech "
            "false-fire rate is the share of random mid-utterance windows at or above θ_h (one hop = 100 ms).", "",
            "| budget | θ_h | generations on pauses | audible FAR | eager on turn ends / held | response time p50 / p90 / mean | mid-speech windows ≥ θ_h | ≈ false eagers per minute of speech |",
            "|---|---|---|---|---|---|---|---|"]
    for b in budgets:
        th_h = fit_eager_head(dev, th_c, b, L)
        t_e = eager_head_times(test, th_h)
        m = simulate(test, th_c, None, L, t_eager=t_e)
        mid_rate = float((test["mid"] >= th_h).mean())
        out.append(f"| {b:.0%} | {th_h:.2f} | {m['gen_pause']:.1%} | {m['audible']:.1%} | {m['eager_rate_complete']:.2f} / {m['held']:.2f} | "
                   f"{m['resp_p50']:.2f} / {m['resp_p90']:.2f} / {m['resp_mean']:.2f} s | {mid_rate:.1%} | {mid_rate * 600:.0f} |")
    out.append("")

    # response latency sensitivity for the chosen table
    out += ["## Response-latency sensitivity (far15 commit, eager table fitted at budget 30 % and L = 0.8 s)", "",
            "| L | rule | generations on pauses | audible FAR | response time p50 / p90 / mean |", "|---|---|---|---|---|"]
    th_e = tables[("far15", 0.3)]
    for Lx in (0.3, 0.5, 0.8, 1.2, 1.6):
        for nm, m in (("stock one-shot", simulate(test, (0.5, 1.01, 1.01), None, Lx)),
                      ("far15 one stage", simulate(test, th_c, None, Lx)),
                      ("far15 + eager 30 %", simulate(test, th_c, th_e, Lx))):
            out.append(f"| {Lx:g} s | {nm} | {m['gen_pause']:.1%} | {m['audible']:.1%} | {m['resp_p50']:.2f} / {m['resp_p90']:.2f} / {m['resp_mean']:.2f} s |")
    out.append("")
    out += ["## Fitted eager tables (dev)", "", "| commit rule | budget | θ_eager at 0.2 / 0.5 / 1.2 s |", "|---|---|---|"]
    for (name, b), th_e in tables.items():
        out.append(f"| {name} | {b:.0%} | {fmt_theta(th_e)} |")
    text = "\n".join(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    pd.DataFrame([dict(commit=name, budget=b, theta_e=th_e, theta_c=RULES[name].thresholds) for (name, b), th_e in tables.items()]).to_csv(
        Path(a.out).with_name("eager_tables.csv"), index=False)
    print(text)


if __name__ == "__main__":
    main()
