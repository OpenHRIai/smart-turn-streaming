"""Stage A2: mine every meeting's silences from the AMI annotations alone (no audio).

    .venv/bin/python scripts/a2_ami_mine.py [--out data/a2/ami_events_all.parquet]

Prints counts per split and per outcome, and writes every event (labelled and excluded) with
its meeting, speaker letter, headset channel and split, so the audio build can cut from it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import ami  # noqa: E402
from eot import turn_rules as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/a2/ami_events_all.parquet")
    a = ap.parse_args()
    ms = ami.meetings()
    rows = []
    per_split: dict[str, Counter] = defaultdict(Counter)
    speakers_per_split: dict[str, set] = defaultdict(set)
    dropped = Counter()
    for meeting, spks in sorted(ms.items()):
        data = []
        for s in spks:
            spk, info = ami.speaker_data(meeting, s)
            dropped[meeting] += info["n_dropped"]
            if spk.vad:
                data.append((spk, s))
        split = ami.split_of(meeting)
        letters = {s.letter: s for _, s in data}
        evs = R.find_all([spk for spk, _ in data], include_excluded=True)
        for e in evs:
            per_split[split][e.reason] += 1
            if e.label is not None:
                speakers_per_split[split].add(letters[e.speaker].global_name or f"{meeting}.{e.speaker}")
            rows.append(dict(meeting=meeting, split=split, speaker=e.speaker, channel=letters[e.speaker].channel,
                             global_name=letters[e.speaker].global_name, t=e.t, label=e.label, reason=e.reason,
                             gap_s=e.gap_s, b_onset_s=e.b_onset_s, b_speaker=e.b_speaker, a_turn_s=e.a_turn_s,
                             a_turn_words=e.a_turn_words, a_text=e.a_text, b_text=e.b_text))
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)

    print(f"{len(ms)} meetings; words without timings dropped: {sum(dropped.values())} "
          f"(top: {dropped.most_common(4)})\n")
    print("| split | meetings | speakers (labelled) | complete | incomplete | no_floor | short_turn | collision | long_silence | tail |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for split in ("train", "dev", "test"):
        c = per_split[split]
        n_meet = df[df.split == split].meeting.nunique()
        print(f"| {split} | {n_meet} | {len(speakers_per_split[split])} | {c['complete']} | {c['incomplete']} | "
              f"{c['no_floor']} | {c['short_turn']} | {c['collision']} | {c['long_silence']} | {c['tail']} |")
    lab = df[df.label.notna()]
    print(f"\nlabelled: {len(lab)}, complete fraction {lab.label.eq('complete').mean():.3f}")
    inc = lab[lab.label == "incomplete"]
    print("incomplete gap_s quantiles:", inc.gap_s.quantile([.25, .5, .75, .9]).round(2).to_dict(),
          " frac >= 1.5 s:", round((inc.gap_s >= 1.5).mean(), 3))
    com = lab[lab.label == "complete"]
    print("complete b_onset_s quantiles:", com.b_onset_s.quantile([.1, .25, .5, .75]).round(2).to_dict(),
          " frac <= 0.2 s:", round((com.b_onset_s <= 0.2).mean(), 3), " frac <= 0.5 s:", round((com.b_onset_s <= 0.5).mean(), 3))
    print("a_turn_words median:", lab.a_turn_words.median(), " a_turn_s median:", round(lab.a_turn_s.median(), 2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
