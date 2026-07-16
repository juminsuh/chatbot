"""
Margin-based comparison of weight configs on val_set (raw, n=100, no bootstrap).

For every item, rank patterns by the weighted pattern_score under a given (thought, emotion,
behavior) weight config:
  - hit  (rank1 == gold): competitor = rank2 (runner-up).
        overall_margin = score(gold) - score(competitor)          -- bigger is better (safer hit)
        field_margin[c] = raw_c(gold) - raw_c(competitor)          -- same, per raw field score
  - miss (rank1 != gold): competitor = rank1 (the wrong prediction).
        overall_margin = score(competitor) - score(gold)          -- smaller is better (closer miss)
        field_margin[c] = raw_c(competitor) - raw_c(gold)

This directly measures how confidently correct hits are, and how close/bad misses are, instead of
collapsing everything to a 0/1 precision@1 signal.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfold_cv import compute_per_item_scores  # reuses the same cached-embedding scoring

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "val_set.json"

WEIGHT_CONFIGS = {
    "baseline":                   (1 / 3, 1 / 3, 1 / 3),
    "thought_up":                 (0.50, 0.25, 0.25),
    "thought_down":               (0.20, 0.40, 0.40),
    "emotion_down":                (0.40, 0.20, 0.40),
    "production":                 (0.45, 0.30, 0.25),
    "cv_tuned_baseline_anchor":   (0.40, 0.40, 0.20),
    "cv_tuned_production_anchor": (0.45, 0.35, 0.20),
}

FIELDS = ("thought", "emotion", "behavior")


def weighted_score(s: dict, weights: tuple[float, float, float]) -> float:
    w_t, w_e, w_b = weights
    return w_t * s["thought"] + w_e * s["emotion"] + w_b * s["behavior"]


def compute_margins(scored: list[dict], weights: tuple[float, float, float]) -> dict:
    hit_overall, miss_overall = [], []
    hit_field = {c: [] for c in FIELDS}
    miss_field = {c: [] for c in FIELDS}

    for s in scored:
        pp = s["per_pattern"]
        gold = s["gold_pattern_id"]
        ranked = sorted(pp.keys(), key=lambda pid: weighted_score(pp[pid], weights), reverse=True)
        rank1, rank2 = ranked[0], ranked[1]

        if rank1 == gold:
            competitor = rank2
            hit_overall.append(weighted_score(pp[gold], weights) - weighted_score(pp[competitor], weights))
            for c in FIELDS:
                hit_field[c].append(pp[gold][c] - pp[competitor][c])
        else:
            competitor = rank1
            miss_overall.append(weighted_score(pp[competitor], weights) - weighted_score(pp[gold], weights))
            for c in FIELDS:
                miss_field[c].append(pp[competitor][c] - pp[gold][c])

    def stats(vals):
        if not vals:
            return (None, None)
        if len(vals) == 1:
            return (float(vals[0]), 0.0)
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)))

    return {
        "n_hit": len(hit_overall),
        "n_miss": len(miss_overall),
        "hit_overall_margin": stats(hit_overall),
        "miss_overall_margin": stats(miss_overall),
        "hit_field_margin": {c: stats(hit_field[c]) for c in FIELDS},
        "miss_field_margin": {c: stats(miss_field[c]) for c in FIELDS},
    }


def fmt(pair):
    m, s = pair
    if m is None:
        return "n/a"
    return f"{m:.4f}±{s:.4f}"


def main():
    val_items = json.load(open(VAL_PATH, encoding="utf-8"))
    assert len(val_items) == 100
    scored = compute_per_item_scores(val_items)

    results = {name: compute_margins(scored, w) for name, w in WEIGHT_CONFIGS.items()}

    print("=== overall margin comparison (val_set, n=100, raw) ===")
    print(f"{'config':28s} {'weights(t/e/b)':16s} {'n_hit':6s} {'n_miss':6s} {'hit margin (bigger=better)':28s} {'miss margin (smaller=better)':28s}")
    for name, w in WEIGHT_CONFIGS.items():
        r = results[name]
        w_str = f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"
        print(f"{name:28s} {w_str:16s} {r['n_hit']:<6d} {r['n_miss']:<6d} {fmt(r['hit_overall_margin']):28s} {fmt(r['miss_overall_margin']):28s}")

    print("\n=== field-level margin breakdown ===")
    for name, w in WEIGHT_CONFIGS.items():
        r = results[name]
        print(f"\n-- {name} ({w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}) --")
        print(f"  {'field':10s} {'hit margin':22s} {'miss margin':22s}")
        for c in FIELDS:
            print(f"  {c:10s} {fmt(r['hit_field_margin'][c]):22s} {fmt(r['miss_field_margin'][c]):22s}")

    out_path = ROOT / "margin_eval_report.json"
    json.dump(
        {name: {**r, "weights": WEIGHT_CONFIGS[name]} for name, r in results.items()},
        open(out_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
