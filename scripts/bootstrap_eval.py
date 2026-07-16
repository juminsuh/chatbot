"""
Bootstrap evaluation on val_set.json using baseline (equal) weights: thought=emotion=behavior=1/3.

Reports:
  1) hit@1 / miss@1 rate over 1000 bootstrap resamples of the 100 val items (mean +- std).
  2) Per-category (thought/emotion/behavior) "gold win rate": among hit items, how often is the
     gold pattern also the top-ranked pattern when ranking by that single category's score alone
     (gold vs. the runner-up in that category); among miss items, same question (the category's
     top-ranked pattern vs. gold). Bootstrapped the same way (mean +- std).

Uses the cached pattern embeddings from pattern_matching.py so only the 100 val items need to be
embedded fresh.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import PATTERN_IDS
from pattern_matching import _get_pattern_vecs, _THOUGHT_ROWS_BY_PATTERN
from embeddings import embed

VAL_PATH = Path(__file__).resolve().parent.parent / "val_set.json"
BASELINE_WEIGHTS = {"thought": 1 / 3, "emotion": 1 / 3, "behavior": 1 / 3}
N_BOOTSTRAP = 1000
SEED = 0


def compute_per_item_scores(items: list[dict]) -> list[dict]:
    thought_vecs, emotion_vecs, behavior_vecs = _get_pattern_vecs()

    t_embeds = embed([it["thought"] for it in items])
    e_embeds = embed([it["emotion"] for it in items])
    b_embeds = embed([it["behavior"] for it in items])

    results = []
    for i, item in enumerate(items):
        t_sims = thought_vecs @ t_embeds[i]
        e_sims = emotion_vecs @ e_embeds[i]
        b_sims = behavior_vecs @ b_embeds[i]

        per_pattern = {}
        for j, pid in enumerate(PATTERN_IDS):
            rows = _THOUGHT_ROWS_BY_PATTERN[pid]
            per_pattern[pid] = {
                "thought": float(np.max(t_sims[rows])) if rows else 0.0,
                "emotion": float(e_sims[j]),
                "behavior": float(b_sims[j]),
            }
        results.append({
            "id": item["id"],
            "gold_pattern_id": item["gold_pattern_id"],
            "per_pattern": per_pattern,
        })
    return results


def baseline_rank(per_pattern: dict) -> list[str]:
    def wsum(s):
        return (
            BASELINE_WEIGHTS["thought"] * s["thought"]
            + BASELINE_WEIGHTS["emotion"] * s["emotion"]
            + BASELINE_WEIGHTS["behavior"] * s["behavior"]
        )
    return sorted(per_pattern.keys(), key=lambda pid: wsum(per_pattern[pid]), reverse=True)


def category_gold_is_top(per_pattern: dict, gold_id: str, category: str) -> bool:
    best_pid = max(per_pattern.keys(), key=lambda pid: per_pattern[pid][category])
    return best_pid == gold_id


def main():
    items = json.load(open(VAL_PATH, encoding="utf-8"))
    assert len(items) == 100, f"expected 100 val items, got {len(items)}"

    scored = compute_per_item_scores(items)

    is_hit = np.zeros(len(scored), dtype=bool)
    cat_gold_top = {c: np.zeros(len(scored), dtype=bool) for c in ("thought", "emotion", "behavior")}

    for i, s in enumerate(scored):
        ranked = baseline_rank(s["per_pattern"])
        is_hit[i] = ranked[0] == s["gold_pattern_id"]
        for c in cat_gold_top:
            cat_gold_top[c][i] = category_gold_is_top(s["per_pattern"], s["gold_pattern_id"], c)

    n = len(scored)
    rng = np.random.default_rng(SEED)

    hit_rates = np.empty(N_BOOTSTRAP)
    miss_rates = np.empty(N_BOOTSTRAP)
    cat_hit_win = {c: [] for c in cat_gold_top}
    cat_miss_win = {c: [] for c in cat_gold_top}

    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        hit_mask = is_hit[idx]
        hit_rates[b] = hit_mask.mean()
        miss_rates[b] = (~hit_mask).mean()

        for c in cat_gold_top:
            vals = cat_gold_top[c][idx]
            hit_vals = vals[hit_mask]
            miss_vals = vals[~hit_mask]
            if hit_vals.size > 0:
                cat_hit_win[c].append(hit_vals.mean())
            if miss_vals.size > 0:
                cat_miss_win[c].append(miss_vals.mean())

    print(f"n={n} bootstrap={N_BOOTSTRAP} weights=thought/emotion/behavior=1/3,1/3,1/3\n")

    print("=== overall (baseline weights) ===")
    print(f"raw hit@1  = {is_hit.mean():.4f} ({is_hit.sum()}/{n})")
    print(f"raw miss@1 = {(~is_hit).mean():.4f} ({(~is_hit).sum()}/{n})")
    print(f"bootstrap hit@1  mean +- std = {hit_rates.mean():.4f} +- {hit_rates.std(ddof=1):.4f}")
    print(f"bootstrap miss@1 mean +- std = {miss_rates.mean():.4f} +- {miss_rates.std(ddof=1):.4f}")

    print("\n=== per-category gold win rate (bootstrap mean +- std) ===")
    print(f"{'category':10s} {'hit-side gold win%':22s} {'miss-side gold win%':22s} {'n_hit_reps':10s} {'n_miss_reps':10s}")
    for c in ("thought", "emotion", "behavior"):
        hv = np.array(cat_hit_win[c])
        mv = np.array(cat_miss_win[c])
        print(
            f"{c:10s} "
            f"{hv.mean():.4f} +- {hv.std(ddof=1):.4f}      "
            f"{mv.mean():.4f} +- {mv.std(ddof=1):.4f}      "
            f"{hv.size:<10d} {mv.size:<10d}"
        )

    out = {
        "n": n,
        "n_bootstrap": N_BOOTSTRAP,
        "weights": BASELINE_WEIGHTS,
        "raw_hit_rate": float(is_hit.mean()),
        "raw_miss_rate": float((~is_hit).mean()),
        "bootstrap_hit_at_1": {"mean": float(hit_rates.mean()), "std": float(hit_rates.std(ddof=1))},
        "bootstrap_miss_at_1": {"mean": float(miss_rates.mean()), "std": float(miss_rates.std(ddof=1))},
        "per_category": {
            c: {
                "hit_side_gold_win_rate": {
                    "mean": float(np.array(cat_hit_win[c]).mean()),
                    "std": float(np.array(cat_hit_win[c]).std(ddof=1)),
                },
                "miss_side_gold_win_rate": {
                    "mean": float(np.array(cat_miss_win[c]).mean()) if cat_miss_win[c] else None,
                    "std": float(np.array(cat_miss_win[c]).std(ddof=1)) if cat_miss_win[c] else None,
                },
            }
            for c in ("thought", "emotion", "behavior")
        },
    }
    out_path = Path(__file__).resolve().parent.parent / "bootstrap_eval_report.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
