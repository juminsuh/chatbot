"""
Weight-sensitivity bootstrap analysis on val_set.json (n=100) and test_set.json (n=50).

For each weight config (thought, emotion, behavior), bootstraps precision@1, precision@2,
regressions (baseline hit -> config miss) and fixes (baseline miss -> config hit), where
"baseline" = equal weights (1/3, 1/3, 1/3).

The val_set bootstrap draws are generated with the same seed/order as the earlier
hit@1/miss@1 analysis (rng seeded with VAL_SEED, 1000 draws of size 100) so it is literally
the same 1000 bootstrap datasets. test_set uses its own independent bootstrap (seeded with
TEST_SEED, 1000 draws of size 50) -- val and test never share resampled indices.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import PATTERN_IDS
from pattern_matching import _get_pattern_vecs, _THOUGHT_ROWS_BY_PATTERN
from embeddings import embed

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "val_set.json"
TEST_PATH = ROOT / "test_set.json"

VAL_SEED = 0     # same seed used in scripts/bootstrap_eval.py
TEST_SEED = 1    # independent stream for test_set
N_BOOTSTRAP = 1000

WEIGHT_CONFIGS = {
    "baseline":     (1 / 3, 1 / 3, 1 / 3),
    "thought_up":   (0.50, 0.25, 0.25),
    "thought_down": (0.20, 0.40, 0.40),
    "emotion_down": (0.40, 0.20, 0.40),
    "production":   (0.45, 0.30, 0.25),
}


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


def rank_by_weights(per_pattern: dict, weights: tuple[float, float, float]) -> list[str]:
    w_t, w_e, w_b = weights

    def wsum(s):
        return w_t * s["thought"] + w_e * s["emotion"] + w_b * s["behavior"]

    return sorted(per_pattern.keys(), key=lambda pid: wsum(per_pattern[pid]), reverse=True)


def evaluate_config(scored: list[dict], weights: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    n = len(scored)
    hit1 = np.zeros(n, dtype=bool)
    hit2 = np.zeros(n, dtype=bool)
    for i, s in enumerate(scored):
        ranked = rank_by_weights(s["per_pattern"], weights)
        hit1[i] = ranked[0] == s["gold_pattern_id"]
        hit2[i] = s["gold_pattern_id"] in ranked[:2]
    return hit1, hit2


def bootstrap_dataset(scored: list[dict], seed: int) -> dict:
    n = len(scored)
    rng = np.random.default_rng(seed)
    idx_draws = [rng.integers(0, n, size=n) for _ in range(N_BOOTSTRAP)]

    hit1_by_config = {}
    hit2_by_config = {}
    for name, weights in WEIGHT_CONFIGS.items():
        hit1, hit2 = evaluate_config(scored, weights)
        hit1_by_config[name] = hit1
        hit2_by_config[name] = hit2

    baseline_hit1 = hit1_by_config["baseline"]

    stats = {name: {"p1": [], "p2": [], "regressions": [], "fixes": []} for name in WEIGHT_CONFIGS}

    for idx in idx_draws:
        base_resampled = baseline_hit1[idx]
        for name in WEIGHT_CONFIGS:
            h1 = hit1_by_config[name][idx]
            h2 = hit2_by_config[name][idx]
            stats[name]["p1"].append(h1.mean())
            stats[name]["p2"].append(h2.mean())
            stats[name]["regressions"].append(int(np.sum(base_resampled & ~h1)))
            stats[name]["fixes"].append(int(np.sum(~base_resampled & h1)))

    summary = {}
    for name, vals in stats.items():
        summary[name] = {
            "precision@1": (float(np.mean(vals["p1"])), float(np.std(vals["p1"], ddof=1))),
            "precision@2": (float(np.mean(vals["p2"])), float(np.std(vals["p2"], ddof=1))),
            "regressions": (float(np.mean(vals["regressions"])), float(np.std(vals["regressions"], ddof=1))),
            "fixes": (float(np.mean(vals["fixes"])), float(np.std(vals["fixes"], ddof=1))),
        }
    return summary


def print_table(title: str, summary: dict):
    print(f"\n=== {title} ===")
    header = f"{'config':14s} {'weights(t/e/b)':18s} {'precision@1':18s} {'precision@2':18s} {'regressions':16s} {'fixes':16s}"
    print(header)
    for name, weights in WEIGHT_CONFIGS.items():
        s = summary[name]
        w_str = f"{weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}"
        p1 = f"{s['precision@1'][0]:.4f}±{s['precision@1'][1]:.4f}"
        p2 = f"{s['precision@2'][0]:.4f}±{s['precision@2'][1]:.4f}"
        reg = f"{s['regressions'][0]:.2f}±{s['regressions'][1]:.2f}"
        fix = f"{s['fixes'][0]:.2f}±{s['fixes'][1]:.2f}"
        print(f"{name:14s} {w_str:18s} {p1:18s} {p2:18s} {reg:16s} {fix:16s}")


def main():
    val_items = json.load(open(VAL_PATH, encoding="utf-8"))
    test_items = json.load(open(TEST_PATH, encoding="utf-8"))
    assert len(val_items) == 100
    assert len(test_items) == 50

    scored_val = compute_per_item_scores(val_items)
    scored_test = compute_per_item_scores(test_items)

    val_summary = bootstrap_dataset(scored_val, VAL_SEED)
    test_summary = bootstrap_dataset(scored_test, TEST_SEED)

    print_table(f"val_set (n=100, bootstrap={N_BOOTSTRAP}, seed={VAL_SEED})", val_summary)
    print_table(f"test_set (n=50, bootstrap={N_BOOTSTRAP}, seed={TEST_SEED})", test_summary)

    out = {
        "weight_configs": WEIGHT_CONFIGS,
        "n_bootstrap": N_BOOTSTRAP,
        "val": {"seed": VAL_SEED, "n": len(val_items), "summary": val_summary},
        "test": {"seed": TEST_SEED, "n": len(test_items), "summary": test_summary},
    }
    out_path = ROOT / "weight_sensitivity_report.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
