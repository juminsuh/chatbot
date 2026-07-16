"""
Same stratified 4-fold CV as scripts/kfold_cv.py, but the tuning objective inside each train
fold is now:

    objective(w) = mean(hit_overall_margin) - mean(miss_overall_margin)

instead of precision@1. Bigger hit margin is better (safer correct predictions), smaller miss
margin is better (less-wrong mistakes), so subtracting the two rewards both at once. If a
candidate weight has zero misses on the train fold, miss_margin is treated as 0 (no penalty,
nothing to be wrong about); symmetric fallback for zero hits.

Reports, per fold: tuned weights, train objective, held-out objective/hit-margin/miss-margin/
precision@1 for both the tuned weight and the baseline weight (1/3,1/3,1/3), so we can see
whether optimizing margin (a continuous objective) generalizes better out-of-fold than
optimizing the previous discrete precision@1 objective did.

test_set is not touched here.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfold_cv import (
    compute_per_item_scores,
    rank_by_weights,
    precision_at_k,
    stratified_kfold_indices,
    weight_grid,
    BASELINE_WEIGHTS,
    TIE_BREAK_ANCHOR,
    K,
    SPLIT_SEED,
    GRID_STEP,
)
from margin_eval import compute_margins

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "val_set.json"


def objective_value(scored: list[dict], weights: tuple[float, float, float]) -> tuple[float, float, float]:
    r = compute_margins(scored, weights)
    hit_mean = r["hit_overall_margin"][0] if r["hit_overall_margin"][0] is not None else 0.0
    miss_mean = r["miss_overall_margin"][0] if r["miss_overall_margin"][0] is not None else 0.0
    return hit_mean - miss_mean, hit_mean, miss_mean


def tune_weights_by_margin(
    train_scored: list[dict], grid: list[tuple[float, float, float]], anchor=TIE_BREAK_ANCHOR
) -> tuple[tuple[float, float, float], float]:
    anchor_arr = np.array(anchor)
    best_obj = -np.inf
    best_candidates = []
    for w in grid:
        obj, _, _ = objective_value(train_scored, w)
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_candidates = [w]
        elif abs(obj - best_obj) <= 1e-12:
            best_candidates.append(w)

    best_w = min(best_candidates, key=lambda w: np.linalg.norm(np.array(w) - anchor_arr))
    return best_w, best_obj


def main():
    val_items = json.load(open(VAL_PATH, encoding="utf-8"))
    assert len(val_items) == 100
    scored = compute_per_item_scores(val_items)

    print(f"=== margin-objective stratified {K}-fold CV (seed={SPLIT_SEED}, grid step={GRID_STEP}, tie-break anchor={TIE_BREAK_ANCHOR}) ===")
    print("objective(w) = mean(hit_overall_margin) - mean(miss_overall_margin)\n")

    folds = stratified_kfold_indices(val_items, K, SPLIT_SEED)
    for i, f in enumerate(folds):
        print(f"fold {i}: n={len(f)}")

    grid = weight_grid(GRID_STEP)
    print(f"weight grid size = {len(grid)}")

    fold_results = []
    all_oof_tuned_hit = np.zeros(len(val_items), dtype=bool)
    all_oof_baseline_hit = np.zeros(len(val_items), dtype=bool)

    for fi in range(K):
        test_idx = folds[fi]
        train_idx = [idx for j, f in enumerate(folds) if j != fi for idx in f]

        train_scored = [scored[i] for i in train_idx]
        test_scored = [scored[i] for i in test_idx]

        tuned_w, train_obj = tune_weights_by_margin(train_scored, grid)

        test_obj_tuned, test_hit_m_tuned, test_miss_m_tuned = objective_value(test_scored, tuned_w)
        test_obj_base, test_hit_m_base, test_miss_m_base = objective_value(test_scored, BASELINE_WEIGHTS)

        test_p1_tuned = precision_at_k(test_scored, tuned_w, 1)
        test_p1_baseline = precision_at_k(test_scored, BASELINE_WEIGHTS, 1)

        for i in test_idx:
            ranked_tuned = rank_by_weights(scored[i]["per_pattern"], tuned_w)
            all_oof_tuned_hit[i] = ranked_tuned[0] == scored[i]["gold_pattern_id"]
            ranked_base = rank_by_weights(scored[i]["per_pattern"], BASELINE_WEIGHTS)
            all_oof_baseline_hit[i] = ranked_base[0] == scored[i]["gold_pattern_id"]

        fold_results.append({
            "fold": fi,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "tuned_weights": tuned_w,
            "train_objective": train_obj,
            "test_objective_tuned": test_obj_tuned,
            "test_objective_baseline": test_obj_base,
            "test_hit_margin_tuned": test_hit_m_tuned,
            "test_hit_margin_baseline": test_hit_m_base,
            "test_miss_margin_tuned": test_miss_m_tuned,
            "test_miss_margin_baseline": test_miss_m_base,
            "test_precision@1_tuned": test_p1_tuned,
            "test_precision@1_baseline": test_p1_baseline,
        })

    print(
        f"\n{'fold':5s} {'tuned w(t/e/b)':16s} {'train obj':10s} {'test obj tuned':15s} {'test obj base':14s} "
        f"{'hitM tuned':11s} {'hitM base':10s} {'missM tuned':12s} {'missM base':11s} {'p@1 tuned':10s} {'p@1 base':9s}"
    )
    for r in fold_results:
        w = r["tuned_weights"]
        w_str = f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"
        print(
            f"{r['fold']:<5d} {w_str:16s} {r['train_objective']:<10.4f} {r['test_objective_tuned']:<15.4f} "
            f"{r['test_objective_baseline']:<14.4f} {r['test_hit_margin_tuned']:<11.4f} {r['test_hit_margin_baseline']:<10.4f} "
            f"{r['test_miss_margin_tuned']:<12.4f} {r['test_miss_margin_baseline']:<11.4f} "
            f"{r['test_precision@1_tuned']:<10.4f} {r['test_precision@1_baseline']:<9.4f}"
        )

    def avg(key):
        return float(np.mean([r[key] for r in fold_results]))

    def std(key):
        return float(np.std([r[key] for r in fold_results], ddof=1))

    print(f"\n=== average across {K} folds ===")
    print(f"train objective                    = {avg('train_objective'):.4f}")
    print(f"held-out objective (tuned)          = {avg('test_objective_tuned'):.4f} +- {std('test_objective_tuned'):.4f}")
    print(f"held-out objective (baseline)       = {avg('test_objective_baseline'):.4f} +- {std('test_objective_baseline'):.4f}")
    print(f"held-out hit margin (tuned)         = {avg('test_hit_margin_tuned'):.4f}")
    print(f"held-out hit margin (baseline)      = {avg('test_hit_margin_baseline'):.4f}")
    print(f"held-out miss margin (tuned)        = {avg('test_miss_margin_tuned'):.4f}")
    print(f"held-out miss margin (baseline)     = {avg('test_miss_margin_baseline'):.4f}")
    print(f"held-out precision@1 (tuned)        = {avg('test_precision@1_tuned'):.4f} +- {std('test_precision@1_tuned'):.4f}")
    print(f"held-out precision@1 (baseline)     = {avg('test_precision@1_baseline'):.4f} +- {std('test_precision@1_baseline'):.4f}")

    print(f"\n=== pooled out-of-fold precision@1 over all 100 val items ===")
    print(f"tuned (fold-specific weights)   : {all_oof_tuned_hit.sum()}/{len(val_items)} = {all_oof_tuned_hit.mean():.4f}")
    print(f"baseline weights                : {all_oof_baseline_hit.sum()}/{len(val_items)} = {all_oof_baseline_hit.mean():.4f}")

    final_w, final_obj = tune_weights_by_margin(scored, grid)
    final_obj_full, final_hit_m, final_miss_m = objective_value(scored, final_w)
    final_p1 = precision_at_k(scored, final_w, 1)
    print(f"\n=== weights tuned on ALL 100 val items by margin objective (candidate final hyperparameter) ===")
    print(
        f"weights = thought={final_w[0]:.2f}, emotion={final_w[1]:.2f}, behavior={final_w[2]:.2f}  "
        f"(objective={final_obj:.4f}, hit_margin={final_hit_m:.4f}, miss_margin={final_miss_m:.4f}, precision@1={final_p1:.4f})"
    )
    print("NOTE: test_set was not touched anywhere in this script.")

    out = {
        "k": K,
        "split_seed": SPLIT_SEED,
        "grid_step": GRID_STEP,
        "tie_break_anchor": TIE_BREAK_ANCHOR,
        "fold_results": [{**r, "tuned_weights": list(r["tuned_weights"])} for r in fold_results],
        "averages": {
            "train_objective": avg("train_objective"),
            "test_objective_tuned": avg("test_objective_tuned"),
            "test_objective_baseline": avg("test_objective_baseline"),
            "test_hit_margin_tuned": avg("test_hit_margin_tuned"),
            "test_hit_margin_baseline": avg("test_hit_margin_baseline"),
            "test_miss_margin_tuned": avg("test_miss_margin_tuned"),
            "test_miss_margin_baseline": avg("test_miss_margin_baseline"),
            "test_precision@1_tuned": avg("test_precision@1_tuned"),
            "test_precision@1_baseline": avg("test_precision@1_baseline"),
        },
        "pooled_oof_precision@1_tuned": float(all_oof_tuned_hit.mean()),
        "pooled_oof_precision@1_baseline": float(all_oof_baseline_hit.mean()),
        "final_weights_fit_on_all_val": {
            "weights": list(final_w),
            "objective": final_obj,
            "hit_margin": final_hit_m,
            "miss_margin": final_miss_m,
            "precision@1": final_p1,
        },
    }
    out_path = ROOT / "kfold_cv_margin_report.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
