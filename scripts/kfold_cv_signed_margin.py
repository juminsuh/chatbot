"""
Same stratified 4-fold CV as scripts/kfold_cv.py, but the tuning objective is now a single
per-item average over ALL n items (fixed denominator, unlike mean(hit_margin)-mean(miss_margin)
in scripts/kfold_cv_margin.py, which re-weights every time an item flips between the hit and
miss pools):

    signed_margin_i = score(gold_i) - score(best non-gold pattern_i)
    objective(w)    = mean_i(signed_margin_i)          over all items, hit or miss

signed_margin_i is positive and equal to the hit margin when gold is top-ranked, negative
(magnitude = miss margin) when it isn't. Averaging it directly over all n items means an item
flipping from hit to miss changes the sum by a continuous amount instead of silently shrinking/
growing the denominator of some other subgroup average -- no more "trade away a fragile hit to
inflate the remaining hit-margin average" degenerate incentive.

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
from margin_eval import compute_margins, weighted_score

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "val_set.json"


def signed_margin_objective(scored: list[dict], weights: tuple[float, float, float]) -> float:
    total = 0.0
    for s in scored:
        pp = s["per_pattern"]
        gold = s["gold_pattern_id"]
        best_non_gold = max((pid for pid in pp if pid != gold), key=lambda pid: weighted_score(pp[pid], weights))
        total += weighted_score(pp[gold], weights) - weighted_score(pp[best_non_gold], weights)
    return total / len(scored)


def tune_weights_by_signed_margin(
    train_scored: list[dict], grid: list[tuple[float, float, float]], anchor=TIE_BREAK_ANCHOR
) -> tuple[tuple[float, float, float], float]:
    anchor_arr = np.array(anchor)
    best_obj = -np.inf
    best_candidates = []
    for w in grid:
        obj = signed_margin_objective(train_scored, w)
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_candidates = [w]
        elif abs(obj - best_obj) <= 1e-12:
            best_candidates.append(w)

    best_w = min(best_candidates, key=lambda w: np.linalg.norm(np.array(w) - anchor_arr))
    return best_w, best_obj


def diagnostics(scored: list[dict], weights: tuple[float, float, float]) -> dict:
    r = compute_margins(scored, weights)
    p1 = precision_at_k(scored, weights, 1)
    return {
        "objective": signed_margin_objective(scored, weights),
        "n_hit": r["n_hit"],
        "n_miss": r["n_miss"],
        "hit_margin": r["hit_overall_margin"][0],
        "miss_margin": r["miss_overall_margin"][0],
        "precision@1": p1,
    }


def main():
    val_items = json.load(open(VAL_PATH, encoding="utf-8"))
    assert len(val_items) == 100
    scored = compute_per_item_scores(val_items)

    print(f"=== signed-margin-objective stratified {K}-fold CV (seed={SPLIT_SEED}, grid step={GRID_STEP}, tie-break anchor={TIE_BREAK_ANCHOR}) ===")
    print("objective(w) = mean_i[ score(gold_i) - score(best_non_gold_i) ]  over all n items\n")

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

        tuned_w, train_obj = tune_weights_by_signed_margin(train_scored, grid)

        test_tuned = diagnostics(test_scored, tuned_w)
        test_base = diagnostics(test_scored, BASELINE_WEIGHTS)

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
            "test_tuned": test_tuned,
            "test_baseline": test_base,
        })

    print(
        f"\n{'fold':5s} {'tuned w(t/e/b)':16s} {'train obj':10s} {'test obj T':11s} {'test obj B':11s} "
        f"{'hitM T':8s} {'hitM B':8s} {'missM T':9s} {'missM B':9s} {'p@1 T':7s} {'p@1 B':7s}"
    )
    for r in fold_results:
        w = r["tuned_weights"]
        w_str = f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"
        t, b = r["test_tuned"], r["test_baseline"]
        print(
            f"{r['fold']:<5d} {w_str:16s} {r['train_objective']:<10.4f} {t['objective']:<11.4f} {b['objective']:<11.4f} "
            f"{t['hit_margin']:<8.4f} {b['hit_margin']:<8.4f} {t['miss_margin'] or 0:<9.4f} {b['miss_margin'] or 0:<9.4f} "
            f"{t['precision@1']:<7.4f} {b['precision@1']:<7.4f}"
        )

    def avg(fn):
        return float(np.mean([fn(r) for r in fold_results]))

    def std(fn):
        return float(np.std([fn(r) for r in fold_results], ddof=1))

    print(f"\n=== average across {K} folds ===")
    print(f"train objective                 = {avg(lambda r: r['train_objective']):.4f}")
    print(f"held-out objective (tuned)      = {avg(lambda r: r['test_tuned']['objective']):.4f} +- {std(lambda r: r['test_tuned']['objective']):.4f}")
    print(f"held-out objective (baseline)   = {avg(lambda r: r['test_baseline']['objective']):.4f} +- {std(lambda r: r['test_baseline']['objective']):.4f}")
    print(f"held-out hit margin (tuned)     = {avg(lambda r: r['test_tuned']['hit_margin']):.4f}")
    print(f"held-out hit margin (baseline)  = {avg(lambda r: r['test_baseline']['hit_margin']):.4f}")
    print(f"held-out miss margin (tuned)    = {avg(lambda r: r['test_tuned']['miss_margin'] or 0):.4f}")
    print(f"held-out miss margin (baseline) = {avg(lambda r: r['test_baseline']['miss_margin'] or 0):.4f}")
    print(f"held-out precision@1 (tuned)    = {avg(lambda r: r['test_tuned']['precision@1']):.4f} +- {std(lambda r: r['test_tuned']['precision@1']):.4f}")
    print(f"held-out precision@1 (baseline) = {avg(lambda r: r['test_baseline']['precision@1']):.4f} +- {std(lambda r: r['test_baseline']['precision@1']):.4f}")

    print(f"\n=== pooled out-of-fold precision@1 over all 100 val items ===")
    print(f"tuned (fold-specific weights)   : {all_oof_tuned_hit.sum()}/{len(val_items)} = {all_oof_tuned_hit.mean():.4f}")
    print(f"baseline weights                : {all_oof_baseline_hit.sum()}/{len(val_items)} = {all_oof_baseline_hit.mean():.4f}")

    final_w, final_obj = tune_weights_by_signed_margin(scored, grid)
    final_diag = diagnostics(scored, final_w)
    print(f"\n=== weights tuned on ALL 100 val items by signed-margin objective (candidate final hyperparameter) ===")
    print(
        f"weights = thought={final_w[0]:.2f}, emotion={final_w[1]:.2f}, behavior={final_w[2]:.2f}  "
        f"(objective={final_obj:.4f}, hit_margin={final_diag['hit_margin']:.4f}, "
        f"miss_margin={final_diag['miss_margin']}, precision@1={final_diag['precision@1']:.4f})"
    )
    print("NOTE: test_set was not touched anywhere in this script.")

    out = {
        "k": K,
        "split_seed": SPLIT_SEED,
        "grid_step": GRID_STEP,
        "tie_break_anchor": TIE_BREAK_ANCHOR,
        "fold_results": [{**r, "tuned_weights": list(r["tuned_weights"])} for r in fold_results],
        "pooled_oof_precision@1_tuned": float(all_oof_tuned_hit.mean()),
        "pooled_oof_precision@1_baseline": float(all_oof_baseline_hit.mean()),
        "final_weights_fit_on_all_val": {
            "weights": list(final_w),
            "objective": final_obj,
            **final_diag,
        },
    }
    out_path = ROOT / "kfold_cv_signed_margin_report.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
