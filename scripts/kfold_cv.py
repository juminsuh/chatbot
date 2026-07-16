"""
Step 1: raw (no bootstrap) hit@1/miss@1 + per-category gold win rate on val_set, baseline weight.
Step 3: stratified (by gold_pattern_id) k=4 fold CV on val_set only.
  - each fold: tune (thought, emotion, behavior) weights on the other k-1 folds (grid search,
    maximize precision@1, ties broken by distance to baseline), evaluate the tuned weights on
    the held-out fold.
  - test_set is untouched here -- reserved for a final one-shot check with whatever weight
    the CV settles on.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import PATTERN_IDS
from pattern_matching import _get_pattern_vecs, _THOUGHT_ROWS_BY_PATTERN
from embeddings import embed

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "val_set.json"

BASELINE_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
TIE_BREAK_ANCHOR = (0.45, 0.30, 0.25)  # current production weights; used only to break ties among equally-good grid points
K = 4
SPLIT_SEED = 42
GRID_STEP = 0.05  # weight grid resolution for tuning


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


def precision_at_k(scored: list[dict], weights: tuple[float, float, float], k: int) -> float:
    if not scored:
        return float("nan")
    hits = 0
    for s in scored:
        ranked = rank_by_weights(s["per_pattern"], weights)
        if s["gold_pattern_id"] in ranked[:k]:
            hits += 1
    return hits / len(scored)


def category_gold_is_top(per_pattern: dict, gold_id: str, category: str) -> bool:
    best_pid = max(per_pattern.keys(), key=lambda pid: per_pattern[pid][category])
    return best_pid == gold_id


def report_step1(scored: list[dict]):
    n = len(scored)
    is_hit = np.zeros(n, dtype=bool)
    cat_gold_top = {c: np.zeros(n, dtype=bool) for c in ("thought", "emotion", "behavior")}

    for i, s in enumerate(scored):
        ranked = rank_by_weights(s["per_pattern"], BASELINE_WEIGHTS)
        is_hit[i] = ranked[0] == s["gold_pattern_id"]
        for c in cat_gold_top:
            cat_gold_top[c][i] = category_gold_is_top(s["per_pattern"], s["gold_pattern_id"], c)

    print(f"=== STEP 1: val_set raw hit@1/miss@1 (baseline weights 1/3,1/3,1/3, n={n}, no bootstrap) ===")
    print(f"hit@1  = {is_hit.sum()}/{n} = {is_hit.mean():.4f}")
    print(f"miss@1 = {(~is_hit).sum()}/{n} = {(~is_hit).mean():.4f}")

    print(f"\n=== per-category gold win rate (raw, no bootstrap) ===")
    print(f"{'category':10s} {'hit-side gold win%':20s} {'miss-side gold win%':20s} {'n_hit':6s} {'n_miss':6s}")
    n_hit = int(is_hit.sum())
    n_miss = int((~is_hit).sum())
    for c in ("thought", "emotion", "behavior"):
        hit_rate = cat_gold_top[c][is_hit].mean() if n_hit else float("nan")
        miss_rate = cat_gold_top[c][~is_hit].mean() if n_miss else float("nan")
        print(f"{c:10s} {hit_rate:<20.4f} {miss_rate:<20.4f} {n_hit:<6d} {n_miss:<6d}")

    return {
        "n": n,
        "hit_at_1": float(is_hit.mean()),
        "miss_at_1": float((~is_hit).mean()),
        "per_category": {
            c: {
                "hit_side_gold_win_rate": float(cat_gold_top[c][is_hit].mean()) if n_hit else None,
                "miss_side_gold_win_rate": float(cat_gold_top[c][~is_hit].mean()) if n_miss else None,
            }
            for c in ("thought", "emotion", "behavior")
        },
    }


def stratified_kfold_indices(items: list[dict], k: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    by_pattern: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        by_pattern.setdefault(item["gold_pattern_id"], []).append(i)

    folds: list[list[int]] = [[] for _ in range(k)]
    for pat, idxs in by_pattern.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)

    for f in folds:
        f.sort()
    return folds


def weight_grid(step: float) -> list[tuple[float, float, float]]:
    steps = round(1.0 / step)
    grid = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            t = i * step
            e = j * step
            b = 1.0 - t - e
            if b < -1e-9:
                continue
            grid.append((round(t, 4), round(e, 4), round(max(b, 0.0), 4)))
    return grid


def tune_weights(train_scored: list[dict], grid: list[tuple[float, float, float]], anchor=TIE_BREAK_ANCHOR) -> tuple[tuple[float, float, float], float]:
    anchor_arr = np.array(anchor)
    best_p1 = -1.0
    best_candidates = []
    for w in grid:
        p1 = precision_at_k(train_scored, w, 1)
        if p1 > best_p1 + 1e-12:
            best_p1 = p1
            best_candidates = [w]
        elif abs(p1 - best_p1) <= 1e-12:
            best_candidates.append(w)

    best_w = min(best_candidates, key=lambda w: np.linalg.norm(np.array(w) - anchor_arr))
    return best_w, best_p1


def main():
    val_items = json.load(open(VAL_PATH, encoding="utf-8"))
    assert len(val_items) == 100
    scored = compute_per_item_scores(val_items)

    step1 = report_step1(scored)

    print(f"\n=== STEP 2: hypothesis ===")
    print(
        "hit 항목에서 thought 단독 gold 승률"
        f" ({step1['per_category']['thought']['hit_side_gold_win_rate']:.4f})이"
        f" emotion({step1['per_category']['emotion']['hit_side_gold_win_rate']:.4f})/"
        f"behavior({step1['per_category']['behavior']['hit_side_gold_win_rate']:.4f})보다 뚜렷하게 높음"
        " -> baseline(1/3,1/3,1/3) 가중치는 thought의 실제 기여도를 과소평가하고 있을 가능성이 있고,"
        " emotion/behavior 비중을 낮추고 thought 비중을 높이는 방향으로 튜닝하면 precision@1이 개선될 것이라는 가설."
    )

    print(f"\n=== STEP 3: stratified {K}-fold CV (seed={SPLIT_SEED}, grid step={GRID_STEP}, tie-break anchor={TIE_BREAK_ANCHOR}) ===")
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

        tuned_w, train_p1 = tune_weights(train_scored, grid)

        test_p1_tuned = precision_at_k(test_scored, tuned_w, 1)
        test_p2_tuned = precision_at_k(test_scored, tuned_w, 2)
        test_p1_baseline = precision_at_k(test_scored, BASELINE_WEIGHTS, 1)
        test_p2_baseline = precision_at_k(test_scored, BASELINE_WEIGHTS, 2)

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
            "train_precision@1": train_p1,
            "test_precision@1_tuned": test_p1_tuned,
            "test_precision@1_baseline": test_p1_baseline,
            "test_precision@2_tuned": test_p2_tuned,
            "test_precision@2_baseline": test_p2_baseline,
        })

    print(f"\n{'fold':5s} {'n_tr':5s} {'n_te':5s} {'tuned w(t/e/b)':18s} {'train p@1':10s} {'test p@1 tuned':16s} {'test p@1 base':16s} {'test p@2 tuned':16s} {'test p@2 base':16s}")
    for r in fold_results:
        w = r["tuned_weights"]
        w_str = f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"
        print(
            f"{r['fold']:<5d} {r['n_train']:<5d} {r['n_test']:<5d} {w_str:18s} "
            f"{r['train_precision@1']:<10.4f} {r['test_precision@1_tuned']:<16.4f} "
            f"{r['test_precision@1_baseline']:<16.4f} {r['test_precision@2_tuned']:<16.4f} "
            f"{r['test_precision@2_baseline']:<16.4f}"
        )

    mean_train_p1 = np.mean([r["train_precision@1"] for r in fold_results])
    mean_test_p1_tuned = np.mean([r["test_precision@1_tuned"] for r in fold_results])
    mean_test_p1_baseline = np.mean([r["test_precision@1_baseline"] for r in fold_results])
    mean_test_p2_tuned = np.mean([r["test_precision@2_tuned"] for r in fold_results])
    mean_test_p2_baseline = np.mean([r["test_precision@2_baseline"] for r in fold_results])
    std_test_p1_tuned = np.std([r["test_precision@1_tuned"] for r in fold_results], ddof=1)
    std_test_p1_baseline = np.std([r["test_precision@1_baseline"] for r in fold_results], ddof=1)

    print(f"\n=== average across {K} folds ===")
    print(f"mean train precision@1        = {mean_train_p1:.4f}")
    print(f"mean held-out precision@1 (tuned)    = {mean_test_p1_tuned:.4f} +- {std_test_p1_tuned:.4f}")
    print(f"mean held-out precision@1 (baseline) = {mean_test_p1_baseline:.4f} +- {std_test_p1_baseline:.4f}")
    print(f"mean held-out precision@2 (tuned)    = {mean_test_p2_tuned:.4f}")
    print(f"mean held-out precision@2 (baseline) = {mean_test_p2_baseline:.4f}")

    print(f"\n=== pooled out-of-fold precision@1 over all 100 val items ===")
    print(f"tuned (fold-specific weights)   : {all_oof_tuned_hit.sum()}/{len(val_items)} = {all_oof_tuned_hit.mean():.4f}")
    print(f"baseline weights                : {all_oof_baseline_hit.sum()}/{len(val_items)} = {all_oof_baseline_hit.mean():.4f}")

    final_w, final_p1 = tune_weights(scored, grid)
    print(f"\n=== weights tuned on ALL 100 val items (candidate final hyperparameter) ===")
    print(f"weights = thought={final_w[0]:.2f}, emotion={final_w[1]:.2f}, behavior={final_w[2]:.2f}  (train precision@1={final_p1:.4f})")
    print("NOTE: test_set was not touched anywhere in this script.")

    out = {
        "step1": step1,
        "kfold": {
            "k": K,
            "split_seed": SPLIT_SEED,
            "grid_step": GRID_STEP,
            "fold_results": [
                {**r, "tuned_weights": list(r["tuned_weights"])} for r in fold_results
            ],
            "mean_train_precision@1": float(mean_train_p1),
            "mean_test_precision@1_tuned": float(mean_test_p1_tuned),
            "mean_test_precision@1_baseline": float(mean_test_p1_baseline),
            "mean_test_precision@2_tuned": float(mean_test_p2_tuned),
            "mean_test_precision@2_baseline": float(mean_test_p2_baseline),
            "pooled_oof_precision@1_tuned": float(all_oof_tuned_hit.mean()),
            "pooled_oof_precision@1_baseline": float(all_oof_baseline_hit.mean()),
        },
        "final_weights_fit_on_all_val": {
            "weights": list(final_w),
            "train_precision@1": final_p1,
        },
    }
    out_path = ROOT / "kfold_cv_report.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
