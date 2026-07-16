import json
from pathlib import Path
from pattern_matching import match_pattern

GOLD_PATH = Path(__file__).parent / "gold_set.json"


def load_gold_set():
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


def rank_patterns(thought: str, emotion: str, behavior: str) -> list[str]:
    match = match_pattern(thought, emotion, behavior, verbose=False)
    return [s["pattern_id"] for s in match["scores"]]


def debug_miss(item: dict) -> None:
    match = match_pattern(item["thought"], item["emotion"], item["behavior"], verbose=False)
    scores = match["scores"]  # sorted desc by pattern_score
    gold_id = item["gold_pattern_id"]

    predicted = scores[0]
    gold_rank, gold = next((i, s) for i, s in enumerate(scores, start=1) if s["pattern_id"] == gold_id)

    def _fmt(label: str, s: dict) -> str:
        return (
            f"    {label} {s['pattern_id']}: pattern_score={s['pattern_score']:.4f} "
            f"(thought={s['thought_score']:.4f}, emotion={s['emotion_score']:.4f}, behavior={s['behavior_score']:.4f})"
        )

    print(f"  [DEBUG] {item['id']}")
    print(f"    thought : {item['thought']}")
    print(f"    emotion : {item['emotion']}")
    print(f"    behavior: {item['behavior']}")
    print(_fmt("predicted(1위, 오답)", predicted))
    print(_fmt(f"gold({gold_rank}위)", gold))
    print(f"    margin(1위 - gold) = {predicted['pattern_score'] - gold['pattern_score']:.4f}")


def precision_at_k(gold_set, k: int) -> dict:
    total = len(gold_set)
    hits = 0
    per_pattern = {}
    misses = []

    for item in gold_set:
        ranked = rank_patterns(item["thought"], item["emotion"], item["behavior"])
        top_k = ranked[:k]
        is_hit = item["gold_pattern_id"] in top_k

        pat = item["gold_pattern_id"]
        per_pattern.setdefault(pat, {"hits": 0, "total": 0})
        per_pattern[pat]["total"] += 1
        if is_hit:
            hits += 1
            per_pattern[pat]["hits"] += 1
        else:
            misses.append({
                "id": item["id"],
                "gold_pattern_id": pat,
                "predicted": ranked[0],
                "top_k": top_k,
            })

    overall = hits / total if total else 0.0
    per_pattern_rate = {
        pat: (v["hits"] / v["total"] if v["total"] else 0.0)
        for pat, v in per_pattern.items()
    }
    return {
        "k": k,
        "overall_precision_at_k": overall,
        "per_pattern": per_pattern_rate,
        "misses": misses,
    }


def main():
    gold_set = load_gold_set()
    easy_set = [g for g in gold_set if g.get("difficulty") != "hard"]
    hard_set = [g for g in gold_set if g.get("difficulty") == "hard"]

    for k in (1, 2, 3):
        result = precision_at_k(easy_set, k)
        print(f"\n=== precision@{k} (main set, n={len(easy_set)}): {result['overall_precision_at_k']:.3f} ===")
        for pat, rate in sorted(result["per_pattern"].items()):
            print(f"  {pat}: {rate:.3f}")
        if result["misses"]:
            print(f"  -- misses (오답으로 예측된 패턴) --")
            for m in result["misses"]:
                print(f"     {m['id']}: gold={m['gold_pattern_id']}  predicted={m['predicted']}  top{k}={m['top_k']}")
            if k == 1:
                # miss@k sets are nested (miss@3 ⊆ miss@2 ⊆ miss@1) since there's
                # exactly one gold pattern per item, so the full score breakdown
                # only needs to run once, here.
                print(f"  -- score breakdown for each miss --")
                item_by_id = {item["id"]: item for item in easy_set}
                for m in result["misses"]:
                    debug_miss(item_by_id[m["id"]])

    if hard_set:
        for k in (1, 3):
            result = precision_at_k(hard_set, k)
            print(f"\n--- hard cases precision@{k} (n={len(hard_set)}): {result['overall_precision_at_k']:.3f} ---")
            for item in hard_set:
                ranked = rank_patterns(item["thought"], item["emotion"], item["behavior"])
                hit = item["gold_pattern_id"] in ranked[:k]
                print(f"  {item['id']}: gold={item['gold_pattern_id']} top1={ranked[0]} hit@{k}={hit}")
            if k == 1 and result["misses"]:
                print(f"  -- score breakdown for each miss --")
                item_by_id = {item["id"]: item for item in hard_set}
                for m in result["misses"]:
                    debug_miss(item_by_id[m["id"]])


if __name__ == "__main__":
    main()
