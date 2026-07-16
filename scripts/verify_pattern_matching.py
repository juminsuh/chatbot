"""Verifies that every pattern in pattern_catalog loads and embeds correctly:
the matching_features.thought list, emotion_text, and behavior_text fields
(not the old top-level thought/emotion/behavior clue lists) each produce a
vector, and match_pattern() runs end-to-end against a sample user utterance.

Run with: python scripts/verify_pattern_matching.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import PATTERN_CATALOG, PATTERN_IDS
import pattern_matching


def check_catalog_shape() -> None:
    print(f"[1] pattern_catalog entries: {len(PATTERN_IDS)}")
    for pid in PATTERN_IDS:
        features = PATTERN_CATALOG[pid]["matching_features"]
        thought_list = features.get("thought")
        emotion_text = features.get("emotion_text")
        behavior_text = features.get("behavior_text")

        assert isinstance(thought_list, list) and len(thought_list) > 0, f"{pid}: matching_features.thought missing/empty"
        assert isinstance(emotion_text, str) and emotion_text.strip(), f"{pid}: matching_features.emotion_text missing/empty"
        assert isinstance(behavior_text, str) and behavior_text.strip(), f"{pid}: matching_features.behavior_text missing/empty"

        print(f"  - {pid}: thought x{len(thought_list)}, emotion_text ok, behavior_text ok")
    print("  all patterns have well-formed matching_features\n")


def check_embeddings_cached() -> None:
    print("[2] precomputing/caching pattern-side embeddings via warmup()...")
    pattern_matching.warmup()
    thought_vecs, emotion_vecs, behavior_vecs = pattern_matching._get_pattern_vecs()

    n_thought_phrases = len(pattern_matching._THOUGHT_TEXTS)
    assert thought_vecs.shape[0] == n_thought_phrases, "thought embedding row count mismatch"
    assert emotion_vecs.shape[0] == len(PATTERN_IDS), "emotion embedding row count mismatch"
    assert behavior_vecs.shape[0] == len(PATTERN_IDS), "behavior embedding row count mismatch"
    assert thought_vecs.shape[1] == emotion_vecs.shape[1] == behavior_vecs.shape[1], "embedding dims mismatch"

    for pid in PATTERN_IDS:
        rows = pattern_matching._THOUGHT_ROWS_BY_PATTERN[pid]
        assert len(rows) == len(PATTERN_CATALOG[pid]["matching_features"]["thought"]), (
            f"{pid}: thought row mapping count mismatch"
        )

    print(f"  thought_vecs shape={thought_vecs.shape}, emotion_vecs shape={emotion_vecs.shape}, "
          f"behavior_vecs shape={behavior_vecs.shape}")
    print("  every pattern's thought/emotion/behavior fields embedded and row-mapped correctly\n")


def check_match_pattern_end_to_end() -> None:
    print("[3] running match_pattern() against a sample consolidated user text...")
    sample_thought = "손을 씻지 않으면 큰일이 날 것 같다는 생각이 자꾸 든다"
    sample_emotion = "찝찝하고 불안한 감정이 계속된다"
    sample_behavior = "하루에도 몇십 번씩 손을 씻고 문단속을 반복해서 확인한다"

    result = pattern_matching.match_pattern(sample_thought, sample_emotion, sample_behavior)

    assert result["pattern_id"] in PATTERN_IDS, "match_pattern returned an unknown pattern_id"
    assert len(result["scores"]) == len(PATTERN_IDS), "scores list should cover every pattern"
    ids_seen = {s["pattern_id"] for s in result["scores"]}
    assert ids_seen == set(PATTERN_IDS), "scores list is missing some pattern(s)"
    # sorted descending by pattern_score
    pattern_scores = [s["pattern_score"] for s in result["scores"]]
    assert pattern_scores == sorted(pattern_scores, reverse=True), "scores not sorted descending"

    print(f"  best match: {result['pattern_id']} (score={result['score']:.4f})")
    print(f"  expected best match is 'compulsive_behavior_loop' given the sample text -- "
          f"{'MATCHES' if result['pattern_id'] == 'compulsive_behavior_loop' else 'DID NOT MATCH, check WEIGHTS/text'}")
    print()


if __name__ == "__main__":
    check_catalog_shape()
    check_embeddings_cached()
    check_match_pattern_end_to_end()
    print("ALL CHECKS PASSED")
