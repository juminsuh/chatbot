import hashlib
from pathlib import Path

import numpy as np

from data import PATTERN_CATALOG, PATTERN_IDS
from embeddings import embed

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "embeddings"

WEIGHTS = {"thought": 1 / 3, "emotion": 1 / 3, "behavior": 1 / 3}
_THOUGHT_PATTERN_IDS: list[str] = []
_THOUGHT_TEXTS: list[str] = []
for _pid in PATTERN_IDS:
    for _text in PATTERN_CATALOG[_pid]["matching_features"]["thought"]:
        _THOUGHT_PATTERN_IDS.append(_pid)
        _THOUGHT_TEXTS.append(_text)

_THOUGHT_ROWS_BY_PATTERN: dict[str, list[int]] = {pid: [] for pid in PATTERN_IDS}
for _row, _pid in enumerate(_THOUGHT_PATTERN_IDS):
    _THOUGHT_ROWS_BY_PATTERN[_pid].append(_row)

_EMOTION_TEXTS: list[str] = [PATTERN_CATALOG[pid]["matching_features"]["emotion_text"] for pid in PATTERN_IDS]
_BEHAVIOR_TEXTS: list[str] = [PATTERN_CATALOG[pid]["matching_features"]["behavior_text"] for pid in PATTERN_IDS]


def _corpus_digest(texts: list[str]) -> str:
    joined = "\x1f".join(texts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def _load_or_compute(texts: list[str], cache_name: str) -> np.ndarray:
    digest = _corpus_digest(texts)
    cache_path = CACHE_DIR / f"{cache_name}_{digest}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    vecs = embed(texts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CACHE_DIR.glob(f"{cache_name}_*.npy"):
        stale.unlink()
    np.save(cache_path, vecs)
    return vecs


_thought_vecs = None
_emotion_vecs = None
_behavior_vecs = None


def _get_pattern_vecs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _thought_vecs, _emotion_vecs, _behavior_vecs
    if _thought_vecs is None:
        _thought_vecs = _load_or_compute(_THOUGHT_TEXTS, "pattern_thought")
        _emotion_vecs = _load_or_compute(_EMOTION_TEXTS, "pattern_emotion")
        _behavior_vecs = _load_or_compute(_BEHAVIOR_TEXTS, "pattern_behavior")
    return _thought_vecs, _emotion_vecs, _behavior_vecs


def warmup() -> None:
    _get_pattern_vecs()


def match_pattern(thought_text: str, emotion_text: str, behavior_text: str, verbose: bool = True) -> dict:
    thought_vecs, emotion_vecs, behavior_vecs = _get_pattern_vecs()
    user_thought_vec, user_emotion_vec, user_behavior_vec = embed(
        [thought_text, emotion_text, behavior_text]
    )

    thought_sims = thought_vecs @ user_thought_vec
    emotion_sims = emotion_vecs @ user_emotion_vec
    behavior_sims = behavior_vecs @ user_behavior_vec

    scores = []
    for i, pid in enumerate(PATTERN_IDS):
        rows = _THOUGHT_ROWS_BY_PATTERN[pid]
        thought_score = float(np.max(thought_sims[rows])) if rows else 0.0
        emotion_score = float(emotion_sims[i])
        behavior_score = float(behavior_sims[i])
        pattern_score = (
            WEIGHTS["thought"] * thought_score
            + WEIGHTS["emotion"] * emotion_score
            + WEIGHTS["behavior"] * behavior_score
        )
        scores.append({
            "pattern_id": pid,
            "thought_score": thought_score,
            "emotion_score": emotion_score,
            "behavior_score": behavior_score,
            "pattern_score": pattern_score,
        })

    scores.sort(key=lambda s: s["pattern_score"], reverse=True)

    if verbose:
        print("[PATTERN MATCH DEBUG] scores (desc):")
        for s in scores:
            print(
                f"  - {s['pattern_id']}: pattern_score={s['pattern_score']:.4f} "
                f"(thought={s['thought_score']:.4f}, emotion={s['emotion_score']:.4f}, "
                f"behavior={s['behavior_score']:.4f})"
            )
        if len(scores) > 1 and abs(scores[0]["pattern_score"] - scores[1]["pattern_score"]) < 1e-3:
            print(
                f"[PATTERN MATCH DEBUG] near-tie between top candidates "
                f"{scores[0]['pattern_id']!r} and {scores[1]['pattern_id']!r} "
                f"-- returning the highest-scoring one as-is, no tie-break threshold applied yet"
            )

    best = scores[0]
    return {"pattern_id": best["pattern_id"], "score": best["pattern_score"], "scores": scores}
