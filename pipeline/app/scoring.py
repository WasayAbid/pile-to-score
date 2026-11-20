"""Map YOLO class names to 0–100 freshness scores per fruit type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FruitType = Literal["banana", "mango", "tomato", "apple"]

# Bump when score tables / guards change — demo prints this so you know the server reloaded.
SCORING_VERSION = "2026-05-25-banana-spoil-tune-v6"

# Legacy single-fruit / fallback point tables (not used for banana/mango basket scoring).
BANANA_CLASS_SCORES: dict[str, int] = {
    "unripe": 62,
    "freshunripe": 72,
    "under-ripened": 62,
    "under_ripened": 62,
    "ripe": 90,
    "freshripe": 100,
    "ripened": 90,
    "overripe": 38,
    "over-ripened": 38,
    "over_ripened": 38,
    "rotten": 12,
}

# Vendor lens: ripe = best to sell today; unripe = still fresh but not peak.
# (Old map had unripe 92 > ripe 85, so green mangoes always looked "fresher" than yellow.)
MANGO_CLASS_SCORES: dict[str, int] = {
    "unripe": 72,
    "raw": 72,
    "partially_ripe": 78,
    "partial": 78,
    "semi_ripe": 78,
    "ripe": 88,
    "overripe": 38,
    "over_ripe": 38,
    "rotten": 12,
    "mature": 80,
}

# Laboro / vendor: green → unripe, half_ripened → partially_ripe, fully_ripened → ripe
# Kaggle / VegNet: unripe · ripe · old · damaged (model may still emit legacy names).
TOMATO_CLASS_SCORES: dict[str, int] = {
    "unripe": 72,
    "green": 72,
    "ripe": 88,
    "old": 38,
    "damaged": 12,
}

# Map YOLO folder names from Colab training → product class names.
TOMATO_MODEL_TO_CANONICAL: dict[str, str] = {
    "unripe": "unripe",
    "green": "unripe",
    "ripe": "ripe",
    "old": "old",
    "overripe": "old",
    "over_ripe": "old",
    "damaged": "damaged",
    "rotten": "damaged",
}

APPLE_CLASS_SCORES: dict[str, int] = {
    "unripe": 88,
    "green": 88,
    "ripe": 84,
    "mature": 80,
    "overripe": 45,
    "rotten": 15,
    "fresh": 90,
    "stale": 35,
}

SCORE_TABLES: dict[FruitType, dict[str, int]] = {
    "banana": BANANA_CLASS_SCORES,
    "mango": MANGO_CLASS_SCORES,
    "tomato": TOMATO_CLASS_SCORES,
    "apple": APPLE_CLASS_SCORES,
}

MIN_CLASSIFY_CONFIDENCE = 0.55
ROTTEN_CAP_THRESHOLD = 60
_BAD_RIPENESS = frozenset({"rotten", "overripe", "over_ripe", "over_ripened"})
BANANA_POINTS_PER_PIECE_MAX = 100

# Basket (stall seg + cls) — vendor retake rules (banana + mango)
BANANA_MIN_PIECE_COUNT = 3
BANANA_MIN_MEAN_CONFIDENCE = 0.38
BANANA_LOW_FRESHNESS_WARN = 25
MANGO_MIN_PIECE_COUNT = 3
MANGO_MIN_MEAN_CONFIDENCE = 0.38
MANGO_LOW_FRESHNESS_WARN = 25
TOMATO_MIN_PIECE_COUNT = 3
TOMATO_MIN_MEAN_CONFIDENCE = 0.38
TOMATO_LOW_FRESHNESS_WARN = 25
APPLE_MIN_PIECE_COUNT = 3
APPLE_MIN_MEAN_CONFIDENCE = 0.38
APPLE_LOW_FRESHNESS_WARN = 25

BASKET_MIN_PIECE_COUNT: dict[FruitType, int] = {
    "banana": BANANA_MIN_PIECE_COUNT,
    "mango": MANGO_MIN_PIECE_COUNT,
    "tomato": TOMATO_MIN_PIECE_COUNT,
    "apple": APPLE_MIN_PIECE_COUNT,
}
BASKET_MIN_MEAN_CONFIDENCE: dict[FruitType, float] = {
    "banana": BANANA_MIN_MEAN_CONFIDENCE,
    "mango": MANGO_MIN_MEAN_CONFIDENCE,
    "tomato": TOMATO_MIN_MEAN_CONFIDENCE,
    "apple": APPLE_MIN_MEAN_CONFIDENCE,
}
BASKET_LOW_FRESHNESS_WARN: dict[FruitType, int] = {
    "banana": BANANA_LOW_FRESHNESS_WARN,
    "mango": MANGO_LOW_FRESHNESS_WARN,
    "tomato": TOMATO_LOW_FRESHNESS_WARN,
    "apple": APPLE_LOW_FRESHNESS_WARN,
}

# Basket scoring matrix (banana + mango): weight × confidence, then spoilage penalty.
# Ripe 1.0 · partially 0.85 · early 0.75 · unripe 0.55 · overripe/rotten 0.15
BANANA_BASKET_CLASS_WEIGHT: dict[str, float] = {
    "ripe": 1.00,
    "freshripe": 1.00,
    "fresh_ripe": 1.00,
    "ripened": 1.00,
    "fresh": 1.00,
    "partially_ripe": 0.85,
    "partial": 0.85,
    "freshunripe": 0.75,
    "early_ripe": 0.75,
    "earlyripe": 0.75,
    "under_ripened": 0.75,
    "unripe": 0.55,
    "green": 0.55,
    # Overripe = brown specks, still often sellable (matches legacy ~38/100 sell pts).
    "overripe": 0.38,
    "over_ripe": 0.38,
    "over_ripened": 0.38,
    "rotten": 0.25,
}

# Same weighted basket tiers as mango: ripe 1.0 · unripe 0.55 · old/damaged 0.15
TOMATO_BASKET_CLASS_WEIGHT: dict[str, float] = {
    "ripe": 1.00,
    "unripe": 0.55,
    "green": 0.55,
    "old": 0.15,
    "damaged": 0.15,
}

MANGO_BASKET_CLASS_WEIGHT: dict[str, float] = {
    "ripe": 1.00,
    "freshripe": 1.00,
    "fresh_ripe": 1.00,
    "ripened": 1.00,
    "partially_ripe": 0.85,
    "partial": 0.85,
    "partial_ripe": 0.85,
    "semi_ripe": 0.85,
    "early_ripe": 0.75,
    "earlyripe": 0.75,
    "mature": 0.75,
    "unripe": 0.55,
    "raw": 0.55,
    "green": 0.55,
    "overripe": 0.15,
    "over_ripe": 0.15,
    "over_ripened": 0.15,
    "rotten": 0.15,
}

APPLE_BASKET_CLASS_WEIGHT: dict[str, float] = {
    "fresh": 1.00,
    "healthy": 1.00,
    "good": 1.00,
    "ripe": 1.00,
    "rotten": 0.15,
    "stale": 0.15,
    "bad": 0.15,
}

BASKET_CLASS_WEIGHTS: dict[FruitType, dict[str, float]] = {
    "banana": BANANA_BASKET_CLASS_WEIGHT,
    "mango": MANGO_BASKET_CLASS_WEIGHT,
    "tomato": TOMATO_BASKET_CLASS_WEIGHT,
    "apple": APPLE_BASKET_CLASS_WEIGHT,
}

# Mango: overripe + rotten both trigger extra basket penalty.
BASKET_SPOIL_PENALTY_CLASSES: dict[FruitType, frozenset[str]] = {
    "banana": frozenset({"rotten"}),
    "mango": frozenset({"overripe", "over_ripe", "over_ripened", "rotten"}),
    "tomato": frozenset({"old", "damaged", "overripe", "over_ripe", "rotten"}),
    "apple": frozenset({"rotten", "stale"}),
}

# Banana: softer penalty tiers — overripe already discounted in class weights.
BANANA_SPOIL_PENALTY_MAX = 12


def _lookup_basket_weight(
    table: dict[str, float], class_name: str, *, exact_only: bool = False
) -> float:
    key = normalize_class(class_name)
    if key in table:
        return table[key]
    if exact_only:
        return 0.70
    for partial in sorted(table.keys(), key=len, reverse=True):
        if partial in key or key in partial:
            return table[partial]
    return 0.70


def basket_class_freshness_weight(fruit_type: FruitType, class_name: str) -> float:
    if fruit_type == "tomato":
        class_name = canonical_tomato_class(class_name)
        return _lookup_basket_weight(
            BASKET_CLASS_WEIGHTS[fruit_type], class_name, exact_only=True
        )
    return _lookup_basket_weight(BASKET_CLASS_WEIGHTS[fruit_type], class_name)


def basket_piece_weighted_contribution(
    fruit_type: FruitType, class_name: str, confidence: float
) -> float:
    conf = max(0.0, min(1.0, float(confidence)))
    return basket_class_freshness_weight(fruit_type, class_name) * conf


def basket_spoil_penalty(fruit_type: FruitType, detections: list[DetectionScore]) -> int:
    """Extra basket penalty from worst-class share (on top of low class weights)."""
    n = len(detections)
    if not n:
        return 0
    spoil = BASKET_SPOIL_PENALTY_CLASSES[fruit_type]
    spoil_n = sum(
        1
        for d in detections
        if (
            canonical_tomato_class(d.class_name)
            if fruit_type == "tomato"
            else normalize_class(d.class_name)
        )
        in spoil
    )
    pct = 100 * spoil_n / n
    if fruit_type == "banana":
        if pct <= 10:
            return 0
        if pct <= 22:
            return 4
        if pct <= 35:
            return 8
        return BANANA_SPOIL_PENALTY_MAX
    if pct <= 5:
        return 0
    if pct <= 15:
        return 5
    if pct <= 25:
        return 10
    return 20


def basket_freshness_tier(score: int) -> str:
    if score >= 90:
        return "Premium Fresh"
    if score >= 75:
        return "Fresh"
    if score >= 60:
        return "Moderately Fresh"
    if score >= 40:
        return "Low Freshness"
    return "Poor Quality"


def _basket_weighted_score(
    fruit_type: FruitType,
    detections: list[DetectionScore],
) -> tuple[int, int, int, list[int], str]:
    """(final, base, penalty, per_piece_pct, detail) — mean(weight×conf)×100 − spoil penalty."""
    n = len(detections)
    if not n:
        return 0, 0, 0, [], ""
    spoil = BASKET_SPOIL_PENALTY_CLASSES[fruit_type]
    contribs = [
        basket_piece_weighted_contribution(fruit_type, d.class_name, d.confidence)
        for d in detections
    ]
    base = round(100 * sum(contribs) / n)
    penalty = basket_spoil_penalty(fruit_type, detections)
    final = max(0, min(100, base - penalty))
    per_piece = [round(c * 100) for c in contribs]
    spoil_n = sum(
        1
        for d in detections
        if (
            canonical_tomato_class(d.class_name)
            if fruit_type == "tomato"
            else normalize_class(d.class_name)
        )
        in spoil
    )
    detail = f"base {base}%"
    if penalty:
        label = "rotten penalty" if fruit_type == "banana" else "spoilage penalty"
        detail += f" - {penalty} {label} ({spoil_n}/{n})"
    return final, base, penalty, per_piece, detail


# Mango aliases (inference imports)
def mango_class_freshness_weight(class_name: str) -> float:
    return basket_class_freshness_weight("mango", class_name)


def mango_piece_weighted_contribution(class_name: str, confidence: float) -> float:
    return basket_piece_weighted_contribution("mango", class_name, confidence)


def _basket_sum_percent(detections: list[DetectionScore]) -> tuple[int, int, int, int]:
    """
    Sum each piece's class score, then express as % of (n * 100).
    Example: 2 unripe (62) + 2 ripe (88) → 300 / 400 → 75%.
    """
    n = len(detections)
    total = sum(d.freshness for d in detections)
    max_possible = n * BANANA_POINTS_PER_PIECE_MAX
    pct = round(100 * total / max_possible) if max_possible else 0
    return pct, total, max_possible, n


def _banana_sum_percent(detections: list[DetectionScore]) -> tuple[int, int, int, int]:
    return _basket_sum_percent(detections)


@dataclass
class DetectionScore:
    class_name: str
    confidence: float
    freshness: int


@dataclass
class FreshnessResult:
    freshness_score: int
    piece_count: int
    mean_confidence: float
    needs_retake: bool
    method: str
    class_histogram: dict[str, int]
    per_piece_scores: list[int]
    model_version: str
    ripeness_label: str = ""
    freshness_tier: str = ""
    scoring_detail: str = ""
    retake_reason: str | None = None
    retake_severity: str = "ok"


def basket_retake_assessment(
    fruit_type: FruitType,
    *,
    piece_count: int,
    mean_confidence: float,
    freshness_score: int,
    stall_weights_present: bool,
) -> tuple[bool, str | None, str]:
    """Returns (needs_retake, retake_reason, severity). severity: block | warn | ok"""
    if not stall_weights_present:
        return True, "model_unavailable", "block"
    if piece_count == 0:
        return True, "no_segments", "block"
    min_pieces = BASKET_MIN_PIECE_COUNT.get(fruit_type, MANGO_MIN_PIECE_COUNT)
    min_conf = BASKET_MIN_MEAN_CONFIDENCE.get(fruit_type, MANGO_MIN_MEAN_CONFIDENCE)
    low_fresh = BASKET_LOW_FRESHNESS_WARN.get(fruit_type, MANGO_LOW_FRESHNESS_WARN)
    if piece_count < min_pieces:
        return True, "too_few_segments", "block"
    if mean_confidence < min_conf:
        return True, "low_confidence", "block"
    if freshness_score <= low_fresh:
        return False, "low_freshness", "warn"
    return False, None, "ok"


def banana_retake_assessment(
    *,
    piece_count: int,
    mean_confidence: float,
    freshness_score: int,
    stall_weights_present: bool,
) -> tuple[bool, str | None, str]:
    return basket_retake_assessment(
        "banana",
        piece_count=piece_count,
        mean_confidence=mean_confidence,
        freshness_score=freshness_score,
        stall_weights_present=stall_weights_present,
    )


def mango_retake_assessment(
    *,
    piece_count: int,
    mean_confidence: float,
    freshness_score: int,
    stall_weights_present: bool,
) -> tuple[bool, str | None, str]:
    return basket_retake_assessment(
        "mango",
        piece_count=piece_count,
        mean_confidence=mean_confidence,
        freshness_score=freshness_score,
        stall_weights_present=stall_weights_present,
    )


def tomato_retake_assessment(
    *,
    piece_count: int,
    mean_confidence: float,
    freshness_score: int,
    stall_weights_present: bool,
) -> tuple[bool, str | None, str]:
    return basket_retake_assessment(
        "tomato",
        piece_count=piece_count,
        mean_confidence=mean_confidence,
        freshness_score=freshness_score,
        stall_weights_present=stall_weights_present,
    )


def apple_retake_assessment(
    *,
    piece_count: int,
    mean_confidence: float,
    freshness_score: int,
    stall_weights_present: bool,
) -> tuple[bool, str | None, str]:
    return basket_retake_assessment(
        "apple",
        piece_count=piece_count,
        mean_confidence=mean_confidence,
        freshness_score=freshness_score,
        stall_weights_present=stall_weights_present,
    )


def normalize_class(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def canonical_tomato_class(name: str) -> str:
    """Map YOLO outputs (e.g. rotten/overripe) → unripe · ripe · old · damaged."""
    key = normalize_class(name)
    return TOMATO_MODEL_TO_CANONICAL.get(key, key)


def tomato_ripeness_display(class_name: str) -> str:
    labels = {
        "unripe": "Unripe",
        "ripe": "Ripe",
        "old": "Old",
        "damaged": "Damaged",
    }
    return labels.get(canonical_tomato_class(class_name), class_name.replace("_", " ").title())


def apple_freshness_display(class_name: str) -> str:
    labels = {
        "fresh": "Fresh",
        "rotten": "Rotten",
        "stale": "Stale",
    }
    key = normalize_class(class_name)
    return labels.get(key, key.replace("_", " ").title())


def class_to_freshness(fruit_type: FruitType, class_name: str) -> int:
    key = normalize_class(class_name)
    if fruit_type == "tomato":
        key = canonical_tomato_class(key)
    table = SCORE_TABLES[fruit_type]
    if key in table:
        return table[key]
    if fruit_type == "tomato":
        return 70
    # Longest label first so "ripe" does not match inside "unripe".
    for partial in sorted(table.keys(), key=len, reverse=True):
        if partial in key or key in partial:
            return table[partial]
    return 70


def aggregate_detections(
    fruit_type: FruitType,
    detections: list[DetectionScore],
    *,
    model_version: str = "unknown",
    method: str = "yolo_detect",
    basket_scoring: Literal["sum_pct", "weighted"] | None = None,
) -> FreshnessResult:
    if not detections:
        return FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=True,
            method=method,
            class_histogram={},
            per_piece_scores=[],
            model_version=model_version,
            retake_reason="no_segments",
            retake_severity="block",
        )

    scores = [d.freshness for d in detections]
    confidences = [d.confidence for d in detections]
    mean_conf = sum(confidences) / len(confidences)
    hist: dict[str, int] = {}
    for d in detections:
        label = (
            canonical_tomato_class(d.class_name)
            if fruit_type == "tomato"
            else d.class_name
        )
        hist[label] = hist.get(label, 0) + 1

    per_piece_scores = scores
    ripeness_summary = ""
    freshness_tier = ""
    scoring_detail = ""

    use_weighted = basket_scoring == "weighted" or (
        basket_scoring is None
        and fruit_type in ("banana", "mango", "tomato")
        and "basket_seg" in method
    )
    use_sum_pct = basket_scoring == "sum_pct"

    if use_sum_pct:
        mean_score, _total_pts, _max_pts, _n = _basket_sum_percent(detections)
        method = f"{method}+sum_pct"
    elif use_weighted:
        mean_score, base, penalty, per_piece_scores, scoring_detail = (
            _basket_weighted_score(fruit_type, detections)
        )
        method = f"{method}+weighted_conf"
        freshness_tier = basket_freshness_tier(mean_score)
        ripeness_summary = freshness_tier
    elif fruit_type == "mango":
        mean_score = round(sum(scores) / len(scores))
        has_bad = any(normalize_class(d.class_name) in _BAD_RIPENESS for d in detections)
        if has_bad:
            mean_score = min(mean_score, ROTTEN_CAP_THRESHOLD)
    else:
        mean_score = round(sum(scores) / len(scores))
        has_bad = any(normalize_class(d.class_name) in _BAD_RIPENESS for d in detections)
        if has_bad:
            mean_score = min(mean_score, ROTTEN_CAP_THRESHOLD)

    needs_retake = False
    retake_reason: str | None = None
    retake_severity = "ok"
    if fruit_type == "banana":
        needs_retake, retake_reason, retake_severity = banana_retake_assessment(
            piece_count=len(detections),
            mean_confidence=mean_conf,
            freshness_score=mean_score,
            stall_weights_present=True,
        )
    elif fruit_type == "mango":
        needs_retake, retake_reason, retake_severity = mango_retake_assessment(
            piece_count=len(detections),
            mean_confidence=mean_conf,
            freshness_score=mean_score,
            stall_weights_present=True,
        )
    elif fruit_type == "tomato":
        needs_retake, retake_reason, retake_severity = tomato_retake_assessment(
            piece_count=len(detections),
            mean_confidence=mean_conf,
            freshness_score=mean_score,
            stall_weights_present=True,
        )

    if ripeness_summary:
        ripeness_label = ripeness_summary
    else:
        dominant = max(hist.keys(), key=lambda k: hist[k]) if hist else ""
        if fruit_type == "tomato" and dominant:
            ripeness_label = tomato_ripeness_display(dominant)
        else:
            ripeness_label = _ripeness_display(dominant) if dominant else ""

    return FreshnessResult(
        freshness_score=max(0, min(100, mean_score)),
        piece_count=len(detections),
        mean_confidence=round(mean_conf, 3),
        needs_retake=needs_retake,
        method=method,
        class_histogram=hist,
        per_piece_scores=per_piece_scores,
        model_version=model_version,
        ripeness_label=ripeness_label,
        freshness_tier=freshness_tier,
        scoring_detail=scoring_detail,
        retake_reason=retake_reason,
        retake_severity=retake_severity,
    )


def _ripeness_display(class_name: str) -> str:
    key = normalize_class(class_name)
    return key.replace("_", " ").title()


def aggregate_classification(
    fruit_type: FruitType,
    class_name: str,
    confidence: float,
    *,
    model_version: str = "unknown",
) -> FreshnessResult:
    label = normalize_class(class_name)
    if fruit_type == "tomato":
        label = canonical_tomato_class(label)
    score = class_to_freshness(fruit_type, label)
    needs_retake = confidence < MIN_CLASSIFY_CONFIDENCE
    method = "yolo_classify" if model_version != "heuristic_v1" else "heuristic_hsv"
    return FreshnessResult(
        freshness_score=score,
        piece_count=1,
        mean_confidence=round(confidence, 3),
        needs_retake=needs_retake,
        method=method,
        class_histogram={label: 1},
        per_piece_scores=[score],
        model_version=model_version,
        ripeness_label=(
            tomato_ripeness_display(label)
            if fruit_type == "tomato"
            else _ripeness_display(label)
        ),
    )
