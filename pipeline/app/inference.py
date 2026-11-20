"""Run YOLO classify (banana c3a2v, mango, apple)."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from app.config import (
    APPLE_CLS_MODEL,
    BANANA_RIPENESS_C3A2V_MODEL,
    active_banana_cls_model,
    resolve_apple_cls_model,
    resolve_apple_stall_seg_model,
    resolve_mango_cls_model,
    resolve_tomato_cls_model,
)
from app.config import (
    resolve_banana_stall_seg_model,
    resolve_mango_stall_seg_model,
    resolve_tomato_stall_seg_model,
)
from app.scoring import (
    DetectionScore,
    FreshnessResult,
    FruitType,
    aggregate_classification,
    aggregate_detections,
    apple_retake_assessment,
    banana_retake_assessment,
    basket_class_freshness_weight,
    basket_piece_weighted_contribution,
    mango_retake_assessment,
    class_to_freshness,
    canonical_tomato_class,
    normalize_class,
    tomato_retake_assessment,
    tomato_ripeness_display,
    apple_freshness_display,
)

if TYPE_CHECKING:
    pass


def _freshness_strict() -> bool:
    """When set (ml demo), missing weights or failed cls → error, no heuristics."""
    return os.environ.get("FRESHNESS_STRICT", "").lower() in ("1", "true", "yes")


_yolo_models: dict[str, object] = {}
_model_mtime: dict[str, float] = {}
# Cache for _weights_fingerprint, keyed like the model cache above — /health
# is unauthenticated and calls get_inference_status() -> _weights_fingerprint
# on every hit; without this it SHA-256's every weights file (up to ~400MB
# total) on every single health check, which a load balancer polling every
# 10s (or an attacker) turns into sustained disk I/O saturation
# (PRODUCTION_READINESS.md "ML Service" P2). Recomputed only when a file's
# mtime actually changes (a real redeploy), same trigger _load_yolo already
# uses to decide whether to reload a model.
_fingerprint_cache: dict[str, tuple[float, str]] = {}

# Dark rot / mold on green mango skin (classifier only knows unripe vs ripe).
# Trigger when rot covers enough of the fruit mask, or one large black pit on frame.
MANGO_SPOILAGE_FRUIT_RATIO = 0.16
MANGO_SPOILAGE_BLOB_FRACTION = 0.02


def _weights_fingerprint(path: Path) -> str:
    if not path.is_file():
        return "missing"

    key = str(path.resolve())
    mtime = path.stat().st_mtime
    cached = _fingerprint_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()[:12]
    _fingerprint_cache[key] = (mtime, digest)
    return digest


def _load_yolo(path: Path):
    key = str(path.resolve())
    if not path.is_file():
        return None
    mtime = path.stat().st_mtime
    if key in _yolo_models and _model_mtime.get(key) == mtime:
        return _yolo_models[key]
    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        _yolo_models[key] = model
        _model_mtime[key] = mtime
        return model
    except Exception:
        return None


def _mango_spoilage_signals(bgr: np.ndarray) -> tuple[float, float]:
    """
    Returns (rot_on_fruit_ratio, largest_rot_blob_fraction_of_frame).
    Ignores white/neutral background; only counts near-black pixels on colored fruit.
    """
    h, w = bgr.shape[:2]
    roi = bgr[int(h * 0.02) : int(h * 0.98), int(w * 0.02) : int(w * 0.98)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    white_bg = (s_ch < 45) & (v_ch > 200)
    # Green / yellow / orange mango flesh (not white backdrop)
    fruit = (
        ((h_ch >= 28) & (h_ch <= 90) & (s_ch >= 30) & (v_ch >= 55))
        | ((h_ch >= 8) & (h_ch <= 28) & (s_ch >= 40) & (v_ch >= 55))
    ) & (~white_bg)

    fruit_u8 = fruit.astype(np.uint8) * 255
    fruit_d = cv2.dilate(
        fruit_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1
    )

    rot = cv2.inRange(gray, 0, 42)
    rot = cv2.bitwise_and(rot, fruit_d)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rot = cv2.morphologyEx(rot, cv2.MORPH_OPEN, k)

    fruit_px = max(int(np.count_nonzero(fruit_u8)), 1)
    rot_px = int(np.count_nonzero(rot))
    rot_ratio = rot_px / fruit_px

    total = max(roi.shape[0] * roi.shape[1], 1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(rot, connectivity=8)
    largest = 0 if n <= 1 else max(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n))
    blob_frac = largest / total
    return rot_ratio, blob_frac


def _guard_mango_spoilage(bgr: np.ndarray, result: FreshnessResult) -> FreshnessResult:
    """
    HF mango cls was trained on clean green vs ripe fruit — no rotten class.
    Large dark lesions on green mangoes are still labeled 'unripe' at high conf.
    """
    rot_ratio, blob_frac = _mango_spoilage_signals(bgr)
    spoiled = rot_ratio >= MANGO_SPOILAGE_FRUIT_RATIO or blob_frac >= MANGO_SPOILAGE_BLOB_FRACTION
    if not spoiled:
        return result

    spoiled_score = class_to_freshness("mango", "rotten")
    hist = dict(result.class_histogram)
    hist["spoiled_spots"] = 1
    model_label = next(iter(result.class_histogram), "unknown")
    return FreshnessResult(
        freshness_score=spoiled_score,
        piece_count=result.piece_count,
        mean_confidence=result.mean_confidence,
        needs_retake=True,
        method="yolo_classify+spoilage_guard",
        class_histogram=hist,
        per_piece_scores=[spoiled_score],
        model_version=result.model_version,
        ripeness_label=f"Spoiled (model said {model_label.replace('_', ' ')})",
    )


# A file well under the 12MB upload cap can still *decompress* to an
# enormous pixel grid (a highly-compressed PNG/JPEG "decompression bomb"),
# and neither cv2.imdecode nor a bare PIL .convert() guard against that —
# OpenCV has no size check at all, and cv2.imdecode decodes eagerly so
# there's no cheap way to peek its dimensions first
# (PRODUCTION_READINESS.md "ML Service" P1). PIL, unlike cv2, parses just
# the header on Image.open() without decoding pixel data, so it's used here
# as a cheap pre-check before the real decode either library would do.
_MAX_IMAGE_PIXELS = 40_000_000  # ~40MP, comfortably above any real phone photo


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            w, h = probe.size
        if w * h > _MAX_IMAGE_PIXELS:
            raise ValueError(f"Image too large: {w}x{h} exceeds {_MAX_IMAGE_PIXELS} pixel limit")
    except ValueError:
        raise
    except Exception:
        pass  # not a format PIL recognizes from the header alone — let the real decode below handle/reject it

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _heuristic_classify(fruit_type: FruitType, bgr: np.ndarray) -> FreshnessResult:
    """Color fallback for mango/apple only when cls weights are missing."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    roi = hsv[int(h * 0.15) : int(h * 0.85), int(w * 0.15) : int(w * 0.85)]
    green = cv2.inRange(roi, (35, 40, 40), (85, 255, 255))
    orange = cv2.inRange(roi, (8, 80, 80), (25, 255, 255))
    total = roi.shape[0] * roi.shape[1] or 1
    pg, po = np.count_nonzero(green) / total, np.count_nonzero(orange) / total
    if fruit_type == "mango":
        label = "unripe" if pg > 0.3 else "ripe" if po > 0.15 else "partially_ripe"
    else:
        label = "unripe" if pg > 0.25 else "ripe"
    return aggregate_classification(fruit_type, label, 0.5, model_version="heuristic_v1")


def _classification_model_path(fruit_type: FruitType) -> Path:
    if fruit_type == "banana":
        return active_banana_cls_model()
    if fruit_type == "mango":
        return resolve_mango_cls_model()
    if fruit_type == "tomato":
        return resolve_tomato_cls_model()
    return resolve_apple_cls_model()


def _banana_classify_bgr(bgr: np.ndarray) -> tuple[str, float, int]:
    """Top class label, confidence, sell-score for one banana crop."""
    path = active_banana_cls_model()
    model = _load_yolo(path)
    if model is None:
        raise FileNotFoundError(
            f"Banana model not found: {path}. "
            "Copy banana_ripeness_c3a2v.pt or banana_fresh_v2.pt into ml-service/models/."
        )
    results = model.predict(source=bgr, verbose=False)
    if not results or results[0].probs is None:
        raise RuntimeError("Banana classifier returned no prediction")
    probs = results[0].probs
    top_i = int(probs.top1)
    conf = float(probs.top1conf.item())
    names = results[0].names or {}
    label = normalize_class(str(names.get(top_i, top_i)))
    return label, conf, class_to_freshness("banana", label)


def run_classify(fruit_type: FruitType, image_bytes: bytes) -> FreshnessResult:
    bgr = _bytes_to_bgr(image_bytes)
    path = _classification_model_path(fruit_type)
    model = _load_yolo(path)

    if fruit_type == "banana":
        label, conf, _ = _banana_classify_bgr(bgr)
        return aggregate_classification(
            "banana",
            label,
            conf,
            model_version=f"banana_cls:{path.name}",
        )

    if model is None:
        if _freshness_strict():
            raise FileNotFoundError(
                f"{fruit_type} classifier not found: {path}. "
                f"Copy weights into ml-service/models/ and restart the demo server."
            )
        # Silent before this — a corrupt/absent weights file produced a
        # plausible-looking heuristic_v1 score with no error and no alert
        # (PRODUCTION_READINESS.md "ML Service" P1). FRESHNESS_STRICT
        # defaults off (not flipped here — some fruit types may legitimately
        # have no dedicated model by design, and turning this on unattended
        # risks breaking those on a guess about product intent), so this is
        # operational visibility, not a behavior change: the response
        # already marks it via model_version="heuristic_v1", this makes it
        # show up in logs/Sentry too.
        logger.warning("Freshness weights missing for %s (%s) — using heuristic_v1 fallback", fruit_type, path)
        out = _heuristic_classify(fruit_type, bgr)
    else:
        results = model.predict(source=bgr, verbose=False)
        if not results or results[0].probs is None:
            if _freshness_strict():
                raise RuntimeError(
                    f"{fruit_type} classifier returned no prediction ({path.name})."
                )
            logger.warning("Freshness classifier returned no prediction for %s (%s) — using heuristic_v1 fallback", fruit_type, path.name)
            out = _heuristic_classify(fruit_type, bgr)
        else:
            probs = results[0].probs
            top_i = int(probs.top1)
            conf = float(probs.top1conf.item())
            names = results[0].names or {}
            label = normalize_class(str(names.get(top_i, top_i)))
            out = aggregate_classification(
                fruit_type,
                label,
                conf,
                model_version=f"{fruit_type}_cls:{path.name}",
            )
    if fruit_type == "mango" and not _freshness_strict():
        out = _guard_mango_spoilage(bgr, out)
    return out


def _mango_classify_bgr(
    bgr: np.ndarray,
    *,
    apply_spoilage_guard: bool = True,
) -> tuple[str, float, int]:
    """Top class label, confidence, sell-score for one mango crop."""
    path = resolve_mango_cls_model()
    model = _load_yolo(path)
    if model is None:
        raise FileNotFoundError(
            f"Mango classifier not found: {path}. "
            "Copy mango_ripeness_cadimedia.pt into ml-service/models/mango/."
        )
    results = model.predict(source=bgr, verbose=False)
    if not results or results[0].probs is None:
        raise RuntimeError("Mango classifier returned no prediction")
    probs = results[0].probs
    top_i = int(probs.top1)
    conf = float(probs.top1conf.item())
    names = results[0].names or {}
    label = normalize_class(str(names.get(top_i, top_i)))
    cls_score = class_to_freshness("mango", label)
    if not apply_spoilage_guard:
        return label, conf, cls_score
    piece = aggregate_classification(
        "mango",
        label,
        conf,
        model_version=f"mango_cls:{path.name}",
    )
    piece = _guard_mango_spoilage(bgr, piece)
    label = next(iter(piece.class_histogram), label)
    return label, piece.mean_confidence, piece.freshness_score


def analyze_freshness(fruit_type: FruitType, image_bytes: bytes) -> FreshnessResult:
    if fruit_type == "banana":
        bgr = _bytes_to_bgr(image_bytes)
        _, agg, _ = analyze_banana_basket(bgr)
        return agg
    if fruit_type == "mango":
        bgr = _bytes_to_bgr(image_bytes)
        _, agg, _ = analyze_mango_basket(bgr)
        return agg
    if fruit_type == "tomato":
        bgr = _bytes_to_bgr(image_bytes)
        if resolve_tomato_stall_seg_model().is_file():
            _, agg, _ = analyze_tomato_basket(bgr)
            return agg
        return classify_tomato_single(bgr)
    if fruit_type == "apple":
        bgr = _bytes_to_bgr(image_bytes)
        if resolve_apple_stall_seg_model().is_file():
            _, agg, _ = analyze_apple_basket(bgr)
            return agg
        return classify_apple_single(bgr)
    raise ValueError(f"Unsupported fruit_type: {fruit_type}")


def analyze_banana_basket(bgr: np.ndarray) -> tuple[list, FreshnessResult, list[dict]]:
    """Stage 1 stall seg → per-crop c3a2v classify → basket score."""
    from app.stage1 import BananaPiece, extract_banana_crops_stall_seg

    path = active_banana_cls_model()
    cls_ok = path.is_file()
    stall_ok = resolve_banana_stall_seg_model().is_file()
    mode = "stall_seg"

    if not stall_ok or not cls_ok:
        missing = []
        if not stall_ok:
            missing.append("banana_stall_seg.pt")
        if not cls_ok:
            missing.append(path.name)
        if _freshness_strict():
            raise FileNotFoundError(
                f"Banana basket missing weights: {', '.join(missing)}"
            )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=True,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"banana_basket:missing:{','.join(missing)}",
            retake_reason="model_unavailable",
            retake_severity="block",
        )
        return [], empty, []

    pieces: list[BananaPiece] = extract_banana_crops_stall_seg(bgr)
    detections: list[DetectionScore] = []
    rows: list[dict] = []

    for idx, piece in enumerate(pieces, start=1):
        label, conf, cls_score = _banana_classify_bgr(piece.crop)
        piece_pct = round(basket_piece_weighted_contribution("banana", label, conf) * 100)
        detections.append(
            DetectionScore(class_name=label, confidence=conf, freshness=piece_pct)
        )
        rows.append(
            {
                "index": idx,
                "class_name": label,
                "ripeness": label.replace("_", " ").title(),
                "confidence": conf,
                "score": piece_pct,
                "class_weight": round(basket_class_freshness_weight("banana", label) * 100),
                "cls_score": cls_score,
            }
        )

    if not detections:
        needs_retake, reason, severity = banana_retake_assessment(
            piece_count=0,
            mean_confidence=0.0,
            freshness_score=0,
            stall_weights_present=stall_ok,
        )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=needs_retake,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"banana_basket:{path.name}:{mode}",
            retake_reason=reason,
            retake_severity=severity,
        )
        return pieces, empty, rows

    agg = aggregate_detections(
        "banana",
        detections,
        model_version=f"banana_basket:{path.name}:{mode}",
        method="basket_seg+classify",
        basket_scoring="weighted",
    )
    needs_retake, reason, severity = banana_retake_assessment(
        piece_count=agg.piece_count,
        mean_confidence=agg.mean_confidence,
        freshness_score=agg.freshness_score,
        stall_weights_present=stall_ok,
    )
    agg.needs_retake = needs_retake
    agg.retake_reason = reason
    agg.retake_severity = severity
    return pieces, agg, rows


def analyze_mango_basket(bgr: np.ndarray) -> tuple[list, FreshnessResult, list[dict]]:
    """Stage 1 mango stall seg → per-crop ripeness cls → basket % ripe count."""
    from app.mango_stage1 import MangoPiece, extract_mango_crops_stall_seg

    cls_path = resolve_mango_cls_model()
    cls_ok = cls_path.is_file()
    stall_ok = resolve_mango_stall_seg_model().is_file()
    mode = "stall_seg"

    if not stall_ok or not cls_ok:
        missing = []
        if not stall_ok:
            missing.append("mango_stall_seg.pt")
        if not cls_ok:
            missing.append(cls_path.name)
        if _freshness_strict():
            raise FileNotFoundError(
                f"Mango basket missing weights: {', '.join(missing)}"
            )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=True,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"mango_basket:missing:{','.join(missing)}",
            retake_reason="model_unavailable",
            retake_severity="block",
        )
        return [], empty, []

    pieces: list[MangoPiece] = extract_mango_crops_stall_seg(bgr)
    detections: list[DetectionScore] = []
    rows: list[dict] = []

    for idx, piece in enumerate(pieces, start=1):
        label, conf, cls_score = _mango_classify_bgr(
            piece.crop, apply_spoilage_guard=False
        )
        spoilage_warning = False
        if not _freshness_strict():
            rot_ratio, blob_frac = _mango_spoilage_signals(piece.crop)
            spoilage_warning = (
                rot_ratio >= MANGO_SPOILAGE_FRUIT_RATIO
                or blob_frac >= MANGO_SPOILAGE_BLOB_FRACTION
            )
        piece_pct = round(basket_piece_weighted_contribution("mango", label, conf) * 100)
        detections.append(
            DetectionScore(class_name=label, confidence=conf, freshness=piece_pct)
        )
        rows.append(
            {
                "index": idx,
                "class_name": label,
                "ripeness": label.replace("_", " ").title(),
                "confidence": conf,
                "score": piece_pct,
                "class_weight": round(basket_class_freshness_weight("mango", label) * 100),
                "cls_score": cls_score,
                "spoilage_warning": spoilage_warning,
                "seg_confidence": piece.confidence,
            }
        )

    if not detections:
        needs_retake, reason, severity = mango_retake_assessment(
            piece_count=0,
            mean_confidence=0.0,
            freshness_score=0,
            stall_weights_present=stall_ok,
        )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=needs_retake,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"mango_basket:{cls_path.name}:{mode}",
            retake_reason=reason,
            retake_severity=severity,
        )
        return pieces, empty, rows

    agg = aggregate_detections(
        "mango",
        detections,
        model_version=f"mango_basket:{cls_path.name}:{mode}",
        method="basket_seg+classify",
        basket_scoring="weighted",
    )
    needs_retake, reason, severity = mango_retake_assessment(
        piece_count=agg.piece_count,
        mean_confidence=agg.mean_confidence,
        freshness_score=agg.freshness_score,
        stall_weights_present=stall_ok,
    )
    agg.needs_retake = needs_retake
    agg.retake_reason = reason
    agg.retake_severity = severity
    return pieces, agg, rows


def _tomato_predict_probs(bgr: np.ndarray) -> tuple[Path, list[tuple[str, float]]]:
    """All class probabilities for one tomato image (sorted desc)."""
    path = resolve_tomato_cls_model()
    model = _load_yolo(path)
    if model is None:
        raise FileNotFoundError(
            f"Tomato classifier not found: {path}. "
            "Copy tomato_ripeness.pt into ml-service/models/tomato/."
        )
    results = model.predict(source=bgr, verbose=False)
    if not results or results[0].probs is None:
        raise RuntimeError("Tomato classifier returned no prediction")
    probs = results[0].probs
    names = results[0].names or {}
    data = probs.data.cpu().numpy()
    ranked = sorted(
        (
            (
                canonical_tomato_class(str(names.get(i, i))),
                float(data[i]),
            )
            for i in range(len(data))
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    return path, ranked


def _tomato_classify_bgr(bgr: np.ndarray) -> tuple[str, float, int]:
    """Top class label, confidence, sell-score for one tomato crop."""
    _, ranked = _tomato_predict_probs(bgr)
    label, conf = ranked[0]
    return label, conf, class_to_freshness("tomato", label)


def classify_tomato_single(bgr: np.ndarray) -> FreshnessResult:
    """Single-image tomato ripeness (Kaggle cls — no stall segmentation)."""
    path, ranked = _tomato_predict_probs(bgr)
    label, conf = ranked[0]
    return aggregate_classification(
        "tomato",
        label,
        conf,
        model_version=f"tomato_cls:{path.name}",
    )


def _apple_predict_probs(bgr: np.ndarray) -> tuple[Path, list[tuple[str, float]]]:
    """All class probabilities for one apple image (sorted desc)."""
    path = resolve_apple_cls_model()
    model = _load_yolo(path)
    if model is None:
        raise FileNotFoundError(
            f"Apple classifier not found: {path}. "
            "Copy apple_cls.pt into ml-service/models/ or models/apple/."
        )
    results = model.predict(source=bgr, verbose=False)
    if not results or results[0].probs is None:
        raise RuntimeError("Apple classifier returned no prediction")
    probs = results[0].probs
    names = results[0].names or {}
    data = probs.data.cpu().numpy()
    ranked = sorted(
        (
            (normalize_class(str(names.get(i, i))), float(data[i]))
            for i in range(len(data))
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    return path, ranked


def _apple_classify_bgr(bgr: np.ndarray) -> tuple[str, float, int]:
    """Top class label, confidence, sell-score for one apple crop."""
    _, ranked = _apple_predict_probs(bgr)
    label, conf = ranked[0]
    return label, conf, class_to_freshness("apple", label)


def classify_apple_single(bgr: np.ndarray) -> FreshnessResult:
    """Single-image apple freshness (fresh vs rotten — no stall segmentation)."""
    path, ranked = _apple_predict_probs(bgr)
    label, conf = ranked[0]
    out = aggregate_classification(
        "apple",
        label,
        conf,
        model_version=f"apple_cls:{path.name}",
    )
    return FreshnessResult(
        freshness_score=out.freshness_score,
        piece_count=out.piece_count,
        mean_confidence=out.mean_confidence,
        needs_retake=out.needs_retake,
        method=out.method,
        class_histogram=out.class_histogram,
        per_piece_scores=out.per_piece_scores,
        model_version=out.model_version,
        ripeness_label=apple_freshness_display(label),
    )


def analyze_tomato_basket(bgr: np.ndarray) -> tuple[list, FreshnessResult, list[dict]]:
    """Stage 1 tomato stall seg → per-crop ripeness cls → weighted basket score."""
    from app.tomato_stage1 import TomatoPiece, extract_tomato_crops_stall_seg

    cls_path = resolve_tomato_cls_model()
    cls_ok = cls_path.is_file()
    stall_ok = resolve_tomato_stall_seg_model().is_file()
    mode = "stall_seg"

    if not stall_ok or not cls_ok:
        missing = []
        if not stall_ok:
            missing.append("tomato_stall_seg.pt")
        if not cls_ok:
            missing.append(cls_path.name)
        if _freshness_strict():
            raise FileNotFoundError(
                f"Tomato basket missing weights: {', '.join(missing)}"
            )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=True,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"tomato_basket:missing:{','.join(missing)}",
            retake_reason="model_unavailable",
            retake_severity="block",
        )
        return [], empty, []

    pieces: list[TomatoPiece] = extract_tomato_crops_stall_seg(bgr)
    detections: list[DetectionScore] = []
    rows: list[dict] = []

    for idx, piece in enumerate(pieces, start=1):
        label, conf, cls_score = _tomato_classify_bgr(piece.crop)
        piece_pct = round(basket_piece_weighted_contribution("tomato", label, conf) * 100)
        detections.append(
            DetectionScore(class_name=label, confidence=conf, freshness=piece_pct)
        )
        rows.append(
            {
                "index": idx,
                "class_name": label,
                "ripeness": tomato_ripeness_display(label),
                "confidence": conf,
                "score": piece_pct,
                "class_weight": round(basket_class_freshness_weight("tomato", label) * 100),
                "cls_score": cls_score,
                "seg_confidence": piece.confidence,
            }
        )

    if not detections:
        needs_retake, reason, severity = tomato_retake_assessment(
            piece_count=0,
            mean_confidence=0.0,
            freshness_score=0,
            stall_weights_present=stall_ok,
        )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=needs_retake,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"tomato_basket:{cls_path.name}:{mode}",
            retake_reason=reason,
            retake_severity=severity,
        )
        return pieces, empty, rows

    agg = aggregate_detections(
        "tomato",
        detections,
        model_version=f"tomato_basket:{cls_path.name}:{mode}",
        method="basket_seg+classify",
        basket_scoring="weighted",
    )
    needs_retake, reason, severity = tomato_retake_assessment(
        piece_count=agg.piece_count,
        mean_confidence=agg.mean_confidence,
        freshness_score=agg.freshness_score,
        stall_weights_present=stall_ok,
    )
    agg.needs_retake = needs_retake
    agg.retake_reason = reason
    agg.retake_severity = severity
    return pieces, agg, rows


def analyze_apple_basket(bgr: np.ndarray) -> tuple[list, FreshnessResult, list[dict]]:
    """Stage 1 apple stall seg → per-crop fresh/rotten cls → weighted basket score."""
    from app.apple_stage1 import ApplePiece, extract_apple_crops_stall_seg

    cls_path = resolve_apple_cls_model()
    cls_ok = cls_path.is_file()
    stall_ok = resolve_apple_stall_seg_model().is_file()
    mode = "stall_seg"

    if not stall_ok or not cls_ok:
        missing = []
        if not stall_ok:
            missing.append("apple_stall_seg.pt")
        if not cls_ok:
            missing.append(cls_path.name)
        if _freshness_strict():
            raise FileNotFoundError(
                f"Apple basket missing weights: {', '.join(missing)}"
            )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=True,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"apple_basket:missing:{','.join(missing)}",
            retake_reason="model_unavailable",
            retake_severity="block",
        )
        return [], empty, []

    pieces: list[ApplePiece] = extract_apple_crops_stall_seg(bgr)
    detections: list[DetectionScore] = []
    rows: list[dict] = []

    for idx, piece in enumerate(pieces, start=1):
        label, conf, cls_score = _apple_classify_bgr(piece.crop)
        piece_pct = round(basket_piece_weighted_contribution("apple", label, conf) * 100)
        detections.append(
            DetectionScore(class_name=label, confidence=conf, freshness=piece_pct)
        )
        rows.append(
            {
                "index": idx,
                "class_name": label,
                "ripeness": apple_freshness_display(label),
                "confidence": conf,
                "score": piece_pct,
                "class_weight": round(basket_class_freshness_weight("apple", label) * 100),
                "cls_score": cls_score,
                "seg_confidence": piece.confidence,
            }
        )

    if not detections:
        needs_retake, reason, severity = apple_retake_assessment(
            piece_count=0,
            mean_confidence=0.0,
            freshness_score=0,
            stall_weights_present=stall_ok,
        )
        empty = FreshnessResult(
            freshness_score=0,
            piece_count=0,
            mean_confidence=0.0,
            needs_retake=needs_retake,
            method="basket_seg+classify",
            class_histogram={},
            per_piece_scores=[],
            model_version=f"apple_basket:{cls_path.name}:{mode}",
            retake_reason=reason,
            retake_severity=severity,
        )
        return pieces, empty, rows

    agg = aggregate_detections(
        "apple",
        detections,
        model_version=f"apple_basket:{cls_path.name}:{mode}",
        method="basket_seg+classify",
        basket_scoring="weighted",
    )
    needs_retake, reason, severity = apple_retake_assessment(
        piece_count=agg.piece_count,
        mean_confidence=agg.mean_confidence,
        freshness_score=agg.freshness_score,
        stall_weights_present=stall_ok,
    )
    agg.needs_retake = needs_retake
    agg.retake_reason = reason
    agg.retake_severity = severity
    if agg.class_histogram:
        top = max(agg.class_histogram.items(), key=lambda kv: kv[1])[0]
        agg.ripeness_label = apple_freshness_display(top)
    return pieces, agg, rows


def get_inference_status() -> dict:
    def _info(path: Path, label: str) -> dict:
        if not path.is_file():
            return {"label": label, "loaded": False, "file": path.name}
        st = path.stat()
        return {
            "label": label,
            "loaded": True,
            "file": path.name,
            "bytes": st.st_size,
            "sha256_12": _weights_fingerprint(path),
            "mtime_iso": __import__("datetime")
            .datetime.fromtimestamp(st.st_mtime)
            .isoformat(timespec="seconds"),
        }

    active = active_banana_cls_model()
    mango_path = resolve_mango_cls_model()
    mango_seg_path = resolve_mango_stall_seg_model()
    tomato_path = resolve_tomato_cls_model()
    tomato_seg_path = resolve_tomato_stall_seg_model()
    tomato_seg_v1_path = resolve_tomato_stall_seg_model("v1")
    apple_path = resolve_apple_cls_model()
    apple_seg_path = resolve_apple_stall_seg_model()
    return {
        "freshness_mode": "per_fruit",
        "banana": _info(active, "banana (active)"),
        "banana_c3a2v": _info(BANANA_RIPENESS_C3A2V_MODEL, "banana_ripeness_c3a2v (6-class)"),
        "banana_stall_seg": _info(resolve_banana_stall_seg_model(), "banana_stall_seg"),
        "mango": _info(mango_path, "mango_ripeness_cadimedia (4-class)"),
        "mango_stall_seg": _info(mango_seg_path, "mango_stall_seg"),
        "tomato": _info(tomato_path, "tomato_ripeness (cls)"),
        "tomato_stall_seg": _info(tomato_seg_path, "tomato_stall_seg (v4)"),
        "tomato_stall_seg_v1": _info(tomato_seg_v1_path, "tomato_stall_segv1 (legacy)"),
        "apple": _info(apple_path, "apple_cls (fresh vs rotten)"),
        "apple_stall_seg": _info(apple_seg_path, "apple_stall_seg"),
        "mango_cls": mango_path.is_file(),
        "mango_seg": mango_seg_path.is_file(),
        "tomato_cls": tomato_path.is_file(),
        "tomato_seg": tomato_seg_path.is_file(),
        "tomato_seg_v1": tomato_seg_v1_path.is_file(),
        "apple_cls": apple_path.is_file(),
        "apple_seg": apple_seg_path.is_file(),
    }
