"""Tomato Stage 1: stall YOLOv8m-seg → one crop per detected tomato."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import (
    CROP_PAD_FRACTION,
    MIN_CROP_SIDE_PX,
    MIN_INFER_SIDE,
    TOMATO_STALL_SEG_CONF,
    resolve_tomato_stall_seg_model,
)

_stall_seg_cache: dict[str, object] = {}


@dataclass(frozen=True)
class TomatoPiece:
    crop: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float = 0.0


def _pad_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1)


def _infer_scale(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side >= MIN_INFER_SIDE:
        return bgr, 1.0
    scale = MIN_INFER_SIDE / side
    infer = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return infer, scale


def _box_to_original(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    if scale == 1.0:
        return box
    return tuple(int(v / scale) for v in box)  # type: ignore[return-value]


def _crop_from_box(
    bgr: np.ndarray,
    box: tuple[int, int, int, int],
    pad_frac: float,
    confidence: float,
) -> TomatoPiece | None:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _pad_box(*box, w, h, pad_frac)
    if x2 - x1 < MIN_CROP_SIDE_PX or y2 - y1 < MIN_CROP_SIDE_PX:
        return None
    crop = bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    return TomatoPiece(crop=crop, box=(x1, y1, x2, y2), confidence=confidence)


def _load_stall_segmenter(*, seg_variant: str = "current"):
    path = resolve_tomato_stall_seg_model(seg_variant)
    key = str(path.resolve())
    if key in _stall_seg_cache:
        return _stall_seg_cache[key]
    if not path.is_file():
        return None
    try:
        from ultralytics import YOLO

        model = YOLO(key)
        _stall_seg_cache[key] = model
        return model
    except Exception:
        return None


def count_tomato_stall_detections(
    bgr: np.ndarray,
    *,
    conf: float | None = None,
    seg_variant: str = "current",
) -> int:
    """Raw YOLO box count before IoU dedupe / min-size filtering."""
    model = _load_stall_segmenter(seg_variant=seg_variant)
    if model is None:
        return 0
    use_conf = TOMATO_STALL_SEG_CONF if conf is None else conf
    infer_bgr, _ = _infer_scale(bgr)
    results = model.predict(
        source=infer_bgr,
        conf=use_conf,
        iou=0.45,
        imgsz=640,
        max_det=500,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return 0
    return len(results[0].boxes)


def extract_tomato_crops_stall_seg(
    bgr: np.ndarray,
    *,
    conf: float | None = None,
    seg_variant: str = "current",
) -> list[TomatoPiece]:
    """One crop per YOLO instance-seg detection from tomato_stall_seg.pt."""
    model = _load_stall_segmenter(seg_variant=seg_variant)
    if model is None:
        return []

    use_conf = TOMATO_STALL_SEG_CONF if conf is None else conf
    infer_bgr, scale = _infer_scale(bgr)
    results = model.predict(
        source=infer_bgr,
        conf=use_conf,
        iou=0.45,
        imgsz=640,
        max_det=500,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    for box in results[0].boxes:
        score = float(box.conf[0]) if box.conf is not None else 0.0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        candidates.append((_box_to_original((x1, y1, x2, y2), scale), score))

    candidates.sort(key=lambda c: c[1], reverse=True)
    pieces: list[TomatoPiece] = []
    for obox, score in candidates:
        piece = _crop_from_box(bgr, obox, CROP_PAD_FRACTION, score)
        if piece is None:
            continue
        if any(_box_iou(piece.box, p.box) > 0.55 for p in pieces):
            continue
        pieces.append(piece)
    return pieces


def draw_tomato_pieces_preview(bgr: np.ndarray, pieces: list[TomatoPiece]) -> np.ndarray:
    preview = bgr.copy()
    for i, p in enumerate(pieces, start=1):
        x1, y1, x2, y2 = p.box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 80, 220), 2)
        cv2.putText(
            preview,
            str(i),
            (x1, max(y1 - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 80, 220),
            2,
            cv2.LINE_AA,
        )
    return preview


def stall_segmenter_status(*, seg_variant: str = "current") -> dict:
    path = resolve_tomato_stall_seg_model(seg_variant)
    loaded = path.is_file()
    return {
        "variant": seg_variant,
        "model": path.name,
        "path": str(path),
        "loaded": loaded,
        "exists": loaded,
        "bytes": path.stat().st_size if loaded else 0,
        "conf": TOMATO_STALL_SEG_CONF,
    }
