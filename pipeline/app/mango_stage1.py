"""Mango Stage 1: trained stall seg (production) + optional YOLO-World zero-shot (demo lab)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import (
    CROP_PAD_FRACTION,
    MANGO_STALL_SEG_CONF,
    MANGO_ZERO_SHOT_CLASSES,
    MANGO_ZERO_SHOT_CONF,
    MANGO_ZERO_SHOT_IMGSZ,
    MANGO_ZERO_SHOT_IOU,
    MANGO_ZERO_SHOT_MODEL,
    MIN_CROP_SIDE_PX,
    MIN_INFER_SIDE,
    resolve_mango_stall_seg_model,
)

_stall_seg = None
_stall_seg_key: str | None = None
_zero_shot_model = None
_zero_shot_key: str | None = None


@dataclass(frozen=True)
class MangoPiece:
    crop: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float


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


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _is_near_duplicate(
    candidate: tuple[int, int, int, int],
    kept: tuple[int, int, int, int],
) -> bool:
    """Drop only near-identical boxes — keep separate mangoes that merely touch/overlap."""
    iou = _box_iou(candidate, kept)
    if iou < 0.72:
        return False
    ca, ka = _box_area(candidate), _box_area(kept)
    if ca == 0 or ka == 0:
        return iou > 0.85
    ratio = min(ca, ka) / max(ca, ka)
    return ratio > 0.82


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
) -> MangoPiece | None:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _pad_box(*box, w, h, pad_frac)
    if x2 - x1 < MIN_CROP_SIDE_PX or y2 - y1 < MIN_CROP_SIDE_PX:
        return None
    crop = bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    return MangoPiece(crop=crop, box=(x1, y1, x2, y2), confidence=confidence)


def _load_stall_segmenter():
    global _stall_seg, _stall_seg_key
    path = resolve_mango_stall_seg_model()
    key = str(path.resolve())
    if _stall_seg is not None and _stall_seg_key == key:
        return _stall_seg
    if not path.is_file():
        return None
    try:
        from ultralytics import YOLO

        _stall_seg = YOLO(key)
        _stall_seg_key = key
        return _stall_seg
    except Exception:
        return None


def extract_mango_crops_stall_seg(bgr: np.ndarray) -> list[MangoPiece]:
    """One crop per instance from mango_stall_seg.pt (trained pile/cart seg)."""
    model = _load_stall_segmenter()
    if model is None:
        return []

    infer_bgr, scale = _infer_scale(bgr)
    results = model.predict(
        source=infer_bgr,
        conf=MANGO_STALL_SEG_CONF,
        iou=0.45,
        imgsz=640,
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
    pieces: list[MangoPiece] = []
    for obox, score in candidates:
        piece = _crop_from_box(bgr, obox, CROP_PAD_FRACTION, score)
        if piece is None:
            continue
        if any(_box_iou(piece.box, p.box) > 0.55 for p in pieces):
            continue
        pieces.append(piece)
    return pieces


def stall_segmenter_status() -> dict:
    path = resolve_mango_stall_seg_model()
    loaded = path.is_file()
    return {
        "model": path.name,
        "path": str(path),
        "loaded": loaded,
        "exists": loaded,
        "bytes": path.stat().st_size if loaded else 0,
        "conf": MANGO_STALL_SEG_CONF,
    }


def _load_zero_shot_detector():
    global _zero_shot_model, _zero_shot_key
    key = f"{MANGO_ZERO_SHOT_MODEL}|{','.join(MANGO_ZERO_SHOT_CLASSES)}"
    if _zero_shot_model is not None and _zero_shot_key == key:
        return _zero_shot_model
    try:
        from ultralytics import YOLO

        model = YOLO(MANGO_ZERO_SHOT_MODEL)
        model.set_classes(list(MANGO_ZERO_SHOT_CLASSES))
        _zero_shot_model = model
        _zero_shot_key = key
        return model
    except Exception as exc:
        print(f"[mango_stage1] YOLO-World load failed: {exc}")
        print("[mango_stage1] Install CLIP: pip install git+https://github.com/ultralytics/CLIP.git")
        return None


def _collect_candidates(
    model,
    infer_bgr: np.ndarray,
    scale: float,
    imgsz: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    results = model.predict(
        source=infer_bgr,
        conf=MANGO_ZERO_SHOT_CONF,
        iou=MANGO_ZERO_SHOT_IOU,
        imgsz=imgsz,
        max_det=300,
        agnostic_nms=True,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    out: list[tuple[tuple[int, int, int, int], float]] = []
    for box in results[0].boxes:
        score = float(box.conf[0]) if box.conf is not None else 0.0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        out.append((_box_to_original((x1, y1, x2, y2), scale), score))
    return out


def extract_mango_crops_zero_shot(bgr: np.ndarray) -> list[MangoPiece]:
    """Detect mangoes with YOLO-World text prompts; return padded bbox crops."""
    model = _load_zero_shot_detector()
    if model is None:
        return []

    infer_bgr, scale = _infer_scale(bgr)
    # Two inference sizes — small mangoes in a pile are often missed at 640 only.
    imgsz_passes = sorted({MANGO_ZERO_SHOT_IMGSZ, min(MANGO_ZERO_SHOT_IMGSZ, 960)}, reverse=True)
    imgsz_passes = list(dict.fromkeys(imgsz_passes))

    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    for imgsz in imgsz_passes:
        candidates.extend(_collect_candidates(model, infer_bgr, scale, imgsz))

    if not candidates:
        return []

    candidates.sort(key=lambda c: c[1], reverse=True)
    pieces: list[MangoPiece] = []
    for obox, score in candidates:
        piece = _crop_from_box(bgr, obox, CROP_PAD_FRACTION, score)
        if piece is None:
            continue
        if any(_is_near_duplicate(piece.box, p.box) for p in pieces):
            continue
        pieces.append(piece)
    return pieces


def draw_mango_pieces_preview(bgr: np.ndarray, pieces: list[MangoPiece]) -> np.ndarray:
    preview = bgr.copy()
    for i, p in enumerate(pieces, start=1):
        x1, y1, x2, y2 = p.box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (232, 163, 23), 2)
        label = f"{i} {p.confidence:.0%}"
        cv2.putText(
            preview,
            label,
            (x1, max(y1 - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (232, 163, 23),
            2,
            cv2.LINE_AA,
        )
    return preview


def mango_zero_shot_status() -> dict:
    model = _load_zero_shot_detector()
    return {
        "model": MANGO_ZERO_SHOT_MODEL,
        "classes": list(MANGO_ZERO_SHOT_CLASSES),
        "conf": MANGO_ZERO_SHOT_CONF,
        "iou": MANGO_ZERO_SHOT_IOU,
        "imgsz": MANGO_ZERO_SHOT_IMGSZ,
        "loaded": model is not None,
        "method": "yolo_world_zero_shot",
    }
