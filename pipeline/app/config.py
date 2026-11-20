import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(BASE_DIR / "models")))
BANANA_MODELS_DIR = MODELS_DIR / "banana"
DATASETS_DIR = BASE_DIR / "datasets"
BANANA_DATASETS_DIR = DATASETS_DIR / "Banana"

GOOGLE_VENDOR_IMAGES_DIR = Path(
    os.environ.get(
        "GOOGLE_VENDOR_IMAGES_DIR",
        str(BANANA_DATASETS_DIR / "real_banana_google_vendor_images"),
    )
)
BANANA_RIPENESS_C3A2V_DATASET = Path(
    os.environ.get(
        "BANANA_RIPENESS_C3A2V_DATASET",
        str(BANANA_DATASETS_DIR / "_probe_banana_ripeness_c3a2v"),
    )
)
BANANA_STALL_SEG_DATASET = Path(
    os.environ.get(
        "BANANA_STALL_SEG_DATASET",
        str(BANANA_DATASETS_DIR / "ALI ME Trained Roboflow"),
    )
)

BANANA_RIPENESS_C3A2V_MODEL = Path(
    os.environ.get(
        "BANANA_RIPENESS_C3A2V_MODEL",
        str(BANANA_MODELS_DIR / "banana_ripeness_c3a2v.pt"),
    )
)


def active_banana_cls_model() -> Path:
    return BANANA_RIPENESS_C3A2V_MODEL


BANANA_STALL_SEG_MODEL = Path(
    os.environ.get(
        "BANANA_STALL_SEG_MODEL",
        str(BANANA_MODELS_DIR / "banana_stall_seg.pt"),
    )
)
BANANA_STALL_SEG_CONF = float(os.environ.get("BANANA_STALL_SEG_CONF", "0.20"))


def resolve_banana_stall_seg_model() -> Path:
    return BANANA_STALL_SEG_MODEL


def default_basket_stage1_mode() -> str:
    return "stall_seg"


MANGO_MODELS_DIR = MODELS_DIR / "mango"


def _apple_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    for name in ("apple", "Apple"):
        p = MODELS_DIR / name
        if p.is_dir():
            dirs.append(p)
    if not dirs:
        dirs.append(MODELS_DIR / "apple")
    return dirs


def resolve_apple_cls_model() -> Path:
    """Colab export → models/apple/apple_cls.pt (not stall seg)."""
    env = os.environ.get("APPLE_CLS_MODEL")
    if env:
        return Path(env)
    for name in ("apple_cls.pt", "apple_ripeness.pt"):
        for base in _apple_model_dirs():
            candidate = base / name
            if candidate.is_file():
                return candidate
        candidate = MODELS_DIR / name
        if candidate.is_file():
            return candidate
    for base in _apple_model_dirs():
        pts = sorted(
            (
                p
                for p in base.glob("*.pt")
                if p.name not in ("apple_stall_seg.pt", "best.pt")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pts:
            return pts[0]
    return MODELS_DIR / "apple_cls.pt"


def resolve_apple_stall_seg_model() -> Path:
    env = os.environ.get("APPLE_STALL_SEG_MODEL")
    if env:
        return Path(env)
    for name in ("apple_stall_seg.pt",):
        for base in _apple_model_dirs():
            candidate = base / name
            if candidate.is_file():
                return candidate
    return _apple_model_dirs()[0] / "apple_stall_seg.pt"


APPLE_STALL_SEG_CONF = float(os.environ.get("APPLE_STALL_SEG_CONF", "0.15"))


APPLE_CLS_MODEL = resolve_apple_cls_model()


def resolve_mango_cls_model() -> Path:
    """Colab export → models/mango/mango_ripeness_cadimedia.pt (not stall seg)."""
    env = os.environ.get("MANGO_CLS_MODEL")
    if env:
        return Path(env)
    for name in (
        "mango_ripeness_cadimedia.pt",
        "mango_cls.pt",
    ):
        candidate = MANGO_MODELS_DIR / name
        if candidate.is_file():
            return candidate
    if MANGO_MODELS_DIR.is_dir():
        pts = sorted(
            (
                p
                for p in MANGO_MODELS_DIR.glob("*.pt")
                if p.name not in ("mango_stall_seg.pt", "best.pt")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pts:
            return pts[0]
    legacy = MODELS_DIR / "mango_cls.pt"
    return legacy


MANGO_STALL_SEG_MODEL = Path(
    os.environ.get(
        "MANGO_STALL_SEG_MODEL",
        str(MANGO_MODELS_DIR / "mango_stall_seg.pt"),
    )
)
MANGO_STALL_SEG_CONF = float(os.environ.get("MANGO_STALL_SEG_CONF", "0.20"))


def resolve_mango_stall_seg_model() -> Path:
    return MANGO_STALL_SEG_MODEL


MANGO_CLS_MODEL = resolve_mango_cls_model()

TOMATO_MODELS_DIR = MODELS_DIR / "tomato"
TOMATO_DATASETS_DIR = DATASETS_DIR / "Tomato"


def resolve_tomato_cls_model() -> Path:
    """Colab export → models/tomato/tomato_ripeness.pt (not stall seg)."""
    env = os.environ.get("TOMATO_CLS_MODEL")
    if env:
        return Path(env)
    for name in (
        "tomato_ripeness.pt",
        "tomato_ripeness_laboro.pt",
        "tomato_cls.pt",
    ):
        candidate = TOMATO_MODELS_DIR / name
        if candidate.is_file():
            return candidate
    if TOMATO_MODELS_DIR.is_dir():
        pts = sorted(
            (
                p
                for p in TOMATO_MODELS_DIR.glob("*.pt")
                if p.name not in ("tomato_stall_seg.pt", "tomato_stall_segv1.pt", "best.pt")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pts:
            return pts[0]
    return TOMATO_MODELS_DIR / "tomato_ripeness.pt"


TOMATO_STALL_SEG_MODEL = Path(
    os.environ.get(
        "TOMATO_STALL_SEG_MODEL",
        str(TOMATO_MODELS_DIR / "tomato_stall_seg.pt"),
    )
)
TOMATO_STALL_SEG_V1_MODEL = Path(
    os.environ.get(
        "TOMATO_STALL_SEG_V1_MODEL",
        str(TOMATO_MODELS_DIR / "tomato_stall_segv1.pt"),
    )
)
TOMATO_STALL_SEG_CONF = float(os.environ.get("TOMATO_STALL_SEG_CONF", "0.10"))
TOMATO_STALL_SEG_DATASET = Path(
    os.environ.get(
        "TOMATO_STALL_SEG_DATASET",
        str(TOMATO_DATASETS_DIR / "Laboro_Tomato_YOLO"),
    )
)
TOMATO_RIPENESS_DATASET = Path(
    os.environ.get(
        "TOMATO_RIPENESS_DATASET",
        str(TOMATO_DATASETS_DIR / "Laboro_Tomato_crops"),
    )
)


def resolve_tomato_stall_seg_model(variant: str | None = None) -> Path:
    """Current (v4) default: tomato_stall_seg.pt · legacy: tomato_stall_segv1.pt."""
    v = (variant or os.environ.get("TOMATO_STALL_SEG_VARIANT", "current")).strip().lower()
    if v in ("v1", "segv1", "legacy", "old"):
        return TOMATO_STALL_SEG_V1_MODEL
    return TOMATO_STALL_SEG_MODEL


TOMATO_CLS_MODEL = resolve_tomato_cls_model()

# Mango Stage 1 experiment — YOLO-World open-vocabulary detect (downloads on first run).
MANGO_ZERO_SHOT_MODEL = os.environ.get("MANGO_ZERO_SHOT_MODEL", "yolov8m-worldv2.pt")
MANGO_ZERO_SHOT_CONF = float(os.environ.get("MANGO_ZERO_SHOT_CONF", "0.08"))
MANGO_ZERO_SHOT_IOU = float(os.environ.get("MANGO_ZERO_SHOT_IOU", "0.25"))
MANGO_ZERO_SHOT_IMGSZ = int(os.environ.get("MANGO_ZERO_SHOT_IMGSZ", "1280"))
MANGO_ZERO_SHOT_CLASSES = tuple(
    c.strip()
    for c in os.environ.get("MANGO_ZERO_SHOT_CLASSES", "mango,mango fruit").split(",")
    if c.strip()
)

API_KEY = os.environ.get("FRESHNESS_API_KEY", "")
CROP_PAD_FRACTION = float(os.environ.get("CROP_PAD", "0.06"))
MIN_CROP_SIDE_PX = int(os.environ.get("MIN_CROP_SIDE", "40"))
MIN_INFER_SIDE = int(os.environ.get("MIN_INFER_SIDE", "640"))
