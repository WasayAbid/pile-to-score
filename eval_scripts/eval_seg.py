"""Evaluate Stage-1 stall segmentation models on held-out test splits + leakage audit."""
import json, re, sys, shutil
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

ML = Path(r"f:/Uni Work/Street-Fresh/street-fresh-bolt-version/ml-service")
SCR = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

COCO_SETS = {
    "banana": (ML / "datasets/Banana/ALI ME Trained Roboflow/Banana Segmentation.coco-segmentation",
               {"train": "train", "valid": "Valid", "test": "Test"}),
    "apple": (ML / "datasets/Apple/Apple Segmenting Dataset.v1i.coco-segmentation",
              {"train": "train", "valid": "valid", "test": "test"}),
    "mango": (SCR / "data/mango_seg", {"train": "train", "valid": "valid", "test": "test"}),
}
MODELS = {f: ML / f"models/{f}/{f}_stall_seg.pt" for f in ["banana", "apple", "mango", "tomato"]}
MODELS["apple"] = ML / "models/Apple/apple_stall_seg.pt"


import numpy as np, cv2


def rle_decode(seg):
    """Decode COCO compressed RLE (string counts) -> HxW uint8 mask."""
    h, w = seg["size"]
    s = seg["counts"]
    if isinstance(s, list):
        cnts = s
    else:
        cnts, p = [], 0
        while p < len(s):
            x, k, more = 0, 0, True
            while more:
                c = ord(s[p]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                p += 1; k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
    flat = np.zeros(h * w, np.uint8); pos, val = 0, 0
    for c in cnts:
        flat[pos:pos + c] = val; pos += c; val ^= 1
    return flat.reshape((w, h)).T


def seg_to_polys(seg):
    if isinstance(seg, list):
        return [p for p in seg if len(p) >= 6]
    m = rle_decode(seg)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in cs if len(c) >= 3]


def coco_to_yolo(src, splits, dst):
    for s, folder in splits.items():
        sd = src / folder
        if not sd.exists():
            continue
        coco = json.load(open(sd / "_annotations.coco.json", encoding="utf-8"))
        imgs = {i["id"]: i for i in coco["images"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)
        for iid, im in imgs.items():
            fp = sd / im["file_name"]
            if not fp.exists():
                continue
            shutil.copy(fp, dst / s / "images" / im["file_name"])
            W, H = im["width"], im["height"]
            lines = []
            for a in anns[iid]:
                if str(a.get("category_id")) == "0":
                    continue
                polys = seg_to_polys(a.get("segmentation") or [])
                if not polys:
                    continue
                poly = max(polys, key=len)
                pts = " ".join(f"{poly[k]/W:.6f} {poly[k+1]/H:.6f}" for k in range(0, len(poly), 2))
                lines.append("0 " + pts)
            (dst / s / "labels" / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['item']\n")
    return dst / "data.yaml"


def source_key(name):
    # Roboflow names: <orig>_<ext>.rf.<hash>.jpg  -> <orig>
    return re.split(r"_(jpg|jpeg|png|webp|avif)\.rf\.", name, flags=re.I)[0].lower()


def dhash(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    out = []
    for img in (im, cv2.flip(im, 1)):
        sm = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        out.append(np.packbits((sm[:, 1:] > sm[:, :-1]).flatten()))
    return out


def near_dupes(root, a, b, thr=8):
    ha = [dhash(p) for p in (root / a / "images").iterdir()]
    hb = [dhash(p)[0] for p in (root / b / "images").iterdir()]
    n = 0
    for h in ha:
        if any(min(np.unpackbits(h[0] ^ x).sum(), np.unpackbits(h[1] ^ x).sum()) <= thr for x in hb):
            n += 1
    return n


def leakage(root):
    keys = {}
    for s in ["train", "valid", "test"]:
        d = root / s / "images"
        keys[s] = {source_key(p.name) for p in d.iterdir()} if d.exists() else set()
    n = {s: len(list((root / s / "images").iterdir())) if (root / s / "images").exists() else 0 for s in keys}
    return dict(images=n, unique_sources={s: len(v) for s, v in keys.items()},
                test_sources_also_in_train=len(keys["test"] & keys["train"]),
                valid_sources_also_in_train=len(keys["valid"] & keys["train"]),
                test_near_duplicates_in_train_dhash=near_dupes(root, "test", "train"))


results = {}
yamls = {f: coco_to_yolo(src, sp, SCR / f"yolo_{f}") for f, (src, sp) in COCO_SETS.items()}
tom = SCR / "yolo_tomato"
# Tomato export mixes box-only rows (5 fields) with polygons; convert boxes to rectangle polygons.
# Mask metrics are therefore not meaningful for tomato -> report box metrics only.
for s in ["train", "valid", "test"]:
    (tom / s / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copytree(SCR / "data/tomato_seg" / s / "images", tom / s / "images", dirs_exist_ok=True)
    for lf in (SCR / "data/tomato_seg" / s / "labels").glob("*.txt"):
        out = []
        for ln in lf.read_text().splitlines():
            v = ln.split()
            if len(v) == 5:
                cx, cy, bw, bh = map(float, v[1:])
                x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                out.append(f"0 {x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1}")
            elif len(v) >= 7:
                out.append("0 " + " ".join(v[1:]))
        (tom / s / "labels" / lf.name).write_text("\n".join(out))
(tom / "data_abs.yaml").write_text(
    f"path: {tom.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['item']\n")
yamls["tomato"] = tom / "data_abs.yaml"

for fruit, y in yamls.items():
    root = Path(y).parent
    model = YOLO(str(MODELS[fruit]))
    r = model.val(data=str(y), split="test", imgsz=640, batch=4, device="cpu", conf=0.001,
                  plots=False, verbose=False, project=str(SCR / "runs"), name=fruit, exist_ok=True)
    results[fruit] = dict(
        mask_mAP50=float(r.seg.map50), mask_mAP50_95=float(r.seg.map),
        mask_P=float(r.seg.mp), mask_R=float(r.seg.mr),
        box_mAP50=float(r.box.map50), box_mAP50_95=float(r.box.map),
        speed_ms=r.speed, leakage=leakage(root),
    )
    print(fruit, json.dumps(results[fruit]), flush=True)

json.dump(results, open(OUT / "seg_results.json", "w"), indent=2)
