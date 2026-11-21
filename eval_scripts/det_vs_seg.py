"""Ablation: YOLOv8m detection vs YOLOv8m segmentation for Stage 1, same data and recipe.
Both trained locally: 15 epochs frozen backbone (freeze=10) + up to 60 epochs fine-tune (patience 15).
Detection reads the same polygon labels (Ultralytics converts them to boxes). Compared on box metrics."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO

S = Path(sys.argv[1]); OUT = S / "results/det_vs_seg.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}
DATA = {"banana": S / "clean_banana/data.yaml", "mango": S / "yolo_mango/data.yaml", "apple": S / "yolo_apple/data.yaml"}
common = dict(imgsz=640, batch=4, device=0, workers=0, optimizer="AdamW", mosaic=0.0, single_cls=True,
              project=str(S / "train_runs"), exist_ok=True, plots=False, verbose=False, seed=0)


def train(fr, task):
    base = "yolov8m-seg.pt" if task == "seg" else "yolov8m.pt"
    if fr == "banana" and task == "seg" and (S / "train_runs/banana_p2/weights/best.pt").exists():
        return S / "train_runs/banana_p2/weights/best.pt"  # identical recipe, already trained
    best = S / f"train_runs/{fr}_{task}_p2/weights/best.pt"
    if best.exists():
        return best
    YOLO(base).train(data=str(DATA[fr]), epochs=15, freeze=10, lr0=1e-3, name=f"{fr}_{task}_p1", **common)
    YOLO(str(S / f"train_runs/{fr}_{task}_p1/weights/best.pt")).train(
        data=str(DATA[fr]), epochs=60, patience=15, lr0=2e-4, name=f"{fr}_{task}_p2", **common)
    return best


def latency(w, imgs, device):
    m = YOLO(str(w)); m.predict(str(imgs[0]), device=device, verbose=False, imgsz=640)
    t = []
    for p in imgs:
        t0 = time.perf_counter(); m.predict(str(p), device=device, verbose=False, imgsz=640, conf=0.2); t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t))


for fr in ["banana", "mango", "apple"]:
    for task in ["det", "seg"]:
        key = f"{fr}_{task}"
        if key in res:
            continue
        w = train(fr, task)
        m = YOLO(str(w))
        r = m.val(data=str(DATA[fr]), split="test", imgsz=640, batch=4, device=0, workers=0, plots=False,
                  verbose=False, project=str(S / "runs"), name=f"dvs_{key}", exist_ok=True)
        imgs = sorted((Path(DATA[fr]).parent / "test/images").iterdir())
        res[key] = dict(box_mAP50=float(r.box.map50), box_mAP50_95=float(r.box.map), box_P=float(r.box.mp), box_R=float(r.box.mr),
                        mask_mAP50=float(r.seg.map50) if task == "seg" else None,
                        params_M=sum(p.numel() for p in m.model.parameters()) / 1e6, size_MB=w.stat().st_size / 1e6,
                        gpu_ms=latency(w, imgs, 0), cpu_ms=latency(w, imgs, "cpu"), weights=str(w))
        print(key, json.dumps({k: v for k, v in res[key].items() if k != "weights"}), flush=True)
        OUT.write_text(json.dumps(res, indent=2))
