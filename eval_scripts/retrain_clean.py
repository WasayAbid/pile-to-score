"""Retrain banana/mango stall segmentation on leak-free splits, evaluate on untouched test split.
Mirrors the original two-phase recipe (frozen backbone -> full fine-tune), shortened for local GPU."""
import json, sys
from pathlib import Path
from ultralytics import YOLO

S = Path(sys.argv[1]); fruits = sys.argv[2:] or ["banana", "mango"]
res_path = S / "results/clean_seg_results.json"
res = json.loads(res_path.read_text()) if res_path.exists() else {}
common = dict(imgsz=640, batch=4, device=0, workers=0, optimizer="AdamW", mosaic=0.0, single_cls=True,
              project=str(S / "train_runs"), exist_ok=True, plots=False, verbose=False, seed=0)
for fr in fruits:
    data = str(S / f"clean_{fr}/data.yaml")
    m = YOLO("yolov8m-seg.pt")
    m.train(data=data, epochs=15, freeze=10, lr0=1e-3, name=f"{fr}_p1", **common)
    p1 = S / f"train_runs/{fr}_p1/weights/best.pt"
    m = YOLO(str(p1))
    m.train(data=data, epochs=60, patience=15, lr0=2e-4, name=f"{fr}_p2", **common)
    best = S / f"train_runs/{fr}_p2/weights/best.pt"
    r = YOLO(str(best)).val(data=data, split="test", imgsz=640, batch=4, device=0, workers=0, plots=False,
                            verbose=False, project=str(S / "runs"), name=f"clean_{fr}", exist_ok=True)
    res[fr] = dict(mask_mAP50=float(r.seg.map50), mask_mAP50_95=float(r.seg.map), mask_P=float(r.seg.mp),
                   mask_R=float(r.seg.mr), box_mAP50=float(r.box.map50), box_mAP50_95=float(r.box.map),
                   gpu_inference_ms=r.speed.get("inference"))
    print(fr, json.dumps(res[fr]), flush=True)
    res_path.write_text(json.dumps(res, indent=2))
