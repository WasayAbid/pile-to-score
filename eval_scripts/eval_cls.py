"""Evaluate Stage-2 ripeness classifiers on held-out splits. Writes results JSON + confusion matrices."""
import json, sys, time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ML = Path(r"f:/Uni Work/Street-Fresh/street-fresh-bolt-version/ml-service")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TASKS = {
    "banana": dict(
        model=ML / "models/banana/banana_ripeness_c3a2v.pt",
        root=ML / "datasets/Banana/_probe_banana_ripeness_c3a2v/test", split="test",
        map={c: c for c in ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]},
    ),
    "mango": dict(
        model=ML / "models/mango/mango_ripeness_cadimedia.pt",
        root=ML / "datasets/Mango/MangoRipenessDetection-Dataset/test", split="test",
        map={"Early Ripe": "partially_ripe", "Partially Ripe": "partially_ripe", "Ripe": "ripe",
             "Unripe": "unripe", "Over Ripe": "overripe"},
    ),
    "tomato": dict(
        model=ML / "models/tomato/tomato_ripeness.pt",
        root=ML / "datasets/Tomato/Tomato Ripeness Dataset/val", split="val",
        map={"Damaged": "rotten", "Old": "overripe", "Ripe": "ripe", "Unripe": "unripe"},
    ),
}


def metrics(cm):
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    return p, r, f


def plot_cm(cm, names, title, path):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2, 1.0 * len(names) + 1.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


results = {}
for fruit, t in TASKS.items():
    model = YOLO(str(t["model"]))
    names = [model.names[i] for i in range(len(model.names))]
    idx = {n: i for i, n in enumerate(names)}
    items = []
    for d in sorted(Path(t["root"]).iterdir()):
        if d.is_dir() and d.name in t["map"]:
            for f in d.iterdir():
                if f.suffix.lower() in IMG_EXT:
                    items.append((f, idx[t["map"][d.name]]))
    cm = np.zeros((len(names), len(names)), int)
    top2 = 0; confs = []; lat = []
    model.predict(str(items[0][0]), verbose=False)  # warmup
    for f, y in items:
        t0 = time.perf_counter()
        r = model.predict(str(f), verbose=False, imgsz=224)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        pr = r.probs
        cm[y, pr.top1] += 1
        top2 += int(y in list(pr.top5)[:2])
        confs.append(float(pr.top1conf))
    p, rc, f1 = metrics(cm)
    acc = np.trace(cm) / cm.sum()
    results[fruit] = dict(
        split=t["split"], n=int(cm.sum()), classes=names, top1=float(acc), top2=top2 / cm.sum(),
        macro_p=float(p.mean()), macro_r=float(rc.mean()), macro_f1=float(f1.mean()),
        per_class={n: dict(p=float(p[i]), r=float(rc[i]), f1=float(f1[i]), support=int(cm[i].sum()))
                   for i, n in enumerate(names)},
        confusion=cm.tolist(), mean_conf=float(np.mean(confs)),
        latency_ms_cpu=dict(mean=float(np.mean(lat)), p50=float(np.median(lat))),
    )
    plot_cm(cm, names, f"{fruit.capitalize()} ripeness ({t['split']}, n={cm.sum()})", OUT / f"cm_{fruit}.png")
    print(fruit, f"top1={acc:.4f} macroF1={f1.mean():.4f} n={cm.sum()}", flush=True)

json.dump(results, open(OUT / "cls_results.json", "w"), indent=2)
