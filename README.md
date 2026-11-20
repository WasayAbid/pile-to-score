# Pile-to-Score

**Estimating basket-level visual freshness of market-stall produce via decoupled instance segmentation and crop-level ripeness classification.**

Muhammad Abdul Wasay Abid — Faculty of Information Technology, University of Central Punjab, Lahore, Pakistan

This repository accompanies the preprint *"Pile-to-Score: Estimating Basket-Level Visual Freshness of Market-Stall Produce via Decoupled Instance Segmentation and Crop-Level Ripeness Classification"* (2026). It contains the trained models, inference pipeline, evaluation scripts and every result reported in the paper.

```
Stall photo ─► Stage 1: class-agnostic YOLOv8m-seg ─► padded crop per fruit
            ─► Stage 2: fruit-specific YOLOv8m-cls ─► weighted basket score + spoilage penalty
            ─► quality gate (retake rules) ─► score 0–100 + freshness tier
```

## Trained models

Download from the [**v1.0 release**](https://github.com/WasayAbid/pile-to-score/releases/tag/v1.0).

| Fruit | Stage | File | Classes | Held-out result |
|-------|-------|------|---------|-----------------|
| Banana | 1 · segmentation | [`banana_stall_seg.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/banana_stall_seg.pt) | fruit | test split contaminated — use leak-free result |
| Banana | 1 · segmentation (leak-free) | [`banana_stall_seg_leakfree.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/banana_stall_seg_leakfree.pt) | fruit | 82.1 mask mAP50 |
| Banana | 2 · ripeness | [`banana_ripeness_c3a2v.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/banana_ripeness_c3a2v.pt) | 6 | 98.6% top-1 (test) |
| Mango | 1 · segmentation | [`mango_stall_seg.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/mango_stall_seg.pt) | fruit | 94.2 mask mAP50 (test) |
| Mango | 2 · ripeness | [`mango_ripeness_cadimedia.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/mango_ripeness_cadimedia.pt) | 4 | 92.0% top-1 (test) |
| Tomato | 1 · segmentation | [`tomato_stall_seg.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/tomato_stall_seg.pt) | fruit | 87.5 mask mAP50 (logged) |
| Tomato | 2 · ripeness | [`tomato_ripeness.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/tomato_ripeness.pt) | 4 | 99.4% top-1 (val) |
| Apple | 1 · segmentation | [`apple_stall_seg.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/apple_stall_seg.pt) | fruit | 80.4 mask mAP50 (test) |
| Apple | 2 · freshness | [`apple_cls.pt`](https://github.com/WasayAbid/pile-to-score/releases/download/v1.0/apple_cls.pt) | 2 | not independently evaluated |

The prototype pipeline uses the full-data segmentation models. `banana_stall_seg_leakfree.pt` was retrained after removing training images that duplicate the banana validation/test images (Sec. 6.2 of the paper); it gives the honest held-out banana estimate. Tomato classifier results are validation results (the dataset has no test split). The basket score is a heuristic and has not been validated against human graders. See `results/NOTES.md` for which result files are authoritative.

## Repository layout

| Path | Contents |
|------|----------|
| `pipeline/app/` | Inference service: Stage 1 segmentation (`stage1.py`, `*_stage1.py`), Stage 2 classification (`inference.py`), basket scoring and retake rules (`scoring.py`), FastAPI server (`main.py`) |
| `eval_scripts/` | Scripts that produce every number in the paper |
| `results/` | Metrics JSONs: classifier results, confusion matrices, leakage audits, segmentation re-evaluation, leak-free retraining, end-to-end runs |
| `paper/` | LaTeX source and figures of the preprint |

## Quick start

```bash
pip install -r pipeline/app/requirements.txt
# place the downloaded weights under pipeline/models/<fruit>/ (banana, mango, tomato, Apple)
cd pipeline
uvicorn app.main:app --port 8000
curl -F "image=@stall.jpg" -F "fruit_type=banana" http://localhost:8000/v1/freshness
```

Or use the models directly:

```python
from ultralytics import YOLO
seg = YOLO("banana_stall_seg.pt")        # one mask per banana
cls = YOLO("banana_ripeness_c3a2v.pt")   # ripeness of one crop
```

## Datasets

The Stage 1 stall-segmentation datasets were built by the author from market photographs. The Stage 2 classifiers were trained on public datasets: [banana-ripeness-c3a2v](https://universe.roboflow.com/mahasarakham-university-yeff9/banana-ripeness-c3a2v), [MangoRipenessDetection](https://universe.roboflow.com/cadimedia/mangoripenessdetection), [VegNet](https://data.mendeley.com/datasets/6nxnjbn9w6/1) (tomato; Suryawanshi et al., Data in Brief 2022) and a public apple fresh/rotten dataset. Datasets are not redistributed here; see their original licences.

## Licence

AGPL-3.0, consistent with the [Ultralytics](https://github.com/ultralytics/ultralytics) framework the models are built on.

## Citation

```bibtex
@misc{abid2026piletoscore,
  title  = {Pile-to-Score: Estimating Basket-Level Visual Freshness of Market-Stall Produce via
            Decoupled Instance Segmentation and Crop-Level Ripeness Classification},
  author = {Abid, Muhammad Abdul Wasay},
  year   = {2026},
  url    = {https://github.com/WasayAbid/pile-to-score}
}
```
