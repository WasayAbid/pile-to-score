# Notes on result files

- `seg_leakage_v2.json` (from `eval_scripts/seg_leak_v2.py`) is the **authoritative** segmentation leakage audit. It trims letterbox padding before comparing images, and every flagged pair was inspected manually.
- The `leakage` fields inside `seg_results.json` come from the first audit version (no border trimming). Its mango flags (10/11) are **false positives** caused by shared black padding. The banana flags (11/11, pixel-identical) are genuine.
- In `clean_seg_results.json`, the **banana** entry is the leak-free retrain reported in the paper (82.1 / 46.6 mask mAP50 / mAP50:95). The mango entry came from a reduced-data retrain triggered by the false-positive v1 audit and is not used in the paper. The mango test split is clean, so the paper reports the re-evaluated 94.2 / 75.2.
- `cls_clean_v2.json` holds classifier accuracy on de-duplicated subsets (with border trimming); `cls_clean.json` is the earlier, near-identical run.
- `det_vs_seg.json` (from `eval_scripts/det_vs_seg.py`): Stage-1 ablation, YOLOv8m detect vs YOLOv8m-seg trained with an identical local recipe (banana on the leak-free split). Box mAP50 differs by at most 1.5 points; the mango/apple seg models in this file are local retrains (apple on the 176 locally available training images), not the released weights.
- `seg_conf_sweep.json` (from `eval_scripts/seg_conf_sweep.py`): recall/precision of Stage-1 instances at IoU >= 0.5 for confidence thresholds 0.10-0.50, using the production post-processing.
- `e2e_results_v2.json` (from `eval_scripts/e2e_v2.py`) supersedes `e2e_results.json`; it also stores mean confidence and the retake reason per photo.
- `gate_sweep.json` / `eval_scripts/gate_sweep.py`: quality-gate sensitivity derived from `e2e_results_v2.json`. The mean-confidence gate never fires for thresholds 0.30-0.50; all retakes come from the segment-count rules.
- Stage-1 instance counts on the test splits are banana 632, mango 254, apple 833 (an earlier draft listed 622/244/821 due to a counting error on label files without trailing newlines).
- Tomato classifier: the VegNet validation split has 725 files; one `.jfif` file is skipped by the evaluation loader, so n = 724.
- Apple segmentation: the local archive is the same Roboflow export used for training (204/21/16) but lacks 28 training and 3 test image files, so re-evaluation uses 13/16 test images.
