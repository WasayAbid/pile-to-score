# Notes on result files

- `seg_leakage_v2.json` (from `eval_scripts/seg_leak_v2.py`) is the **authoritative** segmentation leakage audit. It trims letterbox padding before comparing images, and every flagged pair was inspected manually.
- The `leakage` fields inside `seg_results.json` come from the first audit version (no border trimming). Its mango flags (10/11) are **false positives** caused by shared black padding. The banana flags (11/11, pixel-identical) are genuine.
- In `clean_seg_results.json`, the **banana** entry is the leak-free retrain reported in the paper (82.1 / 46.6 mask mAP50 / mAP50:95). The mango entry came from a reduced-data retrain triggered by the false-positive v1 audit and is not used in the paper. The mango test split is clean, so the paper reports the re-evaluated 94.2 / 75.2.
- `cls_clean_v2.json` holds classifier accuracy on de-duplicated subsets (with border trimming); `cls_clean.json` is the earlier, near-identical run.
