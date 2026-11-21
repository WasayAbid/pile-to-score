"""Quality-gate sensitivity: re-apply the retake rules from scoring.py to e2e_results_v2.json
for several mean-confidence thresholds. Rules: 0 pieces -> no_segments; <3 -> too_few_segments;
mean_conf < thr -> low_confidence (all block). Usage: python gate_sweep.py e2e_results_v2.json"""
import json, sys
from collections import Counter

d = json.load(open(sys.argv[1]))
out = {}
for fr, v in d.items():
    rows = v["rows"]
    o = {}
    for thr in (0.30, 0.38, 0.45, 0.50):
        rej = sum(1 for r in rows if r["n"] == 0 or r["n"] < 3 or r["mean_conf"] < thr)
        o[f"conf{thr}"] = {"accepted": len(rows) - rej, "rejected": rej}
    o["reasons"] = dict(Counter(str(r["reason"]) for r in rows))
    o["min_mean_conf_when_fruit_found"] = min(r["mean_conf"] for r in rows if r["n"] > 0)
    out[fr] = o
print(json.dumps(out, indent=2))
