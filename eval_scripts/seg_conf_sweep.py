"""Stage-1 confidence-threshold sensitivity on labelled (clean) test splits: matched / missed / false detections (box IoU>=0.5).
Mirrors production post-processing: NMS iou=0.45, then greedy removal of crops overlapping an accepted crop by IoU>0.55."""
import json, sys
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO
S=Path(sys.argv[1]); ML=Path(sys.argv[2])
SETS={"banana":(S/"train_runs/banana_p2/weights/best.pt",S/"clean_banana/test"),
      "mango":(ML/"models/mango/mango_stall_seg.pt",S/"yolo_mango/test"),
      "apple":(ML/"models/Apple/apple_stall_seg.pt",S/"yolo_apple/test")}
CONFS=[0.10,0.15,0.20,0.30,0.40,0.50]
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); i=max(0,x2-x1)*max(0,y2-y1)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)
res={}
for fr,(w,d) in SETS.items():
    m=YOLO(str(w)); tot={c:[0,0,0] for c in CONFS}; ngt=0
    for p in sorted((d/"images").iterdir()):
        im=cv2.imdecode(np.fromfile(str(p),np.uint8),1); H,W=im.shape[:2]
        gt=[]
        for ln in (d/"labels"/(p.stem+".txt")).read_text().splitlines():
            v=np.array(ln.split()[1:],float).reshape(-1,2)*[W,H]; gt.append([*v.min(0),*v.max(0)])
        ngt+=len(gt)
        r=m.predict(im,conf=min(CONFS),iou=0.45,imgsz=640,verbose=False,device="cpu")[0]
        dets=sorted(zip(r.boxes.conf.tolist(),r.boxes.xyxy.tolist()),reverse=True)
        for c in CONFS:
            acc=[]
            for s,b in dets:
                if s<c: continue
                if any(iou(b,a)>0.55 for a in acc): continue
                acc.append(b)
            used=set(); tp=0
            for b in acc:
                best=max(range(len(gt)),key=lambda j:iou(b,gt[j]) if j not in used else -1,default=None)
                if best is not None and best not in used and iou(b,gt[best])>=0.5: used.add(best); tp+=1
            tot[c][0]+=tp; tot[c][1]+=len(acc)-tp; tot[c][2]+=len(gt)-tp
    res[fr]={"gt_instances":ngt,"by_conf":{str(c):dict(matched=t[0],false_det=t[1],missed=t[2],recall=t[0]/ngt,precision=t[0]/max(1,t[0]+t[1])) for c,t in tot.items()}}
    print(fr,ngt,{c:(round(v['recall'],3),round(v['precision'],3)) for c,v in res[fr]['by_conf'].items()},flush=True)
json.dump(res,open(S/"results/seg_conf_sweep.json","w"),indent=2)
