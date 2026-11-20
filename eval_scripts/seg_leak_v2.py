"""Near-duplicate audit v2: trims letterbox/padding borders before comparison (v1 was confounded by black padding)."""
import cv2, numpy as np, json, sys
from pathlib import Path
S=Path(sys.argv[1])
def trim(g):
    m=g>12; r=np.where(m.any(1))[0]; c=np.where(m.any(0))[0]
    return g[r[0]:r[-1]+1,c[0]:c[-1]+1] if len(r)>10 and len(c)>10 else g
def v(g):
    x=cv2.resize(trim(g),(32,32),interpolation=cv2.INTER_AREA).astype(float).ravel(); x=(x-x.mean())/(x.std()+1e-6); return x/np.sqrt(len(x))
def load(d): 
    out=[]
    for p in sorted(Path(d).iterdir()):
        g=cv2.imdecode(np.fromfile(str(p),np.uint8),0); out.append((p.name,v(g),v(cv2.flip(g,1))))
    return out
res={}
pairs=[('banana','banana'),('mango','mango'),('apple','apple'),('apple','banana'),('mango','apple'),('banana','mango')]
for te,tr in pairs:
    T=np.array([x[1] for x in load(S/f'yolo_{tr}/train/images')])
    for split in (['test','valid'] if te==tr else ['test']):
        s=[float(max((T@a).max(),(T@b).max())) for _,a,b in load(S/f'yolo_{te}/{split}/images')]
        key=f'{te}_{split}_vs_{tr}_train'; res[key]=dict(n=len(s),max=max(s),median=float(np.median(s)),n_ge_0_95=sum(x>=0.95 for x in s),scores=sorted(round(x,3) for x in s))
        print(key,res[key]['n_ge_0_95'],'/',len(s),'median',round(res[key]['median'],3),'max',round(max(s),3))
json.dump(res,open(S/'results/seg_leakage_v2.json','w'),indent=2)
