import cv2, numpy as np, sys, json
from pathlib import Path
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).parent))
ML=Path(r"f:/Uni Work/Street-Fresh/street-fresh-bolt-version/ml-service")
EXT={'.jpg','.jpeg','.png','.webp','.bmp'}
def trim(g):
    m=g>12; r=np.where(m.any(1))[0]; c=np.where(m.any(0))[0]
    return g[r[0]:r[-1]+1,c[0]:c[-1]+1] if len(r)>10 and len(c)>10 else g
def vec(im):
    im=trim(im)
    v=cv2.resize(im,(24,24),interpolation=cv2.INTER_AREA).astype(np.float32).ravel(); v=(v-v.mean())/(v.std()+1e-6); return v/np.sqrt(len(v))
def load(p): return cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_GRAYSCALE)
T={'banana':(ML/'models/banana/banana_ripeness_c3a2v.pt',ML/'datasets/Banana/_probe_banana_ripeness_c3a2v/test',ML/'datasets/Banana/_probe_banana_ripeness_c3a2v/train',None),
   'tomato':(ML/'models/tomato/tomato_ripeness.pt',ML/'datasets/Tomato/Tomato Ripeness Dataset/val',ML/'datasets/Tomato/Tomato Ripeness Dataset/train',{'Damaged':'rotten','Old':'overripe','Ripe':'ripe','Unripe':'unripe'}),
   'mango':(ML/'models/mango/mango_ripeness_cadimedia.pt',ML/'datasets/Mango/MangoRipenessDetection-Dataset/test',ML/'datasets/Mango/MangoRipenessDetection-Dataset/train',{"Early Ripe":"partially_ripe","Partially Ripe":"partially_ripe","Ripe":"ripe","Unripe":"unripe","Over Ripe":"overripe"})}
out={}; PAIRS={}
for k,(m,te,tr,mp) in T.items():
    TR=[p for p in tr.rglob("*") if p.suffix.lower() in EXT]; R=np.array([vec(load(p)) for p in TR]) #('*') if p.suffix.lower() in EXT])
    model=YOLO(str(m)); idx={v:i for i,v in model.names.items()}
    rows=[]
    for d in te.iterdir():
        for p in d.iterdir():
            if p.suffix.lower() not in EXT: continue
            im=load(p); s=max((R@vec(im)).max(),(R@vec(cv2.flip(im,1))).max())
            y=idx[mp[d.name] if mp else d.name]; pr=model.predict(str(p),verbose=False)[0].probs.top1
            rows.append((s,y==pr)); PAIRS.setdefault(k,[]).append((s,str(p),str(TR[int(np.argmax(np.maximum(R@vec(im),R@vec(cv2.flip(im,1)))))])))
    rows=np.array(rows,dtype=float)
    r={}
    for thr in [0.95,0.90]:
        c=rows[rows[:,0]<thr]; r[f'clean_lt_{thr}']=dict(n=len(c),acc=float(c[:,1].mean()) if len(c) else None)
    d=rows[rows[:,0]>=0.95]; r['dup_ge_0.95']=dict(n=len(d),acc=float(d[:,1].mean()) if len(d) else None)
    r['all']=dict(n=len(rows),acc=float(rows[:,1].mean())); out[k]=r; print(k,r,flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=2)
import pickle; pickle.dump(PAIRS,open(sys.argv[1]+".pairs.pkl","wb"))
