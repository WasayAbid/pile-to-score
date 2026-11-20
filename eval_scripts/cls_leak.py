import cv2, numpy as np, sys, json
from pathlib import Path
ML=Path(r"f:/Uni Work/Street-Fresh/street-fresh-bolt-version/ml-service/datasets")
EXT={'.jpg','.jpeg','.png','.webp','.bmp'}
def feats(root):
    P=[p for p in Path(root).rglob('*') if p.suffix.lower() in EXT]
    F=[];G=[]
    for p in P:
        im=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        for i,dst in ((im,F),(cv2.flip(im,1),G)):
            v=cv2.resize(i,(24,24),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            v=(v-v.mean())/(v.std()+1e-6); dst.append(v/np.sqrt(len(v)))
    return np.array(F),np.array(G)
res={}
for name,te,tr in [('banana',ML/'Banana/_probe_banana_ripeness_c3a2v/test',ML/'Banana/_probe_banana_ripeness_c3a2v/train'),
                   ('mango',ML/'Mango/MangoRipenessDetection-Dataset/test',ML/'Mango/MangoRipenessDetection-Dataset/train'),
                   ('tomato',ML/'Tomato/Tomato Ripeness Dataset/val',ML/'Tomato/Tomato Ripeness Dataset/train')]:
    Ft,Gt=feats(te); Fr,_=feats(tr)
    s=np.maximum((Ft@Fr.T).max(1),(Gt@Fr.T).max(1))
    res[name]=dict(n=len(s),ge_0_95=float((s>=0.95).mean()),ge_0_99=float((s>=0.99).mean()),median=float(np.median(s)))
    print(name,res[name],flush=True)
json.dump(res,open(sys.argv[1],'w'),indent=2)
