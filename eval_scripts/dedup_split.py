import cv2, numpy as np, shutil, json, sys
from pathlib import Path
S=Path(sys.argv[1]); THR=0.85
def v(p):
    im=cv2.imdecode(np.fromfile(str(p),np.uint8),0); o=[]
    for i in (im,cv2.flip(im,1)):
        x=cv2.resize(i,(32,32),interpolation=cv2.INTER_AREA).astype(float).ravel(); x=(x-x.mean())/(x.std()+1e-6); o.append(x/np.sqrt(len(x)))
    return o
rep={}
for fr in ['banana','mango']:
    src=S/f'yolo_{fr}'; dst=S/f'clean_{fr}'
    if dst.exists(): shutil.rmtree(dst)
    held=[]
    for s in ['valid','test']:
        for p in (src/s/'images').iterdir(): held+=v(p)
    H=np.array(held)
    kept=dropped=0
    for s in ['train','valid','test']:
        (dst/s/'images').mkdir(parents=True); (dst/s/'labels').mkdir(parents=True)
        for p in (src/s/'images').iterdir():
            if s=='train' and (H@v(p)[0]).max()>=THR: dropped+=1; continue
            if s=='train': kept+=1
            shutil.copy(p,dst/s/'images'/p.name); shutil.copy(src/s/'labels'/(p.stem+'.txt'),dst/s/'labels'/(p.stem+'.txt'))
    (dst/'data.yaml').write_text(f"path: {dst.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['item']\n")
    rep[fr]=dict(train_kept=kept,train_dropped=dropped); print(fr,rep[fr])
json.dump(rep,open(S/'results/dedup_split.json','w'),indent=2)
