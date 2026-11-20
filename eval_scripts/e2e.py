import sys, time, json, random
from pathlib import Path
import cv2, numpy as np
ML=Path(r"f:/Uni Work/Street-Fresh/street-fresh-bolt-version/ml-service"); sys.path.insert(0,str(ML))
from app import inference as inf
OUT=Path(sys.argv[1]); (OUT/'e2e').mkdir(parents=True,exist_ok=True)
EXT={'.jpg','.jpeg','.png','.webp'}
sets={'banana':(inf.analyze_banana_basket,[ML/'datasets/Banana/real_banana_google_vendor_images/V2-22']),
      'tomato':(inf.analyze_tomato_basket,[ML/'datasets/Tomato/Vendor styled images for dataset']),
      'mango':(inf.analyze_mango_basket,[ML/'datasets/Mango/Mango Segmentiaon Dataset']),
      'apple':(inf.analyze_apple_basket,[ML/'models/Apple', ML/'datasets/Apple/Apple Segmenting Dataset.v1i.coco-segmentation/test'])}
res={}
for fr,(fn,dirs) in sets.items():
    files=[p for d in dirs for p in sorted(d.iterdir()) if p.suffix.lower() in EXT][:25]
    rows=[]
    for i,p in enumerate(files):
        bgr=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_COLOR)
        if bgr is None: continue
        t=time.perf_counter(); out=fn(bgr); ms=(time.perf_counter()-t)*1000
        pieces,result,crops=out
        if i==0 and not rows: print(fr,'result fields:',[k for k in vars(result)][:30])
        r=vars(result)
        rows.append(dict(file=p.name,ms=ms,n=len(pieces),score=r.get('freshness_score'),tier=r.get('freshness_tier'),hist=r.get('class_histogram'),retake=r.get('retake_required') or r.get('needs_retake')))
        if len(rows)<=4:
            vis=bgr.copy()
            for pc,c in zip(pieces,crops):
                b=getattr(pc,'box',None) or getattr(pc,'bbox',None)
                if b is None: continue
                x1,y1,x2,y2=map(int,b[:4]); lab=c.get('class_name') or c.get('label') or ''
                col=(60,180,60) if lab in('ripe','freshripe','fresh') else (0,140,255) if 'unripe' in lab or 'partial' in lab else (40,40,220)
                cv2.rectangle(vis,(x1,y1),(x2,y2),col,max(2,vis.shape[1]//300))
            cv2.imwrite(str(OUT/'e2e'/f'{fr}_{len(rows)}.jpg'),vis)
    res[fr]=dict(n_images=len(rows),mean_ms_cpu=float(np.mean([r['ms'] for r in rows[1:]])),mean_pieces=float(np.mean([r['n'] for r in rows])),rows=rows)
    print(fr,len(rows),'imgs',round(res[fr]['mean_ms_cpu']),'ms',round(res[fr]['mean_pieces'],1),'pieces',flush=True)
json.dump(res,open(OUT/'e2e_results.json','w'),indent=2,default=str)
