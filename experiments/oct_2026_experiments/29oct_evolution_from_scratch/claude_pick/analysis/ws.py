import sys, os, json, numpy as np
sys.path.insert(0, '/share/portal/kk837/gen_mechanics')
from hand_sampler.grammar import hand as H, derive as dv
root='/share/portal/kk837/gen_mechanics/assets/urdf/29oct_niche_seed'
rng=np.random.default_rng(0)
V=0.01
rows=[]
for d in sorted(x for x in os.listdir(root) if x.startswith('seed')):
    h=H.hand_from_dict(json.load(open(f'{root}/{d}/{d}.design.json')))
    ds=dv.dofs(h)
    q=dv.sample_q(h,60000,rng,ds)
    tips=dv.fk(h,q).tips   # N,F,3 palm frame, x = grasp side
    F=tips.shape[1]
    sets=[]; below=[]
    for i in range(F):
        t=tips[:,i]
        m=(t[:,0]>0.025)&(t[:,0]<0.10)&(np.hypot(t[:,1],t[:,2])<0.10)
        below.append(m.mean())
        vox=set(map(tuple,np.floor(t[m]/V).astype(int)))
        sets.append(vox)
    # pinch voxels: voxel reached by >=2 fingers (within 1 voxel dilation ~ pen size)
    from collections import Counter
    def dil(s):
        o=set()
        for (a,b,c) in s:
            for da in (-1,0,1):
                for db in (-1,0,1):
                    for dc in (-1,0,1): o.add((a+da,b+db,c+dc))
        return o
    dsets=[dil(s) for s in sets]
    cnt=Counter(v for s in dsets for v in s)
    n2=sum(1 for v,c in cnt.items() if c>=2); n3=sum(1 for v,c in cnt.items() if c>=3)
    # best pair overlap
    pairs=sorted(((len(dsets[a]&dsets[b]),a,b) for a in range(F) for b in range(a+1,F)),reverse=True)
    # palm-press: tips that reach 20-50mm below palm face (x 0.032-0.062) within palm footprint radius 60mm
    pr=[]
    for i in range(F):
        t=tips[:,i]; r=np.hypot(t[:,1],t[:,2])
        pr.append(((t[:,0]>0.03)&(t[:,0]<0.065)&(r<0.06)).mean())
    nj=len(ds)
    rows.append((d,F,nj,n2,n3,pairs[0][0] if pairs else 0,[round(x,2) for x in below],[round(x,2) for x in pr]))
for r in sorted(rows,key=lambda r:-r[3]):
    print(f"{r[0]:18} F={r[1]} J={r[2]:2} vox2={r[3]:5} vox3={r[4]:5} bestpair={r[5]:5} below={r[6]} palmpress={r[7]}")
