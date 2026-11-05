import sys, os, json, numpy as np
sys.path.insert(0, '/share/portal/kk837/gen_mechanics')
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from hand_sampler.grammar import hand as H, derive as dv
root='/share/portal/kk837/gen_mechanics/assets/urdf/29oct_niche_seed'
rng=np.random.default_rng(1)
P=2000
cent=np.stack([rng.uniform(0.03,0.09,P), rng.uniform(-0.05,0.05,P), rng.uniform(-0.05,0.05,P)],1)
R=Rot.random(P,random_state=2).as_matrix()
tol=0.012
for d in sys.argv[1:]:
    h=H.hand_from_dict(json.load(open(f'{root}/{d}/{d}.design.json')))
    ds=dv.dofs(h); q=dv.sample_q(h,80000,rng,ds); tips=dv.fk(h,q).tips; F=tips.shape[1]
    trees=[cKDTree(tips[:,i]) for i in range(F)]
    nf=np.zeros(P); per=np.zeros(F)
    for k in range(P):
        c=cent[k]; pts=[]
        for ax,hl in ((1,0.015),(2,0.015)):
            n=R[k][:,ax]
            for s in (-0.03,0,0.03):
                for sg in (1,-1): pts.append(c+sg*hl*n+s*R[k][:,0])
        for sg in (1,-1): pts.append(c+sg*0.05*R[k][:,0])
        pts=np.array(pts)
        r=[(trees[i].query(pts,distance_upper_bound=tol)[0]<1).any() for i in range(F)]
        nf[k]=sum(r); per+=r
    print(d, 'mean fingers touching-able', nf.mean().round(2), '>=3:', (nf>=3).mean().round(3), 'per finger', (per/P).round(2))
