import sys, os, json, numpy as np
sys.path.insert(0, '/share/portal/kk837/gen_mechanics')
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from hand_sampler.grammar import hand as H, derive as dv
root='/share/portal/kk837/gen_mechanics/assets/urdf/29oct_niche_seed'
rng=np.random.default_rng(1)
P=3000
# pen poses: centre depth below palm-centre plane 3..9 cm, lateral within 5 cm; random orientation
cent=np.stack([rng.uniform(0.03,0.09,P), rng.uniform(-0.05,0.05,P), rng.uniform(-0.05,0.05,P)],1)
R=Rot.random(P,random_state=2).as_matrix()
tol=0.012
names=sys.argv[1:] or sorted(x for x in os.listdir(root) if x.startswith('seed'))
for d in names:
    h=H.hand_from_dict(json.load(open(f'{root}/{d}/{d}.design.json')))
    ds=dv.dofs(h)
    q=dv.sample_q(h,80000,rng,ds)
    tips=dv.fk(h,q).tips
    F=tips.shape[1]
    trees=[cKDTree(tips[:,i]) for i in range(F)]
    best=np.zeros(P,bool); cnt_pairs=np.zeros(P)
    tri=np.zeros(P,bool)
    for k in range(P):
        c=cent[k]; ok=False
        # side faces (two transverse axes) +/-15mm; also ends +/-50mm
        faces=[]
        for ax,hl in ((1,0.015),(2,0.015),(0,0.05)):
            n=R[k][:,ax]; faces.append((c+hl*n, c-hl*n))
        reach={}
        for fi,(a,b) in enumerate(faces):
            ra=[trees[i].query(a,distance_upper_bound=tol)[0]<1 for i in range(F)]
            rb=[trees[i].query(b,distance_upper_bound=tol)[0]<1 for i in range(F)]
            for i in range(F):
                for j in range(F):
                    if i!=j and ra[i] and rb[j]:
                        ok=True
            reach[fi]=(ra,rb)
        best[k]=ok
    # depth dependence
    sh=cent[:,0]<0.06
    print(f"{d:18} F={F} J={len(ds):2} antipodal_cov={best.mean():.3f}  shallow(<6cm)={best[sh].mean():.3f} deep={best[~sh].mean():.3f}",flush=True)
