"""Reproduce the independent grammar review's numbers (mm, degrees).

Only writes beside this script. feetech_geometry.py is copied unchanged from
E15. Coordinates here: X = flat radial finger direction, Y = palm normal,
Z = tangential. This differs from the repository's palm-coordinate labels.
Sampling reports motor-box intersections, never full hardware certification.
"""

from pathlib import Path
from itertools import product, combinations
from functools import lru_cache
import hashlib
import json
import math
import numpy as np
import feetech_geometry as g

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261013)


def ceil10(x):
    return 10 * math.ceil((x - 1e-10) / 10)


def grid(pitch, inner, outer):
    k = math.ceil(outer / pitch)
    return [(pitch*i, pitch*j) for i in range(-k, k+1)
            for j in range(-k, k+1)
            if inner <= math.hypot(pitch*i, pitch*j) <= outer]


def sat_generic(ca, A, ha, cb, B, hb):
    aa, bb = np.swapaxes(A, -1, -2), np.swapaxes(B, -1, -2)
    cross = np.cross(aa[..., :, None, :], bb[..., None, :, :])
    cross = cross.reshape(cross.shape[:-3] + (9, 3))
    ns = np.concatenate([aa, bb, cross], axis=-2)
    norm = np.linalg.norm(ns, axis=-1)
    valid = norm > 1e-10
    ns = ns / np.maximum(norm[..., None], 1e-30)
    ra = np.abs(np.einsum('...ki,...ij->...kj', ns, A)) @ ha
    rb = np.abs(np.einsum('...ki,...ij->...kj', ns, B)) @ hb
    sep = np.abs(np.einsum('...ki,...i->...k', ns, cb-ca)) - ra - rb
    return np.max(np.where(valid, sep, -np.inf), axis=-1)


def motor_at(base_xz, facing, tilt, kind):
    # e = cos(t)*u+sin(t)*n; c = cos(t)*n-sin(t)*u.
    f, t = np.deg2rad([facing, tilt])
    u = np.array([np.cos(f), 0., np.sin(f)])
    e = np.cos(t)*u + np.sin(t)*g.Y
    c = np.cos(t)*g.Y - np.sin(t)*u
    F = np.column_stack([e, c, np.cross(e, c)])
    A = F @ g.frame(g.KINDS[kind])
    # Top of rectangular two-mm outer enclosure is on palm face Y=0.
    centre0 = 7.5 * A[:, 1]
    depth = centre0[1] + np.abs(A[1]) @ (g.H + 2)
    centre = np.array([base_xz[0], -depth, base_xz[1]]) + centre0
    return centre, A


def chain_poses(kinds, lengths, q, slides):
    q = np.atleast_2d(q)
    R = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    p = np.zeros((len(q), 3))
    centres, frames = [], []
    for j, k in enumerate(kinds):
        a = g.KINDS[k]
        A = g.frame(a)
        c, _ = g.box(np.zeros(3), A, slides[j])
        centres.append(p + R @ c)
        frames.append(R @ A)
        R = R @ g.rotation(a, q[:, j])
        p = p + (R @ g.X) * lengths[j]
    return np.stack(centres, axis=1), np.stack(frames, axis=1)


def chain_stats(kinds, lengths, slides, count=50000):
    lo = np.array([g.LIMITS[k][0] for k in kinds])
    hi = np.array([g.LIMITS[k][1] for k in kinds])
    start = np.array([31.5 if k == 'B' else 0 for k in kinds])
    q = np.vstack([np.zeros(len(kinds)), start,
                   RNG.uniform(lo, hi, (count, len(kinds)))])
    centres, frames = chain_poses(kinds, lengths, q, slides)
    collision = np.zeros(len(q), dtype=bool)
    min_margin = np.full(len(q), np.inf)
    pairs = {}
    for i, j in combinations(range(len(kinds)), 2):
        m = g.sat_margin(centres[:, i], frames[:, i], centres[:, j], frames[:, j])
        min_margin = np.minimum(min_margin, m)
        collision |= m < -1e-8
        if np.any(m[2:] < -1e-8):
            pairs[f'{i+1},{j+1}'] = float(np.mean(m[2:] < -1e-8))
    return {'kinds': kinds, 'lengths': lengths, 'slides': slides,
            'zero_min_sat_mm': float(min_margin[0]),
            'start_min_sat_mm': float(min_margin[1]),
            'random_pose_intersection_fraction': float(collision[2:].mean()),
            'samples': count, 'colliding_pairs': pairs}


def certify_motor_chain(kinds, lengths, slides, step=5.):
    """Continuous lower distance bounds for nonadjacent motor boxes.

    SAT gives a distance lower bound at a sample. Every intervening angle
    is at most step/2 degrees from a sample. The moving downstream box's
    radius about each axis is bounded by intervening link lengths plus its
    maximum radius about its own joint. Sum radius*angle to bound the change
    in distance. Shared upstream rotations cancel. Adjacent pairs use E15's
    analytical separators and are omitted here.
    """
    out = {}
    for i,j in combinations(range(len(kinds)),2):
        if j-i < 2:
            continue
        vals = [g.samples(g.KINDS[kinds[k]],step) for k in range(i,j)]
        angles = np.array(list(product(*vals)))
        smallest = np.inf
        for block in np.array_split(angles,max(1,len(angles)//4096+1)):
            q = np.zeros((len(block),len(kinds)))
            q[:,i:j] = block
            c,A = chain_poses(kinds,lengths,q,slides)
            smallest = min(smallest,float(g.sat_margin(c[:,i],A[:,i],c[:,j],A[:,j]).min()))
        A = g.frame(g.KINDS[kinds[j]])
        c,_ = g.box(np.zeros(3),A,slides[j])
        radius = float(np.linalg.norm(g.vertices(c,A),axis=1).max())
        error = sum(sum(lengths[k:j])+radius for k in range(i,j))*math.radians(step/2)
        out[f'{i+1},{j+1}'] = {'sampled_sat_lower_mm':smallest,
                              'continuous_error_bound_mm':error,
                              'certified_gap_lower_mm':smallest-error,
                              'grid_poses':len(angles)}
    return out


def count_chains(axes, feasible=True):
    """Exact DP counts: length <=250; one to five joints; no geometry tests.

    Feetech zero connections: both axes transverse, distinct lines, and no
    two successive zero links. Ten-mm terminal pad added to E15 point floor.
    Current rules: 0 or 20..90, terminal 10..90, optional coupling at j>0.
    """
    A = [g.direction(*a) for a in axes]
    @lru_cache(None)
    def rec(left, last, previous_zero, budget):
        if left == 1:
            if feasible:
                a = A[last]
                tip = ceil10(2 + 14.5*abs(a[0]) + g.front_sweep(a)*np.sqrt(1-a[0]**2) + 10)
            else:
                tip = 10
            return max(0, (min(90, budget)-tip)//10+1)
        total = 0
        a = A[last]
        for j, b in enumerate(A):
            if feasible:
                floor = ceil10(g.plane_rule(a, b))
                vals = list(range(floor, min(90, budget)+1, 10))
                if (not previous_zero and abs(a[0]) < 1e-8 and abs(b[0]) < 1e-8
                        and abs(a @ b) < 1-1e-8):
                    vals.append(0)
                type_factor = 1
            else:
                vals = [0] + list(range(20, min(90, budget)+1, 10))
                type_factor = 2
            total += type_factor * sum(rec(left-1, j, v == 0, budget-v) for v in vals)
        return total
    return [sum(rec(n, i, False, 250) for i in range(len(A))) for n in range(1, 6)]


def main():
    result = {'seed': 20261013,
              'e15_geometry_sha256': hashlib.sha256((OUT/'feetech_geometry.py').read_bytes()).hexdigest()}
    shell_rad = float(np.linalg.norm([26.5, 16.5, 12]))
    flat_rad = float(np.hypot(26.5, 16.5))
    result['layout'] = {
        'mount_radius': 20, 'radial_bend_required_r': 20+26.5,
        'flat_any_facing_required_r': 20+flat_rad,
        'any_orientation_required_r': 20+shell_rad,
        'orientation_independent_shell_pitch': 2*shell_rad,
        'points_35_55_90': grid(35, 55, 90),
        'points_40_55_90': grid(40, 55, 90),
        'current_empty_grid_count': len(grid(10, 10, 160)),
        'old_continuous_median_r': math.sqrt((10**2+160**2)/2),
        'new_continuous_median_r': math.sqrt((55**2+90**2)/2),
        'new_grid_median_r': float(np.median([math.hypot(*p) for p in grid(40,55,90)]))}
    # Explicit allowed outward-facing counterexample, both with bend first.
    counter = []
    for pitch in [35, 40]:
        p, q = (2*pitch, 0), (2*pitch, pitch)
        c, A = motor_at(p, 270, 0, 'B')
        d, B = motor_at(q, 90, 0, 'B')
        counter.append({'pitch': pitch, 'bases_xz': [p, q], 'facings_deg': [270,90],
                        'bare_sat_mm': float(g.sat_margin(c,A,d,B)),
                        'outer_enclosure_sat_mm': float(sat_generic(c,A,g.H+2,d,B,g.H+2))})
    result['pitch_counterexamples'] = counter
    # All admissible flat-facing/on-kind combinations on neighboring sites.
    points = grid(40,55,90)
    vals, weights = [], []
    for p, q in combinations(points, 2):
        if abs(np.linalg.norm(np.array(p)-q)-40) > 1e-8:
            continue
        fp = [f for f in range(0,360,30) if np.dot(p,[np.cos(np.deg2rad(f)),np.sin(np.deg2rad(f))]) >= -1e-8]
        fq = [f for f in range(0,360,30) if np.dot(q,[np.cos(np.deg2rad(f)),np.sin(np.deg2rad(f))]) >= -1e-8]
        for f, ff, k, kk in product(fp, fq, 'BST', 'BST'):
            c,A = motor_at(p,f,0,k); d,B = motor_at(q,ff,0,kk)
            vals.append(float(g.sat_margin(c,A,d,B)))
            w = {'B':.69,'S':.27,'T':.04}
            weights.append(w[k]*w[kk]/len(fp)/len(fq))
    result['flat_nearest_neighbor_enumeration'] = {
        'state_pairs': len(vals),
        'bare_intersection_fraction_weighted': float(np.average(np.array(vals)<-1e-8,weights=weights)),
        'description':'Each adjacent site pair equally weighted; facing uniform; kinds 69/27/4; no global constraints.'}
    # Flat slab extents at the joint origin; top-flush depth and back protrusion.
    dims = {}
    for k in 'BST':
        A = g.frame(g.KINDS[k]); c = 7.5*A[:,1]
        rad = float(np.abs(A[1])@(g.H+2))
        front = float(c[1]+rad); back = float(c[1]-rad)
        dims[k] = {'signed_normal_interval_mm':[back,front],
                   'depth_below_face_mm':front, 'height_mm':front-back,
                   'protrusion_below_24mm_plate_mm':front-back-24}
    result['palm_depth'] = dims
    result['hinge'] = {'no_penetration_gap_mm':24*math.tan(math.radians(15)),
                        'gap_for_2mm_plane_clearance_mm':2*(12*.5+2)/(1+math.cos(math.radians(30)))}
    coarse = [(az,el) for az in range(0,180,30) for el in [-60,-30,0,30,60]] + [(0,90)]
    ca = [g.direction(*a) for a in coarse]
    floors = [g.plane_rule(a,b) for a in ca for b in ca]
    result['lengths'] = {'all_axis_continuous_max_mm':2+np.hypot(14.5,g.FRONT_RADIUS)+np.hypot(14.5,24.5),
        'coarse_axis_max_mm':float(max(floors)),
        'coarse_rounded_floors':sorted(set(ceil10(x) for x in floors)),
        'on_kind': {a+b: {'unrounded':float(g.plane_rule(g.KINDS[a],g.KINDS[b])),
                          'coarse':ceil10(g.plane_rule(g.KINDS[a],g.KINDS[b]))} for a,b in product('BST',repeat=2)},
        'coarse_tip_with_10mm_pad_max':max(ceil10(2+14.5*abs(a[0])+g.front_sweep(a)*math.sqrt(max(0,1-a[0]**2))+10) for a in ca)}
    result['offsets'] = {a+b:float(g.slide_for_separator(g.KINDS[a],g.KINDS[b])) for a,b in product('BST',repeat=2)}
    result['crossed_221'] = [chain_stats('SBSBB',[0,L,0,L,30],[-26.5,0,-26.5,0,0]) for L in [50,60,80]]
    result['crossed_221_continuous_certificate'] = certify_motor_chain('SBSBB',[0,80,0,80,30],[-26.5,0,-26.5,0,0])
    assert min(p['certified_gap_lower_mm'] for p in result['crossed_221_continuous_certificate'].values()) > 2
    # Adjacent flat fingers, each starts with one of the two crossed orders.
    neighbor = []
    for first,second in product(['SB','BS'],repeat=2):
        c,A=chain_poses(first,[0,30],[0,0],[-26.5,0])
        d,B=chain_poses(second,[0,30],[0,0],[-26.5,0])
        depth_c=float((c[0,:,1]+np.abs(A[0,:,1,:])@(g.H+2)).max())
        depth_d=float((d[0,:,1]+np.abs(B[0,:,1,:])@(g.H+2)).max())
        c=c-depth_c*g.Y
        d=d-depth_d*g.Y+40*g.Z
        margin=min(float(g.sat_margin(c[0,i],A[0,i],d[0,j],B[0,j])) for i,j in product(range(2),repeat=2))
        neighbor.append({'first':first,'second':second,'pitch':40,
                         'joint_depths_below_face_mm':[depth_c,depth_d],
                         'min_bare_sat_mm':margin})
    result['crossed_neighbors'] = neighbor
    # Conservative test of the crossed separator over the entire upstream range.
    crossed_gaps={}
    for a,b in [('S','B'),('B','S')]:
        ds=g.box_distance(*g.pair_poses(g.KINDS[a],g.KINDS[b],0,g.samples(g.KINDS[a],.25),slides=(-26.5,0)))
        assert ds.min() >= 2-1e-8
        crossed_gaps[a+b]=float(ds.min())
    result['crossed_pair_sampled_gap_mm']=crossed_gaps
    # DP search-space comparison, excludes palm joints, geometry and C2.
    counts = {}
    for label,axes in [('kinds',[(0,0),(90,0),(0,90)]),('coarse',coarse)]:
        new = count_chains(axes,True); old = count_chains(axes,False)
        def hand_count(sites,directions,chain_count,n):
            return math.comb(sites,n)*(directions*chain_count)**n
        # New absolute facings: 4 axial sites admit 7, 12 others admit 6 =>100;
        # five tilts, so 500 direction/tilt/base states, avg31.25 per base.
        orient = [5*len([f for f in range(0,360,30) if np.dot(p,[np.cos(np.deg2rad(f)),np.sin(np.deg2rad(f))])>=-1e-8]) for p in points]
        def exact_layout_directions(n):
            return sum(math.prod(orient[i] for i in ids) for ids in combinations(range(len(points)),n))
        counts[label]={'new_chains_by_joint_count':new,'old_chains_by_joint_count':old,
                       'hand_log10':{str(n):{'new':math.log10(exact_layout_directions(n)*sum(new)**n),
                                            'old':math.log10(hand_count(796,60,sum(old),n))} for n in [4,6]}}
    result['space_counts']=counts
    # The first-servo clock/support numbers do not assert complete-pad coplanarity.
    assert len(points)==16
    assert max(ceil10(x) for x in floors)==50
    assert all(x['bare_sat_mm']<0 for x in counter)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
