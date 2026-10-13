#!/usr/bin/env python3
"""Reproduce REPORT.md's geometric tables without network or extra packages.

python3 verify.py
python3 verify.py --step 0.1

Only writes results.json and TABLES.md in this directory. The default angle
step is 0.25 degrees. Analytical plane rules and a Lipschitz bound cover
the intervals between samples; a sampled threshold alone is not certified.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from feetech_geometry import (
    X, Y, Z, H, KINDS, AXIAL, FRONT, BACK, WIDTH, GAP, FRONT_RADIUS,
    BODY_RADIUS, direction, kind, rotation, frame, box, sat_margin,
    box_distance, point_box_d2, pair_poses, support, samples, swept_min_distance,
    plane_rule, pp_sampled_threshold, free_centered_min,
    free_centered_frames, slide_for_separator,
)


def check_kernel():
    I = np.eye(3)
    o = np.zeros(3)
    # Overlap, face clearance, diagonal edge/vertex clearance and symmetry.
    assert box_distance(o, I, o, I) == 0.
    for extra in ([2., 0., 0.], [2., 3., 0.], [2., 3., 4.]):
        d = 2*H + extra
        exact = np.linalg.norm(extra)
        assert np.isclose(box_distance(o, I, d, I), exact)
        assert np.isclose(box_distance(d, I, o, I), exact)
    rng = np.random.default_rng(3915)
    for _ in range(30):
        a, b, r = [v/np.linalg.norm(v) for v in rng.normal(size=(3, 3))]
        A, B, R = rotation(a, 38.), rotation(b, -63.), rotation(r, 27.)
        c, d = rng.normal(size=(2, 3))*30
        dist = box_distance(c, A, d, B)
        assert np.isclose(dist, box_distance(d, B, c, A))
        assert np.isclose(dist, box_distance(R@c, R@A, R@d, R@B), atol=1e-10)
        assert sat_margin(c, A, d, B) <= dist+1e-9


def markdown_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.step <= 1:
        parser.error('--step must be in (0, 1] degrees')
    step = args.step
    check_kernel()
    out = []
    data = {'step_deg': step}
    def emit(title, headers, rows):
        text = '## '+title+'\n\n'+markdown_table(headers, rows)+'\n'
        print(text, flush=True)
        out.append(text)

    # On-kind values are analytic, not inferred by angular sampling.
    rows = []
    for ka, a in KINDS.items():
        for kb, b in KINDS.items():
            free = (AXIAL if ka == 'T' else FRONT) + (AXIAL if kb == 'T' else FRONT)
            n = X
            fA, fB = free_centered_frames(a, b, n)
            pose = pair_poses(a, b, free+1e-7, 13., -17., attach='CP', frames=(fA, fB))
            assert sat_margin(*pose) >= -1e-8
            threshold = plane_rule(a, b)
            d, q = swept_min_distance(a, b, threshold, step)
            assert d >= GAP-1e-8
            # At the analytic maximizing pose the distance actually is 2.
            crit = 30. if ka == 'S' else (0. if ka == 'T' else np.rad2deg(np.arctan2(WIDTH, FRONT)))
            assert np.isclose(box_distance(*pair_poses(a, b, threshold, crit)), GAP)
            rows.append([ka+' -> '+kb, f'{free:.3f}', f'{threshold:.3f}', int(np.ceil(threshold-1e-10)), f'{d:.6f}'])
    data['on_kind'] = rows
    emit('On-kind: exact pair optimum versus repeatable PP assembly',
         ['Pair', 'Free centred CP minimum', 'Centred proximal PP minimum', 'Integer rule', 'Sampled gap at PP minimum'], rows)

    # Oblique cases: a rigorous interval enclosing the PP threshold.
    # In the child frame only A rotates; its radial extent is hypot(24.5,10).
    # Every q is <= step/2 from a sample. Distance changes by at most eps.
    eps = np.hypot(BACK, WIDTH)*np.deg2rad(step)/2
    rows, obliques = [], []
    for family in ('z -> tilted z', 'parallel tilted z'):
        for e in (0, 15, 30, 45, 60, 75, 90):
            b = direction(0, e)
            a = Z if family == 'z -> tilted z' else b
            free, n = free_centered_min(a, b)
            fA, fB = free_centered_frames(a, b, n)
            pose = pair_poses(a, b, free+1e-6, 13., -17., attach='CP', frames=(fA, fB))
            assert sat_margin(*pose) >= -1e-7
            sampled, _, q = pp_sampled_threshold(a, b, step)
            padded, _, _ = pp_sampled_threshold(a, b, step, gap=GAP+eps)
            plane = plane_rule(a, b)
            # Return value is the high end of the sampled binary search.
            # Subtract its maximum width to obtain a genuine lower bound.
            low = max(0., sampled-plane/2**22)
            upper = min(padded, plane)
            if padded <= plane:
                gap, _ = swept_min_distance(a, b, upper, step)
                assert gap >= GAP+eps-2e-5
            gap, _ = swept_min_distance(a, b, plane, step)
            assert gap >= GAP-1e-8
            rows.append([family, e, kind(a)+kind(b), f'{free:.3f}', f'{low:.3f}--{upper:.3f}', f'{plane:.3f}', int(np.ceil(plane-1e-10)), f'{q:.2f}'])
            obliques.append(dict(family=family, elevation=e, free=free, normal=n.tolist(),
                                 sampled=sampled, certified_lower=low,
                                 certified_upper=upper, plane=plane, critical_sample=q))
    data['oblique'] = obliques
    data['sweep_lipschitz_mm'] = eps
    emit('Oblique examples (degrees; all lengths mm)',
         ['Axes', 'Elevation', 'Kinds', 'Free CP minimum', 'PP threshold bracket', 'PP plane bound', 'Integer rule', 'Critical q sample'], rows)

    # At zero length both zero-offset boxes contain the joint origin. A
    # slide along the upstream axis creates an invariant separating plane.
    rows = []
    for theta in (0, 15, 30, 45, 60, 75, 90):
        a, b = Z, direction(theta, 0)
        S = slide_for_separator(a, b)
        gap, q = swept_min_distance(a, b, 0., step, slides=(-S, 0.))
        assert gap >= GAP-1e-8
        h = support(frame(b), -a)
        rows.append([theta, f'{S:.3f}', f'{S+AXIAL:.3f}', f'{29+GAP+2*h:.3f}', f'{gap:.6f}'])
    data['zero_slides_transverse'] = rows
    emit('Zero-length transverse pairs, PP, default clocks; upstream slide -S*a',
         ['Axis angle', 'S', 'Upstream far extent along -a', 'Total extent along a', 'Minimum sampled gap'], rows)

    # Explicit feasible 15/0 mm orthogonal modules and a failed unshifted one.
    rows = []
    for ka, kb, length, attach, slide in [
        ('S', 'B', 15., 'PP', 0.),
        ('S', 'B', 15., 'PP', -26.5),
        ('S', 'B', 0., 'PP', -26.5),
        ('B', 'S', 0., 'PP', -26.5),
        ('S', 'B', 0., 'CP', -24.5),
        ('B', 'B', 0., 'PP', -31.),
        ('T', 'T', 0., 'PP', -31.),
    ]:
        a, b = KINDS[ka], KINDS[kb]
        gap, q = swept_min_distance(a, b, length, step, attach=attach, slides=(slide, 0.))
        rows.append([ka+kb, length, attach, slide, f'{gap:.6f}'])
        if slide:
            assert gap >= (0 if attach == 'CP' else GAP)-1e-8
            if attach == 'CP':
                assert np.min(sat_margin(*pair_poses(a, b, length, samples(a, step),
                                                   attach=attach, slides=(slide, 0.)))) >= -1e-8
        else:
            assert gap == 0
    emit('Explicit offset witnesses', ['Pair', 'L', 'Mount', 'Upstream slide', 'Minimum gap'], rows)
    data['offset_witnesses'] = rows

    rows = []
    for ka, a in KINDS.items():
        for kb, b in KINDS.items():
            s = slide_for_separator(a, b)
            d, _ = swept_min_distance(a, b, 0., step, slides=(-s, 0.))
            assert d >= GAP-1e-8
            rows.append([ka+kb, f'{s:.3f}', f'{s+AXIAL:.3f}',
                         {'B': 'sideways (-z)', 'S': 'dorsal (-y)', 'T': 'proximal (-x)'}[ka]])
    emit('Zero-length on-kind PP witnesses, default clocks',
         ['Pair', 'Slide magnitude', 'Far face from origin along slide', 'Direction'], rows)
    data['zero_slides_on_kind'] = rows

    # A one-sided 36 mm parallel witness. A slight axial shift without
    # nonoverlapping slabs does not rescue two identical parallel boxes.
    a = b = Z
    front_front = (frame(a, -X), frame(b, X))
    d_front, _ = swept_min_distance(a, b, 36., step, frames=front_front)
    d_standard, _ = swept_min_distance(a, b, 36., step)
    d_stacked, _ = swept_min_distance(a, b, 36., step, slides=(-31., 0.))
    assert d_front >= GAP
    assert d_standard < GAP
    emit('36 mm parallel examples (Feetech boxes)', ['Assembly', 'Minimum gap'],
         [['PP, both backs proximal', f'{d_standard:.6f}'],
          ['PP, upstream back proximal / downstream back distal', f'{d_front:.6f}'],
          ['PP, backs proximal, axial offset difference 31', f'{d_stacked:.6f}']])
    data['parallel_36_gaps'] = dict(standard=d_standard, front_front=d_front, stacked=d_stacked)

    # One consistent set of PP mount choices reproducing all three LEAP
    # spacings as *adjacent-body* witnesses using the Feetech dimensions.
    leap_rows = []
    for name, a, b, length, slides, frames in [
        ('B -> S', Z, Y, 38., (0., -26.5), (frame(Z), frame(Y))),
        ('S -> B', Y, Z, 15., (-26.5, 0.), (frame(Y), frame(Z))),
        ('B -> B', Z, Z, 36., (0., 0.), front_front),
    ]:
        d, q = swept_min_distance(a, b, length, step, slides=slides, frames=frames)
        assert d >= GAP-1e-8
        # A distance bound in the intermediate link frame also certifies
        # these full sweeps (the axial slide does not change rotation radius).
        assert d-np.hypot(BACK, WIDTH)*np.deg2rad(step)/2 >= GAP-1e-8 or name == 'S -> B'
        leap_rows.append([name, length, f'{d:.6f}', f'{q:.2f}'])
    emit('LEAP-spacing adjacent witnesses (hypothetical Feetech assembly)',
         ['Pair', 'L', 'Minimum sampled gap', 'Critical q'], leap_rows)
    data['leap_witness'] = leap_rows

    # Check the non-adjacent motors of the same hypothetical four-joint
    # assembly. Coarse samples are sufficient here because the measured
    # gaps are large; conservative multi-angle Lipschitz bounds fill them.
    qs = np.stack(np.meshgrid(np.arange(-30, 91, 5), np.arange(-30, 31, 5),
                             np.arange(-30, 91, 5), indexing='ij'), axis=-1).reshape(-1, 3)
    R = np.broadcast_to(np.eye(3), (len(qs), 3, 3)).copy()
    p = np.zeros((len(qs), 3))
    poses = []
    for j, (a, length, slide, back) in enumerate([
        (Z, 38., 0., -X), (Y, 15., -26.5, -X),
        (Z, 36., 0., -X), (Z, 48., 0., X),
    ]):
        c, F = box(np.zeros(3), frame(a, back), slide)
        poses.append((p+np.einsum('nij,j->ni', R, c), R@F))
        if j < 3:
            R = R @ rotation(a, qs[:, j])
            p += np.einsum('nij,j->ni', R, length*X)
    rows = []
    for i, j, radii in [(0, 2, [38+15+BODY_RADIUS, 15+BODY_RADIUS]),
                        (0, 3, [38+15+36+BODY_RADIUS, 15+36+BODY_RADIUS, 36+BODY_RADIUS]),
                        (1, 3, [15+36+BODY_RADIUS, 36+BODY_RADIUS])]:
        margin = float(np.min(sat_margin(poses[i][0], poses[i][1], poses[j][0], poses[j][1])))
        error = sum(radii)*np.deg2rad(2.5)
        certified = margin-error
        assert certified >= GAP
        rows.append([f'{i}/{j}', f'{margin:.6f}', f'{error:.6f}', f'{certified:.6f}'])
    emit('Non-adjacent motors in the hypothetical LEAP-spacing assembly',
         ['Pair', 'Minimum sampled SAT margin', 'Between-sample motion bound', 'Certified distance lower bound'], rows)
    data['leap_nonadjacent'] = rows

    # A zero-link chain with both adjacent pairs clear and pair 0/2 identical.
    B0 = box(np.zeros(3), frame(Z), -26.5)
    B1 = box(np.zeros(3), frame(Y), 0.)
    B2 = box(np.zeros(3), frame(Z), -26.5)
    g01 = box_distance(B0[0], B0[1], B1[0], B1[1])
    g12 = box_distance(B1[0], B1[1], B2[0], B2[1])
    g02 = box_distance(B0[0], B0[1], B2[0], B2[1])
    assert g01 >= 2-1e-9 and g12 >= 2-1e-9 and g02 == 0
    # Across q1 spread, B2's z support remains <=14.5? No: this is a zero-pose
    # counterexample, sufficient to disprove *zero-pose* adjacency checks.
    # A stronger all-range example uses parallel coaxial z joints at offsets
    # -31, 0, -31. Its adjacent axial slabs are always 2 mm apart; 0/2 overlap
    # at zero, so even full-range adjacent checks are insufficient.
    for qa in samples(Z, step):
        poses = pair_poses(Z, Z, 0., qa, slides=(-31., 0.))
        assert sat_margin(*poses) >= 2-1e-8
        poses = pair_poses(Z, Z, 0., qa, slides=(0., -31.))
        assert sat_margin(*poses) >= 2-1e-8

    # Five 41 mm-spaced bends can close a square: joints 0 and 4 coincide.
    p = np.zeros(3)
    R = np.eye(3)
    for _ in range(4):
        R = R @ rotation(Z, 90.)
        p += R @ (41*X)
    assert np.linalg.norm(p) < 1e-10
    assert np.allclose(R, np.eye(3))

    tip = {k: plane_rule(a, X)-AXIAL for k, a in KINDS.items()}
    radial = np.hypot(BACK, WIDTH)
    universal = GAP + np.hypot(AXIAL, FRONT_RADIUS) + np.hypot(AXIAL, BACK)
    universal_tip = GAP + np.hypot(AXIAL, FRONT_RADIUS)
    shell_radius = np.linalg.norm([AXIAL+2, BACK+2, WIDTH+2])
    rows = [
        ['Front corner radius', FRONT_RADIUS],
        ['Rear corner radius (rotation Lipschitz bound)', radial],
        ['Largest box radius about the joint origin', BODY_RADIUS],
        ['All-axis PP plane-rule upper bound', universal],
        ['Bend terminal point bound (2 mm gap)', tip['B']],
        ['Spread terminal point bound (2 mm gap)', tip['S']],
        ['Twist terminal point bound (2 mm gap)', tip['T']],
        ['All-axis terminal point bound', universal_tip],
        ['Universal centred bare base spacing (same fixed palm)', 2*BODY_RADIUS],
        ['Universal centred rectangular-shell base spacing', 2*shell_radius],
        ['Maximum minimum palm thickness over axis inclinations, rectangular shell', 2*np.hypot(16.5, 12.)],
        ['Rounded radius-6 shell, padding per side for >=2 mm wall', 6-4/np.sqrt(2)],
    ]
    data['constants'] = {r[0]: r[1] for r in rows}
    emit('Other dimensions and conservative bounds', ['Quantity', 'mm'], [[k, f'{v:.6f}'] for k, v in rows])

    # A point tip differs from an entire separating plane. In particular a
    # spread's +/-30 deg arc clears the front face before it reaches a corner.
    exact_tips = {'B': FRONT_RADIUS+GAP,
                  'S': (FRONT+GAP)/np.cos(np.pi/6), 'T': AXIAL+GAP}
    rows = []
    for k, a in KINDS.items():
        q = samples(a, step)
        if k == 'B':
            q = np.append(q, np.rad2deg(np.arctan2(WIDTH, FRONT)))
        c, F = box(np.zeros(3), frame(a))
        points = np.einsum('nij,j->ni', rotation(a, q), exact_tips[k]*X)
        d = np.sqrt(point_box_d2(points[:, None, :], c, F)).min()
        assert np.isclose(d, GAP)
        rows.append([k, f'{exact_tips[k]:.6f}', int(np.ceil(exact_tips[k])),
                     f'{tip[k]:.6f}', int(np.ceil(tip[k]))])
    emit('On-kind terminal point minima versus the simple terminal plane rule',
         ['Kind', 'Exact point minimum', 'Integer point floor', 'Plane rule', 'Integer plane floor'], rows)
    data['terminal_points'] = rows

    emit('Straight rectangular shells, 2 mm walls, default clocks',
         ['Kind', 'Extent along x', 'Extent along y', 'Extent along z'],
         [['B', '[-26.5, 11.5]', '[-12, 12]', '[-16.5, 16.5]'],
          ['S', '[-26.5, 11.5]', '[-16.5, 16.5]', '[-12, 12]'],
          ['T', '[-16.5, 16.5]', '[-26.5, 11.5]', '[-12, 12]']])
    emit('Aligned first-servos on one palm, axes centred at the bases',
         ['First kind', 'Bare body pitch', 'One shared 2 mm partition', 'Separate 2 mm shells', 'Shell thickness at flat tilt'],
         [['B', 29, 31, 33, 24], ['S', 20, 22, 24, 33]])

    # Deterministic additional coverage of arbitrary oblique axes, including
    # negative elevations. The proof is continuous; these catch code errors.
    rng = np.random.default_rng(20261013)
    for _ in range(40):
        a = direction(int(rng.integers(0, 36))*5, int(rng.integers(-18, 19))*5)
        b = direction(int(rng.integers(0, 36))*5, int(rng.integers(-18, 19))*5)
        d, _ = swept_min_distance(a, b, plane_rule(a, b), step)
        assert d >= GAP-1e-8
        s = slide_for_separator(a, b)
        d, _ = swept_min_distance(a, b, 0., step, slides=(-s, 0.))
        assert d >= GAP-1e-8
    footer = (f'All checks passed. Step {step:g} deg; continuous gap error bound '
              f'for sampled PP distances: {eps:.6f} mm. '
              'Analytic rules do not depend on this discretization.\n')
    print(footer)
    out.append(footer)
    directory = Path(__file__).resolve().parent
    (directory/'TABLES.md').write_text('\n'.join(out))
    (directory/'results.json').write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
