# Grammar audit (4-5 Nov 2026)

Proof-checking the hand grammar and the 29 Oct evolution runs' mutations, explaining each design step visually, and
the changes that came out of it. Everything here is Kit-free except the two render helpers.

## Audits of the 29 Oct runs (all clean)

| script | what it checks | result |
|---|---|---|
| `audit_mutations.py` | every archived hand keeps the rules and C1, is canonical, has the right key and niche; every child is one application of its recorded operator to its parent (enumerated exhaustively; add_finger checked backwards) | 2,391 hands, no failures; negative controls (wrong parents, corrupted designs) all caught |
| `audit_build.py` | each hand's spec and URDF match its design (joints, limits by kind, axes, fingertips, effort); a sample rebuilt byte-identically | no failures, 40 rebuilt per run |
| `audit_bookkeeping.py` | every recorded score matches its generation's `fitness.json`; the MAP-Elites map replayed from the seeds reproduces every won/lost count and the final map | no mismatches |

## Explainers (figures under `debug_outputs/grammar_audit/`)

| script | figure |
|---|---|
| `grammar_steps.py --step base / tilt / tilt_c1 / joints / clearance` | how a random hand is built, step by step: base spots and facings, tilt (and why 30/60 deg need a >= 70 mm first link), joint kinds, the clearance check while building |
| `centring_figure.py`, `mount_explained.py` | the old mount-centring rule (a 40 mm re-centring jump lands a servo on the mount) vs the derived mount |
| `children_of.py` | every child one mutation away from a design, by level, drawn with joint-kind colours |

## Changes that came out of it (see the commits)

- `add_finger` copies one of the hand's fingers to a free spot (deterministic, enumerable).
- `FEETECH_COARSE_NOCUT_RULES`: no palm cuts; a hand is its fingers' arrangement; the mount is derived (the clear point
  in the base servos' hull nearest the palm centre, `viability.mount_position_mm`); built hands put the wrist frame at
  the palm centre. The old rule sets are unchanged (old hands rebuild byte-identically).
- Joint tokens: `joint_axis` and `joint_link_vec` (with `joint_origin`), the joint's kinematics at the current pose;
  `joint_link_bbox` optional.

## Render helpers

- `recolour_hands.py`: URDF copies coloured by joint kind (for URDF-based viewers; the template scene ignores URDF
  colours).
- `joint_colours.py`: colours a live template scene's colliders by joint kind and stripes roll tips; used by
  `experiments/30oct_evolution_evaluation/render_archive_videos.py --colour-joints`.
