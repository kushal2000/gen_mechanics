# Copied onto martin/hand-grammar

Copied byte for byte from Kushal's branch `2026-10-06-controlled_wuji_experiments` (commit bbd0d95ce7104ce56bfe3107203f870f6e4bcb7b), `assets/urdf/unified_commercial_hands/`, for the grammar's commercial reference set (GRAMMAR-LOCK item 19):

- `dex3/`: `dex3_left.urdf`, `dex3_left.spec.json`, `vendor_left/LICENSE`, `vendor_left/SOURCE.md`
- `wuji2/`: `wuji2_left.urdf`, `wuji2_right.urdf`, their `.spec.json`, `vendor_left|right/LICENSE` and `SOURCE.md`

The grammar's conform reads joint origins and axes only; the vendor meshes (`dex3/vendor_left/meshes/`, `wuji2/vendor_left|right/meshes/`, 60 files, 19 MB) are copied too so the grammar viewer can draw each conformed hand over its real meshes. The vendor URDFs themselves are not copied. Provenance and licences: each `vendor_*/SOURCE.md` and `LICENSE` (Dex3: Unitree, BSD-3; Wuji v2: Wuji, MIT).
