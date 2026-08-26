# v1_kinematic — frozen before joint coupling

The sampler as it stood once the kinematic parameters were settled: palm
25 x 60 x 60 mm, capsule radius 10 mm, total finger length 100 mm, minimum link
30 mm, one in-plane `splay` per finger, every joint independently actuated.

86 finger designs, 658,244 discrete hands.

Restore with:

    cp checkpoints/v1_kinematic/{sampler,viewer,preview}.py .
