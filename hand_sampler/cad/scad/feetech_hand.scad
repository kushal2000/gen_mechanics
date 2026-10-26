// feetech_hand.scad: printable parts for hand-grammar hands built from
// Feetech HL-3915-C001 (or the plastic-case HL-2915-C001) servos.
//
// A generated hand file (hand.scad) includes this library, can override any
// setting in the first block, and places the modules below with the
// grammar's frames (hand_sampler/cad). Units: millimetres.
//
// Servo frame S (every servo-local module): origin on the output axis at the
// case's mid-plane; X along the axis toward the driven horn, Y the servo's
// back, Z its 20 mm width (hand_sampler/cad/servo.py).
//
// How a joint is built: the servo's case sits in a housing in the part
// before the joint (side plates on the two horn faces, screwed through the
// case's far holes, a back wall, side walls). The part after the joint holds
// the horns in a yoke: a U (two arms on the horns and a bridge ahead of the
// case's swing), or one arm on the driven horn for a joint whose axis leans
// along the finger. Every part is its positive pieces, less the servo and
// screw keep-outs, less what its neighbours sweep over the joints' ranges.

// ---------------------------------------------------------------- settings
FIT = 0.1;          // printed socket around a servo case (Martin, 10/24: 0.1 mm a side)
CASE_CLEAR = 0.5;   // a turning servo case to a printed part
PART_CLEAR = 1.0;   // two printed parts that turn against each other
WALL = 2.0;         // housing plates, back wall, bridge
ARM_GAP = 0.1;      // each yoke arm's inner face to its horn's face
ARM_DRIVEN = 3.0;   // the arm on the driven horn (4 x M2 x 5)
ARM_IDLER = 2.0;    // the arm on the idler horn (2 x the servo's box screws)
ARM_R = 9.5;        // an arm's disc around the axis
HEAD_SEAT = 0.6;    // case screws: counterbore depth in the side plates
SWEEP_STEP = 5;     // degrees between swept copies
HORN_REACH = 18.6;  // how far out along the axis the horns' clearance reaches (an arm's outer face and a gap)
PLUG_OUT = 9;       // a servo plug and its cable bend beyond the case face
$fn = 40;

// ------------------------------------------------- the servo (HL-3915-C001)
CASE_X = [-11.5, 11.5];
CASE_Y = [-9.5, 24.5];
CASE_Z = [-10.0, 10.0];
HORN_R = 8.0;
DRIVEN_X = [11.5, 14.5];
IDLER_X = [-14.5, -12.1];
HORN_HOLE_R = 6.0;            // 8 x M2 tapped on this radius, every 45 degrees from +Z
HEAD_R = 2.1;                 // the horns' centre screw heads
HEAD_X = 17.0;
STUB_R = 2.5;                 // the idler stub (idler horn removed)
STUB_X = -14.0;
FAR_HOLES = [[22.5, -8.0], [22.5, 8.0]];   // (Y, Z) of the case holes the housings use
NEAR_HOLES = [[-7.5, -8.0], [-7.5, 8.0]];  // the holes by the horns (a full-face cradle's too)
NEAR_SEAT = 1.0;                           // their heads sit deeper: the next link's arm passes over them
SPLINE_R = 2.9;                            // the output spline and the idler stub (2.5) past the case faces
CONN_Y = [6.1, 11.85];        // the 3-pin sockets, one in each Z face
CONN_X = [-7.7, -4.4];

M2_CLEAR = 2.4;
M2_HEAD = 4.0;                // pan head 3.5 mm, with room
M2_PILOT = 1.7;
M3_CLEAR = 3.4;
M3_PILOT = 2.5;               // an M3 screw cutting its own thread in a cradle's pad
M3_NUT = 5.7;                 // across flats (5.5), with room
M3_HEAD = 6.0;

// ------------------------------------------------------------------ helpers
module box_lh(lo, hi) translate(lo) cube([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]);

// a cylinder along X from x0 to x1
module cyl_x(r, x0, x1) translate([x0, 0, 0]) rotate([0, 90, 0]) cylinder(r = r, h = x1 - x0);

// a cylinder along X through the point (0, y, z)
module cyl_x_at(y, z, r, x0, x1) translate([0, y, z]) cyl_x(r, x0, x1);

// children (ONE convex piece) swept about X from lo to hi degrees: chained
// hulls of rotated copies (a little more than the true sweep)
module sweep_x(lo, hi, step = SWEEP_STEP) {
    n = max(1, ceil((hi - lo) / step));
    for (k = [0 : n - 1]) hull() {
        rotate([lo + (hi - lo) * k / n, 0, 0]) children();
        rotate([lo + (hi - lo) * (k + 1) / n, 0, 0]) children();
    }
}

// children (ONE convex piece, in the second joint's S frame) swept about the
// second joint's X over lo2..hi2 and then, placed by T (that frame in the
// first joint's S frame), about the first joint's X over lo1..hi1: a
// crossing's two joints, coarser steps
module sweep_x2(T, lo1, hi1, lo2, hi2, step = 10) {
    n1 = max(1, ceil((hi1 - lo1) / step));
    n2 = max(1, ceil((hi2 - lo2) / step));
    for (k = [0 : n1 - 1], m = [0 : n2 - 1]) hull() {
        for (a = [lo1 + (hi1 - lo1) * k / n1, lo1 + (hi1 - lo1) * (k + 1) / n1],
             b = [lo2 + (hi2 - lo2) * m / n2, lo2 + (hi2 - lo2) * (m + 1) / n2])
            rotate([a, 0, 0]) multmatrix(T) rotate([b, 0, 0]) children();
    }
}

// --------------------------------------------------------- the servo itself
module servo_model(idler_horn = true) {
    color("dimgray") box_lh([CASE_X[0], CASE_Y[0], CASE_Z[0]], [CASE_X[1], CASE_Y[1], CASE_Z[1]]);
    color("silver") cyl_x(HORN_R, DRIVEN_X[0], DRIVEN_X[1]);
    color("black") cyl_x(HEAD_R, DRIVEN_X[1], HEAD_X);
    if (idler_horn) {
        color("silver") cyl_x(HORN_R, IDLER_X[0], IDLER_X[1]);
        color("black") cyl_x(HEAD_R, -HEAD_X, IDLER_X[0]);
    } else {
        color("silver") cyl_x(STUB_R, STUB_X, CASE_X[0]);
    }
}

// the arm screws a yoke takes (for the assembly pictures)
module arm_screws(single = false) {
    color("goldenrod") {
        for (a = [45, 135, 225, 315]) translate([ARM_X0 + ARM_DRIVEN, HORN_HOLE_R * sin(a), HORN_HOLE_R * cos(a)]) {
            cyl_x(1.75, 0, 1.4); cyl_x(1.0, -5, 0);
        }
        if (!single) for (a = [90, 270]) translate([-ARM_X0 - ARM_IDLER, HORN_HOLE_R * sin(a), HORN_HOLE_R * cos(a)]) {
            cyl_x(1.75, -1.4, 0); cyl_x(1.0, 0, 3.5);
        }
    }
}

// the case screws a cradle takes
module case_screws(full = false) {
    color("goldenrod") for (yz = full ? concat(FAR_HOLES, NEAR_HOLES) : FAR_HOLES, s = [-1, 1])
        translate([s * (HOUSING_X - HEAD_SEAT), yz[0], yz[1]]) {
            if (s > 0) { cyl_x(1.75, 0, 1.4); cyl_x(1.0, -3.5, 0); }
            else { cyl_x(1.75, -1.4, 0); cyl_x(1.0, 0, 3.5); }
        }
}

// the case grown by g (a convex piece: what a turning case sweeps)
module case_box(g = 0) box_lh([CASE_X[0] - g, CASE_Y[0] - g, CASE_Z[0] - g], [CASE_X[1] + g, CASE_Y[1] + g, CASE_Z[1] + g]);

// ------------------------------------------- the housing (part before a joint)
SOCKET_WALL = 1.2;                           // a socket's thin walls on the case's 34 x 23 faces
HOUSING_X = CASE_X[1] + FIT + WALL;          // its side plates' outer faces (the case screws go through)
HOUSING_Y = CASE_Y[1] + FIT + WALL;          // its back wall's outer face
HOUSING_Z = CASE_Z[1] + FIT + SOCKET_WALL;   // its thin walls' outer faces
SOCKET_Y0 = FAR_HOLES[0][0] - 5.5;           // where a socket starts: just ahead of the far screws' bosses

// a box around the case (convex; the keep-outs hollow it): side plates,
// back wall, side walls, open at the front
module housing_box(g = 0) box_lh([-HOUSING_X - g, CASE_Y[0] - g, -HOUSING_Z - g], [HOUSING_X + g, HOUSING_Y + g, HOUSING_Z + g]);

// a cradle (bracket-frame links, the palm): a socket the case's back end
// slides into (0.1 mm a side), its two plates as far as the far screws'
// bosses, thin walls on the other two faces, the back plate (convex; the
// case's keep-out hollows it)
module cradle_socket(g = 0) box_lh([-HOUSING_X - g, SOCKET_Y0 - g, -HOUSING_Z - g], [HOUSING_X + g, HOUSING_Y + g, HOUSING_Z + g]);

// a full-face cradle's plate on the horn face s (Martin, 10/24: where a spread
// and a bend meet at a knuckle, both the near and the far holes take screws)
// (2 mm past the case's front, round the near holes' bosses)
module cradle_plate(s, g = 0) {
    if (s > 0) box_lh([CASE_X[1] + FIT - g, CASE_Y[0] - 2 - g, -HOUSING_Z - g], [HOUSING_X + g, HOUSING_Y + g, HOUSING_Z + g]);
    else box_lh([-HOUSING_X - g, CASE_Y[0] - 2 - g, -HOUSING_Z - g], [CASE_X[0] - FIT + g, HOUSING_Y + g, HOUSING_Z + g]);
}

// a link's connector between two rectangles, each given as a frame (origin
// at its centre, its plane the frame's y-z) and a size: their hull, hollow
// inside with its walls and both ends TUBE_WALL thick (solid when short)
TUBE_WALL = 2.5;
module tube(A, sa, B, sb, g = 0) {
    module rect(M, sz, d) multmatrix(M) translate([0, -sz[0] / 2 - d, -sz[1] / 2 - d]) cube([0.01, sz[0] + 2 * d, sz[1] + 2 * d]);
    ca = [A[0][3], A[1][3], A[2][3]];
    cb = [B[0][3], B[1][3], B[2][3]];
    len = norm(cb - ca);
    u = len > 0 ? (cb - ca) / len : [0, 0, 0];
    difference() {
        hull() { rect(A, sa, g); rect(B, sb, g); }
        if (g == 0 && len > 2 * TUBE_WALL + 2)
            hull() { translate(u * TUBE_WALL) rect(A, sa, -TUBE_WALL); translate(-u * TUBE_WALL) rect(B, sb, -TUBE_WALL); }
    }
}

// what a housing leaves empty: the case and its way in from the front, the
// horns (or the idler stub) and their way in, the four case screws, the
// connector windows (and the plug on the cable side: +1 / -1 along Z, 0 none)
// (`channel`: how far ahead of the case its way in is kept clear: a closed
// housing's servo slides in from the front; a cradle's, open on two more
// sides, only needs its own front clear)
// (`full`: full-face plates; the case goes in with its horns off, so the
// plates keep a round hole for each horn and a slot only for the spline)
module housing_keepout(single = false, cable = 0, channel = 80, full = false) {
    box_lh([CASE_X[0] - FIT, CASE_Y[0] - channel, CASE_Z[0] - FIT], [CASE_X[1] + FIT, CASE_Y[1] + FIT, CASE_Z[1] + FIT]);
    if (full) {
        cyl_x(HORN_R + CASE_CLEAR, DRIVEN_X[0] - FIT, HORN_REACH);
        if (!single) cyl_x(HORN_R + CASE_CLEAR, -HORN_REACH, -(DRIVEN_X[0] - FIT));
        for (s = [-1, 1]) hull() {
            cyl_x(SPLINE_R, s > 0 ? CASE_X[1] : -HOUSING_X - 1, s > 0 ? HOUSING_X + 1 : CASE_X[0]);
            translate([0, -channel, 0]) cyl_x(SPLINE_R, s > 0 ? CASE_X[1] : -HOUSING_X - 1, s > 0 ? HOUSING_X + 1 : CASE_X[0]);
        }
        if (single) cyl_x(STUB_R + 0.4, -HORN_REACH, CASE_X[0] + FIT);
        for (yz = NEAR_HOLES, s = [-1, 1]) {
            cyl_x_at(yz[0], yz[1], M2_CLEAR / 2, s > 0 ? CASE_X[1] : -HOUSING_X - 3, s > 0 ? HOUSING_X + 3 : CASE_X[0]);
            cyl_x_at(yz[0], yz[1], M2_HEAD / 2, s > 0 ? HOUSING_X - NEAR_SEAT : -HOUSING_X - 3, s > 0 ? HOUSING_X + 3 : -(HOUSING_X - NEAR_SEAT));
        }
    } else {
    // the driven horn: a keyhole through the plate, open to the front
    hull() {
        cyl_x(HORN_R + PART_CLEAR, DRIVEN_X[0] - FIT, HORN_REACH);
        translate([0, -channel, 0]) cyl_x(HORN_R + PART_CLEAR, DRIVEN_X[0] - FIT, HORN_REACH);
    }
    if (single) {
        hull() {
            cyl_x(STUB_R + 0.4, -HORN_REACH, CASE_X[0] + FIT);
            translate([0, -channel, 0]) cyl_x(STUB_R + 0.4, -HORN_REACH, CASE_X[0] + FIT);
        }
    } else {
        hull() {
            cyl_x(HORN_R + PART_CLEAR, -HORN_REACH, -(DRIVEN_X[0] - FIT));
            translate([0, -channel, 0]) cyl_x(HORN_R + PART_CLEAR, -HORN_REACH, -(DRIVEN_X[0] - FIT));
        }
    }
    }
    // case screws through both side plates, heads seated HEAD_SEAT deep
    for (yz = FAR_HOLES, s = [-1, 1]) {
        cyl_x_at(yz[0], yz[1], M2_CLEAR / 2, s > 0 ? CASE_X[1] : -HOUSING_X - 3, s > 0 ? HOUSING_X + 3 : CASE_X[0]);
        cyl_x_at(yz[0], yz[1], M2_HEAD / 2, s > 0 ? HOUSING_X - HEAD_SEAT : -HOUSING_X - 3, s > 0 ? HOUSING_X + 3 : -(HOUSING_X - HEAD_SEAT));
    }
    // connector windows through both side walls; the plug on the cable side
    for (s = [-1, 1])
        box_lh([CONN_X[0] - 1.5, CONN_Y[0] - 1.3, s > 0 ? CASE_Z[1] - 2 : -HOUSING_Z - 1],
               [CONN_X[1] + 1.5, CONN_Y[1] + 1.3, s > 0 ? HOUSING_Z + 1 : CASE_Z[0] + 2]);
    if (cable != 0)
        box_lh([CONN_X[0] - 1.5, CONN_Y[0] - 1.3, cable > 0 ? CASE_Z[1] : -CASE_Z[1] - PLUG_OUT],
               [CONN_X[1] + 1.5, CONN_Y[1] + 1.3, cable > 0 ? CASE_Z[1] + PLUG_OUT : -CASE_Z[1]]);
}

// a screwdriver's way to the four case screws, `reach` out from the side
// plates (cut through the part's own plate, tube and arms, never through
// another servo's cradle)
module case_screw_tunnels(reach = 30, full = false) {
    for (yz = full ? concat(FAR_HOLES, NEAR_HOLES) : FAR_HOLES, s = [-1, 1])
        cyl_x_at(yz[0], yz[1], 1.9, s > 0 ? HOUSING_X : -HOUSING_X - reach, s > 0 ? HOUSING_X + reach : -HOUSING_X);
}

// ----------------------------------------------- the yoke (part after a joint)
ARM_X0 = DRIVEN_X[1] + ARM_GAP;              // both arms' inner faces (the horns' faces are at +/-14.5)

// an arm's disc (convex): s = +1 the driven arm, -1 the idler arm
module arm_disc(s, g = 0) {
    t = s > 0 ? ARM_DRIVEN : ARM_IDLER;
    if (s > 0) cyl_x(ARM_R + g, ARM_X0 - g, ARM_X0 + t + g);
    else cyl_x(ARM_R + g, -ARM_X0 - t - g, -ARM_X0 + g);
}

// the slab an arm lies in (for clipping the bridge into the arm's outline)
module arm_slab(s, g = 0) {
    t = s > 0 ? ARM_DRIVEN : ARM_IDLER;
    if (s > 0) box_lh([ARM_X0 - g, -200, -200], [ARM_X0 + t + g, 200, 200]);
    else box_lh([-ARM_X0 - t - g, -200, -200], [-ARM_X0 + g, 200, 200]);
}

// what an arm leaves empty: the centre screw head, the horn screws (the
// driven arm on 4 holes, the idler arm on 2), and a way in for the
// screwdriver: each hole runs on outward through whatever else is there
SCREWDRIVER_REACH = 25;
module arm_keepout(s, holes) {
    t = s > 0 ? ARM_DRIVEN : ARM_IDLER;
    x0 = s > 0 ? ARM_X0 - 1 : -ARM_X0 - t - SCREWDRIVER_REACH;
    x1 = s > 0 ? ARM_X0 + t + SCREWDRIVER_REACH : -ARM_X0 + 1;
    xa0 = s > 0 ? ARM_X0 - 1 : -ARM_X0 - t;            // the arm's own thickness
    xa1 = s > 0 ? ARM_X0 + t : -ARM_X0 + 1;
    cyl_x(HEAD_R + 0.4, xa0, xa1);
    cyl_x(HEAD_R + 0.6, s > 0 ? ARM_X0 + t : x0, s > 0 ? x1 : -ARM_X0 - t);
    for (a = holes) {
        cyl_x_at(HORN_HOLE_R * sin(a), HORN_HOLE_R * cos(a), M2_CLEAR / 2, xa0, xa1);
        cyl_x_at(HORN_HOLE_R * sin(a), HORN_HOLE_R * cos(a), M2_HEAD / 2, s > 0 ? ARM_X0 + t : x0, s > 0 ? x1 : -ARM_X0 - t);
    }
}

// ------------------------------------------------------------- the palm
// the palm plate: an outline (y, z) extruded over the plate's thickness
module plate(outline, x0, x1) translate([x0, 0, 0]) rotate([0, 90, 0]) linear_extrude(height = x1 - x0)
    polygon([for (p = outline) [-p[1], p[0]]]);

// a screw from the palm's front through its plate into the mount's column
module column_screw(y, z) {
    translate([0, y, z]) cyl_x(M3_CLEAR / 2, -PLATE_HALF - 1, -PLATE_HALF + 3.5);
}

// ------------------------------------------------------- the bay and the mount
// The bay frame: x along the palm normal (toward the palm), y along the bay's
// long axis, z across it; its origin on the mount's axis at the bay floor's
// outer face (where the arm's flange is). (cy, cz): the bay's centre.
PLATE_HALF = 12;
BAY_L = 76;  BAY_W = 60;  BAY_D = 22;
BAY_FLOOR = 4.5;  BAY_WALL = 2.0;  BAY_LID = 3.0;  STANDOFF = 4.0;
SKIRT_T = 3.0;  SKIRT_H = 3.5;  SKIRT_FIT = 0.2;
SKIRT_SCREWS_Z = [-20, 20];             // horizontal M3 screws through the skirt's short sides
MOUNT_TOP = 40;                         // the palm's back face, from the bay floor's outer face
COLUMN_R = 20;  BORE_R = 7;                // the cables' bore: seven PicoBlade cables, one plug at a time
BOARD = [69.5, 38.0, 1.6];
BOARD_HOLES = [[2.6, 2.6], [66.9, 2.6], [2.6, 35.4], [66.9, 35.4]];
USB_Z = [7.1, 17.1];  USB_H = 4.4;      // board y range and height above the board
XT30_Z = [16.8, 31.2];  XT30_H = 6.1;
ISO_BOLT_ANGLES = [45, 135, 225, 315];  // check against your robot's flange drawing
ISO_PIN_ANGLE = 0;
ISO_R = [15.75, 25.0];                  // ISO 9409-1-31.5 (M5) to -50 (M6) bolt circles
SLOT_W = 6.6;  PIN_W = 6.2;

function board_yz(cy, cz, p) = [cy - BOARD[0] / 2 + p[0], cz - BOARD[1] / 2 + p[1]];

module bay_cup(cy, cz) {
    top = BAY_D - BAY_LID;               // the cup's rim, under the lid
    bx0 = BAY_FLOOR + STANDOFF;          // the board's underside
    difference() {
        union() {
            box_lh([0, cy - BAY_L / 2, cz - BAY_W / 2], [top, cy + BAY_L / 2, cz + BAY_W / 2]);
        }
        // the inside, with bosses left for the skirt screws
        difference() {
            box_lh([BAY_FLOOR, cy - BAY_L / 2 + BAY_WALL, cz - BAY_W / 2 + BAY_WALL], [top + 1, cy + BAY_L / 2 - BAY_WALL, cz + BAY_W / 2 - BAY_WALL]);
            for (sy = [-1, 1], z = SKIRT_SCREWS_Z)
                box_lh([top - 7, sy > 0 ? cy + BAY_L / 2 - BAY_WALL - 6 : cy - BAY_L / 2, cz + z - 4],
                       [top + 1, sy > 0 ? cy + BAY_L / 2 : cy - BAY_L / 2 + BAY_WALL + 6, cz + z + 4]);
        }
        // skirt screws (M3, self-tapping into the bosses)
        for (sy = [-1, 1], z = SKIRT_SCREWS_Z)
            translate([top - SKIRT_H / 2, sy > 0 ? cy + BAY_L / 2 + 1 : cy - BAY_L / 2 - 1, cz + z])
                rotate([sy > 0 ? 90 : -90, 0, 0]) cylinder(r = 1.3, h = 9);
        // the arm's flange: radial slots for both ISO bolt circles (button heads inside), the pin's slot
        for (a = ISO_BOLT_ANGLES) hull() for (r = ISO_R) translate([-1, r * cos(a), r * sin(a)]) rotate([0, 90, 0]) cylinder(r = SLOT_W / 2, h = BAY_FLOOR + 2);
        hull() for (r = ISO_R) translate([-1, r * cos(ISO_PIN_ANGLE), r * sin(ISO_PIN_ANGLE)]) rotate([0, 90, 0]) cylinder(r = PIN_W / 2, h = BAY_FLOOR - 1.5);
        // the board's ports: USB-C and XT30 through the short wall at the board's left edge
        for (w = [[USB_Z, USB_H], [XT30_Z, XT30_H]])
            box_lh([bx0 - 0.5, cy - BAY_L / 2 - 1, cz - BOARD[1] / 2 + w[0][0]], [bx0 + BOARD[2] + w[1], cy - BAY_L / 2 + BAY_WALL + 1, cz - BOARD[1] / 2 + w[0][1]]);
    }
    // the board's standoffs, M2 pilots
    for (h = BOARD_HOLES) let(p = board_yz(cy, cz, h)) difference() {
        translate([BAY_FLOOR - 0.01, p[0], p[1]]) rotate([0, 90, 0]) cylinder(r = 3, h = STANDOFF + 0.01);
        translate([BAY_FLOOR - 3, p[0], p[1]]) rotate([0, 90, 0]) cylinder(r = M2_PILOT / 2, h = STANDOFF + 4);
    }
}

// the board where it sits (for the preview)
module board_model(cy, cz) {
    color("darkgreen") translate([BAY_FLOOR + STANDOFF, cy - BOARD[0] / 2, cz - BOARD[1] / 2]) cube([BOARD[2], BOARD[0], BOARD[1]]);
}

// the lid, its skirt over the cup's rim, the column up to the palm's back
// face; a bore for the cables; screw holes for the column's top (`screws`:
// [y, z] in the bay frame, M3 self-tapping from the palm's split plane)
module mount(cy, cz, screws, bore = [0, 0]) {
    top = BAY_D - BAY_LID;
    difference() {
        union() {
            box_lh([top, cy - BAY_L / 2 - SKIRT_FIT - SKIRT_T, cz - BAY_W / 2 - SKIRT_FIT - SKIRT_T], [BAY_D, cy + BAY_L / 2 + SKIRT_FIT + SKIRT_T, cz + BAY_W / 2 + SKIRT_FIT + SKIRT_T]);
            box_lh([top - SKIRT_H, cy - BAY_L / 2 - SKIRT_FIT - SKIRT_T, cz - BAY_W / 2 - SKIRT_FIT - SKIRT_T], [top, cy + BAY_L / 2 + SKIRT_FIT + SKIRT_T, cz + BAY_W / 2 + SKIRT_FIT + SKIRT_T]);
            cyl_x(COLUMN_R, BAY_D - 0.01, MOUNT_TOP);
        }
        box_lh([top - SKIRT_H - 1, cy - BAY_L / 2 - SKIRT_FIT, cz - BAY_W / 2 - SKIRT_FIT], [top + 0.01, cy + BAY_L / 2 + SKIRT_FIT, cz + BAY_W / 2 + SKIRT_FIT]);
        translate([0, bore[0], bore[1]]) cyl_x(BORE_R, top - 1, MOUNT_TOP + 1);
        for (sy = [-1, 1], z = SKIRT_SCREWS_Z)
            translate([top - SKIRT_H / 2, sy > 0 ? cy + BAY_L / 2 - 1 : cy - BAY_L / 2 + 1, cz + z])
                rotate([sy > 0 ? -90 : 90, 0, 0]) cylinder(r = M3_CLEAR / 2, h = 10);
        for (q = screws) translate([MOUNT_TOP - 12, q[0], q[1]]) rotate([0, 90, 0]) cylinder(r = 1.3, h = 13);
    }
}
