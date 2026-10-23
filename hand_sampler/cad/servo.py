"""The Feetech HL-3915-C001 servo as the build sees it, in millimetres.

Measured from Feetech's STEP model (`FEETECH_HL_3915_Servo_Motor.stp`, the
robotopian.com listing, 2026-10-23; kept in `outputs/build_ref/`, not in
git) and the HLS3915M specification A/0 (2025-02-28). The plastic-case
HL-2915-C001 is taken to have the same outline (Martin, 2026-10-23).

The servo frame S: origin on the output axis at the case's mid-plane; X
along the axis toward the driven horn, Y the servo's back (the grammar's v:
the case runs from 9.5 mm ahead of the axis to 24.5 behind it), Z = X x Y
(the grammar's w, the 20 mm width). A servo placed with its driven horn on
the grammar's +a has S = (a, v, w); flipped (driven on -a), S = (-a, v, -w).

    case          X -11.5..11.5, Y -9.5..24.5, Z -10..10 (23 x 34 x 20)
    horns         16 mm across, the driven one X 11.5..14.5 (flange from 12),
                  the idler -14.5..-12.1; 8 x M2 tapped on a 12 mm circle
                  every 45 degrees (0 along +Z)
    centre screws heads 4.2 mm across, out to |X| = 17.0
    idler stub    5 mm across, out to X = -14.0 once the idler horn is off
    case holes    4 x M2 tapped on each horn face at Y -7.5 / 22.5, Z +/-8
    connectors    a 3-pin 1.25 mm socket on each Z face, Y 6.1..11.85,
                  X -7.7..-4.4 (toward the idler), 1.6 mm inside the face
    box screws    8 per servo: M2 x 3.5 mm, pan head 3.5 across, 1.4 tall
                  (Martin, 2026-10-23)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

Range = Tuple[float, float]


@dataclass(frozen=True)
class Screw:
    name: str
    thread_mm: float          # nominal diameter
    length_mm: float          # under the head
    head_d_mm: float
    head_h_mm: float
    clear_d_mm: float         # clearance hole in a printed part
    pilot_d_mm: float         # a hole it taps into printed plastic


M2_BOX = Screw("M2 x 3.5 pan head (in the servo box)", 2.0, 3.5, 3.5, 1.4, 2.4, 1.7)
M2X5 = Screw("M2 x 5 pan head", 2.0, 5.0, 3.5, 1.4, 2.4, 1.7)
M3X10 = Screw("M3 x 10 socket head", 3.0, 10.0, 5.5, 3.0, 3.4, 2.6)
M4X50 = Screw("M4 x 50 socket head", 4.0, 50.0, 7.0, 4.0, 4.5, 3.4)


@dataclass(frozen=True)
class Servo:
    name: str
    case_x: Range = (-11.5, 11.5)
    case_y: Range = (-9.5, 24.5)
    case_z: Range = (-10.0, 10.0)
    horn_r: float = 8.0
    driven_horn_x: Range = (11.5, 14.5)
    idler_horn_x: Range = (-14.5, -12.1)
    horn_hole_circle_r: float = 6.0
    horn_hole_count: int = 8
    centre_head_r: float = 2.1
    centre_head_x: float = 17.0
    idler_stub_r: float = 2.5
    idler_stub_x: float = -14.0
    case_holes_yz: Tuple[Tuple[float, float], ...] = ((-7.5, -8.0), (-7.5, 8.0), (22.5, -8.0), (22.5, 8.0))
    connector_y: Range = (6.1, 11.85)
    connector_x: Range = (-7.7, -4.4)
    connector_inset: float = 1.6
    box_screws: int = 8
    box_screw: Screw = M2_BOX
    stall_nm: float = 1.39
    mass_g: float = 35.8

    def scad_params(self) -> Dict[str, object]:
        """The numbers the OpenSCAD library reads (`SERVO = [...]`)."""
        return {
            "case_x": list(self.case_x), "case_y": list(self.case_y), "case_z": list(self.case_z),
            "horn_r": self.horn_r, "driven_horn_x": list(self.driven_horn_x),
            "idler_horn_x": list(self.idler_horn_x), "horn_hole_circle_r": self.horn_hole_circle_r,
            "centre_head_r": self.centre_head_r, "centre_head_x": self.centre_head_x,
            "idler_stub_r": self.idler_stub_r, "idler_stub_x": self.idler_stub_x,
            "connector_y": list(self.connector_y), "connector_x": list(self.connector_x),
        }


HL3915 = Servo("Feetech HL-3915-C001 (metal case)")
HL2915 = Servo("Feetech HL-2915-C001 (plastic case)")
SERVOS = {"HL3915": HL3915, "HL2915": HL2915}
