"""Motion primitives for the LEAP hand workcell.

All primitives operate on a :class:`~dexsuite.env.DexSuiteEnv`. Wrist motions are
interpolated (jerk-free) so grasped objects are not flung; finger closure is
ramped through :func:`grasp_pose`. Parameters here were tuned empirically against
the physics of the vendored LEAP model and are surfaced through ``config.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Open / ready pose: fingers straight, slightly splayed; thumb abducted clear.
OPEN_POSE: Dict[str, float] = {
    "if_mcp": 0.0, "if_rot": -0.30, "if_pip": 0.0, "if_dip": 0.0,
    "mf_mcp": 0.0, "mf_rot": 0.00, "mf_pip": 0.0, "mf_dip": 0.0,
    "rf_mcp": 0.0, "rf_rot": 0.30, "rf_pip": 0.0, "rf_dip": 0.0,
    "th_cmc": 1.60, "th_axl": 0.0, "th_mcp": 0.0, "th_ipl": -0.60,
}

# Grasp-pocket offset (metres) of the enveloping convergence zone from the wrist
# frame origin. An object placed here is reliably caged when the fingers close.
POCKET = np.array([0.07, 0.03])

# Index-fingertip offset from the wrist frame when in the pointing/press pose.
INDEX_TIP = np.array([0.02, -0.008])


def grasp_pose(c: float) -> Dict[str, float]:
    """Finger targets for closure fraction ``c`` in [0, 1] (0 = open, 1 = firm)."""
    c = float(np.clip(c, 0.0, 1.0))
    p = dict(OPEN_POSE)
    for f in ("if", "mf", "rf"):
        p[f + "_rot"] = OPEN_POSE[f + "_rot"] * (1.0 - c)
        p[f + "_mcp"] = 0.80 * c
        p[f + "_pip"] = 1.30 * c
        p[f + "_dip"] = 0.70 * c
    p["th_cmc"] = 1.55
    p["th_axl"] = 0.20 + 0.90 * c
    p["th_mcp"] = 0.20 + 1.50 * c
    p["th_ipl"] = -0.30 + 1.10 * c
    return p


@dataclass
class SkillParams:
    approach_z: float = 0.06     # wrist-z when hovering above a pick/place point
    grasp_z: float = 0.00        # wrist-z at the grasp pocket (pedestal height)
    lift_z: float = 0.16         # wrist-z while carrying
    move_steps: int = 350        # steps for a lateral/vertical wrist move
    close_steps: int = 500       # steps to ramp the grasp closed
    open_steps: int = 1600       # steps to ramp the grasp open (slow = no fling)
    settle_steps: int = 250


class Hand:
    """Stateful skill controller bound to an environment."""

    def __init__(self, env, params: SkillParams | None = None):
        self.env = env
        self.p = params or SkillParams()
        self._closure = 0.0

    # ---- low-level moves --------------------------------------------------
    def move_wrist(self, x, y, z, yaw=0.0, pitch=0.0, roll=0.0, steps=None):
        steps = steps or self.p.move_steps
        cur = self.env.get_wrist_target()
        tgt = np.array([x, y, z, yaw, pitch, roll], dtype=float)
        for i in range(steps):
            a = (i + 1) / steps
            self.env.set_wrist_target(cur + (tgt - cur) * a)
            self.env.step(1)

    def set_closure(self, c_to, steps):
        c_from = self._closure
        for i in range(steps):
            c = c_from + (c_to - c_from) * (i + 1) / steps
            self.env.set_finger_targets(grasp_pose(c))
            self.env.step(1)
        self._closure = c_to

    def open_hand(self):
        self.env.set_finger_targets(OPEN_POSE)
        self._closure = 0.0

    # ---- composite skills -------------------------------------------------
    def goto_above(self, xy):
        wx, wy = xy[0] - POCKET[0], xy[1] - POCKET[1]
        self.move_wrist(wx, wy, self.p.approach_z)
        return wx, wy

    def pick_at(self, xy) -> bool:
        """Pick an object resting at world ``xy`` on a pedestal. Returns grasp ok."""
        wx, wy = self.goto_above(xy)
        self.move_wrist(wx, wy, self.p.grasp_z)
        self.set_closure(1.0, self.p.close_steps)
        self.env.step(self.p.settle_steps)
        self.move_wrist(wx, wy, self.p.lift_z)
        return int(np.count_nonzero(self.env.touch() > 0.05)) >= 2

    def reorient(self, obj_name, roll=1.5, hold=180, steps=460):
        """In-hand reorientation: roll the grasped object to a held target pose,
        then return to neutral for placement. Returns (degrees_rotated, still_held).
        """
        import numpy as _np
        t = self.env.get_wrist_target()
        q0 = self.env.object_pose(obj_name)[3:].copy()
        self.move_wrist(t[0], t[1], t[2], roll=roll, steps=steps)
        self.env.step(hold)
        qp = self.env.object_pose(obj_name)[3:].copy()
        held = self.env.object_pose(obj_name)[2] > 0.55
        self.move_wrist(t[0], t[1], t[2], roll=0.0, steps=steps)
        deg = float(_np.degrees(2 * _np.arccos(min(1.0, abs(float(_np.dot(q0, qp)))))))
        return deg, held

    def place_in(self, xy):
        wx, wy = xy[0] - POCKET[0], xy[1] - POCKET[1]
        self.move_wrist(wx, wy, self.p.lift_z, steps=700)
        self.move_wrist(wx, wy, self.p.approach_z, steps=300)
        self.set_closure(0.0, self.p.open_steps)
        self.env.step(500)

    def press_button(self, xy, depth_z, finger="if"):
        """Single-finger press: extend the index finger and push the cap down."""
        wx, wy = xy[0] - INDEX_TIP[0], xy[1] - INDEX_TIP[1]
        # point only the index finger down, curl the others out of the way
        pose = dict(OPEN_POSE)
        pose.update({"if_rot": 0.0, "mf_mcp": 1.2, "mf_pip": 1.4,
                     "rf_mcp": 1.2, "rf_pip": 1.4, "th_cmc": 1.6,
                     "if_mcp": 0.0, "if_pip": 0.0, "if_dip": 0.0})
        self.env.set_finger_targets(pose)
        self.move_wrist(wx, wy, self.p.approach_z)
        self.move_wrist(wx, wy, depth_z, steps=300)
        self.env.step(300)
        pressed = self.env.button_depress()
        self.move_wrist(wx, wy, self.p.approach_z, steps=300)
        self.open_hand()
        return pressed
