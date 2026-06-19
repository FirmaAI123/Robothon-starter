"""Gait controller: a central-pattern-generator (CPG) trot with PD torque control.

Diagonal leg pairs (FL+RR / FR+RL) swing in anti-phase. Each leg's thigh sweeps
to propel the body and lifts during swing; the knee bends to clear the ground.
A steering command biases the left/right stride to yaw. Joint targets are tracked
by PD torques on the Go2's 12 motors — fully closed-loop physics, no kinematics.
Parameters were tuned empirically (forward walk ≈0.4 m/s, stable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import LEGS

# nominal standing pose (Go2 "home"): hip, thigh, calf per leg
STAND = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}
PHASE = {"FL": 0.0, "RR": 0.0, "FR": np.pi, "RL": np.pi}      # trot pairing
SIDE = {"FL": +1, "RL": +1, "FR": -1, "RR": -1}              # left / right


@dataclass
class GaitParams:
    freq: float = 2.0          # gait cycles / s
    thigh_lift: float = 0.55   # swing lift amplitude (thigh)
    calf_lift: float = 1.0     # swing knee-bend amplitude
    sweep: float = 0.50        # stride / forward propulsion
    kp: float = 55.0           # PD position gain
    kd: float = 2.2            # PD damping gain


class TrotController:
    def __init__(self, env, params: GaitParams | None = None):
        self.env = env
        self.p = params or GaitParams()

    def joint_targets(self, t: float, forward: float = 1.0, turn: float = 0.0) -> np.ndarray:
        """Desired joint angles (12,) in env actuator order for time ``t``."""
        des = []
        for lg in LEGS:
            ph = 2 * np.pi * self.p.freq * t + PHASE[lg]
            s = np.sin(ph)
            lift = max(0.0, s)
            sweep = (self.p.sweep * forward + turn * SIDE[lg]) * np.cos(ph)
            des.append(STAND["hip"])
            des.append(STAND["thigh"] + sweep + self.p.thigh_lift * lift)
            des.append(STAND["calf"] - self.p.calf_lift * lift)
        return np.array(des)

    def pd_torque(self, q_des: np.ndarray) -> np.ndarray:
        q = self.env.joint_pos
        v = self.env.joint_vel
        return self.p.kp * (q_des - q) - self.p.kd * v

    def act(self, t: float, forward: float = 1.0, turn: float = 0.0):
        self.env.set_motor_torques(self.pd_torque(self.joint_targets(t, forward, turn)))

    # ---- gait phase (for HUD) ----
    def stance_swing(self, t: float):
        return {lg: ("swing" if np.sin(2 * np.pi * self.p.freq * t + PHASE[lg]) > 0
                     else "stance") for lg in LEGS}


# ---- autonomous patrol: traverse terrain, then a multi-turn course ----
def patrol_command(t: float):
    """Returns (forward, turn, label) for an autonomous patrol over time:
    walk across the terrain course (ramp, bumps, step), then steer through a
    left/right turning sequence."""
    if t < 9.0:
        return 1.0, 0.0, "Traversing terrain: ramp, bumps, step (open-loop CPG)"
    if t < 12.5:
        return 0.5, 0.20, "Turning left"
    if t < 15.5:
        return 1.0, 0.0, "Walking on new heading"
    if t < 19.0:
        return 0.5, -0.20, "Turning right"
    if t < 22.0:
        return 1.0, 0.0, "Walking forward"
    return 0.0, 0.0, "Halting (settling to stand)"
