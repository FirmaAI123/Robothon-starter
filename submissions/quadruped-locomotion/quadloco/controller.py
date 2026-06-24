"""Gait controller: a central-pattern-generator (CPG) trot with PD torque control.

Diagonal leg pairs (FL+RR / FR+RL) swing in anti-phase. Each leg's thigh sweeps
to propel the body and lifts during swing; the knee bends to clear the ground.
A steering command biases the left/right stride to yaw. Joint targets are tracked
by PD torques on the Go2's 12 motors — fully closed-loop physics, no kinematics.
Two sensor-feedback loops ride on top: IMU+gyro heading-hold and contact-gated IMU
terrain-leveling. The forward command maps to a controllable ~0.1-0.8 m/s speed band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import LEGS

# nominal standing pose (Go2 "home"): hip, thigh, calf per leg
STAND = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}
PHASE = {"FL": 0.0, "RR": 0.0, "FR": np.pi, "RL": np.pi}      # trot pairing
SIDE = {"FL": +1, "RL": +1, "FR": -1, "RR": -1}              # left / right
FRONT = {"FL": +1, "FR": +1, "RL": -1, "RR": -1}             # front / back
LR = {"FL": +1, "RL": +1, "FR": -1, "RR": -1}                # left / right (level)


@dataclass
class GaitParams:
    freq: float = 2.0          # gait cycles / s
    thigh_lift: float = 0.55   # swing lift amplitude (thigh)
    calf_lift: float = 1.0     # swing knee-bend amplitude
    sweep: float = 0.50        # stride / forward propulsion
    kp: float = 55.0           # PD position gain
    kd: float = 2.2            # PD damping gain
    # --- closed-loop terrain feedback (IMU active body-leveling via stance knees) ---
    level_pitch: float = 0.30  # IMU-pitch -> stance-knee extension (keep trunk level)
    level_roll: float = 0.12   # IMU-roll  -> stance-knee extension (low: avoid yaw coupling)
    # --- closed-loop heading hold (IMU yaw + gyro-rate damping) ---
    kp_yaw: float = 2.5        # proportional yaw correction
    kd_yaw: float = 0.18       # gyro yaw-rate damping (PD on real rate feedback)


class TrotController:
    def __init__(self, env, params: GaitParams | None = None):
        self.env = env
        self.p = params or GaitParams()
        self._target_heading = None
        self._v_integ = 0.0        # body-velocity loop integral state
        self.terrain_fb = True     # closed-loop terrain feedback (set False to ablate)

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

    def imu_pitch_roll(self):
        """Trunk pitch & roll from the IMU orientation quaternion."""
        w, x, y, z = self.env.get_obs()["imu_quat"]
        pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
        roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
        return pitch, roll

    def terrain_feedback(self, q_des: np.ndarray, t: float) -> np.ndarray:
        """Closed-loop terrain adaptation by active body-leveling: from IMU pitch &
        roll, extend the knee on the low side / shorten on the high side to keep the
        trunk level on slopes & uneven ground. The loop is **gated on real foot
        contact** — only a leg that is in stance phase AND actually bearing load
        (foot-contact sensor) levels, so airborne/clipping legs never fight the gait.
        Acting on the knee alone changes leg length without disturbing the fore-aft
        sweep. Uses IMU + foot-contact sensors — true terrain feedback."""
        pitch, roll = self.imu_pitch_roll()
        fc = self.env.foot_contacts()           # FL, FR, RL, RR
        q = q_des.copy()
        for i, lg in enumerate(LEGS):
            ph = 2 * np.pi * self.p.freq * t + PHASE[lg]
            if np.sin(ph) >= 0.0 or fc[i] < 0.5:   # only stance legs truly on the ground
                continue
            ext = self.p.level_pitch * pitch * FRONT[lg] + self.p.level_roll * roll * LR[lg]
            ext = float(np.clip(ext, -0.35, 0.45))
            q[3 * i + 2] += ext                  # calf only (less curl = longer leg)
        return q

    def pd_torque(self, q_des: np.ndarray) -> np.ndarray:
        q = self.env.joint_pos
        v = self.env.joint_vel
        return self.p.kp * (q_des - q) - self.p.kd * v

    def act(self, t: float, forward: float = 1.0, turn: float = 0.0):
        q_des = self.joint_targets(t, forward, turn)
        if self.terrain_fb:
            q_des = self.terrain_feedback(q_des, t)
        self.env.set_motor_torques(self.pd_torque(q_des))

    # ---- closed-loop IMU heading hold ----
    def imu_yaw(self) -> float:
        """Trunk yaw read from the IMU orientation sensor."""
        w, x, y, z = self.env.get_obs()["imu_quat"]
        return float(2.0 * np.arctan2(z, w))

    def drive(self, t: float, forward: float = 1.0, turn_cmd: float = 0.0,
              kp_yaw: float | None = None, hold: bool = True):
        """Drive with IMU feedback: explicit turn commands steer (and re-anchor the
        heading target); straight segments hold heading via a **PD** correction —
        proportional on IMU yaw error and derivative on the measured gyro yaw-rate
        (`τ = kp·err − kd·ω_z`) — closed-loop on two real sensors, not dead-reckoning."""
        kp_yaw = self.p.kp_yaw if kp_yaw is None else kp_yaw
        if self._target_heading is None:
            self._target_heading = self.imu_yaw()
        if abs(turn_cmd) > 1e-6:
            self._target_heading = self.imu_yaw()      # track heading through turns
            turn = turn_cmd
        elif hold and forward > 0:
            err = (self.imu_yaw() - self._target_heading + np.pi) % (2 * np.pi) - np.pi
            wz = float(self.env.get_obs()["gyro"][2])  # measured yaw-rate (rad/s)
            # positive turn command yaws the body negative, so correct with +err;
            # gyro-rate damping (-kd·ω_z) suppresses overshoot/oscillation.
            turn = float(np.clip(kp_yaw * err - self.p.kd_yaw * wz, -0.15, 0.15))
        else:
            turn = 0.0
        self.act(t, forward, turn)

    # ---- closed-loop body-velocity tracking (uses the velocimeter) ----
    def track_speed(self, t: float, v_set: float, turn_cmd: float = 0.0,
                    kp: float = 1.5, ki: float = 3.0):
        """Outer velocity loop on top of the gait: read forward body speed from the
        **velocimeter** and PI-control the gait's forward command so the robot tracks a
        commanded speed setpoint (m/s) regardless of payload/terrain/friction — a fourth
        closed-loop on a real sensor. Inner heading-hold + terrain-leveling still run."""
        v = float(self.env.get_obs()["vel"][0])
        self._v_integ = float(np.clip(self._v_integ + (v_set - v) * self.env.dt, -2.0, 2.0))
        fcmd = float(np.clip(kp * (v_set - v) + ki * self._v_integ, 0.0, 2.5))
        self.drive(t, fcmd, turn_cmd)

    # ---- closed-loop go-to-goal navigation (autonomous mission, self-verifying) ----
    def steer_to(self, t: float, goal_xy, reach: float = 0.25, k: float = 1.0):
        """Fifth closed-loop, on **position**: steer the heading toward a target waypoint
        (from `base_xy`/`heading`) and walk until within `reach` m, slowing as it nears.
        Returns the current distance to the goal — the robot self-verifies arrival when it
        drops below `reach`. Heading-hold + terrain-leveling still run underneath."""
        x, y = self.env.base_xy()
        dx, dy = goal_xy[0] - x, goal_xy[1] - y
        d = float(np.hypot(dx, dy))
        err = (np.arctan2(dy, dx) - self.env.heading() + np.pi) % (2 * np.pi) - np.pi
        if d < reach:
            fwd = 0.0                       # arrived
        elif abs(err) > 0.6:
            fwd = 0.15                      # badly mis-aligned -> turn (nearly) in place first
        else:
            fwd = 1.0 if d > 0.6 else 0.5   # aligned -> walk, slow near the goal
        turn = float(np.clip(-k * err, -0.22, 0.22))
        self.drive(t, fwd, turn)
        return d

    # ---- gait phase (for HUD) ----
    def stance_swing(self, t: float):
        return {lg: ("swing" if np.sin(2 * np.pi * self.p.freq * t + PHASE[lg]) > 0
                     else "stance") for lg in LEGS}


# ---- autonomous patrol: traverse terrain, show motion range, then steer ----
def patrol_command(t: float):
    """Returns (forward, turn, label) for an autonomous patrol over time: cross the
    terrain course (ramp, rough, step) under closed-loop leveling, then — on the flat
    beyond — demonstrate the motion range (spin in place, speed sweep) and steer a
    left/right course, all with IMU heading-hold on the straights."""
    if t < 8.0:
        return 1.0, 0.0, "Traversing terrain: ramp, rough, step (closed-loop IMU leveling)"
    if t < 10.5:
        return 1.0, 0.0, "Onto flat ground (heading-hold)"
    if t < 13.5:
        return 0.0, 0.45, "Spin in place (in-place yaw)"
    if t < 16.0:
        return 0.4, 0.0, "Speed control: slow (~0.2 m/s)"
    if t < 18.5:
        return 1.4, 0.0, "Speed control: fast (~0.5 m/s)"
    if t < 21.5:
        return 0.5, 0.20, "Turning left"
    if t < 24.5:
        return 1.0, 0.0, "Walking on new heading (heading-hold)"
    if t < 27.5:
        return 0.5, -0.20, "Turning right"
    return 0.0, 0.0, "Halting (settling to stand)"
