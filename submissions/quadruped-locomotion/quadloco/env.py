"""QuadLoco environment: a Unitree Go2 quadruped on a terrain course (ramp, rough
heightfield, step), driven by real torque actuators with full physics (contacts,
friction) — every frame advances ``mj_step``, not a kinematic puppet. Exposes an
IMU/proprioception observation, foot-contact detection against all ground geoms,
external-push (``xfrc_applied``) disturbances, rendering, and per-step hooks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
SCENE = os.path.join(ASSETS, "scene.xml")
FLAT = os.path.join(ASSETS, "flat.xml")     # terrain-free scene for speed-control eval
CARGO = os.path.join(ASSETS, "cargo.xml")   # loco-manipulation: carry a payload on a tray
MISSION_SCENE = os.path.join(ASSETS, "mission.xml")  # flat + goal markers for the go-to-goal mission demo

LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = [f"{lg}_{j}" for lg in LEGS for j in ("hip", "thigh", "calf")]  # 12, actuator order


@dataclass
class EnvConfig:
    scene: str = SCENE
    seed: int = 0
    randomize: bool = False
    mass_jitter: float = 0.0      # fractional trunk-mass jitter
    yaw_jitter: float = 0.0       # rad of initial heading jitter
    friction_jitter: float = 0.0  # fractional ground-friction jitter
    sensor_noise: float = 0.0     # IMU/gyro/velocimeter noise scale (real-world sensing)
    render_w: int = 1280
    render_h: int = 720


class QuadEnv:
    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.jpos = np.array([self._jqadr(j) for j in JOINTS])
        self.jvel = np.array([self._jdadr(j) for j in JOINTS])
        self.act = np.array([self._aid(j) for j in JOINTS])
        self.foot_gid = {lg: self._gid(lg) for lg in LEGS}
        self.floor_gid = self._gid("floor")
        # every ground geom a foot can stand on (floor + terrain), so contact
        # sensing works on the ramp / rough patch / step, not only the flat floor.
        self.terrain_gids = {self._gid(n) for n in ("floor", "ramp", "rough", "step")} - {-1}
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        # optional cargo (loco-manipulation): a tray welded to the trunk + a free payload
        self.tray_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray")
        self.payload_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.has_cargo = self.tray_bid >= 0 and self.payload_bid >= 0
        self._renderer: Optional[mujoco.Renderer] = None
        self._hooks: list = []
        self._push_force = np.zeros(3)
        self._push_steps = 0
        self._noise_rng = np.random.default_rng(self.cfg.seed + 777)  # seeded sensor noise
        self._fill_heightfield(12345 if not self.cfg.randomize else self.cfg.seed)
        if self.cfg.randomize and self.cfg.friction_jitter > 0:
            self._randomize_friction()
        self.reset()

    def _randomize_friction(self):
        """Jitter the ground sliding friction (per terrain geom) for robustness eval."""
        for g in self.terrain_gids:
            f = 1.0 + self.rng.uniform(-self.cfg.friction_jitter, self.cfg.friction_jitter)
            self.model.geom_friction[g, 0] *= f

    def _fill_heightfield(self, seed=12345):
        """Procedurally fill the 'rough' heightfield with smooth random terrain
        (deterministic by seed) so the patch is genuinely uneven, not flat. The
        seed varies the bumps per trial under randomization (same seed -> same
        terrain, preserving determinism)."""
        hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "rough")
        if hid < 0:
            return
        nr, nc = int(self.model.hfield_nrow[hid]), int(self.model.hfield_ncol[hid])
        rng = np.random.default_rng(seed)
        coarse = rng.random((nr // 6 + 2, nc // 6 + 2))
        big = np.repeat(np.repeat(coarse, 6, 0), 6, 1)[:nr, :nc]
        for _ in range(3):                      # smooth
            p = np.pad(big, 1, mode="edge")
            big = sum(p[i:i + nr, j:j + nc] for i in range(3) for j in range(3)) / 9.0
        big = (big - big.min()) / (big.max() - big.min() + 1e-9)
        adr = self.model.hfield_adr[hid]
        self.model.hfield_data[adr:adr + nr * nc] = big.flatten().astype(np.float64)

    # ---- id helpers ----
    def _aid(self, n):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def _jqadr(self, n):
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")
        return self.model.jnt_qposadr[j]

    def _jdadr(self, n):
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")
        return self.model.jnt_dofadr[j]

    def _gid(self, n):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)

    def _sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
        return self.data.sensordata[adr:adr + dim].copy()

    # ---- hooks ----
    def add_step_hook(self, fn):
        self._hooks.append(fn)

    def clear_step_hooks(self):
        self._hooks.clear()

    # ---- lifecycle ----
    def _place_cargo(self):
        """Seat the tray on the trunk's back and the payload on the tray (the padded
        keyframe zeroes their free-joint qpos, so we position them explicitly)."""
        bp = self.data.xpos[self.base_bid].copy()
        for bid, dz in ((self.tray_bid, 0.055), (self.payload_bid, 0.10)):
            adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
            self.data.qpos[adr:adr + 3] = [bp[0], bp[1], bp[2] + dz]
            self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]

    def reset(self) -> Dict:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)   # "home" standing pose
        self.data.xfrc_applied[:] = 0.0       # clear any external pushes (determinism)
        self._push_steps = 0
        if self.has_cargo:
            mujoco.mj_forward(self.model, self.data)   # get base pose, then seat cargo
            self._place_cargo()
        if self.cfg.randomize:
            if self.cfg.mass_jitter > 0:
                f = 1.0 + self.rng.uniform(-self.cfg.mass_jitter, self.cfg.mass_jitter)
                self.model.body_mass[self.base_bid] *= f
            if self.cfg.yaw_jitter > 0:
                yaw = self.rng.uniform(-self.cfg.yaw_jitter, self.cfg.yaw_jitter)
                self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, n: int = 1):
        for _ in range(n):
            if self._push_steps > 0:                       # external disturbance
                self.data.xfrc_applied[self.base_bid, :3] = self._push_force
                self._push_steps -= 1
                if self._push_steps == 0:
                    self.data.xfrc_applied[self.base_bid, :3] = 0.0
            mujoco.mj_step(self.model, self.data)
            for h in self._hooks:
                h(self)

    # ---- external disturbance (closed-loop recovery testing) ----
    def apply_push(self, force_xyz, steps: int):
        """Apply a constant external force (N) to the trunk for the next ``steps``
        physics steps — a real ``data.xfrc_applied`` impulse, used to test the
        controller's closed-loop recovery."""
        self._push_force = np.asarray(force_xyz, dtype=float)
        self._push_steps = int(steps)

    @property
    def pushing(self) -> bool:
        return self._push_steps > 0

    # ---- control ----
    def set_motor_torques(self, tau: np.ndarray):
        lo = self.model.actuator_ctrlrange[self.act, 0]
        hi = self.model.actuator_ctrlrange[self.act, 1]
        self.data.ctrl[self.act] = np.clip(tau, lo, hi)

    @property
    def joint_pos(self):
        return self.data.qpos[self.jpos].copy()

    @property
    def joint_vel(self):
        return self.data.qvel[self.jvel].copy()

    # ---- observation ----
    @property
    def dt(self):
        return self.model.opt.timestep

    def base_xy(self):
        return self.data.xpos[self.base_bid][:2].copy()

    def base_height(self):
        return float(self.data.xpos[self.base_bid][2])

    def heading(self):
        q = self.data.qpos[3:7]
        return float(2 * np.arctan2(q[3], q[0]))

    def upright(self):
        # cos angle between body-z and world-z
        zaxis = self.data.xmat[self.base_bid].reshape(3, 3)[:, 2]
        return float(zaxis[2])

    # ---- cargo (loco-manipulation) ----
    def payload_offset(self):
        """(horizontal offset, height above tray) of the payload vs the tray center."""
        if not self.has_cargo:
            return (0.0, 0.0)
        p = self.data.xpos[self.payload_bid]
        t = self.data.xpos[self.tray_bid]
        return float(np.hypot(p[0] - t[0], p[1] - t[1])), float(p[2] - t[2])

    def payload_on(self) -> bool:
        """True while the payload is still riding on the tray (not slid/fallen off)."""
        if not self.has_cargo:
            return True
        horiz, dz = self.payload_offset()
        return horiz < 0.13 and dz > 0.0

    def foot_contacts(self):
        c = {lg: 0.0 for lg in LEGS}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            for lg in LEGS:
                fg = self.foot_gid[lg]
                if (g1 == fg and g2 in self.terrain_gids) or (g2 == fg and g1 in self.terrain_gids):
                    c[lg] = 1.0
        return np.array([c[lg] for lg in LEGS])

    def get_obs(self) -> Dict:
        quat = self._sensor("imu_quat"); gyro = self._sensor("imu_gyro")
        acc = self._sensor("imu_acc"); vel = self._sensor("imu_vel")
        if self.cfg.sensor_noise > 0:                # emulate real (noisy) proprioception
            s = self.cfg.sensor_noise
            quat = quat + self._noise_rng.normal(0, 0.01 * s, 4)   # ~1.1 deg @ s=1
            quat = quat / (np.linalg.norm(quat) + 1e-9)
            gyro = gyro + self._noise_rng.normal(0, 0.05 * s, 3)   # rad/s
            vel = vel + self._noise_rng.normal(0, 0.03 * s, 3)     # m/s
        return {
            "joint_pos": self.joint_pos, "joint_vel": self.joint_vel,
            "imu_quat": quat, "gyro": gyro, "acc": acc, "vel": vel,
            "base_pos": self._sensor("base_pos"), "foot_contacts": self.foot_contacts(),
            "height": self.base_height(), "heading": self.heading(),
            "upright": self.upright(), "time": float(self.data.time),
        }

    # ---- rendering ----
    def renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.cfg.render_h, self.cfg.render_w)
        return self._renderer

    def render(self, camera="track_cam"):
        r = self.renderer()
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
