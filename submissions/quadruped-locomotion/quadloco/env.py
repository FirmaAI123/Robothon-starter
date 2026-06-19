"""QuadLoco environment: a Unitree Go2 quadruped on flat ground, driven by real
torque actuators with full physics (contacts, friction) — every frame advances
``mj_step``, not a kinematic puppet. Exposes an IMU/proprioception observation,
foot-contact detection, rendering, and per-step hooks for the recorder/eval.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
SCENE = os.path.join(ASSETS, "scene.xml")

LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = [f"{lg}_{j}" for lg in LEGS for j in ("hip", "thigh", "calf")]  # 12, actuator order


@dataclass
class EnvConfig:
    scene: str = SCENE
    seed: int = 0
    randomize: bool = False
    mass_jitter: float = 0.0      # fractional trunk-mass jitter
    yaw_jitter: float = 0.0       # rad of initial heading jitter
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
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._renderer: Optional[mujoco.Renderer] = None
        self._hooks: list = []
        self.reset()

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
    def reset(self) -> Dict:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)   # "home" standing pose
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
            mujoco.mj_step(self.model, self.data)
            for h in self._hooks:
                h(self)

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

    def foot_contacts(self):
        c = {lg: 0.0 for lg in LEGS}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            for lg in LEGS:
                fg = self.foot_gid[lg]
                if (g1 == fg and g2 == self.floor_gid) or (g2 == fg and g1 == self.floor_gid):
                    c[lg] = 1.0
        return np.array([c[lg] for lg in LEGS])

    def get_obs(self) -> Dict:
        return {
            "joint_pos": self.joint_pos, "joint_vel": self.joint_vel,
            "imu_quat": self._sensor("imu_quat"), "gyro": self._sensor("imu_gyro"),
            "acc": self._sensor("imu_acc"), "vel": self._sensor("imu_vel"),
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
