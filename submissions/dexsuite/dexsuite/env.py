"""DexSuite environment: a LEAP dexterous hand on a 6-DOF actuated wrist over a
tabletop workcell, with touch / force / IMU sensing and RGB-D cameras.

The environment is deliberately thin: it owns the MuJoCo model + data, exposes a
clean observation dict, and offers low-level target setters. Task logic lives in
``skills.py`` (motion primitives) and ``planner.py`` (the autonomous task).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mujoco
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
SCENE = os.path.join(ASSETS, "scene.xml")

# Wrist actuators, in order: x, y, z (slides) then yaw, pitch, roll (hinges).
WRIST_ACTS = ["wrist_x_act", "wrist_y_act", "wrist_z_act",
              "wrist_yaw_act", "wrist_pitch_act", "wrist_roll_act"]

# 16 finger joints (index, middle, ring, thumb), matching the LEAP convention.
FINGER_JOINTS = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]
FINGERTIPS = ["if", "mf", "rf", "th"]
OBJECTS = ["cube", "ball"]


@dataclass
class EnvConfig:
    scene: str = SCENE
    seed: int = 0
    # domain randomization magnitudes (used by the data collector)
    randomize: bool = False
    pos_jitter: float = 0.0          # metres of XY jitter on manipulands
    mass_jitter: float = 0.0         # fractional mass jitter
    render_w: int = 1280
    render_h: int = 720


class DexSuiteEnv:
    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.wrist_act = np.array([self._aid(n) for n in WRIST_ACTS])
        self.finger_act = {n: self._aid(n + "_act") for n in FINGER_JOINTS}
        self.finger_act_ids = np.array([self.finger_act[n] for n in FINGER_JOINTS])
        self._renderer: Optional[mujoco.Renderer] = None
        self._default_object_qpos = self._read_object_qpos()
        self._hooks: list = []          # callables invoked once per sim step
        self.reset()

    def add_step_hook(self, fn) -> None:
        """Register ``fn(env)`` to run after every simulation step."""
        self._hooks.append(fn)

    def clear_step_hooks(self) -> None:
        self._hooks.clear()

    # ---- id helpers -------------------------------------------------------
    def _aid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise KeyError(f"actuator {name!r} not found")
        return i

    def _sensor(self, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr:adr + dim].copy()

    def _jqadr(self, joint: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        return self.model.jnt_qposadr[jid]

    def _read_object_qpos(self) -> Dict[str, np.ndarray]:
        out = {}
        for o in OBJECTS:
            a = self._jqadr(o + "_free")
            out[o] = self.model.qpos0[a:a + 7].copy()
        return out

    # ---- lifecycle --------------------------------------------------------
    def reset(self) -> Dict:
        mujoco.mj_resetData(self.model, self.data)
        for o in OBJECTS:
            a = self._jqadr(o + "_free")
            qpos = self._default_object_qpos[o].copy()
            if self.cfg.randomize and self.cfg.pos_jitter > 0:
                qpos[0] += self.rng.uniform(-self.cfg.pos_jitter, self.cfg.pos_jitter)
                qpos[1] += self.rng.uniform(-self.cfg.pos_jitter, self.cfg.pos_jitter)
            self.data.qpos[a:a + 7] = qpos
        if self.cfg.randomize and self.cfg.mass_jitter > 0:
            for o in OBJECTS:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o)
                f = 1.0 + self.rng.uniform(-self.cfg.mass_jitter, self.cfg.mass_jitter)
                self.model.body_mass[bid] = max(1e-3, self.model.body_mass[bid] * f)
        # start at home, hand open
        self.data.ctrl[self.wrist_act] = [0, 0, 0.06, 0, 0, 0]
        from .skills import OPEN_POSE  # local import to avoid cycle
        self.set_finger_targets(OPEN_POSE)
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            for hook in self._hooks:
                hook(self)

    # ---- control ----------------------------------------------------------
    def set_wrist_target(self, pose6) -> None:
        self.data.ctrl[self.wrist_act] = np.asarray(pose6, dtype=float)

    def get_wrist_target(self) -> np.ndarray:
        return self.data.ctrl[self.wrist_act].copy()

    def set_finger_targets(self, pose: Dict[str, float]) -> None:
        for name, val in pose.items():
            self.data.ctrl[self.finger_act[name]] = val

    # ---- observation ------------------------------------------------------
    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    def touch(self) -> np.ndarray:
        return np.array([float(self._sensor(t + "_touch")[0]) for t in FINGERTIPS])

    def object_pose(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.concatenate([self.data.xpos[bid].copy(), self.data.xquat[bid].copy()])

    def button_depress(self) -> float:
        return float(self._sensor("btn_pos")[0])

    def get_obs(self) -> Dict:
        return {
            "finger_qpos": np.array([self.data.qpos[self._jqadr(j)] for j in FINGER_JOINTS]),
            "wrist_qpos": self.data.qpos[:6].copy(),
            "touch": self.touch(),
            "wrist_force": self._sensor("wrist_force"),
            "wrist_torque": self._sensor("wrist_torque"),
            "palm_acc": self._sensor("palm_acc"),
            "palm_gyro": self._sensor("palm_gyro"),
            "palm_pos": self._sensor("palm_pos"),
            "palm_quat": self._sensor("palm_quat"),
            "cube_pose": self.object_pose("cube"),
            "ball_pose": self.object_pose("ball"),
            "button": self.button_depress(),
            "time": float(self.data.time),
        }

    # ---- rendering --------------------------------------------------------
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.cfg.render_h, self.cfg.render_w)
        return self._renderer

    def render(self, camera: str = "scene_cam", depth: bool = False) -> np.ndarray:
        r = self.renderer()
        if depth:
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=camera)
            out = r.render()
            r.disable_depth_rendering()
            return out
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
