"""PandaStack environment: a Franka Panda (7-DOF arm + parallel gripper) on a table
with three free cubes, driven by real physics (``mj_step``, contacts). Exposes the
end-effector grasp point + arm Jacobian for closed-loop IK, cube poses (ground truth
and a noisy *sensed* estimate), grasp-force from the contact buffer, and rendering.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
SCENE = os.path.join(ASSETS, "franka", "stack_scene.xml")

CUBES = ["cube1", "cube2", "cube3"]
CUBE_START = np.array([[0.40, 0.15], [0.57, 0.10], [0.43, -0.15]])  # (x,y) on the table
TABLE_TOP = 0.40
CUBE_H = 0.05
GRASP_OFFSET = np.array([0.0, 0.0, 0.103])   # hand-origin -> grasp point (down the hand axis)


@dataclass
class EnvConfig:
    scene: str = SCENE
    seed: int = 0
    randomize: bool = False
    pos_jitter: float = 0.03     # m, start-position jitter of cubes
    pose_noise: float = 0.0      # m, std of the SENSED cube-pose estimate (real perception)
    render_w: int = 1280
    render_h: int = 720


class PandaEnv:
    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.noise_rng = np.random.default_rng(self.cfg.seed + 991)
        self.hand = self._bid("hand")
        self.arm_q = np.array([self._jqadr(f"joint{i}") for i in range(1, 8)])
        self.arm_dof = np.array([self._jdadr(f"joint{i}") for i in range(1, 8)])
        self.finger_q = self._jqadr("finger_joint1")
        self.grip_act = self._aid("actuator8")
        self.arm_act = np.array([self._aid(f"actuator{i}") for i in range(1, 8)])
        self.cube_bid = [self._bid(c) for c in CUBES]
        self.cube_qadr = [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self.cube_bid]
        self.cube_gid = [self._gid(c) for c in CUBES]
        self.finger_gids = {g for g in range(self.model.ngeom)
                            if self.model.geom_bodyid[g] in
                            (self._bid("left_finger"), self._bid("right_finger"))}
        self._renderer = None
        self._hooks = []
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))
        self.reset()

    # ---- id helpers ----
    def _bid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
    def _gid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
    def _aid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def _jqadr(self, n): return self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def _jdadr(self, n): return self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]

    def add_step_hook(self, fn): self._hooks.append(fn)

    # ---- lifecycle ----
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)   # arm "home"
        for i, adr in enumerate(self.cube_qadr):
            xy = CUBE_START[i].copy()
            if self.cfg.randomize:
                xy = xy + self.rng.uniform(-self.cfg.pos_jitter, self.cfg.pos_jitter, 2)
            self.data.qpos[adr:adr + 3] = [xy[0], xy[1], TABLE_TOP + CUBE_H / 2 + 0.002]
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        return self

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            for h in self._hooks:
                h(self)

    # ---- control ----
    def set_arm(self, q):
        lo = self.model.actuator_ctrlrange[self.arm_act, 0]
        hi = self.model.actuator_ctrlrange[self.arm_act, 1]
        self.data.ctrl[self.arm_act] = np.clip(q, lo, hi)

    def set_grip(self, g):
        self.data.ctrl[self.grip_act] = g     # 255 = open, 0 = closed

    # ---- kinematics / perception ----
    def grasp_point(self):
        return self.data.xpos[self.hand] + self.data.xmat[self.hand].reshape(3, 3) @ GRASP_OFFSET

    def arm_jac(self):
        mujoco.mj_jac(self.model, self.data, self._jp, self._jr, self.grasp_point(), self.hand)
        return self._jp[:, self.arm_dof]

    def arm_qpos(self):
        return self.data.qpos[self.arm_q].copy()

    def cube_pos(self, i):
        return self.data.xpos[self.cube_bid[i]].copy()

    def sensed_pos(self, i):
        """Cube position as a real sensor would report it: ground truth + noise/bias."""
        p = self.cube_pos(i)
        if self.cfg.pose_noise > 0:
            p = p + self.noise_rng.normal(0, self.cfg.pose_noise, 3) * np.array([1, 1, 0.3])
        return p

    def grasp_force(self):
        """Sum of finger contact normal forces (N) against any cube — measured grip."""
        f = np.zeros(6); tot = 0.0
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            if (c.geom1 in self.finger_gids) ^ (c.geom2 in self.finger_gids):
                other = c.geom2 if c.geom1 in self.finger_gids else c.geom1
                if other in self.cube_gid:
                    mujoco.mj_contactForce(self.model, self.data, k, f)
                    tot += abs(f[0])
        return float(tot)

    def holding(self, i):
        """Is cube i currently in the gripper (near the grasp point, off the table)?"""
        return float(np.linalg.norm(self.cube_pos(i) - self.grasp_point())) < 0.06

    # ---- rendering ----
    def render(self, camera="track_cam"):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.cfg.render_h, self.cfg.render_w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
