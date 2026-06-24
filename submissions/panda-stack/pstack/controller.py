"""Closed-loop manipulation controller for the Panda: damped-least-squares IK that
servos the gripper grasp point to Cartesian targets, grasp/place primitives, and an
autonomous **stacking** state machine that *self-verifies* each step and *recovers*
from detected failures (re-grasp / re-place) instead of aborting.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .env import TABLE_TOP, CUBE_H

OPEN, CLOSE = 255.0, 0.0
BASE_XY = np.array([0.48, 0.0])     # where the tower is built
GRASP_F = 0.5                       # N, min finger contact force confirming a real grip
                                    # (empty close reads ~0 N; a held cube reads ~2.5 N)


class StackController:
    def __init__(self, env, max_retry=2):
        self.env = env
        self.max_retry = max_retry
        d = mujoco.MjData(env.model); mujoco.mj_resetDataKeyframe(env.model, d, 0)
        self.qhome = d.qpos[env.arm_q].copy()
        self.log = []                # (cube, attempt, event)
        self.phase = "Ready"         # for the demo HUD
        self.placed = [False, False, False]

    # ---- closed-loop IK servo (position task + nullspace home bias) ----
    def ik_servo(self, target, grip, steps, kp=5.0, closed_loop=True):
        target = np.asarray(target, float)
        I3, I7 = np.eye(3), np.eye(7)
        for _ in range(steps):
            gp = self.env.grasp_point(); J = self.env.arm_jac()
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * I3, kp * (target - gp))
            if closed_loop:
                N = I7 - np.linalg.pinv(J) @ J
                dq = dq + N @ (0.3 * (self.qhome - self.env.arm_qpos()))
            self.env.set_arm(self.env.arm_qpos() + np.clip(dq, -0.06, 0.06))
            self.env.set_grip(grip); self.env.step(1)
        return float(np.linalg.norm(target - self.env.grasp_point()))

    def open_loop_reach(self, target, grip, steps):
        """Ablation baseline: solve IK once to the target, command it, execute open-loop."""
        gp = self.env.grasp_point(); J = self.env.arm_jac()
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), np.asarray(target, float) - gp)
        q = self.env.arm_qpos() + dq
        for _ in range(steps):
            self.env.set_arm(q); self.env.set_grip(grip); self.env.step(1)
        return float(np.linalg.norm(np.asarray(target, float) - self.env.grasp_point()))

    # ---- primitives ----
    def grasp(self, i, closed_loop=True):
        """Grasp cube i using its SENSED pose; returns True if securely held after lift."""
        s = self.env.sensed_pos(i); cz = TABLE_TOP + CUBE_H / 2
        reach = self.ik_servo if closed_loop else self.open_loop_reach
        reach([s[0], s[1], cz + 0.13], OPEN, 350)      # above
        reach([s[0], s[1], cz + 0.004], OPEN, 350)     # descend to cube
        self.ik_servo([s[0], s[1], cz + 0.004], CLOSE, 250)   # close (always closed-loop hold)
        self.ik_servo([s[0], s[1], cz + 0.22], CLOSE, 350)    # lift
        # verify: cube held near the gripper, lifted off the table, AND the fingers are
        # actually pressing it (measured contact force from the buffer) — not an empty close.
        secure = (self.env.holding(i)
                  and self.env.cube_pos(i)[2] > cz + 0.12
                  and self.env.grasp_force() > GRASP_F)
        return bool(secure)

    def place(self, i, target_xyz):
        t = np.asarray(target_xyz, float)
        self.ik_servo([t[0], t[1], t[2] + 0.20], CLOSE, 350)   # carry above target
        self.ik_servo([t[0], t[1], t[2] + 0.012], CLOSE, 400)  # lower onto the stack
        self.ik_servo([t[0], t[1], t[2] + 0.012], OPEN, 200)   # release
        self.ik_servo([t[0], t[1], t[2] + 0.20], OPEN, 300)    # retreat up

    def recover(self):
        """After a detected failure: open, lift clear, return toward home posture."""
        gp = self.env.grasp_point()
        self.ik_servo([gp[0], gp[1], TABLE_TOP + 0.32], OPEN, 300)

    def placed_ok(self, i, tz, tol=0.045):
        c = self.env.cube_pos(i)
        return (np.hypot(c[0] - BASE_XY[0], c[1] - BASE_XY[1]) < tol) and (abs(c[2] - tz) < 0.03)

    # ---- autonomous stacking mission (self-verifying + self-correcting) ----
    def stack(self, order=(0, 1, 2)):
        self.log = []; self.placed = [False, False, False]
        for level, i in enumerate(order):
            tz = TABLE_TOP + CUBE_H * (level + 0.5)
            done = False
            for attempt in range(self.max_retry + 1):
                self.phase = (f"Grasping cube {i + 1}" if attempt == 0
                              else f"Re-grasping cube {i + 1} (recovered)")
                got = self.grasp(i)
                if not got:                                   # missed / slipped on lift
                    self.log.append((i, attempt, "grasp_failed"))
                    self.phase = f"Grasp missed — self-correcting"; self.recover(); continue
                self.phase = f"Placing cube {i + 1} on the tower (level {level + 1})"
                self.place(i, [BASE_XY[0], BASE_XY[1], tz])
                if self.placed_ok(i, tz):
                    self.log.append((i, attempt, "placed")); self.placed[i] = True; done = True; break
                self.log.append((i, attempt, "place_off"))    # placement out of tolerance
                self.phase = "Placement off — self-correcting"; self.recover()
            if not done:
                self.phase = f"Failed cube {i + 1}"; return False
        self.phase = "Tower complete"
        return True

    def tower_stable(self, order=(0, 1, 2), settle=600):
        """Verify the tower stays standing after release (post-release settle check)."""
        for _ in range(settle):
            self.env.step(1)
        for level, i in enumerate(order):
            c = self.env.cube_pos(i)
            if np.hypot(c[0] - BASE_XY[0], c[1] - BASE_XY[1]) > 0.06:
                return False
            if c[2] < TABLE_TOP + CUBE_H * (level + 0.5) - 0.03:    # toppled / fell a level
                return False
        return True
