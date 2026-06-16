"""Autonomous task planner: a deterministic state machine that runs the full
DexSuite long-horizon task and reports per-stage success metrics.

Task: sort two parts from their pick fixtures into colour-matched bins (with an
in-hand reorientation en route), then press a confirmation button.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .env import DexSuiteEnv
from .skills import Hand, SkillParams

# Bin centres (world XY) and the button location, matched to scene.xml.
BIN = {"cube": (-0.24, 0.15), "ball": (-0.24, -0.15)}
BUTTON_XY = (-0.02, 0.20)
BUTTON_PRESS_Z = -0.06
PRESS_TARGET = 0.012   # metres of cap travel that counts as a press


@dataclass
class StageResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TaskResult:
    stages: List[StageResult] = field(default_factory=list)
    success_rate: float = 0.0

    def add(self, name, ok, detail=""):
        self.stages.append(StageResult(name, ok, detail))

    def finalize(self):
        self.success_rate = np.mean([s.ok for s in self.stages]) if self.stages else 0.0
        return self


def _in_bin(pos, bin_xy, tol=0.085):
    return abs(pos[0] - bin_xy[0]) < tol and abs(pos[1] - bin_xy[1]) < tol and pos[2] < 0.5


def run_task(env: DexSuiteEnv, params: SkillParams | None = None,
             reorient: bool = True, verbose: bool = True, recorder=None) -> TaskResult:
    hand = Hand(env, params)
    res = TaskResult()
    env.reset()

    def phase(t):
        if recorder is not None:
            recorder.set_phase(t)

    def tick(t):
        if recorder is not None:
            recorder.tick(t)

    def cam(name):
        if recorder is not None:
            recorder.set_camera(name)

    phase("Initializing workcell")
    env.step(150)

    labels = {"cube": "red cube", "ball": "blue ball"}
    for obj in ("cube", "ball"):
        start = env.object_pose(obj)[:3].copy()
        cam("track_cam")
        phase(f"Grasping {labels[obj]} from pick fixture")
        contacts = hand.pick_at(start[:2])
        lift_z = env.object_pose(obj)[2]
        lifted = lift_z > 0.6   # object cleared its pedestal -> grasp succeeded
        res.add(f"grasp_{obj}", lifted,
                f"lift_z={lift_z:.3f}, fingertip_contacts={int(np.count_nonzero(env.touch()>0.05))}")
        if lifted:
            tick(f"Grasped {labels[obj]}")

        if reorient:
            cam("track_cam")
            phase(f"In-hand reorientation of {labels[obj]} (~90 deg)")
            deg, held = hand.reorient(obj, roll=1.5)
            ok = held and deg > 45.0
            res.add(f"reorient_{obj}", ok, f"in-hand rotation ~{deg:.0f} deg, still held={held}")
            if ok:
                tick(f"Reoriented {labels[obj]} ~{deg:.0f}deg in-hand")

        cam("scene_cam")
        phase(f"Placing {labels[obj]} in colour-matched bin")
        hand.place_in(BIN[obj])
        pos = env.object_pose(obj)[:3]
        placed = _in_bin(pos, BIN[obj])
        res.add(f"place_{obj}", placed, f"final=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
        if placed:
            tick(f"Sorted {labels[obj]} into bin")
        hand.move_wrist(0.0, 0.0, hand.p.lift_z, steps=300)

    cam("scene_cam")
    phase("Fingertip precision: pressing confirmation button")
    pressed = hand.press_button(BUTTON_XY, BUTTON_PRESS_Z)
    ok = abs(pressed) >= PRESS_TARGET
    res.add("press_button", ok, f"cap travel={abs(pressed)*1000:.1f} mm")
    if ok:
        tick("Confirmation button pressed")
    phase("Task complete")
    env.step(120)

    res.finalize()
    if verbose:
        for s in res.stages:
            print(f"[{'PASS' if s.ok else 'FAIL'}] {s.name:16s} {s.detail}")
        print(f"success_rate={res.success_rate:.2f}")
    return res
