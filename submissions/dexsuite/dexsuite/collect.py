"""Automated data collection.

Runs the autonomous policy under domain randomization and records synchronized
trajectories — proprioception, tactile, wrist force/torque, IMU, object poses,
RGB + depth from the wrist camera, and the commanded action — to compressed
``.npz`` shards plus a JSON manifest. This turns the environment into a
reproducible imitation-learning dataset generator.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from .env import DexSuiteEnv, EnvConfig
from .planner import run_task
from .skills import SkillParams


class _Logger:
    def __init__(self, env, stride: int = 10, with_images: bool = True):
        self.env = env
        self.stride = stride
        self.with_images = with_images
        self.k = 0
        self.buf: Dict[str, List] = {}
        env.add_step_hook(self._on_step)

    def _push(self, key, val):
        self.buf.setdefault(key, []).append(np.asarray(val, dtype=np.float32))

    def _on_step(self, env):
        self.k += 1
        if self.k % self.stride:
            return
        obs = env.get_obs()
        self._push("finger_qpos", obs["finger_qpos"])
        self._push("wrist_qpos", obs["wrist_qpos"])
        self._push("touch", obs["touch"])
        self._push("wrist_ft", np.concatenate([obs["wrist_force"], obs["wrist_torque"]]))
        self._push("imu", np.concatenate([obs["palm_acc"], obs["palm_gyro"]]))
        self._push("cube_pose", obs["cube_pose"])
        self._push("ball_pose", obs["ball_pose"])
        self._push("action", env.data.ctrl.copy())
        self._push("time", [obs["time"]])
        if self.with_images:
            from PIL import Image
            small = Image.fromarray(env.render("wrist_cam")).resize((128, 72))
            self.buf.setdefault("wrist_rgb", []).append(np.asarray(small, dtype=np.uint8))

    def dump(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.buf.items()}


def collect(out_dir: str = "dataset", episodes: int = 5, stride: int = 10,
            with_images: bool = True, seed: int = 0,
            pos_jitter: float = 0.004, mass_jitter: float = 0.15) -> str:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"episodes": [], "stride": stride, "with_images": with_images}
    for ep in range(episodes):
        env = DexSuiteEnv(EnvConfig(seed=seed + ep, randomize=True,
                                    pos_jitter=pos_jitter, mass_jitter=mass_jitter))
        log = _Logger(env, stride=stride, with_images=with_images)
        res = run_task(env, SkillParams(), verbose=False)
        data = log.dump()
        path = os.path.join(out_dir, f"episode_{ep:03d}.npz")
        np.savez_compressed(path, **data)
        env.close()
        steps = int(len(data.get("time", [])))
        manifest["episodes"].append({
            "file": os.path.basename(path), "seed": seed + ep,
            "frames": steps, "success_rate": float(res.success_rate),
            "stages": {s.name: bool(s.ok) for s in res.stages},
        })
        print(f"episode {ep}: {steps} frames, success={res.success_rate:.2f} -> {path}")
    keys = sorted({k for ep in [data] for k in ep})
    manifest["fields"] = keys
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    n = sum(e["frames"] for e in manifest["episodes"])
    print(f"wrote {episodes} episodes ({n} frames) to {out_dir}/")
    return out_dir


if __name__ == "__main__":
    collect()
