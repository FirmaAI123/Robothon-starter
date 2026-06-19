"""Render the annotated demo video by running the autonomous patrol."""
from __future__ import annotations

from .env import QuadEnv, EnvConfig
from .controller import TrotController, patrol_command
from .video import Recorder, ACCENT, WHITE, DIM, GREEN


def record_demo(out_path="demo.mp4", seed=0, fps=30, every=12, secs=16.0):
    env = QuadEnv(EnvConfig(seed=seed))
    env.reset()
    ctl = TrotController(env)
    rec = Recorder(env, ctl, every=every)
    rec.title_card([
        ("QuadLoco", 54, ACCENT),
        ("Physics-Based Quadruped Locomotion", 26, WHITE),
        ("Unitree Go2 · 12 torque motors · CPG trot + PD · MuJoCo", 20, DIM),
        ("real dynamics & contacts — every frame is mj_step", 20, DIM),
    ], n=42)
    n = int(secs / env.dt)
    last = None
    for i in range(n):
        t = i * env.dt
        fwd, turn, label = patrol_command(t)
        if label != last:
            rec.set_phase(label)
            last = label
        ctl.act(t, fwd, turn)
        env.step(1)
    import numpy as np
    dist = float(np.linalg.norm(env.base_xy() - rec.x0))
    rec.title_card([
        ("Results", 34, GREEN),
        (f"autonomous patrol: {dist:.1f} m traversed, stayed upright", 20, WHITE),
        ("closed-loop torque control · real contacts · no kinematic scripting", 20, DIM),
        ("reproducible · headless · no GPU", 20, DIM),
    ], n=48)
    rec.save(out_path, fps=fps)
    env.close()
    print(f"wrote {out_path} ({len(rec.frames)} frames, {len(rec.frames)/fps:.1f}s)")
    return out_path


if __name__ == "__main__":
    record_demo()
