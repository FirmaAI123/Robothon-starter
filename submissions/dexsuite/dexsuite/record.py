"""Render the annotated demo video by running the autonomous task."""
from __future__ import annotations

from .env import DexSuiteEnv, EnvConfig
from .planner import run_task
from .video import Recorder, ACCENT, WHITE, DIM, GREEN


def record_demo(out_path: str = "demo.mp4", seed: int = 0, fps: int = 30,
                every: int = 18) -> str:
    env = DexSuiteEnv(EnvConfig(seed=seed))
    rec = Recorder(env, every=every)
    rec.title_card([
        ("DexSuite", 54, ACCENT),
        ("Dexterous In-Hand & Pick-Place Manipulation", 26, WHITE),
        ("LEAP Hand · 22-DOF · MuJoCo", 20, DIM),
        ("FFAI Robothon 2026", 20, DIM),
    ], n=42)
    res = run_task(env, recorder=rec, verbose=True)
    rec.title_card([
        ("Task complete", 34, GREEN),
        (f"stages passed: {sum(s.ok for s in res.stages)}/{len(res.stages)}", 26, WHITE),
        ("autonomous · touch-sensed · reproducible", 20, DIM),
    ], n=48)
    path = rec.save(out_path, fps=fps)
    env.close()
    print(f"wrote {path} ({len(rec.frames)} frames, {len(rec.frames)/fps:.1f}s)")
    return path


if __name__ == "__main__":
    record_demo()
