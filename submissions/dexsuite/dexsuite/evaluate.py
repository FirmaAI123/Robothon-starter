"""Quantitative evaluation: run many randomized trials and report aggregate
statistics — per-stage success rates and manipulation-quality metrics. This is
the empirical evidence behind every claim in the README.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .env import DexSuiteEnv, EnvConfig
from .planner import run_task
from .skills import SkillParams


def evaluate(trials: int = 20, seed: int = 0, randomize: bool = True,
             pos_jitter: float = 0.003, mass_jitter: float = 0.10,
             out_json: str | None = None) -> Dict:
    stage_pass: Dict[str, int] = defaultdict(int)
    metrics: Dict[str, List[float]] = defaultdict(list)
    overall = []
    for i in range(trials):
        env = DexSuiteEnv(EnvConfig(seed=seed + i, randomize=randomize,
                                    pos_jitter=pos_jitter, mass_jitter=mass_jitter))
        res = run_task(env, SkillParams(), verbose=False, gait_ball=False)
        env.close()
        for s in res.stages:
            stage_pass[s.name] += int(s.ok)
        for k, v in res.metrics.items():
            metrics[k].append(v)
        overall.append(res.success_rate)

    def stat(xs):
        a = np.asarray(xs, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    summary = {
        "trials": trials, "randomized": randomize,
        "pos_jitter_m": pos_jitter, "mass_jitter_frac": mass_jitter,
        "overall_success_rate": float(np.mean(overall)),
        "stage_success_rate": {k: stage_pass[k] / trials for k in stage_pass},
        "metrics": {k: stat(v) for k, v in metrics.items()},
    }

    order = ["grasp_cube", "reorient_cube", "place_cube",
             "grasp_ball", "reorient_ball", "place_ball", "press_button"]
    print(f"\n=== DexSuite evaluation: {trials} randomized trials "
          f"(pos±{pos_jitter*1000:.0f}mm, mass±{mass_jitter*100:.0f}%) ===")
    print(f"{'stage':18s} success")
    for k in order:
        if k in stage_pass:
            print(f"{k:18s} {stage_pass[k]}/{trials}  ({stage_pass[k]/trials*100:4.0f}%)")
    print(f"{'OVERALL':18s} {np.mean(overall)*100:4.0f}%")
    print("\nmanipulation metrics (mean ± std):")
    for k in ("lift_z_cube", "lift_z_ball", "reorient_deg_cube",
              "gait_deg_ball", "button_mm"):
        if k in metrics:
            s = stat(metrics[k])
            print(f"  {k:18s} {s['mean']:6.2f} ± {s['std']:4.2f}  "
                  f"[{s['min']:.2f}, {s['max']:.2f}]")

    if out_json:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {out_json}")
    return summary


def gait_benchmark(trials: int = 20, seed: int = 0, randomize: bool = True,
                   pos_jitter: float = 0.003, mass_jitter: float = 0.10) -> Dict:
    """Standalone benchmark of the finger-gaiting in-hand rotation capability:
    grasp the ball, then roll it with the fingers (wrist fixed). Reports how often
    the object stays secured and the rotation achieved."""
    from .skills import Hand
    held = 0
    degs: List[float] = []
    for i in range(trials):
        env = DexSuiteEnv(EnvConfig(seed=seed + i, randomize=randomize,
                                    pos_jitter=pos_jitter, mass_jitter=mass_jitter))
        env.reset(); env.step(120)
        h = Hand(env)
        h.pick_at(env.object_pose("ball")[:2])
        deg, ok = h.inhand_spin("ball", cycles=8, amp=0.26, closure=0.90)
        env.close()
        held += int(ok)
        if ok:
            degs.append(deg)
    a = np.asarray(degs, dtype=float)
    print(f"\n=== Finger-gaiting benchmark: {trials} randomized trials "
          f"(wrist held fixed) ===")
    print(f"object kept secured: {held}/{trials} ({held/trials*100:.0f}%)")
    if len(a):
        print(f"in-hand rotation when secured: {a.mean():.0f} ± {a.std():.0f} deg "
              f"(max {a.max():.0f}deg)")
    return {"trials": trials, "secured_rate": held / trials,
            "rotation_deg": {"mean": float(a.mean()) if len(a) else 0.0,
                             "max": float(a.max()) if len(a) else 0.0}}


if __name__ == "__main__":
    evaluate()
    gait_benchmark()
