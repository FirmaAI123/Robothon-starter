"""Quantitative locomotion evaluation: many randomized trials (trunk-mass and
initial-heading jitter), reporting forward distance, mean speed, trunk-upright
rate, and fall rate with 95% Wilson confidence intervals."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from .env import QuadEnv, EnvConfig
from .controller import TrotController, GaitParams


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def walk_trial(seed, secs=10.0, randomize=True):
    env = QuadEnv(EnvConfig(seed=seed, randomize=randomize,
                            mass_jitter=0.12, yaw_jitter=0.05))
    env.reset()
    ctl = TrotController(env)
    x0 = env.base_xy().copy()
    min_up = 1.0
    n = int(secs / env.dt)
    fell = False
    for i in range(n):
        t = i * env.dt
        ctl.act(t, forward=1.0, turn=0.0)
        env.step(1)
        min_up = min(min_up, env.upright())
        if env.base_height() < 0.15:
            fell = True
            break
    dist = float(env.base_xy()[0] - x0[0])
    env.close()
    return {"dist": dist, "speed": dist / secs, "min_up": min_up, "fell": fell}


def evaluate(trials=20, secs=10.0, out_json=None) -> Dict:
    rows = [walk_trial(i, secs) for i in range(trials)]
    upright = sum(not r["fell"] and r["min_up"] > 0.5 for r in rows)
    dists = np.array([r["dist"] for r in rows])
    speeds = np.array([r["speed"] for r in rows])
    lo, hi = wilson_ci(upright, trials)
    print(f"\n=== QuadLoco evaluation: {trials} randomized trials over the TERRAIN course "
          f"(ramp+bumps+step; trunk mass ±12%, heading ±0.05 rad, {secs:.0f}s) ===")
    print(f"stayed upright (no fall): {upright}/{trials} ({upright/trials*100:.0f}%)  "
          f"95% CI [{lo*100:.0f}%, {hi*100:.0f}%]")
    print(f"forward distance: {dists.mean():.2f} ± {dists.std():.2f} m  "
          f"[{dists.min():.2f}, {dists.max():.2f}]")
    print(f"mean speed:       {speeds.mean():.2f} ± {speeds.std():.2f} m/s")
    summary = {
        "trials": trials, "secs": secs,
        "upright_rate": upright / trials, "upright_ci": [lo, hi],
        "distance_m": {"mean": float(dists.mean()), "std": float(dists.std()),
                       "min": float(dists.min()), "max": float(dists.max())},
        "speed_mps": {"mean": float(speeds.mean()), "std": float(speeds.std())},
    }
    if out_json:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out_json}")
    return summary


if __name__ == "__main__":
    evaluate()
