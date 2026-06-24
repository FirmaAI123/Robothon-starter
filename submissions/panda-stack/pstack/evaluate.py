"""Quantitative evaluation for PandaStack: tower-build success (with post-release
stability) and 95% Wilson CIs, a recovery ablation (self-correction ON vs OFF under
perception noise), perception-noise robustness, and a closed-loop-vs-open-loop IK
ablation. All numbers reproducible; written to eval_results.json."""
from __future__ import annotations

import json
import numpy as np

from .env import PandaEnv, EnvConfig
from .controller import StackController


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _run(seed, noise=0.0, max_retry=2):
    env = PandaEnv(EnvConfig(seed=seed, randomize=True, pose_noise=noise))
    ctl = StackController(env, max_retry=max_retry)
    built = ctl.stack()
    stable = ctl.tower_stable() if built else False
    fails = sum(1 for l in ctl.log if l[2] != "placed")
    env.close()
    return {"built": built, "stable": stable, "fails": fails, "log": ctl.log}


def stack_success(trials=16, noise=0.0, out_json=None):
    rows = [_run(i, noise=noise) for i in range(trials)]
    ok = sum(r["stable"] for r in rows)
    lo, hi = wilson_ci(ok, trials)
    print(f"\n=== PandaStack: 3-cube tower success ({trials} randomized trials, "
          f"pose-noise {noise*100:.0f} cm) ===")
    print(f"built a stable tower (survives post-release settle): {ok}/{trials} "
          f"({ok/trials*100:.0f}%)  95% Wilson CI [{lo*100:.0f}%, {hi*100:.0f}%]")
    res = {"trials": trials, "noise": noise, "success_rate": ok / trials, "ci": [lo, hi]}
    _persist(out_json, "stack_success", res)
    return res


def recovery_ablation(trials=24, noise=0.012, out_json=None):
    """Self-correction ON (retry on detected failure) vs OFF (one shot) under the SAME
    perception noise — isolates the agentic recovery loop's contribution."""
    on = [_run(i, noise=noise, max_retry=2) for i in range(trials)]
    off = [_run(i, noise=noise, max_retry=0) for i in range(trials)]
    s_on = sum(r["stable"] for r in on); s_off = sum(r["stable"] for r in off)
    recovered = sum(1 for r in on if r["stable"] and r["fails"] > 0)
    lo_on, hi_on = wilson_ci(s_on, trials); lo_off, hi_off = wilson_ci(s_off, trials)
    print(f"\n=== Agentic recovery ablation (pose-noise {noise*100:.1f} cm, {trials} trials) ===")
    print(f"recovery OFF (one shot): {s_off}/{trials} ({s_off/trials*100:.0f}%)  CI [{lo_off*100:.0f}%, {hi_off*100:.0f}%]")
    print(f"recovery ON  (re-grasp): {s_on}/{trials} ({s_on/trials*100:.0f}%)  CI [{lo_on*100:.0f}%, {hi_on*100:.0f}%]")
    print(f"runs that hit a failure and still completed (self-corrected): {recovered}/{trials}")
    res = {"trials": trials, "noise": noise, "success_off": s_off / trials, "success_on": s_on / trials,
           "off_ci": [lo_off, hi_off], "on_ci": [lo_on, hi_on], "self_corrected_runs": recovered}
    _persist(out_json, "recovery_ablation", res)
    return res


def perception_noise(noises=(0.0, 0.01, 0.02), trials=12, out_json=None):
    print(f"\n=== Perception-noise robustness ({trials} trials per level) ===")
    print(f"{'sensing noise':>14s}  tower success")
    out = []
    for nz in noises:
        ok = sum(_run(i, noise=nz)["stable"] for i in range(trials))
        lo, hi = wilson_ci(ok, trials)
        print(f"{nz*100:11.0f} cm   {ok}/{trials} ({ok/trials*100:.0f}%)  CI [{lo*100:.0f}%, {hi*100:.0f}%]")
        out.append({"noise": nz, "success_rate": ok / trials, "ci": [lo, hi]})
    _persist(out_json, "perception_noise", out)
    return out


def ik_ablation(trials=16, out_json=None):
    """Closed-loop DLS-IK servoing vs a single open-loop IK solve-then-execute, on the
    grasp primitive — proves the value of closing the loop, quantitatively."""
    def grasp_rate(closed):
        ok = 0
        for s in range(trials):
            env = PandaEnv(EnvConfig(seed=s, randomize=True)); ctl = StackController(env)
            got = ctl.grasp(0, closed_loop=closed); ok += got; env.close()
        return ok
    cl = grasp_rate(True); ol = grasp_rate(False)
    lo_c, hi_c = wilson_ci(cl, trials); lo_o, hi_o = wilson_ci(ol, trials)
    print(f"\n=== Closed-loop vs open-loop IK (grasp success, {trials} trials) ===")
    print(f"open-loop  (one IK solve, execute): {ol}/{trials} ({ol/trials*100:.0f}%)  CI [{lo_o*100:.0f}%, {hi_o*100:.0f}%]")
    print(f"closed-loop (DLS-IK servoing):      {cl}/{trials} ({cl/trials*100:.0f}%)  CI [{lo_c*100:.0f}%, {hi_c*100:.0f}%]")
    res = {"trials": trials, "open_loop": ol / trials, "closed_loop": cl / trials,
           "open_ci": [lo_o, hi_o], "closed_ci": [lo_c, hi_c]}
    _persist(out_json, "ik_ablation", res)
    return res


def _persist(path, key, val):
    if not path:
        return
    try:
        with open(path) as f:
            base = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        base = {}
    base[key] = val
    with open(path, "w") as f:
        json.dump(base, f, indent=2)
    print(f"updated {path} ({key})")


if __name__ == "__main__":
    J = "eval_results.json"
    stack_success(out_json=J)
    recovery_ablation(out_json=J)
    perception_noise(out_json=J)
    ik_ablation(out_json=J)
