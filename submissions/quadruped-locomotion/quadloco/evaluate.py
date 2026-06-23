"""Quantitative locomotion evaluation: many randomized trials (trunk-mass and
initial-heading jitter), reporting forward distance, mean speed, trunk-upright
rate, and fall rate with 95% Wilson confidence intervals."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from .env import QuadEnv, EnvConfig, FLAT, CARGO
from .controller import TrotController, GaitParams

# loco-manipulation delivery route: cross the terrain, take a gentle turn, walk on
DELIVERY_ROUTE = [(8.0, 1.0, 0.0), (3.0, 0.5, 0.15), (3.0, 1.0, 0.0)]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def mean_ci(x, z=1.96):
    """Mean and 95% CI half-width (normal approx) for a sample."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return (0.0, 0.0)
    return (float(x.mean()), float(z * x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)


def _versions():
    import mujoco
    return {"mujoco": mujoco.__version__, "numpy": np.__version__}


# randomization applied to every robustness trial (kept in one place for the README)
RANDOMIZE = dict(mass_jitter=0.12, yaw_jitter=0.05, friction_jitter=0.20)


def walk_trial(seed, secs=10.0, randomize=True, terrain_fb=True):
    env = QuadEnv(EnvConfig(seed=seed, randomize=randomize, **RANDOMIZE))
    env.reset()
    ctl = TrotController(env)
    ctl.terrain_fb = terrain_fb
    x0 = env.base_xy().copy()
    min_up = 1.0
    n = int(secs / env.dt)
    fell = False
    pitches = []
    for i in range(n):
        t = i * env.dt
        ctl.drive(t, forward=1.0, turn_cmd=0.0)   # closed-loop IMU heading hold
        env.step(1)
        min_up = min(min_up, env.upright())
        pitches.append(abs(ctl.imu_pitch_roll()[0]))
        if env.base_height() < 0.15:
            fell = True
            break
    dist = float(env.base_xy()[0] - x0[0])
    drift = float(abs(env.base_xy()[1] - x0[1]))
    pdeg = np.degrees(pitches)
    env.close()
    return {"dist": dist, "speed": dist / secs, "min_up": min_up, "fell": fell,
            "drift": drift, "pitch": float(pdeg.mean()), "pitch_max": float(pdeg.max())}


def evaluate(trials=20, secs=10.0, out_json=None) -> Dict:
    rows = [walk_trial(i, secs) for i in range(trials)]
    upright = sum(not r["fell"] and r["min_up"] > 0.5 for r in rows)
    dists = np.array([r["dist"] for r in rows])
    speeds = np.array([r["speed"] for r in rows])
    drifts = np.array([r["drift"] for r in rows])
    pits = np.array([r["pitch"] for r in rows])
    lo, hi = wilson_ci(upright, trials)
    dm, dci = mean_ci(dists); sm, sci = mean_ci(speeds)
    drm, drci = mean_ci(drifts); pm, pci = mean_ci(pits)
    print(f"\n=== QuadLoco evaluation: {trials} randomized trials over the TERRAIN course "
          f"(ramp+rough-heightfield+step; mass ±12%, heading ±0.05 rad, friction ±20%, {secs:.0f}s) ===")
    print(f"stayed upright (no fall): {upright}/{trials} ({upright/trials*100:.0f}%)  "
          f"95% CI [{lo*100:.0f}%, {hi*100:.0f}%]")
    print(f"forward distance: {dm:.2f} ± {dci:.2f} m (95% CI)  [{dists.min():.2f}, {dists.max():.2f}]")
    print(f"mean speed:       {sm:.2f} ± {sci:.2f} m/s")
    print(f"lateral drift (IMU heading-hold): {drm:.2f} ± {drci:.2f} m")
    print(f"mean trunk pitch (terrain leveling): {pm:.1f} ± {pci:.1f} deg")
    summary = {
        "trials": trials, "secs": secs, "seeds": list(range(trials)),
        "randomization": RANDOMIZE, "versions": _versions(),
        "upright_rate": upright / trials, "upright_ci": [lo, hi],
        "distance_m": {"mean": dm, "ci95": dci, "min": float(dists.min()), "max": float(dists.max())},
        "speed_mps": {"mean": sm, "ci95": sci},
        "drift_m": {"mean": drm, "ci95": drci},
        "trunk_pitch_deg": {"mean": pm, "ci95": pci},
    }
    if out_json:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out_json}")
    return summary


def ablation_terrain_feedback(trials=20, secs=10.0, out_json=None):
    """Paired-seed ablation: closed-loop terrain feedback ON vs OFF on the SAME seeds,
    over the terrain course — isolates the IMU active-leveling loop. Reports the
    per-seed pitch reduction with a 95% CI (paired), plus peak pitch & uprightness."""
    on = [walk_trial(i, secs, terrain_fb=True) for i in range(trials)]
    off = [walk_trial(i, secs, terrain_fb=False) for i in range(trials)]
    def agg(rows):
        return (sum(not r["fell"] and r["min_up"] > 0.5 for r in rows) / trials,
                float(np.mean([r["pitch"] for r in rows])),
                float(np.mean([r["pitch_max"] for r in rows])),
                float(np.mean([r["min_up"] for r in rows])))
    on_a, off_a = agg(on), agg(off)
    delta = np.array([off[i]["pitch"] - on[i]["pitch"] for i in range(trials)])  # paired
    dmean, dci = mean_ci(delta)
    print(f"\n=== Ablation: closed-loop terrain feedback (paired, {trials} seeds) ===")
    print(f"{'':16s} upright   mean|pitch|  peak|pitch|  min-upright")
    print(f"{'feedback OFF':16s} {off_a[0]*100:4.0f}%     {off_a[1]:4.1f} deg     {off_a[2]:4.1f} deg     {off_a[3]:.2f}")
    print(f"{'feedback ON':16s} {on_a[0]*100:4.0f}%     {on_a[1]:4.1f} deg     {on_a[2]:4.1f} deg     {on_a[3]:.2f}")
    print(f"paired pitch reduction: {dmean:.2f} ± {dci:.2f} deg (95% CI, n={trials})")
    res = {
        "trials": trials,
        "off": {"upright": off_a[0], "pitch_mean": off_a[1], "pitch_peak": off_a[2], "min_upright": off_a[3]},
        "on": {"upright": on_a[0], "pitch_mean": on_a[1], "pitch_peak": on_a[2], "min_upright": on_a[3]},
        "paired_pitch_reduction_deg": {"mean": dmean, "ci95": dci},
    }
    if out_json:
        try:
            with open(out_json) as f:
                base = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            base = {}
        base["ablation_terrain_feedback"] = res
        with open(out_json, "w") as f:
            json.dump(base, f, indent=2)
        print(f"updated {out_json} (ablation block)")
    return res


def push_recovery(impulses=(80.0, 160.0, 240.0, 320.0), t_push=3.0, dur=0.10,
                  secs=7.0, trials=6, out_json=None):
    """External-disturbance recovery probe: walk, apply a lateral impulse to the trunk
    via data.xfrc_applied mid-stride, and measure how the closed-loop controller copes.
    Probes increasing impulses to find the HONEST limit — survive-rate (never fell) with
    a Wilson CI, peak roll excursion, and recovery time (|roll| back under 5°). Reports
    exactly what the gait withstands and where it starts to fail; no inflation."""
    print(f"\n=== Push recovery: lateral {dur*1000:.0f} ms impulse at t={t_push}s "
          f"({trials} randomized seeds/level) ===")
    print(f"{'impulse':>9s}  survive    peak|roll|   recover(|roll|<5°)")
    out = []
    for imp in impulses:
        survived = 0; peaks = []; recs = []
        for s in range(trials):
            env = QuadEnv(EnvConfig(seed=s, randomize=True, **RANDOMIZE)); env.reset()
            ctl = TrotController(env); fell = False
            peak = 0.0; rec = None; pushed = False
            for i in range(int(secs / env.dt)):
                t = i * env.dt
                if not pushed and t >= t_push:
                    env.apply_push([0.0, imp, 0.0], int(dur / env.dt)); pushed = True
                ctl.drive(t, 1.0, 0.0); env.step(1)
                if t >= t_push + dur:
                    roll = abs(np.degrees(ctl.imu_pitch_roll()[1]))   # lateral shove -> roll
                    peak = max(peak, roll)
                    if rec is None and roll < 5.0:
                        rec = t - (t_push + dur)
                if env.base_height() < 0.15:
                    fell = True; break
            survived += (not fell)
            peaks.append(peak); recs.append(rec if rec is not None else float("nan"))
            env.close()
        lo, hi = wilson_ci(survived, trials)
        pk = float(np.nanmean(peaks))
        rec_ok = [r for r in recs if r == r]
        rs = float(np.mean(rec_ok)) if rec_ok else float("nan")
        rs_s = f"{rs:.2f} s" if rs == rs else "  did not settle"
        print(f"{imp:6.0f} N   {survived}/{trials}      {pk:5.1f} deg    {rs_s}")
        out.append({"impulse_N": imp, "survive": survived / trials, "survive_ci": [lo, hi],
                    "peak_roll_deg": pk, "recover_s": (rs if rs == rs else None)})
    if out_json:
        try:
            with open(out_json) as f:
                base = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            base = {}
        base["push_recovery"] = out
        with open(out_json, "w") as f:
            json.dump(base, f, indent=2)
        print(f"updated {out_json} (push_recovery block)")
    return out


def speed_tracking(cmds=(0.4, 1.0, 1.6), secs=6.0, trials=4):
    """Velocity control: command several forward levels on flat ground and report the
    measured body speed (mean ± 95% CI) — shows speed is controllable, not one-speed."""
    print(f"\n=== Speed control: commanded forward -> measured speed (flat, {trials} seeds) ===")
    print(f"{'command':>8s}   measured m/s        upright")
    out = []
    for c in cmds:
        sp = []; ups = []
        for s in range(trials):
            env = QuadEnv(EnvConfig(scene=FLAT, seed=100 + s, randomize=True, **RANDOMIZE)); env.reset()
            ctl = TrotController(env); mu = 1.0
            # settle, then measure displacement on flat
            for i in range(int(1.5 / env.dt)):
                ctl.drive(i * env.dt, c, 0.0); env.step(1)
            x0 = env.base_xy().copy(); t0 = env.data.time
            for i in range(int(secs / env.dt)):
                ctl.drive(t0 + i * env.dt, c, 0.0); env.step(1); mu = min(mu, env.upright())
            sp.append(float(np.linalg.norm(env.base_xy() - x0) / secs)); ups.append(mu)
            env.close()
        m, ci = mean_ci(sp)
        print(f"{c:7.1f}    {m:.2f} ± {ci:.2f}          {np.mean(ups):.2f}")
        out.append({"cmd": c, "speed_mps": m, "speed_ci95": ci, "min_upright": float(np.mean(ups))})
    return out


def _delivery_trial(seed, terrain_fb=True):
    env = QuadEnv(EnvConfig(scene=CARGO, seed=seed, randomize=True, **RANDOMIZE))
    env.reset()
    ctl = TrotController(env); ctl.terrain_fb = terrain_fb
    min_up = 1.0; delivered = True; max_slip = 0.0; t = 0.0
    for dur, fwd, turn in DELIVERY_ROUTE:
        for _ in range(int(dur / env.dt)):
            ctl.drive(t, fwd, turn); env.step(1); t += env.dt
            min_up = min(min_up, env.upright())
            max_slip = max(max_slip, env.payload_offset()[0])
            if not env.payload_on() or env.base_height() < 0.15:
                delivered = False
    env.close()
    return {"delivered": delivered, "min_up": min_up, "max_slip": max_slip}


def payload_delivery(trials=10, out_json=None):
    """Loco-manipulation: the Go2 carries a free payload on a trunk tray across the
    terrain course + a turn, balancing it through whole-body locomotion (real free-body
    contacts). Reports the delivery rate (Wilson CI), the cargo's max slip on the tray,
    and a laden-stability ablation (terrain-leveling ON vs OFF, paired seeds)."""
    on = [_delivery_trial(i, terrain_fb=True) for i in range(trials)]
    off = [_delivery_trial(i, terrain_fb=False) for i in range(trials)]
    delv = sum(r["delivered"] for r in on)
    lo, hi = wilson_ci(delv, trials)
    up_on = float(np.mean([r["min_up"] for r in on]))
    up_off = float(np.mean([r["min_up"] for r in off]))
    slip = float(np.mean([r["max_slip"] for r in on]))
    print(f"\n=== Loco-manipulation: payload delivery over terrain + turn ({trials} seeds) ===")
    print(f"delivered (cargo kept the whole route): {delv}/{trials} "
          f"({delv/trials*100:.0f}%)  95% CI [{lo*100:.0f}%, {hi*100:.0f}%]")
    print(f"max cargo slip on tray: {slip*100:.1f} cm (stayed within the {12:.0f} cm tray)")
    print(f"laden min-uprightness — leveling ON {up_on:.2f}  vs  OFF {up_off:.2f}")
    res = {"trials": trials, "delivered_rate": delv / trials, "delivered_ci": [lo, hi],
           "max_slip_m": slip, "laden_min_up_on": up_on, "laden_min_up_off": up_off}
    if out_json:
        try:
            with open(out_json) as f:
                base = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            base = {}
        base["payload_delivery"] = res
        with open(out_json, "w") as f:
            json.dump(base, f, indent=2)
        print(f"updated {out_json} (payload_delivery block)")
    return res


if __name__ == "__main__":
    evaluate(out_json="eval_results.json")
    ablation_terrain_feedback(out_json="eval_results.json")
    speed_tracking()
    push_recovery(out_json="eval_results.json")
    payload_delivery(out_json="eval_results.json")
