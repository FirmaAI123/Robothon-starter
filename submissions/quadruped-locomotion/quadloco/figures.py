"""Generate the README figures from real eval data (matplotlib, headless Agg).

matplotlib is imported lazily here only — the core walk/eval/record paths do not
depend on it. Run via ``python run.py figures``."""
from __future__ import annotations

import json
import os

import numpy as np

from .env import QuadEnv, EnvConfig
from .controller import TrotController, patrol_command
from .evaluate import RANDOMIZE

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
ACCENT = "#e87a5a"; GREEN = "#5ac878"; DIMC = "#9aa0ac"; BG = "#0f1115"


def _pitch_series(seed, terrain_fb, secs=10.0):
    env = QuadEnv(EnvConfig(seed=seed, randomize=True, **RANDOMIZE)); env.reset()
    ctl = TrotController(env); ctl.terrain_fb = terrain_fb
    ts, ps = [], []
    for i in range(int(secs / env.dt)):
        t = i * env.dt
        ctl.drive(t, 1.0, 0.0); env.step(1)
        ts.append(t); ps.append(abs(np.degrees(ctl.imu_pitch_roll()[0])))
    env.close()
    return np.array(ts), np.array(ps)


def make_ablation_plot(path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    js = json.load(open(os.path.join(os.path.dirname(FIGDIR), "eval_results.json")))
    ab = js["ablation_terrain_feedback"]
    os.makedirs(FIGDIR, exist_ok=True)
    path = path or os.path.join(FIGDIR, "ablation.png")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=BG)
    # left: mean & peak trunk pitch, OFF vs ON
    labels = ["mean |pitch|", "peak |pitch|"]
    off = [ab["off"]["pitch_mean"], ab["off"]["pitch_peak"]]
    on = [ab["on"]["pitch_mean"], ab["on"]["pitch_peak"]]
    x = np.arange(len(labels)); w = 0.36
    a0.bar(x - w / 2, off, w, label="feedback OFF", color=DIMC)
    a0.bar(x + w / 2, on, w, label="feedback ON", color=GREEN)
    for xi, (o, n) in enumerate(zip(off, on)):
        a0.text(xi - w / 2, o + 0.4, f"{o:.1f}°", ha="center", color="white", fontsize=9)
        a0.text(xi + w / 2, n + 0.4, f"{n:.1f}°", ha="center", color="white", fontsize=9)
    a0.set_xticks(x); a0.set_xticklabels(labels)
    a0.set_ylabel("trunk pitch (deg)")
    a0.set_title(f"Terrain-leveling ablation (paired, n={ab['trials']})", color="white", fontsize=11)
    a0.legend(facecolor=BG, edgecolor=DIMC, labelcolor="white", fontsize=9)

    # right: pitch vs time for one paired seed, OFF vs ON (smoothed for readability)
    def smooth(v, k=40):
        return np.convolve(v, np.ones(k) / k, mode="same")
    ts, p_off = _pitch_series(3, False, secs=6.0)   # terrain-crossing window
    _, p_on = _pitch_series(3, True, secs=6.0)
    a1.plot(ts, smooth(p_off), color=DIMC, lw=1.8, label="feedback OFF")
    a1.plot(ts, smooth(p_on), color=GREEN, lw=1.8, label="feedback ON")
    a1.set_xlabel("time (s)"); a1.set_ylabel("|trunk pitch| (deg, smoothed)")
    a1.set_title("Trunk pitch crossing the terrain (seed 3)", color="white", fontsize=11)
    a1.legend(facecolor=BG, edgecolor=DIMC, labelcolor="white", fontsize=9)

    for ax in (a0, a1):
        ax.set_facecolor(BG); ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color(DIMC)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def make_hero(path=None):
    """Grab one annotated HUD frame mid-terrain via the demo Recorder."""
    from .video import Recorder
    os.makedirs(FIGDIR, exist_ok=True)
    path = path or os.path.join(FIGDIR, "hero.png")
    env = QuadEnv(EnvConfig(seed=0)); env.reset()
    ctl = TrotController(env)
    rec = Recorder(env, ctl, every=10_000)     # don't auto-collect frames
    for i in range(int(2.8 / env.dt)):          # ~2.8 s in: climbing the ramp
        t = i * env.dt; fwd, turn, lab = patrol_command(t)
        rec.set_phase(lab); ctl.drive(t, fwd, turn); env.step(1)
    from PIL import Image
    Image.fromarray(rec._compose()).save(path)
    env.close()
    print(f"wrote {path}")
    return path


def make_cargo_hero(path=None):
    """Grab a frame of the Go2 carrying the payload across the terrain."""
    from .env import CARGO
    from .video import Recorder
    from .evaluate import DELIVERY_ROUTE
    os.makedirs(FIGDIR, exist_ok=True)
    path = path or os.path.join(FIGDIR, "cargo.png")
    env = QuadEnv(EnvConfig(scene=CARGO, seed=0)); env.reset()
    ctl = TrotController(env)
    rec = Recorder(env, ctl, every=10_000)
    rec.set_phase("Delivering payload over terrain")
    t = 0.0
    for _ in range(int(5.5 / env.dt)):          # ~5.5 s in: carrying over the rough patch
        ctl.drive(t, 1.0, 0.0); env.step(1); t += env.dt
    from PIL import Image
    Image.fromarray(rec._compose()).save(path)
    env.close()
    print(f"wrote {path}")
    return path


def make_all():
    make_hero()
    make_ablation_plot()
    make_cargo_hero()


if __name__ == "__main__":
    make_all()
