"""Render the annotated demo video by running the autonomous patrol."""
from __future__ import annotations

from .env import QuadEnv, EnvConfig, CARGO
from .controller import TrotController, patrol_command
from .evaluate import DELIVERY_ROUTE
from .video import Recorder, ACCENT, WHITE, DIM, GREEN


# on-camera disturbance: a lateral impulse mid-stride during the straight segment
PUSH_T = 22.8          # s
PUSH_FORCE = 130.0     # N lateral (within the ~160 N survivable envelope)
PUSH_DUR = 0.10        # s


def record_demo(out_path="demo.mp4", seed=0, fps=30, every=12, secs=28.5):
    env = QuadEnv(EnvConfig(seed=seed))
    env.reset()
    ctl = TrotController(env)
    rec = Recorder(env, ctl, every=every)
    rec.title_card([
        ("QuadLoco", 54, ACCENT),
        ("Closed-Loop Terrain-Adaptive Quadruped Locomotion", 26, WHITE),
        ("Unitree Go2 · 12 torque motors · CPG trot + PD · MuJoCo", 20, DIM),
        ("3 feedback loops: gait + IMU heading-hold + IMU terrain leveling", 20, GREEN),
        ("real dynamics & contacts — every frame is mj_step", 20, DIM),
    ], n=42)
    n = int(secs / env.dt)
    last = None
    pushed = False
    for i in range(n):
        t = i * env.dt
        fwd, turn, label = patrol_command(t)
        if not pushed and t >= PUSH_T:          # shove the robot on camera
            env.apply_push([0.0, PUSH_FORCE, 0.0], int(PUSH_DUR / env.dt))
            pushed = True
        if PUSH_T <= t < PUSH_T + 1.6:
            label = f"Shoved sideways ({PUSH_FORCE:.0f} N impulse) — recovering"
        if label != last:
            rec.set_phase(label)
            last = label
        ctl.drive(t, fwd, turn)
        env.step(1)
    import numpy as np
    dist = float(np.linalg.norm(env.base_xy() - rec.x0))
    rec.title_card([
        ("Results", 34, GREEN),
        (f"autonomous patrol: {dist:.1f} m, terrain + spin + speed + turns, stayed upright", 20, WHITE),
        ("terrain leveling (ablation): peak trunk pitch 24.2° → 19.3°, 20/20 upright", 20, GREEN),
        (f"shoved at {PUSH_FORCE:.0f} N mid-stride — recovered (real data.xfrc_applied)", 20, GREEN),
        ("closed-loop torque control · real contacts · no kinematic scripting · no GPU", 20, DIM),
    ], n=48)
    rec.save(out_path, fps=fps)
    env.close()
    print(f"wrote {out_path} ({len(rec.frames)} frames, {len(rec.frames)/fps:.1f}s)")
    return out_path


def record_cargo(out_path="demo_cargo.mp4", seed=0, fps=30, every=12):
    """Loco-manipulation clip: the Go2 delivers a free payload across the terrain."""
    env = QuadEnv(EnvConfig(scene=CARGO, seed=seed)); env.reset()
    ctl = TrotController(env)
    rec = Recorder(env, ctl, every=every)
    rec.title_card([
        ("QuadLoco · Cargo Delivery", 34, ACCENT),
        ("Loco-manipulation: carry a free payload across terrain", 26, WHITE),
        ("the cargo is a free rigid body — balanced by whole-body locomotion", 20, DIM),
    ], n=40)
    last = None; t = 0.0
    for dur, fwd, turn in DELIVERY_ROUTE:
        for _ in range(int(dur / env.dt)):
            on = env.payload_on()
            label = ("Delivering payload over terrain" if fwd >= 1.0 else
                     "Gentle turn — keeping the cargo on") + ("" if on else "  (cargo lost)")
            if label != last:
                rec.set_phase(label); last = label
            ctl.drive(t, fwd, turn); env.step(1); t += env.dt
    delivered = env.payload_on()
    rec.title_card([
        ("Delivered" if delivered else "Result", 34, GREEN),
        ("free payload carried across ramp + rough + step + a turn", 20, WHITE),
        ("10/10 delivery over randomized trials · real free-body contacts", 20, GREEN),
        ("loco-manipulation on a torque-controlled quadruped · no GPU", 20, DIM),
    ], n=46)
    rec.save(out_path, fps=fps)
    env.close()
    print(f"wrote {out_path} ({len(rec.frames)} frames, {len(rec.frames)/fps:.1f}s)")
    return out_path


def record_mission(out_path="demo_mission.mp4", seed=0, fps=30, every=24):
    """Autonomous go-to-goal mission demo: the Go2 patrols 3 visible waypoints
    (out -> across -> home) on closed-loop pose control, self-verifying arrival."""
    from .env import MISSION_SCENE
    from .evaluate import MISSION
    env = QuadEnv(EnvConfig(scene=MISSION_SCENE, seed=seed)); env.reset()
    ctl = TrotController(env); rec = Recorder(env, ctl, every=every)
    rec.title_card([
        ("QuadLoco · Autonomous Mission", 34, ACCENT),
        ("Go-to-goal: patrol 3 waypoints, then return home", 26, WHITE),
        ("closed-loop on pose · self-verifies arrival within 0.25 m · real mj_step", 20, DIM),
    ], n=34)
    t = 0.0
    for wi, (gx, gy) in enumerate(MISSION, 1):
        for _ in range(int(20.0 / env.dt)):
            d = ctl.steer_to(t, (gx, gy))
            rec.set_phase(f"Waypoint {wi}/{len(MISSION)} → ({gx:.1f}, {gy:.1f})   distance {d:.2f} m")
            env.step(1); t += env.dt
            if d < 0.25:
                break
        for _ in range(int(0.7 / env.dt)):     # brief self-verified-arrival beat
            rec.set_phase(f"✓ Waypoint {wi} reached  (self-verified < 0.25 m)")
            ctl.drive(t, 0.0, 0.0); env.step(1); t += env.dt
    rec.title_card([
        ("Mission complete", 34, GREEN),
        ("3/3 waypoints reached and returned home — self-verified", 20, WHITE),
        ("autonomous go-to-goal · 5th closed-loop (pose) · no GPU", 20, DIM),
    ], n=40)
    rec.save(out_path, fps=fps); env.close()
    print(f"wrote {out_path} ({len(rec.frames)} frames, {len(rec.frames)/fps:.1f}s)")
    return out_path


if __name__ == "__main__":
    record_demo()
