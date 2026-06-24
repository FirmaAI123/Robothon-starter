#!/usr/bin/env python3
"""QuadLoco entrypoint.

    python run.py walk            # headless autonomous patrol, prints metrics
    python run.py record [out]    # render the annotated demo video
    python run.py eval [N]        # full eval: robustness + ablation + speed + push (CIs)
    python run.py figures         # regenerate README figures from eval data
    python run.py info            # model / sensor summary
"""
import sys

PATROL_SECS = 28.5


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    mode = argv[0] if argv else "walk"

    if mode == "walk":
        import numpy as np
        from quadloco.env import QuadEnv, EnvConfig
        from quadloco.controller import TrotController, patrol_command
        env = QuadEnv(EnvConfig(seed=0)); env.reset()
        ctl = TrotController(env); x0 = env.base_xy().copy(); min_up = 1.0
        for i in range(int(PATROL_SECS / env.dt)):
            t = i * env.dt; fwd, turn, _ = patrol_command(t)
            ctl.drive(t, fwd, turn); env.step(1); min_up = min(min_up, env.upright())
        dist = float(np.linalg.norm(env.base_xy() - x0))
        print(f"patrol {PATROL_SECS:.0f}s: traversed {dist:.2f} m, final height {env.base_height():.2f} m, "
              f"min uprightness {min_up:.2f}, upright={min_up > 0.5}")
        return 0 if min_up > 0.5 else 1

    if mode == "record":
        from quadloco.record import record_demo
        record_demo(argv[1] if len(argv) > 1 else "demo.mp4")
        return 0

    if mode == "record-cargo":
        from quadloco.record import record_cargo
        record_cargo(argv[1] if len(argv) > 1 else "demo_cargo.mp4")
        return 0

    if mode == "eval":
        from quadloco.evaluate import (evaluate, ablation_terrain_feedback, speed_tracking,
                                       speed_setpoint_tracking, push_recovery,
                                       payload_delivery, noise_robustness)
        n = int(argv[1]) if len(argv) > 1 else 20
        evaluate(trials=n, out_json="eval_results.json")
        ablation_terrain_feedback(trials=n, out_json="eval_results.json")
        speed_tracking()
        speed_setpoint_tracking(out_json="eval_results.json")
        push_recovery(out_json="eval_results.json")
        payload_delivery(out_json="eval_results.json")
        noise_robustness(out_json="eval_results.json")
        return 0

    if mode == "figures":
        from quadloco.figures import make_all
        make_all()
        return 0

    if mode == "info":
        from quadloco.env import QuadEnv
        import mujoco
        env = QuadEnv()
        m = env.model
        print(f"model: nq={m.nq} nu={m.nu} nbody={m.nbody} nsensor={m.nsensor} "
              f"timestep={m.opt.timestep}")
        print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                             for i in range(m.nu)])
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
