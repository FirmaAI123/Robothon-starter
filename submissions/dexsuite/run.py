#!/usr/bin/env python3
"""DexSuite entrypoint.

Usage:
    python run.py autonomous          # run the task headless, print stage results
    python run.py record [out.mp4]    # render the annotated demo video
    python run.py collect [dir] [N]   # generate a randomized dataset of N episodes
    python run.py teleop              # interactive keyboard teleoperation (needs display)
    python run.py info                # print model / sensor summary
"""
import sys


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    mode = argv[0] if argv else "autonomous"

    if mode == "autonomous":
        from dexsuite.env import DexSuiteEnv
        from dexsuite.planner import run_task
        from dexsuite import config as C
        cfg = C.load()
        res = run_task(DexSuiteEnv(C.env_config(cfg)), C.skill_params(cfg))
        return 0 if res.success_rate >= 0.99 else 1

    if mode == "record":
        from dexsuite.record import record_demo
        from dexsuite import config as C
        rc = C.load().get("record", {})
        out = argv[1] if len(argv) > 1 else rc.get("out", "demo.mp4")
        record_demo(out, fps=rc.get("fps", 30), every=rc.get("capture_every", 7))
        return 0

    if mode == "collect":
        from dexsuite.collect import collect
        from dexsuite import config as C
        cc = C.load().get("collect", {})
        out = argv[1] if len(argv) > 1 else cc.get("out_dir", "dataset")
        n = int(argv[2]) if len(argv) > 2 else cc.get("episodes", 5)
        collect(out, episodes=n, stride=cc.get("stride", 10),
                with_images=cc.get("with_images", True),
                pos_jitter=cc.get("pos_jitter", 0.004),
                mass_jitter=cc.get("mass_jitter", 0.15))
        return 0

    if mode == "eval":
        from dexsuite.evaluate import evaluate, gait_benchmark
        n = int(argv[1]) if len(argv) > 1 else 20
        evaluate(trials=n, out_json="eval_results.json")
        gait_benchmark(trials=n)
        return 0

    if mode == "teleop":
        from dexsuite.teleop import teleop
        teleop()
        return 0

    if mode == "info":
        from dexsuite.env import DexSuiteEnv
        import mujoco
        env = DexSuiteEnv()
        m = env.model
        print(f"model: nq={m.nq} nu={m.nu} nbody={m.nbody} "
              f"nsensor={m.nsensor} timestep={m.opt.timestep}")
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
        print("sensors:", ", ".join(names))
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
