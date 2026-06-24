#!/usr/bin/env python3
"""PandaStack entrypoint.

    python run.py stack            # run one autonomous stacking mission, print result
    python run.py eval [N]         # quantitative eval (success + CIs, recovery & IK ablations)
    python run.py record [out]     # render the annotated stacking demo video
    python run.py info             # model summary
"""
import sys


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    mode = argv[0] if argv else "stack"

    if mode == "stack":
        from pstack.env import PandaEnv, EnvConfig
        from pstack.controller import StackController
        env = PandaEnv(EnvConfig(seed=0, randomize=True)); ctl = StackController(env)
        built = ctl.stack(); stable = ctl.tower_stable() if built else False
        hs = [round(float(env.cube_pos(i)[2]), 3) for i in range(3)]
        print(f"stack mission: built={built}, stable_tower={stable}, cube heights={hs}, log={[l[2] for l in ctl.log]}")
        return 0 if stable else 1

    if mode == "eval":
        from pstack.evaluate import stack_success, recovery_ablation, perception_noise, ik_ablation
        J = "eval_results.json"   # writes all metrics; deterministic anchors reproduce exactly,
        #                           noise-driven rates vary slightly by MuJoCo build (see README)
        stack_success(out_json=J)
        recovery_ablation(out_json=J)
        perception_noise(out_json=J)
        ik_ablation(out_json=J)
        return 0

    if mode == "record":
        from pstack.record import record_demo
        record_demo(argv[1] if len(argv) > 1 else "demo.mp4")
        return 0

    if mode == "info":
        from pstack.env import PandaEnv
        import mujoco
        env = PandaEnv(); m = env.model
        print(f"model: nq={m.nq} nu={m.nu} nbody={m.nbody} timestep={m.opt.timestep}")
        print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
