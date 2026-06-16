"""Interactive keyboard teleoperation via the MuJoCo passive viewer.

Controls (held keys repeat):
    W / S   wrist +x / -x          I / K   pitch +/-
    A / D   wrist +y / -y          J / L   yaw +/-
    R / F   wrist +z / -z (up/dn)  U / O   roll +/-
    SPACE   close grasp            G       open grasp
    ESC     quit

This mode needs a display and so is excluded from headless CI; the autonomous,
record, and collect modes are fully headless.
"""
from __future__ import annotations

import numpy as np

from .env import DexSuiteEnv
from .skills import grasp_pose


def teleop():
    import mujoco
    import mujoco.viewer

    env = DexSuiteEnv()
    env.reset()
    tgt = env.get_wrist_target()
    state = {"closure": 0.0}
    step = np.array([0.01, 0.01, 0.01, 0.04, 0.04, 0.04])

    def on_key(keycode):
        c = chr(keycode) if 0 <= keycode < 0x110000 else ""
        c = c.lower()
        m = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1),
             "r": (2, +1), "f": (2, -1), "l": (3, +1), "j": (3, -1),
             "i": (4, +1), "k": (4, -1), "u": (5, +1), "o": (5, -1)}
        if c in m:
            idx, sgn = m[c]
            tgt[idx] += sgn * step[idx]
        elif keycode == 32:  # space -> close
            state["closure"] = min(1.0, state["closure"] + 0.2)
        elif c == "g":
            state["closure"] = max(0.0, state["closure"] - 0.2)

    print(__doc__)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        while v.is_running():
            env.set_wrist_target(tgt)
            env.set_finger_targets(grasp_pose(state["closure"]))
            env.step(1)
            v.sync()


if __name__ == "__main__":
    teleop()
