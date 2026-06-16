"""DexSuite test suite.

Covers model integrity, observation shapes, deterministic reproducibility, and a
full end-to-end run of the autonomous task (the headline correctness guarantee).
Rendering tests are skipped automatically when no GL backend is available.
"""
import numpy as np
import pytest

from dexsuite.env import DexSuiteEnv, EnvConfig
from dexsuite.planner import run_task
from dexsuite.skills import Hand, grasp_pose, OPEN_POSE


@pytest.fixture(scope="module")
def env():
    e = DexSuiteEnv(EnvConfig(seed=0))
    yield e
    e.close()


def test_model_integrity(env):
    m = env.model
    assert m.nu == 22                      # 6 wrist + 16 finger actuators
    assert m.nq == 37                       # 6 wrist + 16 finger + 2*freejoint(7) + button
    # all touch + force + imu sensors present
    import mujoco
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)}
    for s in ("if_touch", "mf_touch", "rf_touch", "th_touch",
              "wrist_force", "wrist_torque", "palm_acc", "palm_gyro", "btn_pos"):
        assert s in names


def test_obs_shapes(env):
    obs = env.reset()
    assert obs["finger_qpos"].shape == (16,)
    assert obs["wrist_qpos"].shape == (6,)
    assert obs["touch"].shape == (4,)
    assert obs["wrist_force"].shape == (3,)
    assert obs["cube_pose"].shape == (7,)


def test_grasp_pose_bounds():
    for c in (0.0, 0.5, 1.0):
        p = grasp_pose(c)
        assert set(p) == set(OPEN_POSE)
        assert all(np.isfinite(list(p.values())))


def test_determinism():
    r1 = run_task(DexSuiteEnv(EnvConfig(seed=0)), verbose=False)
    r2 = run_task(DexSuiteEnv(EnvConfig(seed=0)), verbose=False)
    assert [s.ok for s in r1.stages] == [s.ok for s in r2.stages]


def test_single_grasp_lifts(env):
    env.reset()
    env.step(120)
    hand = Hand(env)
    xy = env.object_pose("cube")[:2]
    hand.pick_at(xy)
    assert env.object_pose("cube")[2] > 0.6   # cube cleared its pedestal


def test_full_task_succeeds():
    res = run_task(DexSuiteEnv(EnvConfig(seed=0)), verbose=False)
    assert res.success_rate >= 0.99, [(s.name, s.ok, s.detail) for s in res.stages]


def test_inhand_finger_gaiting():
    """Finger-gaiting rolls the grasped ball with the WRIST HELD FIXED."""
    env = DexSuiteEnv(EnvConfig(seed=0))
    env.reset(); env.step(120)
    hand = Hand(env)
    hand.pick_at(env.object_pose("ball")[:2])
    wrist_before = env.get_wrist_target().copy()
    deg, held = hand.inhand_spin("ball", cycles=6, amp=0.22, closure=0.90)
    assert held, "object dropped during finger-gaiting"
    assert deg > 25.0, f"insufficient in-hand rotation: {deg:.0f} deg"
    # the wrist target must not have moved: the fingers did the work
    assert np.allclose(env.get_wrist_target(), wrist_before, atol=1e-6)


def test_render_smoke(env):
    try:
        img = env.render("scene_cam")
    except Exception as exc:           # no GL backend in this environment
        pytest.skip(f"rendering unavailable: {exc}")
    assert img.shape == (env.cfg.render_h, env.cfg.render_w, 3)
    assert img.mean() > 1.0            # not an all-black frame
