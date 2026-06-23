"""QuadLoco tests: model integrity, that control is real physics (mj_step moves
the robot), forward locomotion, and determinism."""
import numpy as np
import pytest

from quadloco.env import QuadEnv, EnvConfig
from quadloco.controller import TrotController, patrol_command


@pytest.fixture(scope="module")
def env():
    e = QuadEnv(EnvConfig(seed=0))
    yield e
    e.close()


def test_model_integrity(env):
    assert env.model.nu == 12          # 12 torque motors
    import mujoco
    names = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
             for i in range(env.model.nsensor)}
    for s in ("imu_quat", "imu_gyro", "imu_acc", "base_pos"):
        assert s in names


def test_obs_shapes(env):
    o = env.reset()
    assert o["joint_pos"].shape == (12,)
    assert o["foot_contacts"].shape == (4,)
    assert o["gyro"].shape == (3,)


def test_walks_forward():
    env = QuadEnv(EnvConfig(seed=0)); env.reset()
    ctl = TrotController(env)
    x0 = env.base_xy()[0]
    min_up = 1.0
    for i in range(int(8.0 / env.dt)):
        ctl.drive(i * env.dt, 1.0, 0.0)   # production path: gait + heading-hold + terrain feedback
        env.step(1)
        min_up = min(min_up, env.upright())
    assert env.base_xy()[0] - x0 > 1.0, "robot did not walk forward"
    assert min_up > 0.5, "robot fell over"
    env.close()


def test_is_real_physics():
    """Torques must drive motion: zero command -> the robot does NOT hold the
    scripted pose (it sags under gravity), proving this is dynamics, not kinematics."""
    env = QuadEnv(EnvConfig(seed=0)); env.reset()
    h0 = env.base_height()
    for _ in range(300):
        env.set_motor_torques(np.zeros(12))
        env.step(1)
    assert env.base_height() < h0 - 0.02, "no gravity response — not real physics"
    env.close()


def test_heading_hold_reduces_drift():
    """Closed-loop IMU heading-hold keeps the robot straighter than open-loop."""
    def drift(use_drive):
        env = QuadEnv(EnvConfig(seed=7, randomize=True, mass_jitter=0.12, yaw_jitter=0.05))
        env.reset(); ctl = TrotController(env); y0 = env.base_xy()[1]
        for i in range(int(10.0 / env.dt)):
            (ctl.drive if use_drive else ctl.act)(i * env.dt, 1.0, 0.0)
            env.step(1)
        d = abs(env.base_xy()[1] - y0); env.close(); return d
    assert drift(True) < drift(False) - 0.3, "heading-hold did not reduce drift"


def test_terrain_feedback_levels_trunk():
    """Closed-loop terrain feedback (IMU active leveling) keeps the trunk flatter
    over the terrain course than the same gait with the loop ablated."""
    from quadloco.evaluate import walk_trial
    on = np.mean([walk_trial(s, 8.0, terrain_fb=True)["pitch"] for s in range(4)])
    off = np.mean([walk_trial(s, 8.0, terrain_fb=False)["pitch"] for s in range(4)])
    assert on < off, f"terrain feedback should reduce trunk pitch ({on:.1f} !< {off:.1f})"


def test_recovers_from_push():
    """The closed-loop gait survives a bounded lateral impulse (real xfrc_applied)
    and recovers uprightness — disturbance rejection, not luck."""
    env = QuadEnv(EnvConfig(seed=1)); env.reset(); ctl = TrotController(env)
    for i in range(int(3.0 / env.dt)):
        ctl.drive(i * env.dt, 1.0, 0.0); env.step(1)
    env.apply_push([0.0, 120.0, 0.0], int(0.10 / env.dt))   # 120 N lateral, 0.1 s
    for i in range(int(3.0 / env.dt)):
        ctl.drive(3.0 + i * env.dt, 1.0, 0.0); env.step(1)
    assert env.base_height() > 0.15, "fell after a 120 N lateral impulse"
    assert env.upright() > 0.7, "did not recover uprightness after the push"
    env.close()


def test_speed_control_monotone():
    """Commanded forward maps monotonically to measured speed (controllable, upright)."""
    from quadloco.evaluate import speed_tracking
    res = speed_tracking(cmds=(0.4, 1.6), trials=2)
    assert res[1]["speed_mps"] > res[0]["speed_mps"] + 0.05, "faster command not faster"
    assert all(r["min_upright"] > 0.5 for r in res), "fell during speed sweep"


def test_payload_delivery():
    """Loco-manipulation: the Go2 carries a free payload across the terrain + a turn
    without dropping it or falling — real free-body cargo, balanced by locomotion."""
    from quadloco.evaluate import _delivery_trial
    r = _delivery_trial(0, terrain_fb=True)
    assert r["delivered"], "payload was dropped during delivery"
    assert r["min_up"] > 0.5, "robot fell while carrying cargo"


def test_determinism():
    def run():
        e = QuadEnv(EnvConfig(seed=0)); e.reset(); c = TrotController(e)
        for i in range(500):
            c.act(i * e.dt, 1.0); e.step(1)
        x = e.base_xy()[0]; e.close(); return x
    assert abs(run() - run()) < 1e-9
