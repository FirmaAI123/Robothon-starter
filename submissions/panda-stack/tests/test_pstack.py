"""PandaStack tests: model integrity, real physics (mj_step), closed-loop grasp,
autonomous tower build, and determinism."""
import numpy as np
import pytest

from pstack.env import PandaEnv, EnvConfig, TABLE_TOP
from pstack.controller import StackController, OPEN


@pytest.fixture(scope="module")
def env():
    e = PandaEnv(EnvConfig(seed=0))
    yield e
    e.close()


def test_model_integrity(env):
    assert env.model.nu == 8                 # 7 arm + 1 gripper actuator
    assert len(env.cube_bid) == 3            # three cubes


def test_is_real_physics():
    """A cube held in the air falls under gravity when stepped — dynamics, not kinematics."""
    e = PandaEnv(EnvConfig(seed=0))
    adr = e.cube_qadr[0]
    e.data.qpos[adr:adr + 3] = [0.7, 0.32, 0.75]    # drop a cube in mid-air, clear of the arm
    import mujoco; mujoco.mj_forward(e.model, e.data)
    z0 = e.cube_pos(0)[2]
    for _ in range(200):
        e.step(1)
    assert e.cube_pos(0)[2] < z0 - 0.15, "cube did not fall — not real physics"
    e.close()


def test_closed_loop_grasp():
    """Closed-loop IK grasps and lifts a cube; open-loop is the weaker baseline."""
    e = PandaEnv(EnvConfig(seed=0)); c = StackController(e)
    assert c.grasp(0, closed_loop=True), "closed-loop grasp failed"
    e.close()


def test_stacks_tower():
    """The autonomous mission builds a stable 3-cube tower (seed 0)."""
    e = PandaEnv(EnvConfig(seed=0, randomize=True)); c = StackController(e)
    assert c.stack(), "stacking mission did not complete"
    assert c.tower_stable(), "tower did not stay standing after release"
    e.close()


def test_self_verification_flags_failure():
    """placed_ok / holding actually discriminate (a far cube is not 'placed')."""
    e = PandaEnv(EnvConfig(seed=1)); c = StackController(e)
    assert not c.placed_ok(0, TABLE_TOP + 0.025), "verifier passed an un-placed cube"
    e.close()


def test_determinism():
    def run():
        e = PandaEnv(EnvConfig(seed=0)); c = StackController(e)
        for _ in range(300):
            c.ik_servo(e.grasp_point() + np.array([0.05, 0, 0]), OPEN, 1)
        x = e.grasp_point().copy(); e.close(); return x
    assert np.allclose(run(), run(), atol=1e-9), "non-deterministic"
