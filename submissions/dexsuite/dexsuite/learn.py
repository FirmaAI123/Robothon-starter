"""Closing the loop: learn a grasping policy from the suite's own demonstrations.

We (1) generate randomized scripted reach-grasp-lift demonstrations, (2) train a
small behavioural-cloning MLP — implemented from scratch in NumPy so the learning
loop needs *no extra dependencies* — and (3) deploy it closed-loop on unseen
randomized object positions, reporting grasp success. Data → train → deploy, all
inside the submission.

Observation (13): object XY relative to palm (2), wrist joint pos (6),
fingertip touch (4), task phase (1).
Action (7): wrist target (6) + grasp closure (1).
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from .env import DexSuiteEnv, EnvConfig
from .skills import Hand, grasp_pose, POCKET, SkillParams

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "bc_policy.npz")
HORIZON = 90          # control steps per reach-grasp episode (decimated)
SUBSTEPS = 18         # sim steps per control step


def _obs(env, phase) -> np.ndarray:
    o = env.get_obs()
    palm_xy = o["palm_pos"][:2]
    rel = o["ball_pose"][:2] - palm_xy
    return np.concatenate([rel, o["wrist_qpos"], o["touch"], [phase]]).astype(np.float32)


def _scripted_action(env, hand, target_xy, k) -> Tuple[np.ndarray, float]:
    """Open-loop reach-grasp schedule used to *generate* demonstrations."""
    wx, wy = target_xy[0] - POCKET[0], target_xy[1] - POCKET[1]
    if k < 25:                     # hover above
        wz, c = hand.p.approach_z, 0.0
    elif k < 40:                   # descend to pocket
        wz, c = hand.p.grasp_z, 0.0
    elif k < 60:                   # close
        wz, c = hand.p.grasp_z, (k - 40) / 20.0
    else:                          # lift
        wz, c = hand.p.lift_z, 1.0
    return np.array([wx, wy, wz, 0, 0, 0], dtype=np.float32), float(c)


def collect_bc_data(n_demos=60, seed=0):
    X, Y = [], []
    for i in range(n_demos):
        env = DexSuiteEnv(EnvConfig(seed=seed + i, randomize=True,
                                    pos_jitter=0.012, mass_jitter=0.12))
        env.reset(); env.step(60)
        hand = Hand(env)
        tgt = env.object_pose("ball")[:2].copy()
        for k in range(HORIZON):
            X.append(_obs(env, k / HORIZON))
            a_w, c = _scripted_action(env, hand, tgt, k)
            Y.append(np.concatenate([a_w, [c]]).astype(np.float32))
            env.set_wrist_target(a_w); env.set_finger_targets(grasp_pose(c))
            env.step(SUBSTEPS)
        env.close()
    return np.asarray(X), np.asarray(Y)


class MLP:
    """2-hidden-layer ReLU MLP with Adam, NumPy only."""
    def __init__(self, di, dh, do, seed=0):
        r = np.random.default_rng(seed)
        self.p = {
            "W1": r.normal(0, np.sqrt(2/di), (di, dh)), "b1": np.zeros(dh),
            "W2": r.normal(0, np.sqrt(2/dh), (dh, dh)), "b2": np.zeros(dh),
            "W3": r.normal(0, np.sqrt(2/dh), (dh, do)), "b3": np.zeros(do),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x, cache=False):
        z1 = x @ self.p["W1"] + self.p["b1"]; a1 = np.maximum(0, z1)
        z2 = a1 @ self.p["W2"] + self.p["b2"]; a2 = np.maximum(0, z2)
        y = a2 @ self.p["W3"] + self.p["b3"]
        if cache:
            self._c = (x, z1, a1, z2, a2)
        return y

    def step(self, x, ytrue, lr=2e-3):
        y = self.forward(x, cache=True)
        x_, z1, a1, z2, a2 = self._c
        n = x.shape[0]
        g = {}
        dy = 2 * (y - ytrue) / n
        g["W3"] = a2.T @ dy; g["b3"] = dy.sum(0)
        da2 = dy @ self.p["W3"].T; dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.p["W2"].T; dz1 = da1 * (z1 > 0)
        g["W1"] = x_.T @ dz1; g["b1"] = dz1.sum(0)
        self.t += 1
        for k in self.p:                 # Adam
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean((y - ytrue) ** 2))

    def save(self, path):
        np.savez(path, **{k: v for k, v in self.p.items()},
                 xmu=self.xmu, xsd=self.xsd, ymu=self.ymu, ysd=self.ysd)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(13, 128, 7)
        for k in net.p:
            net.p[k] = d[k]
        net.xmu, net.xsd, net.ymu, net.ysd = d["xmu"], d["xsd"], d["ymu"], d["ysd"]
        return net


def train(n_demos=60, epochs=300, seed=0):
    X, Y = collect_bc_data(n_demos, seed)
    net = MLP(13, 128, 7, seed=1)
    net.xmu, net.xsd = X.mean(0), X.std(0) + 1e-6
    net.ymu, net.ysd = Y.mean(0), Y.std(0) + 1e-6
    Xn = (X - net.xmu) / net.xsd
    Yn = (Y - net.ymu) / net.ysd
    rng = np.random.default_rng(0)
    bs = 256
    for ep in range(epochs):
        idx = rng.permutation(len(Xn))
        loss = 0.0
        for j in range(0, len(idx), bs):
            b = idx[j:j + bs]
            loss = net.step(Xn[b], Yn[b])
        if (ep + 1) % 50 == 0:
            print(f"epoch {ep+1:3d}  train_mse={loss:.4f}")
    net.save(WEIGHTS)
    print(f"saved policy -> {WEIGHTS}  ({len(X)} samples from {n_demos} demos)")
    return net


def rollout_on(env, net) -> bool:
    """Run the learned policy closed-loop on an existing env (lets a video recorder
    capture the rollout via the env's step hooks). Returns grasp success."""
    for k in range(HORIZON):
        x = _obs(env, k / HORIZON)
        a = (net.forward(((x - net.xmu) / net.xsd)[None])[0] * net.ysd + net.ymu)
        env.set_wrist_target(a[:6])
        env.set_finger_targets(grasp_pose(float(np.clip(a[6], 0, 1))))
        env.step(SUBSTEPS)
    return env.object_pose("ball")[2] > 0.6


def rollout(net, seed, randomize=True):
    env = DexSuiteEnv(EnvConfig(seed=seed, randomize=randomize,
                                pos_jitter=0.012, mass_jitter=0.12))
    env.reset(); env.step(60)
    ok = rollout_on(env, net)
    env.close()
    return ok


def evaluate_policy(trials=20, seed=500):
    net = MLP.load(WEIGHTS)
    ok = sum(rollout(net, seed + i) for i in range(trials))
    print(f"\n=== Learned BC grasp policy: closed-loop on {trials} UNSEEN randomized "
          f"object placements ===")
    print(f"grasp success: {ok}/{trials} ({ok/trials*100:.0f}%)")
    return ok / trials


if __name__ == "__main__":
    train()
    evaluate_policy()
