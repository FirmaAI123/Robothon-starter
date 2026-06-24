"""Annotated demo recorder for PandaStack: a live HUD (mission phase, per-cube
grasp/place ticks, measured grasp force, tower height) over a tracking camera.
Frames are captured by hooking the env's per-step callback while the real mission runs."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACCENT = (232, 162, 76); GREEN = (90, 200, 120); DIM = (150, 156, 168)
WHITE = (238, 240, 245); PANEL = (18, 20, 26); RED = (224, 92, 72)
from .env import TABLE_TOP, CUBE_H


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


class Recorder:
    def __init__(self, env, ctl, every=18, cam="track_cam"):
        self.env = env; self.ctl = ctl; self.every = every; self.cam = cam
        self.frames: List[np.ndarray] = []; self._k = 0
        self.fonts = {s: _font(s) for s in (20, 26, 34, 54)}
        env.add_step_hook(self._on_step)

    def _on_step(self, env):
        self._k += 1
        if self._k % self.every == 0:
            self.frames.append(self._compose())

    def _compose(self):
        img = Image.fromarray(self.env.render(self.cam)).convert("RGB")
        W, H = img.size; d = ImageDraw.Draw(img, "RGBA")
        # top bar
        d.rectangle([0, 0, W, 56], fill=(*PANEL, 210))
        d.text((20, 12), "PandaStack", font=self.fonts[34], fill=ACCENT)
        d.text((210, 20), "Closed-Loop Block Stacking · Franka Panda · MuJoCo",
               font=self.fonts[20], fill=WHITE)
        d.text((W - 150, 18), f"t = {self.env.data.time:5.2f}s", font=self.fonts[20], fill=DIM)
        # bottom phase bar
        d.rectangle([0, H - 52, W, H], fill=(*PANEL, 210))
        ph = self.ctl.phase
        col = GREEN if ("complete" in ph or "Placing" in ph) else (RED if "correct" in ph or "miss" in ph.lower() else WHITE)
        d.text((20, H - 44), ph, font=self.fonts[26], fill=col)
        # left telemetry
        px, py = 18, 78
        d.rectangle([px, py, px + 250, py + 168], fill=(*PANEL, 170))
        d.text((px + 12, py + 10), "cubes placed", font=self.fonts[20], fill=WHITE)
        for i in range(3):
            cx = px + 30 + i * 56; ok = self.ctl.placed[i]
            d.ellipse([cx, py + 38, cx + 22, py + 60], fill=GREEN if ok else DIM)
            d.text((cx + 4, py + 40), str(i + 1), font=self.fonts[20], fill=(20, 20, 20))
        gf = self.env.grasp_force()
        d.text((px + 12, py + 76), f"grasp force {gf:5.1f} N", font=self.fonts[20],
               fill=GREEN if gf > 1 else DIM)
        # tower height (max cube z above table)
        th = max(self.env.cube_pos(i)[2] for i in range(3)) - TABLE_TOP
        d.text((px + 12, py + 104), f"tower height {th*100:4.1f} cm", font=self.fonts[20], fill=WHITE)
        d.text((px + 12, py + 132), "closed-loop IK · self-verify · recover",
               font=self.fonts[20], fill=ACCENT)
        return np.asarray(img)

    def title_card(self, lines: List[Tuple[str, int, tuple]], n=40):
        W, H = self.env.cfg.render_w, self.env.cfg.render_h
        img = Image.new("RGB", (W, H), (10, 11, 15)); d = ImageDraw.Draw(img)
        y = H // 2 - 22 * len(lines)
        for text, size, color in lines:
            w = d.textlength(text, font=self.fonts[size])
            d.text(((W - w) / 2, y), text, font=self.fonts[size], fill=color); y += size + 18
        self.frames.extend([np.asarray(img)] * n)

    def save(self, path, fps=30):
        import imageio.v2 as imageio
        imageio.mimsave(path, self.frames, fps=fps, quality=8, macro_block_size=8)
        return path
