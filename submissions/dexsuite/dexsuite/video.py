"""Annotated demo-video recorder.

A :class:`Recorder` registers a per-step hook on the environment, periodically
renders the scene camera, composites a live sensor HUD (phase caption, fingertip
touch bars, wrist force, button state, success ticks) and a wrist-camera inset,
and writes an MP4. The video is produced *by running the submitted code* — a
hard requirement of the hackathon rules.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACCENT = (232, 122, 90)
GREEN = (90, 200, 120)
DIM = (150, 156, 168)
WHITE = (238, 240, 245)
PANEL = (18, 20, 26)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


class Recorder:
    def __init__(self, env, every: int = 18, main_cam: str = "scene_cam",
                 inset_cam: str = "wrist_cam"):
        self.env = env
        self.every = every
        self.main_cam = main_cam
        self.inset_cam = inset_cam
        self.frames: List[np.ndarray] = []
        self._k = 0
        self.phase = "initializing"
        self.ticks: List[str] = []
        self.fonts = {s: _font(s) for s in (20, 26, 34, 54)}
        env.add_step_hook(self._on_step)

    # ---- planner-facing API ----------------------------------------------
    def set_phase(self, text: str):
        self.phase = text

    def set_camera(self, name: str):
        self.main_cam = name

    def tick(self, text: str):
        self.ticks.append(text)

    # ---- capture ----------------------------------------------------------
    def _on_step(self, env):
        self._k += 1
        if self._k % self.every == 0:
            self.frames.append(self._compose())

    def _compose(self) -> np.ndarray:
        main = self.env.render(self.main_cam)
        img = Image.fromarray(main).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img, "RGBA")

        # title bar
        d.rectangle([0, 0, W, 56], fill=(*PANEL, 210))
        d.text((20, 12), "DexSuite", font=self.fonts[34], fill=ACCENT)
        d.text((168, 20), "Dexterous Manipulation · LEAP Hand · MuJoCo",
               font=self.fonts[20], fill=WHITE)
        obs = self.env.get_obs()
        d.text((W - 150, 18), f"t = {obs['time']:5.2f}s", font=self.fonts[20], fill=DIM)

        # phase caption
        d.rectangle([0, H - 52, W, H], fill=(*PANEL, 210))
        d.text((20, H - 44), self.phase, font=self.fonts[26], fill=WHITE)

        # left sensor panel
        px, py, pw = 18, 78, 232
        d.rectangle([px, py, px + pw, py + 250], fill=(*PANEL, 170))
        d.text((px + 12, py + 8), "FINGERTIP TOUCH", font=self.fonts[20], fill=DIM)
        touch = obs["touch"]
        for i, (lab, val) in enumerate(zip(["INDEX", "MIDDLE", "RING", "THUMB"], touch)):
            y = py + 40 + i * 30
            d.text((px + 12, y), lab, font=self.fonts[20], fill=WHITE)
            bw = int(np.clip(val / 8.0, 0, 1) * 96)
            d.rectangle([px + 110, y + 4, px + 110 + 96, y + 18], outline=DIM)
            d.rectangle([px + 110, y + 4, px + 110 + bw, y + 18],
                        fill=GREEN if val > 0.05 else DIM)
        fmag = float(np.linalg.norm(obs["wrist_force"]))
        d.text((px + 12, py + 168), f"WRIST FORCE  {fmag:5.1f} N",
               font=self.fonts[20], fill=WHITE)
        btn = obs["button"] * 1000
        d.text((px + 12, py + 196), f"BUTTON       {abs(btn):4.1f} mm",
               font=self.fonts[20], fill=GREEN if abs(btn) > 8 else WHITE)

        # success ticks
        for i, t in enumerate(self.ticks[-6:]):
            d.text((px + 12, py + 232 + i * 0), "", font=self.fonts[20], fill=GREEN)
        ty = py + 262
        for t in self.ticks[-7:]:
            d.text((px, ty), f"✓ {t}", font=self.fonts[20], fill=GREEN)
            ty += 26

        # wrist-cam inset
        try:
            inset = self.env.render(self.inset_cam)
            ins = Image.fromarray(inset).resize((256, 144))
            img.paste(ins, (W - 268, H - 210))
            d.rectangle([W - 268, H - 210, W - 12, H - 66], outline=ACCENT, width=2)
            d.text((W - 262, H - 208), "wrist cam", font=self.fonts[20], fill=WHITE)
        except Exception:
            pass
        return np.asarray(img)

    # ---- output -----------------------------------------------------------
    def title_card(self, lines: List[Tuple[str, int, tuple]], n: int = 36):
        W, H = self.env.cfg.render_w, self.env.cfg.render_h
        img = Image.new("RGB", (W, H), (10, 11, 15))
        d = ImageDraw.Draw(img)
        y = H // 2 - 22 * len(lines)
        for text, size, color in lines:
            w = d.textlength(text, font=self.fonts[size])
            d.text(((W - w) / 2, y), text, font=self.fonts[size], fill=color)
            y += size + 18
        frame = np.asarray(img)
        self.frames.extend([frame] * n)

    def save(self, path: str, fps: int = 30):
        import imageio.v2 as imageio
        imageio.mimsave(path, self.frames, fps=fps, quality=8, macro_block_size=8)
        return path
