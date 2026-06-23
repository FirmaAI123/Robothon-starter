"""Annotated demo-video recorder for the quadruped, with a live locomotion HUD
(speed, distance, heading, per-foot stance/swing, trunk uprightness) and a body-
tracking camera. The video is produced by running the submitted code."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACCENT = (232, 122, 90)
GREEN = (90, 200, 120)
DIM = (150, 156, 168)
WHITE = (238, 240, 245)
PANEL = (18, 20, 26)
LEGS = ["FL", "FR", "RL", "RR"]


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
        self.env = env
        self.ctl = ctl
        self.every = every
        self.cam = cam
        self.frames: List[np.ndarray] = []
        self._k = 0
        self.phase = ""
        self.x0 = env.base_xy().copy()
        self._last_xy = env.base_xy().copy()
        self._last_t = 0.0
        self._speed = 0.0
        self._push_flash = 0
        self.fonts = {s: _font(s) for s in (20, 26, 34, 54)}
        env.add_step_hook(self._on_step)

    def set_phase(self, t):
        self.phase = t

    def _on_step(self, env):
        self._k += 1
        if self._k % self.every == 0:
            self.frames.append(self._compose())

    def _compose(self):
        img = Image.fromarray(self.env.render(self.cam)).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img, "RGBA")
        o = self.env.get_obs()
        # speed estimate
        xy = self.env.base_xy()
        dt = max(1e-3, o["time"] - self._last_t)
        self._speed = 0.8 * self._speed + 0.2 * (np.linalg.norm(xy - self._last_xy) / dt)
        self._last_xy, self._last_t = xy.copy(), o["time"]

        d.rectangle([0, 0, W, 56], fill=(*PANEL, 210))
        d.text((20, 12), "QuadLoco", font=self.fonts[34], fill=ACCENT)
        d.text((192, 20), "Physics-Based Quadruped Locomotion · Go2 · MuJoCo",
               font=self.fonts[20], fill=WHITE)
        d.text((W - 150, 18), f"t = {o['time']:5.2f}s", font=self.fonts[20], fill=DIM)
        if getattr(self.env, "has_cargo", False):
            ok = self.env.payload_on()
            d.text((W - 200, 64), f"CARGO: {'ON TRAY' if ok else 'LOST'}",
                   font=self.fonts[26], fill=GREEN if ok else (224, 92, 72))

        d.rectangle([0, H - 52, W, H], fill=(*PANEL, 210))
        d.text((20, H - 44), self.phase, font=self.fonts[26], fill=WHITE)

        # external-push banner (flashes while a disturbance is applied / recovering)
        if getattr(self.env, "pushing", False):
            self._push_flash = 45
        if self._push_flash > 0:
            self._push_flash -= 1
            msg = "EXTERNAL PUSH → RECOVERING"
            tw = d.textlength(msg, font=self.fonts[34])
            d.rectangle([(W - tw) / 2 - 18, 70, (W + tw) / 2 + 18, 116], fill=(*ACCENT, 230))
            d.text(((W - tw) / 2, 76), msg, font=self.fonts[34], fill=(20, 12, 10))

        # left telemetry panel
        px, py = 18, 78
        d.rectangle([px, py, px + 250, py + 214], fill=(*PANEL, 170))
        dist = float(np.linalg.norm(xy - self.x0))
        hd = np.degrees(o["heading"])
        pitch_deg = abs(np.degrees(self.ctl.imu_pitch_roll()[0]))
        fb_on = getattr(self.ctl, "terrain_fb", False)
        d.text((px + 12, py + 10), f"distance   {dist:5.2f} m", font=self.fonts[20], fill=WHITE)
        d.text((px + 12, py + 36), f"speed      {self._speed:5.2f} m/s", font=self.fonts[20], fill=WHITE)
        d.text((px + 12, py + 62), f"heading    {hd:+5.0f} deg", font=self.fonts[20], fill=WHITE)
        # trunk pitch + closed-loop terrain-leveling indicator
        d.text((px + 12, py + 88), f"trunk pitch {pitch_deg:4.1f} deg", font=self.fonts[20], fill=WHITE)
        d.ellipse([px + 200, py + 90, px + 216, py + 106], fill=GREEN if fb_on else DIM)
        d.text((px + 12, py + 110), "terrain leveling (IMU)", font=self.fonts[20],
               fill=GREEN if fb_on else DIM)
        # uprightness bar
        up = o["upright"]
        d.text((px + 12, py + 136), "uprightness", font=self.fonts[20], fill=WHITE)
        d.rectangle([px + 150, py + 140, px + 150 + 80, py + 154], outline=DIM)
        d.rectangle([px + 150, py + 140, px + 150 + int(max(0, up) * 80), py + 154],
                    fill=GREEN if up > 0.6 else ACCENT)
        # foot stance/swing dots
        d.text((px + 12, py + 166), "feet (● stance)", font=self.fonts[20], fill=WHITE)
        fc = o["foot_contacts"]
        for i, lg in enumerate(LEGS):
            cx = px + 30 + i * 56
            col = GREEN if fc[i] > 0.5 else DIM
            d.ellipse([cx, py + 192, cx + 16, py + 208], fill=col)
            d.text((cx - 4, py + 210), lg, font=self.fonts[20], fill=DIM)
        return np.asarray(img)

    def title_card(self, lines: List[Tuple[str, int, tuple]], n=40):
        W, H = self.env.cfg.render_w, self.env.cfg.render_h
        img = Image.new("RGB", (W, H), (10, 11, 15))
        d = ImageDraw.Draw(img)
        y = H // 2 - 22 * len(lines)
        for text, size, color in lines:
            w = d.textlength(text, font=self.fonts[size])
            d.text(((W - w) / 2, y), text, font=self.fonts[size], fill=color)
            y += size + 18
        self.frames.extend([np.asarray(img)] * n)

    def save(self, path, fps=30):
        import imageio.v2 as imageio
        imageio.mimsave(path, self.frames, fps=fps, quality=8, macro_block_size=8)
        return path
