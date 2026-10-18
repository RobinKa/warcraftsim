"""A side panel for whole-game videos (video.render_replay's overlay): what each side had and did,
and for a policy what it thought (its value against the return that followed, rewards, entropy).

The data is a trace the game's player wrote while it played (fullgame/selfplay.py, play.py,
collect.py: trace_step), one row per step:
    {"t", "game_time", "material": {p: value}, "res": {p: [gold, lumber, food used, food cap]},
     "orders": {p: {label: count}}, and for policies "value", "reward", "entropy": {p: float}}
and a header: {"sides": [{"player", "name", "kind"}], "title", "gamma", "outcome": {p: +1/0/-1}}.
Player keys are strings (JSON).
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..overlay import AGENT_COLORS, BG, DIM, FG, PLAYER_COLORS, PLAYER_COLOR_NAMES, EpisodeOverlay, _font
from ..protocol import Command, Observation
from .trace import trace_step  # noqa: F401 (the players import it from here too)

SIDE_COLORS = [AGENT_COLORS[0], AGENT_COLORS[1]]


def _name(code: str) -> str:
    """A hero's or an ability's name in the game (data/w3ids.json), else its code."""
    from ..data.w3g import object_kinds
    return (object_kinds().get(code) or (None, code))[1]


def _hero_abilities() -> dict[str, tuple[str, ...]]:
    from ..data.abilities import hero_abilities
    return hero_abilities()


class FullGameOverlay:
    PANEL_W = 420
    END_SECONDS = 2.5
    _chart = EpisodeOverlay._chart  # (uses self.f_small only)

    def __init__(self, trace: dict):
        self.steps = trace["steps"]
        self.T = max(len(self.steps), 1)
        self.sides = trace["sides"]  # [{"player", "name", "kind"}] for players 0 and 1
        self.title = trace.get("title", "")
        self.outcome = {int(k): v for k, v in (trace.get("outcome") or {}).items()}
        self.target = trace.get("target")  # the learner's target strategy (its items' times, and when each was made)
        gamma = trace.get("gamma", 0.997)
        keys = [str(s["player"]) for s in self.sides]
        col = lambda name: np.array([[r.get(name, {}).get(k, np.nan) for k in keys] for r in self.steps],  # noqa: E731
                                    float).reshape(len(self.steps), len(keys))
        self.value, self.reward, self.entropy = col("value"), col("reward"), col("entropy")
        self.material = col("material")
        self.has_policy = np.isfinite(self.value).any(axis=0)  # per side
        rew = np.nan_to_num(self.reward)
        self.returns = np.zeros_like(rew)  # discounted return-to-go
        acc = np.zeros(len(keys))
        for t in reversed(range(len(self.steps))):
            acc = rew[t] + gamma * acc
            self.returns[t] = acc
        self.cum = np.cumsum(rew, axis=0)
        nxt = np.vstack([self.value[1:], np.zeros((1, len(keys)))]) if len(self.steps) else self.value
        self.td = rew + gamma * np.nan_to_num(nxt) - self.value
        # the value and the return without the shaping's potential: with potential shaping the value learns the
        # outcome less the shaping's weight times the material lead (the potential goes back to 0 at the end),
        # so ahead in material it reads low and far behind high (+1.6 in a game it lost). Added back: an
        # estimate of the outcome (-1 a loss, +1 a win) and the discounted outcome that followed
        self.win, self.win_returns, sh = self.value, self.returns, trace.get("shaping") or {}
        if sh.get("weight") and len(self.steps) and len(keys) == 2:
            res = lambda k: np.array([[(r.get("res", {}).get(kk) or [0, 0])[k] for kk in keys] for r in self.steps],  # noqa: E731
                                     float)
            pot = self.material - sh.get("penalty", 0.0) * (res(0) + res(1))
            lead = sh["weight"] * (pot - pot[:, ::-1]) / sh["scale"]
            self.win, self.win_returns = self.value + lead, self.returns + lead
        self.f_title, self.f, self.f_small, self.f_bold = _font(14, True), _font(12), _font(11), _font(12, True)
        self.f_big = _font(30, True)
        self.in_game = False
        self.w = self.h = 0

    # ---- the renderer's interface (video.render_replay) ------------------------------------------
    def begin(self, setup, orders: dict, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.canvas = Image.new("RGB", (w + self.PANEL_W, h), BG)
        self._last_panel = None

    @property
    def size(self) -> tuple[int, int]:
        return self.w + self.PANEL_W, self.h

    def markers(self, obs: Observation, commands: Sequence[Command]) -> list[Command]:
        return []  # whole games: too many units for marks; the game's health bars show

    def render_step(self, frames: list[bytes], t: int, obs_a: Observation, obs_b: Observation | None,
                    commands: Sequence[Command], camera=None) -> list[bytes]:
        panel = self._panel(min(t, self.T - 1), obs_a)
        self.canvas.paste(panel, (self.w, 0))
        self._last_panel = panel
        out = []
        for raw in frames:
            self.canvas.paste(Image.frombytes("RGB", (self.w, self.h), raw, "raw", "BGRX"), (0, 0))
            out.append(self.canvas.tobytes())
        return out

    def render_end(self, last_frame: bytes, fps: int) -> list[bytes]:
        img = Image.frombytes("RGB", (self.w, self.h), last_frame, "raw", "BGRX")
        d = ImageDraw.Draw(img, "RGBA")
        o = self.outcome.get(self.sides[0]["player"], 0)
        winner = self.sides[0] if o > 0 else self.sides[1] if o < 0 else None
        text, color = ((f"{winner['name']} WINS"[:28], SIDE_COLORS[self.sides.index(winner)]) if winner
                       else ("TIE (time limit)", (235, 205, 60)))
        box = (self.w // 2 - 220, self.h // 2 - 60, self.w // 2 + 220, self.h // 2 + 40)
        d.rectangle(box, fill=(0, 0, 0, 170))
        d.text((self.w // 2, self.h // 2 - 30), text, font=self.f_big, fill=color, anchor="mm")
        if len(self.steps):
            m = self.material[-1]
            sub = f"material {m[0]:.0f} vs {m[1]:.0f}"
            rets = [f"{s['name'][:12]} return {self.cum[-1, i]:+.2f}" for i, s in enumerate(self.sides)
                    if np.isfinite(self.reward[:, i]).any()]
            d.text((self.w // 2, self.h // 2 + 12), " · ".join([sub, *rets]), font=self.f, fill=FG, anchor="mm")
        self.canvas.paste(img, (0, 0))
        if self._last_panel is not None:
            self.canvas.paste(self._last_panel, (self.w, 0))
        return [self.canvas.tobytes()] * int(self.END_SECONDS * fps)

    # ---- the panel ----------------------------------------------------------------------------
    def _panel(self, t: int, obs: Observation) -> Image.Image:
        img = Image.new("RGB", (self.PANEL_W, self.h), BG)
        d = ImageDraw.Draw(img)
        x0, x1 = 12, self.PANEL_W - 12
        y = 8
        d.text((x0, y), self.title[:58], font=self.f_title, fill=FG)
        y += 20
        for i, s in enumerate(self.sides):
            p = s["player"]
            d.rectangle([x0, y + 2, x0 + 10, y + 12], fill=PLAYER_COLORS[p % len(PLAYER_COLORS)])
            d.text((x0 + 16, y), f"{'AB'[i]}: {s['name']} ({PLAYER_COLOR_NAMES[p % len(PLAYER_COLOR_NAMES)]})"[:60],
                   font=self.f_bold, fill=SIDE_COLORS[i])
            y += 16
        row = self.steps[t] if self.steps else {}
        gt = row.get("game_time", obs.game_time)
        line = f"t {gt:5.1f} s   step {t + 1}/{self.T}"
        if len(self.steps) and np.isfinite(self.reward[t, 0]):
            line += f"   reward {self.reward[t, 0]:+.3f}   return {self.cum[t, 0]:+.2f}"
        d.text((x0, y + 2), line, font=self.f, fill=FG)
        y += 22
        if len(self.steps) and self.has_policy.any():
            series = []
            for i in range(len(self.sides)):
                if self.has_policy[i]:
                    series.append((self.win[:, i], SIDE_COLORS[i], False))
                    if np.isfinite(self.reward[:, i]).any():
                        series.append((self.win_returns[:, i], tuple(int(c * 0.6) for c in SIDE_COLORS[i]), True))
            vt = " / ".join(f"{self.win[t, i]:+.2f}" for i in range(len(self.sides)) if self.has_policy[i])
            shaped = self.win is not self.value
            self._chart(d, (x0, y, x1, y + 78), series, t,
                        ("win estimate (V + shaping × lead) · dashed: what followed" if shaped
                         else "value V(s) · dashed: discounted return that followed"), vt)
            y += 86
            if np.isfinite(self.reward[:, 0]).any():
                # (a symmetric log scale: the steps' rewards are ~0.001-0.02, the end's ~1, which made the rest flat;
                # the TD error's line, on it, covered the bars)
                sl = lambda x: np.sign(x) * np.log10(1 + np.abs(x) / 1e-3)  # noqa: E731
                self._chart(d, (x0, y, x1, y + 48), [(sl(np.nan_to_num(self.reward[:, 0])), None, "bars")],
                            t, "A's reward per step (symlog)", f"{self.reward[t, 0]:+.3f}")
                y += 56
        if len(self.steps):
            mat = [(self.material[:, i], SIDE_COLORS[i], False) for i in range(len(self.sides))]
            self._chart(d, (x0, y, x1, y + 64), mat, t, "material: units and buildings (cost × hp left)",
                        " vs ".join(f"{self.material[t, i]:.0f}" for i in range(len(self.sides))))
            y += 72
        # economy now
        res = row.get("res", {})
        for i, s in enumerate(self.sides):
            r = res.get(str(s["player"]))
            if r:
                d.text((x0, y), f"{'AB'[i]}  gold {r[0]:>5}   lumber {r[1]:>5}   food {r[2]}/{r[3]}",
                       font=self.f, fill=SIDE_COLORS[i])
                y += 16
        y += 6
        # the heroes: level, the abilities learned (a dead one dimmed)
        hs = row.get("heroes")
        if hs is not None:
            d.text((x0, y), "heroes: level · abilities learned", font=self.f_small, fill=DIM)
            y += 15
            for i, s in enumerate(self.sides):
                mine = hs.get(str(s["player"])) or []
                if not mine:
                    y = self._wrapped(d, f"{'AB'[i]}: –", x0, x1, y, SIDE_COLORS[i]) + 2
                for code, level, alive, levels in mine:
                    slots = _hero_abilities().get(code, ())
                    learned = ", ".join(f"{_name(a)} {lv}" for a, lv in zip(slots, levels) if lv > 0) or "none"
                    color = SIDE_COLORS[i] if alive else tuple(int(c * 0.5) for c in SIDE_COLORS[i])
                    y = self._wrapped(d, f"{'AB'[i]}: {_name(code)} {level}{'' if alive else ' (dead)'}: {learned}",
                                      x0, x1, y, color) + 2
            y += 4
        if self.target:  # A's target: each item made (by when), due and missing, or still to come
            y = self._target(d, x0, x1, y, row.get("game_time", obs.game_time)) + 4
        # this step's orders (and the last second's, so they stay readable)
        d.text((x0, y), "orders (this step and the one before)", font=self.f_small, fill=DIM)
        y += 15
        for i, s in enumerate(self.sides):
            c: Counter = Counter()
            for k in (t - 1, t):
                if 0 <= k < len(self.steps):
                    c.update(self.steps[k].get("orders", {}).get(str(s["player"]), {}))
            text = ", ".join(f"{name} ×{n}" if n > 1 else name for name, n in c.most_common(6)) or "–"
            y = self._wrapped(d, f"{'AB'[i]}: {text}", x0, x1, y, SIDE_COLORS[i]) + 4
        if len(self.steps) and np.isfinite(self.entropy).any() and y + 50 < self.h:
            ent = [(self.entropy[:, i], SIDE_COLORS[i], False) for i in range(len(self.sides))
                   if np.isfinite(self.entropy[:, i]).any()]
            self._chart(d, (x0, self.h - 46, x1, self.h - 8), ent, t, "policy entropy (per unit, mean)",
                        " / ".join(f"{self.entropy[t, i]:.2f}" for i in range(len(self.sides))
                                   if np.isfinite(self.entropy[t, i])))
        return img

    def _target(self, d: ImageDraw.ImageDraw, x0: int, x1: int, y: int, now: float) -> int:
        """A's target build order at game second `now`: made (bright, ✓), due and missing (red, ✗), to come (dim)."""
        tg = self.target
        src = {"human": "a person's game", "ai": "the AI's game"}.get(tg.get("src", ""), "a game")
        d.text((x0, y), f"A's target build ({src}): ✓ made · ✗ due, missing · later", font=self.f_small, fill=DIM)
        y += 15
        made = tg.get("made") or {}
        x = x0
        for code, due in sorted(tg["t"].items(), key=lambda kv: kv[1]):
            m = made.get(code)
            if m is not None and m <= now:
                text, color = f"{_name(code)} ✓", SIDE_COLORS[0]
            elif due <= now:
                text, color = f"{_name(code)} ✗", (232, 96, 80)
            else:
                text, color = _name(code), DIM
            w = d.textlength(text + "  ", font=self.f_small)
            if x + w > x1 and x > x0:
                x, y = x0, y + 14
            d.text((x, y), text, font=self.f_small, fill=color)
            x += w
        return y + 14

    def _wrapped(self, d: ImageDraw.ImageDraw, text: str, x0: int, x1: int, y: int, color) -> int:
        words, line = text.split(" "), ""
        for w in words:
            trial = (line + " " + w).strip()
            if d.textlength(trial, font=self.f) > x1 - x0 and line:
                d.text((x0, y), line, font=self.f, fill=color)
                y += 15
                line = "   " + w
            else:
                line = trial
        if line:
            d.text((x0, y), line, font=self.f, fill=color)
            y += 15
        return y
