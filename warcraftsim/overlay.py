"""Replay video overlay: which units are the agent's, what it ordered, and what the policy thinks.

Drawn on every frame of a replay video (video.render_replay(..., overlay=...)):

* on the game footage (scenario maps: the camera stays on the scenario center, whose projection
  was calibrated with scripts/calibrate_camera.py): a ring and slot label under every agent unit
  (A0, A1, ... = the rows of the side panel), slot labels on enemy units (E0, ... = the attack
  targets the policy chooses from), and the orders of the current step: move arrows, attack lines
  with a crosshair on the target, stop markers. Positions are interpolated between steps.
* a side panel: which side is the agent; the value V(s) the policy predicted next to the
  discounted return that actually followed; reward and TD error per step; team hit points; per
  unit the policy's action probabilities with the action it sampled; the policy's entropy; and an
  outcome card at the end.

Values and probabilities come from evaluating the checkpoint saved last before the episode ended
(warcraftsim.puffer.policy) on the observations the agent saw, so they are those of a policy
at most one checkpoint interval newer than the one that acted.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .protocol import Command, ImmediateOrder, Observation, PointOrder, TargetOrder, Unit

# Ground plane (relative to the camera target) -> normalized window coordinates, for the default
# game camera in a 16:9 window (scripts/calibrate_camera.py; reprojection error < 0.2 px).
GROUND_TO_SCREEN = np.array([
    (0.000415183543, 0.000170282438, 0.499508483),
    (-7.99792066e-08, -0.000474973268, 0.408600129),
    (-1.35883146e-07, 0.000340551895, 1),
])

PLAYER_COLORS = [(255, 3, 3), (0, 66, 255), (28, 230, 185), (84, 0, 129), (255, 252, 0), (254, 138, 14)]
PLAYER_COLOR_NAMES = ["red", "blue", "teal", "purple", "yellow", "orange"]
AGENT_COLORS = [(70, 255, 120), (255, 170, 40)]  # rings/labels of agent A, B
KIND_COLORS = {"noop": (125, 125, 135), "stop": (235, 205, 60), "retreat": (176, 131, 240), "move": (80, 190, 245),
               "attack": (245, 80, 60), "cast": (245, 110, 210)}


def _ability_orders() -> dict[str, tuple[str, float]]:
    """Order string -> (ability name, area) of the hero abilities (empty without the game data)."""
    try:
        from .data.abilities import ability_info
        return {i.order: (i.name, max(i.area, default=0.0)) for i in ability_info().values() if i.order}
    except Exception:
        return {}


def _hero_abilities(u: Unit) -> list[tuple[str, int, str, float, float]]:
    """A hero's learned abilities: (name, level, state, cooldown share left, seconds left); state
    is "ready", "cooldown", "mana" (not enough) or "passive"."""
    try:
        from .data.abilities import ability_info, hero_abilities
        from .env import ability_ready
        info_by_code, slots = ability_info(), hero_abilities().get(u.type, ())
    except Exception:
        return []
    out = []
    for k, code in enumerate(slots):
        if k >= len(u.abilities) or not u.abilities[k][0]:
            continue
        level, left = u.abilities[k]
        info = info_by_code.get(code)
        if info is None:
            continue
        full = info.at(info.cooldown, level) or 1.0
        if not info.castable:
            state = "passive"
        elif left > 0:
            state = "cooldown"
        elif not ability_ready(u, k, info):
            state = "mana"
        else:
            state = "ready"
        out.append((info.name, level, state, min(left / full, 1.0), left))
    return out


def _unit_name(unit_type: str) -> str:
    try:
        from .data.objects import unit_names
        return unit_names().get(unit_type, unit_type)
    except Exception:
        return unit_type


def _hp_color(frac: float) -> tuple[int, int, int]:
    return (70, 210, 90) if frac > 0.6 else (230, 200, 60) if frac > 0.3 else (235, 70, 60)


ABILITY_COLORS = {"ready": (90, 220, 110), "cooldown": (120, 124, 134), "mana": (90, 150, 255),
                  "passive": (150, 150, 160)}


def _ability_name(unit_type: str, slot: int) -> str:
    try:
        from .data.abilities import ability_info, hero_abilities
        return ability_info()[hero_abilities()[unit_type][slot]].name
    except Exception:
        return f"ability {slot + 1}"
PALETTE = [(80, 190, 245), (120, 220, 140), (235, 205, 60), (245, 140, 60), (245, 80, 60), (200, 110, 230),
           (110, 130, 240), (60, 200, 200), (180, 180, 180)]
BG, FG, DIM = (18, 20, 26), (235, 235, 240), (140, 145, 155)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
        try:
            return ImageFont.truetype(f"{d}/{name}", size)
        except OSError:
            pass
    return ImageFont.load_default()


def _steps(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def _slots(obs: Observation, player: int, slots: dict) -> tuple[list[Unit | None], list[Unit | None]]:
    """The agent's own and enemy units by slot, assigned like the environments assign them (a unit
    keeps its slot for the episode; None: dead)."""
    from .env import _slot_units

    return _slot_units(obs, player, slots.setdefault(player, ([], [])), 64, 64)


class EpisodeOverlay:
    PANEL_W = 400
    END_SECONDS = 2.0

    def __init__(self, task, trace: dict, outputs: list | None = None, gamma: float = 0.99, title: str = "",
                 policy_step: int | None = None):
        self.task = task
        self.actions = np.asarray(trace["actions"])  # [T, A, heads]
        self.rewards = np.asarray(trace["rewards"], np.float64)  # [T, A]
        self.T, self.A = self.actions.shape[:2]
        self.outcomes = np.asarray(trace.get("outcomes", np.zeros(self.A)))
        self.outputs = outputs  # per agent: policy.PolicyOutput or None
        self.title = title
        self.policy_step = policy_step
        self.returns = np.zeros_like(self.rewards)  # discounted return-to-go G_t
        acc = np.zeros(self.A)
        for t in reversed(range(self.T)):
            acc = self.rewards[t] + gamma * acc
            self.returns[t] = acc
        self.cum = np.cumsum(self.rewards, axis=0)
        self.values = self.td = self.entropy = None
        if outputs:
            self.values = np.stack([o.values[:self.T] for o in outputs], axis=1)  # [T, A]
            nxt = np.vstack([self.values[1:], np.zeros((1, self.A))])
            self.td = self.rewards + gamma * nxt - self.values
            self.entropy = np.stack([o.entropy[:self.T].mean(axis=1) for o in outputs], axis=1)
        self.f_title, self.f, self.f_small, self.f_bold = _font(14, True), _font(12), _font(11), _font(12, True)
        self.f_label, self.f_big = _font(12, True), _font(30, True)
        self.w = self.h = 0
        self.world = False
        self._max_hp: dict[str, float] = {}  # per side, from the first observation
        self._slot_ids: dict = {}  # per agent player: unit ids by slot

    # ---- setup ----------------------------------------------------------------------------------

    def begin(self, setup, orders: dict[str, int], w: int, h: int) -> None:
        """`orders`: order name -> id (the first observation's Observation.orders)."""
        self.w, self.h = w, h
        self.order_names = {oid: name for name, oid in (orders or {}).items()}  # order id -> name
        self._casts = _ability_orders()
        sc = setup.scenario
        self.world = sc is not None and abs(w / h - 16 / 9) < 0.02  # calibrated camera and aspect
        self.camera = sc.resolved_center() if sc is not None else (0.0, 0.0)
        self.target = sc.target if sc is not None else None
        self.agent_players = [i for i, s in enumerate(setup.slots) if s.kind == "agent"][:self.A]
        self.sides = []
        for i, s in enumerate(setup.slots):
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            cname = PLAYER_COLOR_NAMES[i % len(PLAYER_COLOR_NAMES)]
            if i in self.agent_players:
                a = self.agent_players.index(i)
                who = "AGENT" if self.A == 1 else f"AGENT {'AB'[a]}"
                ring = "green" if a == 0 else "orange"
                self.sides.append((color, f"{who} ({cname}, {ring} rings)", AGENT_COLORS[a]))
            else:
                what = {"ai": f"built-in AI ({s.difficulty})", "scripted": "scripted AI", "idle": "idle units"}
                self.sides.append((color, f"{what.get(s.kind, s.kind)} ({cname})", FG))
        self.canvas = Image.new("RGB", (w + self.PANEL_W, h), BG)
        self._last_panel = None

    @property
    def size(self) -> tuple[int, int]:
        return self.w + self.PANEL_W, self.h

    def _screen(self, x: float, y: float) -> tuple[float, float]:
        p = GROUND_TO_SCREEN @ (x - self.camera[0], y - self.camera[1], 1.0)
        return p[0] / p[2] * self.w, p[1] / p[2] * self.h

    # ---- frames ---------------------------------------------------------------------------------

    def render_step(self, frames: list[bytes], t: int, obs_a: Observation, obs_b: Observation | None,
                    commands: Sequence[Command]) -> list[bytes]:
        """The frames captured while step t (from observation obs_a to obs_b) was simulated."""
        t = min(t, self.T - 1)
        panel = self._panel(t, obs_a)
        self.canvas.paste(panel, (self.w, 0))
        self._last_panel = panel
        labels: dict[int, tuple[str, tuple, bool]] = {}
        for a, p in enumerate(self.agent_players):
            own, enemy = _slots(obs_a, p, self._slot_ids)
            for i, u in enumerate(own):
                if u is not None:
                    labels[u.id] = (f"{'AB'[a]}{i}", AGENT_COLORS[a], True)
            if self.A == 1:
                for i, u in enumerate(enemy):
                    if u is not None:
                        labels.setdefault(u.id, (f"E{i}", (215, 215, 220), False))
        owners = {u.id: u.owner for u in obs_a.units}
        orders = [c for c in commands if isinstance(c, (PointOrder, TargetOrder, ImmediateOrder))
                  and owners.get(c.unit) in self.agent_players]
        pos_a = {u.id: (u.x, u.y) for u in obs_a.units if u.alive}
        pos_b = {u.id: (u.x, u.y) for u in obs_b.units if u.alive} if obs_b is not None else {}
        units_a = {u.id: u for u in obs_a.units if u.alive}
        units_b = {u.id: u for u in obs_b.units} if obs_b is not None else {}
        out = []
        for k, raw in enumerate(frames):
            img = Image.frombytes("RGB", (self.w, self.h), raw, "raw", "BGRX")
            if self.world:
                self._draw_world(img, (k + 1) / len(frames), pos_a, pos_b, labels, orders, owners,
                                 units_a, units_b)
            self.canvas.paste(img, (0, 0))
            out.append(self.canvas.tobytes())
        return out

    def render_end(self, last_frame: bytes, fps: int) -> list[bytes]:
        """An outcome card over the last frame."""
        img = Image.frombytes("RGB", (self.w, self.h), last_frame, "raw", "BGRX")
        d = ImageDraw.Draw(img, "RGBA")
        o = float(self.outcomes[0]) if len(self.outcomes) else 0.0
        if self.A == 1:
            text, color = {1: ("AGENT WINS", (70, 255, 120)), -1: ("AGENT LOSES", (255, 90, 70))}.get(
                int(o), ("DRAW (time limit)", (235, 205, 60)))
        else:
            text, color = {1: ("AGENT A WINS", AGENT_COLORS[0]), -1: ("AGENT B WINS", AGENT_COLORS[1])}.get(
                int(o), ("DRAW (time limit)", (235, 205, 60)))
        box = (self.w // 2 - 190, self.h // 2 - 60, self.w // 2 + 190, self.h // 2 + 40)
        d.rectangle(box, fill=(0, 0, 0, 170))
        d.text((self.w // 2, self.h // 2 - 30), text, font=self.f_big, fill=color, anchor="mm")
        ret = ", ".join(f"{'AB'[a] if self.A > 1 else ''}{' ' if self.A > 1 else ''}return {self.cum[-1, a]:+.2f}"
                        for a in range(self.A))
        d.text((self.w // 2, self.h // 2 + 12), ret, font=self.f, fill=FG, anchor="mm")
        self.canvas.paste(img, (0, 0))
        if self._last_panel is not None:
            self.canvas.paste(self._last_panel, (self.w, 0))
        frame = self.canvas.tobytes()
        return [frame] * int(self.END_SECONDS * fps)

    def _draw_world(self, img: Image.Image, f: float, pos_a: dict, pos_b: dict, labels: dict,
                    orders: list, owners: dict, units_a: dict | None = None, units_b: dict | None = None) -> None:
        d = ImageDraw.Draw(img)

        def at(uid: int):
            a = pos_a.get(uid)
            if a is None:
                return None
            b = pos_b.get(uid, a)
            return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f

        def ring(x: float, y: float, r: float, color, width: int = 2) -> None:
            pts = [self._screen(x + r * math.cos(k * math.pi / 12), y + r * math.sin(k * math.pi / 12))
                   for k in range(25)]
            d.line(pts, fill=color, width=width)

        if self.target is not None:
            ring(self.target[0], self.target[1], self.target[2], (255, 215, 0), 3)
            sx, sy = self._screen(self.target[0], self.target[1])
            d.text((sx, sy), "target", font=self.f_small, fill=(255, 215, 0), anchor="mm")
        for c in orders:
            src = at(c.unit)
            if src is None:
                continue
            name = self.order_names.get(c.order, "?")
            sx, sy = self._screen(*src)
            cast = self._casts.get(name)
            if cast is not None:  # a hero ability: its name, and where it goes (its area if any)
                color = KIND_COLORS["cast"]
                spell, area = cast
                if isinstance(c, PointOrder):
                    dst = (c.x, c.y)
                elif isinstance(c, TargetOrder):
                    dst = at(c.target)
                else:
                    dst = None
                if dst is not None:
                    ex, ey = self._screen(*dst)
                    d.line([(sx, sy), (ex, ey)], fill=color, width=3)
                    ring(dst[0], dst[1], max(area, 45.0), color, 2)
                else:
                    ring(src[0], src[1], max(area, 60.0), color, 3)
                d.text((sx, sy - 30), spell, font=self.f_label, fill=color, anchor="mb", stroke_width=2,
                       stroke_fill=(0, 0, 0))
            elif isinstance(c, PointOrder):
                color = KIND_COLORS["attack" if name == "attack" else "move"]
                ex, ey = self._screen(c.x, c.y)
                d.line([(sx, sy), (ex, ey)], fill=color, width=2)
                ang = math.atan2(ey - sy, ex - sx)
                d.polygon([(ex, ey), (ex - 9 * math.cos(ang - 0.45), ey - 9 * math.sin(ang - 0.45)),
                           (ex - 9 * math.cos(ang + 0.45), ey - 9 * math.sin(ang + 0.45))], fill=color)
            elif isinstance(c, TargetOrder):
                dst = at(c.target)
                if dst is None:
                    continue
                color = KIND_COLORS["attack"] if owners.get(c.target) not in self.agent_players else (150, 255, 150)
                ex, ey = self._screen(*dst)
                d.line([(sx, sy), (ex, ey)], fill=color, width=2)
                d.ellipse([ex - 7, ey - 5, ex + 7, ey + 5], outline=color, width=2)
                d.line([(ex - 11, ey), (ex + 11, ey)], fill=color)
                d.line([(ex, ey - 9), (ex, ey + 9)], fill=color)
            elif isinstance(c, ImmediateOrder) and name in ("stop", "holdposition"):
                d.rectangle([sx - 5, sy - 5, sx + 5, sy + 5], outline=KIND_COLORS["stop"], width=2)
        for uid, (label, color, mine) in labels.items():
            p = at(uid)
            if p is None:
                continue
            if mine:
                ring(p[0], p[1], 42, color)
            sx, sy = self._screen(*p)
            d.text((sx, sy + 13), label, font=self.f_label, fill=color, anchor="mt", stroke_width=2,
                   stroke_fill=(0, 0, 0))
            ua = (units_a or {}).get(uid)
            if ua is not None:
                self._unit_bars(d, sx, sy, ua, (units_b or {}).get(uid) or ua, f)

    def _unit_bars(self, d: ImageDraw.ImageDraw, sx: float, sy: float, ua: Unit, ub: Unit, f: float) -> None:
        """Hit points and mana above a unit (between the step's two observations), and under them
        a square per learned hero ability: green ready, grey filling up while it cools down,
        blue outline without the mana for it, a dot for a passive one."""
        w, x0, y0 = 40, sx - 20, sy - 36
        hp = ua.hp + (ub.hp - ua.hp) * f if ub.alive else ua.hp * (1 - f)
        frac = max(0.0, min(hp / max(ua.max_hp, 1), 1.0))
        d.rectangle([x0 - 1, y0 - 1, x0 + w + 1, y0 + 5], fill=(0, 0, 0))
        d.rectangle([x0, y0, x0 + w * frac, y0 + 4], fill=_hp_color(frac))
        y = y0 + 6
        if ua.max_mana > 0:
            mana = ua.mana + (ub.mana - ua.mana) * f
            d.rectangle([x0 - 1, y - 1, x0 + w + 1, y + 3], fill=(0, 0, 0))
            d.rectangle([x0, y, x0 + w * max(0.0, min(mana / ua.max_mana, 1.0)), y + 2], fill=(90, 150, 255))
            y += 5
        if ua.abilities:
            abilities = _hero_abilities(ua)
            x = sx - (len(abilities) * 10 - 2) / 2
            for _, _, state, left, _ in abilities:
                color = ABILITY_COLORS[state]
                if state == "passive":
                    d.ellipse([x + 2, y + 2, x + 6, y + 6], fill=color)
                elif state == "mana":
                    d.rectangle([x, y, x + 8, y + 8], outline=color, width=2)
                else:
                    d.rectangle([x, y, x + 8, y + 8], fill=(40, 42, 48) if state == "cooldown" else color,
                                outline=(0, 0, 0))
                    if state == "cooldown" and left < 0.85:  # the part already cooled down
                        d.rectangle([x + 1, y + 1 + 7 * left, x + 7, y + 7], fill=color)
                x += 10

    # ---- side panel -----------------------------------------------------------------------------

    def _panel(self, t: int, obs: Observation) -> Image.Image:
        img = Image.new("RGB", (self.PANEL_W, self.h), BG)
        d = ImageDraw.Draw(img)
        x0, x1 = 12, self.PANEL_W - 12
        y = 8
        d.text((x0, y), self.title, font=self.f_title, fill=FG)
        y += 19
        pol = (f"policy: checkpoint at {_steps(self.policy_step)} agent steps" if self.policy_step is not None
               else "policy: no checkpoint yet (values unavailable)")
        d.text((x0, y), pol, font=self.f_small, fill=DIM)
        y += 18
        for color, text, text_color in self.sides:
            d.rectangle([x0, y + 2, x0 + 10, y + 12], fill=color)
            d.text((x0 + 16, y), text, font=self.f_bold if text_color != FG else self.f, fill=text_color)
            y += 16
        y += 4
        cum = self.cum[t]
        d.text((x0, y), f"t {obs.game_time:5.1f} s   step {t + 1}/{self.T}   reward {self.rewards[t, 0]:+.3f}"
                        f"   return {cum[0]:+.2f}", font=self.f, fill=FG)
        y += 20
        two = self.A > 1
        ch = 72 if two else 92
        # value vs what actually followed
        series = []
        for a in range(self.A):
            col = AGENT_COLORS[a]
            if self.values is not None:
                series.append((self.values[:, a], col, False))
            series.append((self.returns[:, a], tuple(int(c * 0.6) for c in col), True))
        vtxt = (f"{self.values[t, 0]:+.2f}" + (f" / {self.values[t, 1]:+.2f}" if two else "")
                if self.values is not None else "")
        self._chart(d, (x0, y, x1, y + ch), series, t, "value V(s) · dashed: discounted return that followed",
                    vtxt)
        y += ch + 8
        rew = [(self.rewards[:, 0], None, "bars")]
        if self.td is not None:
            rew.append((self.td[:, 0], (235, 205, 60), False))
        self._chart(d, (x0, y, x1, y + 50), rew, t, "reward per step (bars)  ·  TD error r + γV' − V (line)",
                    f"{self.td[t, 0]:+.3f}" if self.td is not None else "")
        y += 58
        y = self._hp_bars(d, obs, x0, x1, y) + 4
        y = self._hero_rows(d, obs, x0, x1, y) + 4
        y = self._action_rows(d, obs, t, x0, x1, y)
        if self.entropy is not None and y + 46 < self.h:
            ent = [(self.entropy[:, a], AGENT_COLORS[a], False) for a in range(self.A)]
            self._chart(d, (x0, self.h - 44, x1, self.h - 8), ent, t, "policy entropy (mean per action head)",
                        f"{self.entropy[t, 0]:.2f}")
        return img

    def _chart(self, d: ImageDraw.ImageDraw, box, series, t: int, label: str, value: str = "") -> None:
        x0, y0, x1, y1 = box
        d.text((x0, y0), label, font=self.f_small, fill=DIM)
        if value:
            d.text((x1, y0), value, font=self.f_small, fill=FG, anchor="ra")
        y0 += 14
        d.rectangle([x0, y0, x1, y1], outline=(45, 48, 58))
        vals = np.concatenate([np.asarray(s[0], float) for s in series])
        lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        n = len(series[0][0])
        xs = lambda i: x0 + 1 + (x1 - x0 - 2) * i / max(n - 1, 1)  # noqa: E731
        ys = lambda v: y1 - 1 - (y1 - y0 - 2) * (v - lo) / (hi - lo)  # noqa: E731
        d.line([(x0, ys(0)), (x1, ys(0))], fill=(60, 64, 76))
        for values, color, style in series:
            values = np.asarray(values, float)
            if style == "bars":
                for i, v in enumerate(values):
                    c = (90, 200, 120) if v >= 0 else (230, 90, 80)
                    if i > t:
                        c = tuple(int(ci * 0.35) for ci in c)
                    d.line([(xs(i), ys(0)), (xs(i), ys(v))], fill=c)
                continue
            pts = [(xs(i), ys(v)) for i, v in enumerate(values)]
            if style:  # dashed
                for i in range(0, len(pts) - 1, 2):
                    d.line(pts[i:i + 2], fill=color, width=1)
            else:
                d.line(pts[:t + 1], fill=color, width=2)
                d.line(pts[t:], fill=tuple(int(c * 0.45) for c in color), width=1)
        px = xs(t)
        d.line([(px, y0), (px, y1)], fill=(230, 230, 235))
        for values, color, style in series:
            if style is False:
                v = float(np.asarray(values)[t])
                d.ellipse([px - 3, ys(v) - 3, px + 3, ys(v) + 3], fill=color)

    def _hp_bars(self, d: ImageDraw.ImageDraw, obs: Observation, x0: int, x1: int, y: int) -> int:
        p0 = self.agent_players[0] if self.agent_players else 0
        groups = [("A" if self.A > 1 else "agent", [u for u in obs.units if u.owner == p0], AGENT_COLORS[0])]
        others = [u for u in obs.units if u.owner != p0 and u.owner in obs.players]
        groups.append(("B" if self.A > 1 else "enemy", others, AGENT_COLORS[1] if self.A > 1 else (230, 90, 80)))
        for name, units, color in groups:
            alive = [u for u in units if u.alive]
            mx = self._max_hp.setdefault(name, max(sum(u.max_hp for u in alive), 1))  # dead units drop out
            hp = sum(u.hp for u in alive)
            d.text((x0, y), name, font=self.f_small, fill=color)
            bx0, bx1 = x0 + 48, x1 - 118
            d.rectangle([bx0, y + 2, bx1, y + 11], outline=(60, 64, 76))
            d.rectangle([bx0, y + 2, bx0 + (bx1 - bx0) * hp / mx, y + 11], fill=color)
            d.text((x1, y), f"{hp:.0f}/{mx:.0f} hp · {len(alive)} units", font=self.f_small, fill=FG, anchor="ra")
            y += 15
        return y

    def _hero_rows(self, d: ImageDraw.ImageDraw, obs: Observation, x0: int, x1: int, y: int) -> int:
        """Per hero: level, mana, and its learned abilities with their level and state."""
        rows = []
        for a, p in enumerate(self.agent_players):
            own, enemy = _slots(obs, p, self._slot_ids)
            rows += [(f"{'AB'[a]}{i}", AGENT_COLORS[a], u) for i, u in enumerate(own) if u is not None and u.is_hero]
            if self.A == 1:
                rows += [(f"E{i}", (215, 215, 220), u) for i, u in enumerate(enemy) if u is not None and u.is_hero]
        for label, color, u in rows:
            d.text((x0, y), f"{label} {_unit_name(u.type)}  level {u.hero_level}", font=self.f_bold, fill=color)
            if u.max_mana:
                bx0, bx1 = x1 - 150, x1 - 76
                d.rectangle([bx0, y + 4, bx1, y + 10], outline=(60, 64, 76))
                d.rectangle([bx0, y + 4, bx0 + (bx1 - bx0) * min(u.mana / u.max_mana, 1.0), y + 10],
                            fill=(90, 150, 255))
                d.text((x1, y), f"{u.mana}/{u.max_mana} mana", font=self.f_small, fill=DIM, anchor="ra")
            y += 14
            x = x0 + 10
            abilities = _hero_abilities(u)
            if not abilities:
                d.text((x, y), "no abilities learned", font=self.f_small, fill=DIM)
            for name, level, state, _, seconds in abilities:
                text = f"{name} {level} " + {"ready": "ready", "mana": "no mana", "passive": "passive",
                                              "cooldown": f"{seconds:.0f}s"}[state]
                width = d.textlength(text, font=self.f_small) + 14
                if x + width > x1 and x > x0 + 10:
                    x, y = x0 + 10, y + 13
                d.text((x, y), text, font=self.f_small, fill=ABILITY_COLORS[state])
                x += width
            y += 15
        return y

    def _action_rows(self, d: ImageDraw.ImageDraw, obs: Observation, t: int, x0: int, x1: int, y: int) -> int:
        task = self.task
        g = max(task.group_size, 1)
        labels = task.head_labels or tuple(tuple(str(i) for i in range(n)) for n in task.act_sizes)
        groups = task.num_atns // g
        bottom = self.h - (50 if self.entropy is not None else 8)
        rows = []
        for a, p in enumerate(self.agent_players):
            own, enemy = _slots(obs, p, self._slot_ids)
            for i in range(groups):
                rows.append((a, i, own[i] if i < len(own) else None, enemy))
        head = "per unit: policy probabilities (bar) and the sampled action"
        d.text((x0, y), head, font=self.f_small, fill=DIM)
        y += 15
        row_h = max(11, min(18, (bottom - y) // max(len(rows), 1)))
        bar_x0, bar_x1 = x0 + 30, x0 + 170
        for a, i, unit, enemy in rows:
            if y + row_h > bottom:
                break
            name = f"{'AB'[a]}{i}"
            if unit is None:
                d.text((x0, y), f"{name}  —", font=self.f_small, fill=(70, 74, 84))
                y += row_h
                continue
            d.text((x0, y), name, font=self.f_bold, fill=AGENT_COLORS[a])
            h0 = i * g
            chosen = int(self.actions[t, a, h0])
            names = labels[h0]
            probs = self.outputs[a].probs[h0][t] if self.outputs else None
            bh = min(row_h - 4, 10)
            if probs is not None:
                cx = bar_x0
                for v, pv in enumerate(probs):
                    wv = (bar_x1 - bar_x0) * pv
                    color = KIND_COLORS.get(names[v], PALETTE[v % len(PALETTE)])
                    d.rectangle([cx, y + 2, cx + wv, y + 2 + bh], fill=color)
                    if v == chosen:
                        d.rectangle([cx - 1, y + 1, cx + wv + 1, y + 3 + bh], outline=(255, 255, 255))
                    cx += wv
            text = names[chosen] if chosen < len(names) else str(chosen)
            if probs is not None:
                text += f" {probs[chosen]:.0%}"
            detail = task.detail_heads.get(chosen)
            for off in (detail if isinstance(detail, tuple) else () if detail is None else (detail,)):
                hd = h0 + off  # e.g. move -> direction; attack -> target; cast -> ability, target
                dv = int(self.actions[t, a, hd])
                dname = labels[hd][dv] if dv < len(labels[hd]) else str(dv)
                slot = dname[:1] == "E" and dname[1:].isdigit()  # else a rule (semantic targets)
                if slot and self.A > 1:
                    dname = f"{'BA'[a]}{dname[1:]}"
                if slot and names[chosen] == "attack" and (dv >= len(enemy) or enemy[dv] is None):
                    dname += " (none: ignored)"
                if names[chosen] == "cast" and dname.startswith("ability"):
                    dname = _ability_name(unit.type, dv)
                text += f" → {dname}"
                if self.outputs:
                    text += f" {self.outputs[a].probs[hd][t][dv]:.0%}"
            d.text((bar_x1 + 8, y), text, font=self.f_small, fill=KIND_COLORS.get(names[chosen], FG))
            y += row_h
        return y
