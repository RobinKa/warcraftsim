"""Evaluate a PufferLib 5.0 checkpoint in numpy: values, action probabilities, entropies.

The default PufferLib policy (the one warcraftsim trains) is, without biases,

    linear encoder (hidden x obs) -> num_layers x MinGRU (3*hidden x hidden each)
    -> linear decoder ((sum(act_sizes) + 1) x hidden: action logits, then the value)

and a checkpoint is the flat fp32 parameter buffer in that registration order, each tensor
padded to a multiple of 8 floats (see puffercpu.c in PufferLib). Used to show "what the agent
thinks" next to replay videos.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # = 1 / (1 + exp(-x)), without overflow for large -x


@dataclass
class PolicyOutput:
    values: np.ndarray  # [T]
    probs: list[np.ndarray]  # per action head: [T, n]
    entropy: np.ndarray  # [T, heads]

    def logp(self, actions: np.ndarray) -> np.ndarray:
        """Log-probability of the taken actions [T, heads] under the policy, per head."""
        idx = np.asarray(actions, int)
        return np.stack([np.log(np.maximum(p[np.arange(len(p)), idx[:, h]], 1e-12))
                         for h, p in enumerate(self.probs)], axis=1)


class PufferPolicy:
    def __init__(self, path: str | os.PathLike, obs_size: int, act_sizes: tuple[int, ...],
                 hidden: int = 128, layers: int = 2):
        w = np.fromfile(path, np.float32)
        self.path = Path(path)
        self.act_sizes = tuple(act_sizes)
        pos = 0

        def take(n: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal pos
            if pos + n > len(w):
                raise ValueError(f"{path}: checkpoint too small for obs {obs_size}, hidden {hidden}, "
                                 f"layers {layers}, actions {act_sizes}")
            t = w[pos:pos + n].reshape(shape)
            pos = (pos + n + 7) & ~7
            return t

        atn_sum = sum(act_sizes)
        self.encoder = take(hidden * obs_size, (hidden, obs_size))
        self.decoder = take((atn_sum + 1) * hidden, (atn_sum + 1, hidden))
        self.gru = [take(3 * hidden * hidden, (3 * hidden, hidden)) for _ in range(layers)]
        if pos != len(w):
            raise ValueError(f"{path}: {len(w)} floats, expected {pos} (wrong architecture?)")
        self.hidden = hidden

    def initial_state(self) -> list[np.ndarray]:
        """The recurrent state at the start of an episode (the trainer zeroes it after a terminal)."""
        return [np.zeros(self.hidden, np.float32) for _ in self.gru]

    def step(self, o: np.ndarray, state: list[np.ndarray]) -> np.ndarray:
        """One observation [obs_size] -> the decoder output [sum(act_sizes) + 1] (logits per head,
        then the value); `state` (from initial_state) is updated in place."""
        h = self.hidden
        x = self.encoder @ np.asarray(o, np.float32)
        for i, wl in enumerate(self.gru):
            c = wl @ x
            hid, gate, hw = c[:h], c[h:2 * h], c[2 * h:]
            h_tilde = np.where(hid >= 0, hid + 0.5, _sigmoid(hid))
            out = state[i] + _sigmoid(gate) * (h_tilde - state[i])
            state[i] = out
            s = _sigmoid(hw)
            x = s * out + (1 - s) * x
        return self.decoder @ x

    def run(self, obs: np.ndarray, masks: np.ndarray | None = None) -> PolicyOutput:
        """Evaluate one episode of observations [T, obs_size], from a zero recurrent state.
        `masks` [T, sum(act_sizes)] (0: not possible) are applied as the trainer samples."""
        state = self.initial_state()
        outs = [self.step(o, state) for o in np.asarray(obs, np.float32)]
        dec = np.asarray(outs)
        probs, ents, at = [], [], 0
        for n in self.act_sizes:
            logits = dec[:, at:at + n].astype(np.float64)
            if masks is not None:
                m = np.asarray(masks[:, at:at + n]) > 0
                logits = np.where(m | ~m.any(axis=1, keepdims=True), logits, -np.inf)
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            probs.append(p)
            ents.append(-(p * np.log(np.maximum(p, 1e-12))).sum(axis=1))
            at += n
        return PolicyOutput(values=dec[:, -1].astype(np.float64), probs=probs, entropy=np.stack(ents, axis=1))


def checkpoint_at(checkpoint_dir: str | os.PathLike, when: float) -> Path | None:
    """The newest checkpoint written at or before wall time `when` (else the oldest one)."""
    files = sorted(Path(checkpoint_dir).rglob("*.bin"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if re.fullmatch(r"\d+\.bin", f.name)]
    if not files:
        return None
    before = [f for f in files if f.stat().st_mtime <= when]
    return before[-1] if before else files[0]


def checkpoint_step(path: Path) -> int:
    """The trainer's agent-step count a checkpoint was saved at (its file name)."""
    return int(path.stem)
