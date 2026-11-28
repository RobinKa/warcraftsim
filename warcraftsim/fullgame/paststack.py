"""--infer-past-stack S: a round's past-snapshot calls of one architecture as ONE network call.

Why: past snapshots (the league's members, the exploiters' versions) make ~85% of the inference calls (fgself-271:
~690 of ~810 a second) at ~1.5 rows each, and a call of a 10-layer network is a chain of ~150 small kernels whatever its
rows: its cost is per call, not per row. Here an architecture's S slots hold their weights stacked ([S, ...]: one flat
[S, total] tensor laid out as selfplay.flat_layout per slot, each buffer [S, ...]) and one call runs every slot's rows
through the same network code under torch.func.vmap over torch.func.functional_call: batched matrix multiplies, one
kernel chain for the round's ~6 past calls. A slot's rows are rows s*B .. s*B + B - 1 of the call's input [S * B, W] (the
rows Inference._launch packs; B: rows per slot, a bucket of `rows`), its answers the same rows of the output. Rows of a slot
nobody asked for this round hold an earlier call's valid input; their answers are not read (a row's answer does not depend
on the others').

The same numbers as the slots' own calls: every op is per slot and per row; the sampling's uniform draws come from the
caller (model.noise_shapes: torch.compile cannot trace a draw from a generator inside vmap; drawn here, outside it, from
the server's generator), so with the same draws a stacked call gives a slot's rows exactly what its own call gives
(tests/unit/test_past_stack.py). The weights of a snapshot go into a slot as they go into a PastNet slot: its flat
weights (selfplay.pack_flat) into the slot's row of the flat tensor, its buffers into the slot's part of each (dst(),
the C++ loop's copies: csrc/infer_loop.cpp's Stack)."""

from __future__ import annotations

import torch
from torch.func import functional_call, vmap

from . import features as fx
from .model import FullGameNet, noise_shapes

STACK_ROWS = (1, 2, 4)  # --infer-past-stack-rows: the rows per slot a stacked call pads to (more: several calls)


def stack_rows(text, top: int = 16) -> tuple[int, ...]:
    """--infer-past-stack-rows -> the sorted row buckets (each from 1 to `top`); empty: STACK_ROWS."""
    if isinstance(text, (list, tuple)):
        given = {int(x) for x in text}
    else:
        given = {int(x) for x in str(text or "").split(",") if x.strip()}
    if any(b < 1 or b > top for b in given):
        raise ValueError(f"--infer-past-stack-rows {sorted(given)}: each from 1 to {top}")
    return tuple(sorted(given)) if given else STACK_ROWS


def row_width(net: FullGameNet, E: int) -> int:
    """A packed input row's floats for E entities (Inference._launch's layout)."""
    d = net.config["d"] if net.memory else 0
    return E * (fx.F + 3) + net.config["G"] + 1 + net.config["n_orders"] + d


def blank_rows(net: FullGameNet, rows: int, E: int) -> torch.Tensor:
    """[rows, W] valid padding rows (as Inference._launch's: one entity in view, every order available)."""
    G, NO = net.config["G"], net.config["n_orders"]
    x = torch.zeros(rows, row_width(net, E))
    a = E * fx.F + G
    x[:, a + 2 * E] = 1.0  # (the mask's first entity: attention over nothing gives NaNs)
    x[:, a + 3 * E + 1:a + 3 * E + 1 + NO] = 1.0
    return x


def pack_rows(net: FullGameNet, x: torch.Tensor, sts: list[dict], E: int, at: int = 0) -> None:
    """View steps (dicts: ent, type, cur, glob, n, n_own, avail, h, shared_noise) into rows at.. of x [R, W] (CPU), as
    Inference._launch packs them (the entities past a row's own zero, masked)."""
    G, NO = net.config["G"], net.config["n_orders"]
    d = net.config["d"] if net.memory else 0
    a_glob, a_typ = E * fx.F, E * fx.F + G
    a_own = a_typ + 3 * E
    for i, st in enumerate(sts):
        r = x[at + i]
        r.zero_()
        n = int(st["n"])
        r[:a_glob].view(E, fx.F)[:n] = torch.as_tensor(st["ent"], dtype=torch.float32)[:n]
        r[a_glob:a_typ] = torch.as_tensor(st["glob"], dtype=torch.float32)
        r[a_typ:a_typ + n] = torch.as_tensor(st["type"][:n], dtype=torch.float32)
        r[a_typ + E:a_typ + E + n] = torch.as_tensor(st["cur"][:n], dtype=torch.float32)
        r[a_typ + 2 * E:a_typ + 2 * E + max(n, 1)] = 1.0
        r[a_own] = float(min(int(st["n_own"]), fx.MAX_OWN))
        avail = st.get("avail")
        r[a_own + 1:a_own + 1 + NO] = 1.0 if avail is None else torch.as_tensor(avail, dtype=torch.float32)
        if st.get("shared_noise"):
            r[a_own + 1] = 2.0
        if d and st.get("h") is not None:
            r[a_own + 1 + NO:] = torch.as_tensor(st["h"], dtype=torch.float32)


class _Call(torch.nn.Module):
    """selfplay.packed_call on the network as a module (functional_call swaps its weights for a slot's)."""

    def __init__(self, net: FullGameNet, bf16: bool, mixed: bool):
        super().__init__()
        self.net, self.bf16, self.mixed = net, bf16, mixed
        c = net.config
        self.G, self.NO, self.d = c["G"], c["n_orders"], c["d"] if net.memory else 0

    def forward(self, x, noise):
        from .selfplay import packed_call
        pairs = {r: (noise[r], noise[r + "1"]) for r in noise if not r.endswith("1")}
        return packed_call(self.net, x, self.G, self.NO, self.d, self.bf16, None, False, self.mixed, noise=pairs)


class PastStack:
    """An architecture's S past-snapshot slots as one call (see the module's doc). `rows`: the rows per slot its calls pad
    to; `entities`: the entity counts (fx.MAX_ENT always); `fuse`: the stacked call compiled (torch.compile, as
    Inference's) before its CUDA graphs are captured; `bf16`, `mixed`, `tf32`: Inference's precision flags; `generator`:
    the sampling's (the server's own: its graphs register it)."""

    def __init__(self, src: FullGameNet, slots: int, device, rows=STACK_ROWS, entities=(64,), fuse: bool = False,
                 bf16: bool = False, mixed: bool = False, tf32: bool = False, generator: torch.Generator | None = None,
                 capture_mode: str = "global"):
        from .selfplay import flat_layout, pack_flat
        if slots < 1:
            raise ValueError("a stack of at least one slot")
        self.S, self.device = int(slots), torch.device(device)
        self.rows = tuple(sorted(set(int(r) for r in rows)))
        self.entities = tuple(sorted(set(int(e) for e in entities) | {fx.MAX_ENT}))
        cuda = self.device.type == "cuda"
        self.fuse, self.bf16 = bool(fuse) and cuda, (bool(bf16) or bool(mixed)) and cuda
        self.mixed = bool(mixed) and self.bf16
        self.tf32 = cuda and ((bool(tf32) and not self.bf16) or self.mixed)
        self.generator, self.capture_mode = generator, capture_mode
        net = FullGameNet(**src.config).to(self.device).eval()
        net.load_state_dict(src.state_dict())
        for p in net.parameters():
            p.requires_grad_(False)
        for layer in net.transformer.layers:  # (the encoder over [L, N, d] tokens: FullGameNet.encode; same weights)
            layer.self_attn.batch_first = False
        self.net = net
        self.config = dict(src.config)
        named = list(net.named_parameters())
        offsets, self.total = flat_layout([p for _, p in named])
        one = pack_flat([p for _, p in named])
        self.flat = one.unsqueeze(0).repeat(self.S, 1).contiguous()  # [S, total]: slot s's weights a row (one copy)
        self.params = {"net." + n: self.flat[:, at:at + p.numel()].view((self.S,) + tuple(p.shape))
                       for (n, p), at in zip(named, offsets)}
        self.buffers = {"net." + n: b.detach().unsqueeze(0).repeat((self.S,) + (1,) * b.dim()).contiguous()
                        for n, b in net.named_buffers()}
        self.holding: list[str | None] = [None] * self.S
        self.module = _Call(net, self.bf16, self.mixed)
        self._core = None
        self.graphs: dict[tuple, tuple] = {}  # (rows per slot, entities) -> (input, graph, output)
        self.pool = torch.cuda.graph_pool_handle() if cuda else None  # (its graphs run one at a time, on one stream)
        self.stream = torch.cuda.Stream(device=self.device) if cuda else None

    # the slots' weights ---------------------------------------------------------------------------------------------
    def dst(self, s: int) -> list[tuple[int, int]]:
        """Slot s's places as the C++ loop copies a snapshot into them: [(address, bytes)], its flat weights first, then
        each buffer, in the order of the snapshot's (pack_flat(src.parameters()), src.buffers()): NativeServer._snapshot."""
        f = self.flat[s]
        return [(f.data_ptr(), f.numel() * f.element_size())] + \
            [(b[s].data_ptr(), b[s].numel() * b.element_size()) for b in self.buffers.values()]

    @torch.no_grad()
    def load(self, s: int, src: FullGameNet, key: str | None = None) -> None:
        """A snapshot's weights and buffers into slot s (the Python side's copy; the C++ loop's is dst()'s)."""
        from .selfplay import pack_flat
        self.copy_in(s, pack_flat(src.parameters()), list(src.buffers()), key)

    @torch.no_grad()
    def copy_in(self, s: int, flat: torch.Tensor, buffers: list, key: str | None = None) -> None:
        """A snapshot packed as NativeServer._snapshot packs it (its flat weights, its buffers) into slot s: one copy each,
        as the C++ loop's into dst(s)."""
        if len(buffers) != len(self.buffers) or flat.numel() != self.total:
            raise ValueError(f"a snapshot of {flat.numel()} weights and {len(buffers)} buffers into a stack of "
                             f"{self.total} and {len(self.buffers)}")
        self.flat[s].copy_(flat, non_blocking=True)
        for mine, b in zip(self.buffers.values(), buffers):
            mine[s].copy_(b, non_blocking=True)
        self.holding[s] = key

    # the call ---------------------------------------------------------------------------------------------------------
    def _core_fn(self):
        if self._core is None:
            module = self.module

            def one(p, b, x, noise):
                return functional_call(module, (p, b), (x, noise))

            def core(params, buffers, x, noise):
                return vmap(one, in_dims=(0, 0, 0, 0))(params, buffers, x, noise)
            if self.fuse:
                torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)
                torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 1024)
                core = torch.compile(core, dynamic=False)
            self._core = core
        return self._core

    def draw(self, B: int, E: int) -> dict:
        """The call's uniform draws: {role: [S, B, ...], role + "1": the shared draw} (model.noise_shapes)."""
        out = {}
        for role, (u, one) in noise_shapes(self.net, B, E).items():
            out[role] = torch.rand((self.S,) + u, device=self.device, generator=self.generator)
            out[role + "1"] = torch.rand((self.S,) + one, device=self.device, generator=self.generator)
        return out

    def call(self, x: torch.Tensor, noise: dict | None = None) -> torch.Tensor:
        """[S * B, W] packed rows (slot s's at s * B ..) -> [S * B, cols] answers (selfplay.packed_call's)."""
        S = self.S
        B = x.shape[0] // S
        c = self.config
        E = (x.shape[1] - c["G"] - 1 - c["n_orders"] - (c["d"] if c.get("memory") else 0)) // (fx.F + 3)
        if noise is None:
            noise = self.draw(B, E)
        fast = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)  # (the encoder layers' fused inference kernel has no batching
        try:                                             # rule: vmap would run it slot by slot)
            out = self._core_fn()(self.params, self.buffers, x.view(S, B, x.shape[1]), noise)
        finally:
            torch.backends.mha.set_fastpath_enabled(fast)
        return out.reshape(S * B, out.shape[-1])

    def graph(self, B: int, E: int) -> tuple:
        """The call's CUDA graph for B rows per slot and E entities: (its input [S * B, W], the graph, its output). The
        graph reads the slots' weights where they are: a snapshot copied into a slot is used."""
        from .selfplay import matmul_precision
        key = (int(B), int(E))
        if key not in self.graphs:
            x = blank_rows(self.net, self.S * key[0], key[1]).to(self.device)
            with torch.no_grad(), matmul_precision(self.tf32):
                for _ in range(2):  # (outside the capture: what the first calls set up, the compile among it)
                    self.call(x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                if self.generator is not None:
                    graph.register_generator_state(self.generator)
                with torch.cuda.graph(graph, pool=self.pool, stream=self.stream, capture_error_mode=self.capture_mode):
                    out = self.call(x)
            self.graphs[key] = (x, graph, out)
        return self.graphs[key]

    def warm(self) -> None:
        """Every shape's graph now (the native loop replays them; none is captured later)."""
        for B in self.rows:
            for E in self.entities:
                self.graph(B, E)

    def bucket(self, rows: int, widest: int) -> tuple[int, int] | None:
        B = next((b for b in self.rows if rows <= b), None)
        E = next((e for e in self.entities if widest <= e), None)
        return None if B is None or E is None else (B, E)
