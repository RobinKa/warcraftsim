"""The inference server's loop in C++ inside the learner's process (selfplay --infer-native; csrc/infer_loop.cpp).

As a process of its own the server took turns on the GPU with the learner (WSL has no MPS: an update took
5.1 s in the run against 2.2 s alone); as a Python thread of the learner's process its loop waited for the
learner's GIL. Here Python sets up (the networks, every call's CUDA graph, the past-snapshot slots) and a
C++ thread runs the rounds without the GIL. Python is called back only to reload the current network, to
load a new past snapshot, and for the statistics.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import torch

from . import features as fx

_EXT = None


def extension():
    """The compiled loop (torch's JIT build: once, then from its cache)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        # the CUDA headers and runtime from the pip packages torch depends on: one package per library
        # with CUDA 12 (nvidia/cuda_runtime/...), one for all with CUDA 13 (nvidia/cu13/...)
        nv = Path(torch.__file__).resolve().parent.parent / "nvidia"
        incs = sorted(glob.glob(str(nv / "*" / "include")))
        cudart = sorted(glob.glob(str(nv / "*" / "lib" / "libcudart.so.*")))
        if not cudart:
            raise FileNotFoundError(f"no libcudart in {nv}")
        rt = Path(cudart[-1])
        os.environ["PATH"] = f"{Path.home() / '.local' / 'bin'}:{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"  # (ninja)
        _EXT = load(name="w3_infer_loop", sources=[str(Path(__file__).with_name("csrc") / "infer_loop.cpp")],
                    extra_include_paths=incs, extra_cflags=["-O2"],
                    extra_ldflags=["-ltorch_cuda", "-lc10_cuda", f"-L{rt.parent}", f"-l:{rt.name}", f"-Wl,-rpath,{rt.parent}"])
    return _EXT


class NativeServer:
    """The server's setup in Python and its loop in C++ (start(), stop(), failed())."""

    def __init__(self, cfg: dict, conns: list, device):
        from .selfplay import Inference, Nets, PastNet, SharedRows, warm_inference
        assert cfg.get("shared_rows"), "--infer-native needs --infer-shm 1"
        assert not cfg.get("exploiter_share"), "--infer-native serves the current network and past snapshots only"
        self.cfg, self.conns, self.device = cfg, conns, device
        self.nets = Nets(cfg, device)
        while self.nets.current is None:
            time.sleep(1)
            self.nets.reload()
        self.generator = torch.Generator(device=device)
        self.generator.seed()
        top = cfg["infer_batch"]
        fuse = bool(cfg.get("infer_fuse", 0))
        self.shared = SharedRows(*cfg["shared_rows"])
        self.infer = Inference(self.nets, device, top, True, buckets=tuple(b for b in (8, 24, 64) if b <= top) or (top,),
                               thread=False, entities=tuple(cfg.get("infer_entities", (48, 96))), fuse=fuse,
                               generator=self.generator, capture_mode="thread_local")
        self.past = Inference(self.nets, device, 16, True, buckets=(4, 16), thread=False,
                              entities=tuple(cfg.get("infer_past_entities", (64,))), fuse=fuse,
                              generator=self.generator, capture_mode="thread_local")
        self.swap = PastNet(device, slots=cfg.get("infer_past_slots", 1))
        t0 = time.time()
        league = Path(cfg["run_dir"]) / "league.json"
        keys = [m["path"] for m in json.loads(league.read_text()).get("members", []) if m.get("path")] if league.exists() else []
        warm_inference(self.nets, self.infer, self.past, self.swap, keys)
        print(f"native inference: graphs captured in {time.time() - t0:.0f} s", flush=True)
        self.arch_ids: dict[str, int] = {}
        self.kept: dict[str, tuple] = {}  # past snapshots' weights the loop copies from (kept alive here)
        self.stats_path = Path(cfg["run_dir"]) / "inference.jsonl"
        self.loop = self._build()

    def _width(self, net, E: int) -> int:
        d = net.config["d"] if net.memory else 0
        return E * (fx.F + 3) + net.config["G"] + 1 + net.config["n_orders"] + d

    def _build(self):
        ext = extension()
        loop = ext.Loop()
        sh, net = self.shared, self.nets.current
        d = net.config["d"] if net.memory else 0
        loop.set_rows([a.ctypes.data for a in (sh.ent, sh.typ, sh.cur, sh.glob, sh.n_own, sh.avail, sh.h, sh.out)],
                      fx.MAX_ENT, fx.F, fx.MAX_OWN, net.config["G"], net.config["n_orders"], d)
        loop.set_fds([c.fileno() for c in self.conns])
        loop.set_live(self.infer.max_batch, list(self.infer.buckets), list(self.infer.entities))
        graphs = self.infer._graphs[net]
        for B in self.infer.buckets:
            for E in self.infer.entities:
                x, graph, out = graphs[(B, self._width(net, E))]
                loop.add_live_shape(B, E, x.shape[1], out.shape[1], graph, x.data_ptr(), out.data_ptr())
        for (arch, slot), (slot_net, flat, buffers, _) in sorted(self.swap.hosts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            a = self.arch_ids.setdefault(arch, len(self.arch_ids))
            dst = [(flat.data_ptr(), flat.numel() * flat.element_size())] + \
                  [(b.data_ptr(), b.numel() * b.element_size()) for b in buffers]
            k = loop.add_slot(a, 16, list(self.past.buckets), list(self.past.entities), dst)
            assert k == slot, (k, slot)
            pg = self.past._graphs[slot_net]
            for B in self.past.buckets:
                for E in self.past.entities:
                    x, graph, out = pg[(B, self._width(slot_net, E))]
                    loop.add_slot_shape(a, k, B, E, x.shape[1], out.shape[1], graph, x.data_ptr(), out.data_ptr())
        loop.set_callbacks(self._reload, self._snapshot, self._stats)
        loop.period = self.cfg.get("infer_period", 0.0)
        loop.version = self.nets.version
        return loop

    def _reload(self) -> int:  # (from the loop's thread, the GIL held, on the current network's stream)
        self.nets.reload()
        return int(self.nets.version)

    def _snapshot(self, key: str):  # (from the loop's thread: a past snapshot it has not seen yet)
        src = self.nets.get(key)
        arch = json.dumps(src.config, sort_keys=True, default=str)
        flat = torch.cat([p.data.reshape(-1) for p in src.parameters()])
        buffers = list(src.buffers())
        torch.cuda.current_stream().synchronize()  # (made before the loop's streams copy from them)
        self.kept[key] = (flat, buffers)
        if arch not in self.arch_ids:
            raise RuntimeError(f"a past snapshot of an architecture without slots: {key}")
        return self.arch_ids[arch], [(flat.data_ptr(), flat.numel() * flat.element_size())] + \
            [(b.data_ptr(), b.numel() * b.element_size()) for b in buffers]

    def _stats(self, s: dict) -> None:
        dt = max(s["dt"], 1e-9)
        calls, rows, rounds = max(s["calls"], 1), max(s["rows"], 1), max(s["rounds"], 1)
        with open(self.stats_path, "a") as f:
            f.write(json.dumps({"time": time.time(), "calls_per_s": round(s["calls"] / dt, 1),
                                "rows_per_s": round(s["rows"] / dt, 1), "rows_per_call": round(s["rows"] / calls, 2),
                                "rounds_per_s": round(s["rounds"] / dt, 1), "past_calls_per_s": round(s["past"] / dt, 1),
                                "round_ms": round(1000 * s["busy"] / rounds, 2), "wait_ms": round(1000 * s["wait"] / rounds, 2),
                                "busy": round(1 - s["idle"] / dt, 3), "entities": round(s["ents"] / rows, 1),
                                "entities_max_mean": round(s["ents_max"] / calls, 1), "native": True}) + "\n")

    def start(self) -> None:
        self.loop.start()

    def stop(self) -> None:
        self.loop.stop()

    def failed(self) -> str | None:
        return self.loop.error() if self.loop.failed() else None
