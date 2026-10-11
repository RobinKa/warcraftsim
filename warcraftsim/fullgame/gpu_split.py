"""The GPU's SMs split between the learner and the inference loop (CUDA green contexts; native Linux, driver
12.8+). In one process both ran on all SMs: the learner's long kernels held the SMs while the inference
loop's small, latency-bound calls queued behind them (a round ~10 ms in the run, ~2 ms alone). With a part
of their own the inference calls never wait for the learner's, and the learner's kernels run on the rest.

torch.cuda.GreenContext takes the first N SMs and drops the remainder: two of them overlap. Here one split
gives both parts, each a green context; work goes to a part through streams made in it. CUDA graphs captured
as usual replay on such a stream within its part (scripts/check_gpu_split.py).
"""

from __future__ import annotations

import torch
from torch.cuda._utils import _check_cuda_bindings as _check, _cuda_bindings_driver as _drv


class GpuSplit:
    """`small` SMs for one part (the inference loop's), the rest for the other (the learner's)."""

    def __init__(self, small: int, device_id: int = 0) -> None:
        torch.cuda.init()
        torch.cuda.synchronize(device_id)  # (the primary context exists)
        dev = _check(_drv.cuDeviceGet(device_id))
        sm = _check(_drv.cuDeviceGetDevResource(dev, _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.total = sm.sm.smCount
        groups, n, rest = _check(_drv.cuDevSmResourceSplitByCount(1, sm, 0, small))
        assert n == 1, n
        self.device_id = device_id
        self.small_sms, self.rest_sms = groups[0].sm.smCount, rest.sm.smCount
        self._ctxs = []
        for res in (groups[0], rest):
            desc = _check(_drv.cuDevResourceGenerateDesc([res], 1))
            self._ctxs.append(_check(_drv.cuGreenCtxCreate(desc, dev, _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)))
        self._streams = []

    def stream(self, part: int, priority: int = 0) -> torch.cuda.ExternalStream:
        """A new stream in part 0 (the small one) or 1 (the rest). priority: as torch's (lower = higher)."""
        s = _check(_drv.cuGreenCtxStreamCreate(self._ctxs[part], _drv.CUstream_flags.CU_STREAM_NON_BLOCKING, priority))
        self._streams.append(s)  # (kept: the ExternalStream does not own it)
        return torch.cuda.ExternalStream(int(s), self.device_id)
