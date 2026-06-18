from types import NoneType
from typing import Any, NamedTuple

import torch
import torch.nn as nn

from ..interpolation import LinearInterpolation
from ..problems import InitialValueProblem
from ..terms import ODETerm
from ..typing import AcceptTensor, DataTensor, StatusTensor, TimeTensor
from .base import StepResult


class LinearInterpolationData(NamedTuple):
    t0: TimeTensor
    dt: TimeTensor
    y0: DataTensor
    y1: DataTensor


class Euler(nn.Module):
    def __init__(self, term: ODETerm | None):
        super().__init__()

        self.term = term

    def init(
        self,
        term: ODETerm | None,
        problem: InitialValueProblem,
        f0: DataTensor | None,
        *,
        stats: dict[str, Any],
        args: Any,
    ) -> NoneType:
        return None

    def step(
        self,
        term: ODETerm | None,
        running: AcceptTensor,
        y0: DataTensor,
        t0: TimeTensor,
        dt: TimeTensor,
        state: NoneType,
        *,
        stats: dict[str, Any],
        args: Any,
    ) -> tuple[StepResult, LinearInterpolationData, NoneType, StatusTensor | None]:
        term = term if term is not None else self.term
        assert term is not None

        # Convert dt into the data dtype for dtype stability
        dt_data = dt.to(dtype=y0.dtype)

        y1 = torch.addcmul(y0, dt_data[:, None], term.vf(t0, y0, stats, args))

        return (
            StepResult(y1, None),
            LinearInterpolationData(t0, dt, y0, y1),
            state,
            None,
        )

    def merge_states(
        self, accept: AcceptTensor, current: NoneType, previous: NoneType
    ) -> NoneType:
        return None

    def convergence_order(self):
        return 1

    def build_interpolation(self, data: LinearInterpolationData):
        return LinearInterpolation(data.t0, data.dt, data.y0, data.y1)
