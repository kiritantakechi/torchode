from collections.abc import Callable
from typing import Any, Final

import torch
import torch.nn as nn

from .problems import InitialValueProblem
from .typing import DataTensor, TimeTensor


class ODETerm(nn.Module):
    with_args: Final[bool]
    with_stats: Final[bool]

    def __init__(
        self,
        f: Callable[[TimeTensor, DataTensor], DataTensor],
        *,
        with_stats: bool = True,
        with_args: bool = False,
    ):
        """Initialize an ODE term of the form `dy/dt = f(t, y)`.

        Arguments
        ---------
        f
            Right-hand side of the ODE
        with_stats
            If true, track statistics such as the number of function evaluations. If your
            dynamics are very fast to evaluate, disabling this can improve the performance
            of the solver by 1-2%.
        with_args
            If true, `f` will be passed an additional static argument in the third
            position.
        """

        super().__init__()

        self.f = f
        self.with_stats = with_stats
        self.with_args = with_args

    def init(self, problem: InitialValueProblem, stats: dict[str, Any]):
        if not self.with_stats:
            return
        # There is no reason for these to be on the GPU
        stats["n_f_evals"] = torch.zeros(
            problem.batch_size, device="cpu", dtype=torch.long
        )

    def vf(
        self, t: TimeTensor, y: DataTensor, stats: dict[str, Any], args: Any
    ) -> DataTensor:
        """Evaluate the vector field."""
        if self.with_stats:
            stats["n_f_evals"].add_(1)

        if self.with_args:
            return self.f(t, y, args)
        else:
            return self.f(t, y)
