import pytest
import torch
from problems import get_problem
from pytest import approx

from torchode import AutoDiffAdjoint, Dopri5, Euler, Heun, IntegralController, Tsit5


@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5, Euler])
def test_can_be_compiled(step_method):
    _, term, problem = get_problem("sine", [[0.1, 0.15, 1.0], [1.0, 1.9, 2.0]])
    step_size_controller = IntegralController(1e-3, 1e-3, term=term)
    adjoint = AutoDiffAdjoint(step_method(term), step_size_controller)
    compiled = torch.compile(adjoint)

    dt0 = torch.tensor([0.01, 0.01]) if step_method is Euler else None
    solution = adjoint.solve(problem, dt0=dt0)
    solution_compiled = compiled.solve(problem, dt0=dt0)

    assert solution.ts == approx(solution_compiled.ts)
    assert solution.ys == approx(solution_compiled.ys, abs=1e-3, rel=1e-3)


@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5, Euler])
def test_passing_term_dynamically_equals_fixed_term(step_method):
    _, term, problem = get_problem("sine", [[0.1, 0.15, 1.0], [1.0, 1.9, 2.0]])

    dt0 = torch.tensor([0.01, 0.01]) if step_method is Euler else None

    controller = IntegralController(1e-3, 1e-3)
    adjoint = AutoDiffAdjoint(step_method(None), controller)
    solution = adjoint.solve(problem, term, dt0=dt0)

    controller_compiled = IntegralController(1e-3, 1e-3, term=term)
    adjoint_compiled = AutoDiffAdjoint(step_method(term), controller_compiled)
    solution_compiled = torch.compile(adjoint_compiled).solve(problem, dt0=dt0)

    assert solution.ts == approx(solution_compiled.ts)
    assert solution.ys == approx(solution_compiled.ys, abs=1e-3, rel=1e-3)
