"""Integration with TMR scheduling and factory pool state."""

from __future__ import annotations

from ....ds import FactoryPool, QubitAngleTracker
from src.star.simulation import calculate_success_rate

from .initial import demand_proportional_initial
from .objectives.stochastic_coverage import StochasticCoverageObjective
from .optimizers.greedy_marginal_gain import optimize_greedy
from .problem import AssignmentProblem, build_assignment_problem, normalize_angle


def _administrative_qubit_for_factory(
    problem: AssignmentProblem,
    factory_id: int,
    angle: float,
) -> int:
    """Pick nearest qubit needing ``angle`` for tracker bookkeeping."""
    fi = problem.factory_index[factory_id]
    candidates = problem.qubits_by_angle.get(angle, [])
    if not candidates:
        return problem.qubit_ids[0]

    best_qubit = candidates[0]
    best_dist = int(problem.distances[fi, problem.qubit_index[best_qubit]])

    for qubit_id in candidates[1:]:
        qi = problem.qubit_index[qubit_id]
        dist = int(problem.distances[fi, qi])
        if dist < best_dist or (dist == best_dist and qubit_id < best_qubit):
            best_dist = dist
            best_qubit = qubit_id

    return best_qubit


def apply_factory_angle_assignment(
    assignment: dict[int, float],
    problem: AssignmentProblem,
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    code_distance: int,
) -> None:
    """Write factory->angle assignment into pool and qubit trackers."""
    for factory_id, angle in assignment.items():
        angle = normalize_angle(angle)
        qubit_id = _administrative_qubit_for_factory(problem, factory_id, angle)
        success_rate = calculate_success_rate(angle, code_distance)
        qubit_trackers[qubit_id].add_factory(factory_id, angle)
        factory_pool.assign_factory(factory_id, angle, qubit_id, success_rate)


def optimize_factory_angle_assignment(
    factories: list,
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    code_distance: int,
    *,
    include_lookahead: bool,
    allocation_level_range: range | None = None,
) -> dict[int, float] | None:
    """Run stochastic-coverage greedy optimization and apply the result."""
    problem = build_assignment_problem(
        factories,
        qubit_trackers,
        logic_qubit_locations,
        code_distance,
        include_lookahead=include_lookahead,
        allocation_level_range=allocation_level_range,
    )
    if problem is None:
        return None

    objective = StochasticCoverageObjective(problem)
    initial = demand_proportional_initial(problem)
    assignment = optimize_greedy(problem, objective, initial)
    apply_factory_angle_assignment(
        assignment,
        problem,
        factory_pool,
        qubit_trackers,
        code_distance,
    )
    return assignment
