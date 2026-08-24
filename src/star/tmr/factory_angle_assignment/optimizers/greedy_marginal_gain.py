"""Greedy marginal-gain optimizer for factory-to-angle assignment."""

from __future__ import annotations

from ..objectives.base import AssignmentObjective
from ..problem import AssignmentProblem, normalize_angle


def optimize_greedy(
    problem: AssignmentProblem,
    objective: AssignmentObjective,
    initial_assignment: dict[int, float],
    *,
    max_rounds: int | None = None,
) -> dict[int, float]:
    """
    Iteratively move one factory to the angle with largest cost reduction.

    Stops when no single-factory move improves the objective.
    """
    assignment = {
        factory_id: normalize_angle(angle)
        for factory_id, angle in initial_assignment.items()
    }
    rounds = 0

    while max_rounds is None or rounds < max_rounds:
        best_delta = 0.0
        best_factory_id: int | None = None
        best_angle: float | None = None

        for factory_id in problem.factory_ids:
            current_angle = assignment[factory_id]
            for candidate_angle in problem.angles:
                if candidate_angle == current_angle:
                    continue
                delta = objective.delta_cost(assignment, factory_id, candidate_angle)
                if delta < best_delta:
                    best_delta = delta
                    best_factory_id = factory_id
                    best_angle = candidate_angle

        if best_factory_id is None or best_angle is None:
            break

        assignment[best_factory_id] = best_angle
        rounds += 1

    return assignment
