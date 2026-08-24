"""Initial factory-to-angle assignments for greedy optimization."""

from __future__ import annotations

from collections import defaultdict

from ..angle_collection import integer_allocation

from .problem import AssignmentProblem, normalize_angle


def demand_proportional_initial(
    problem: AssignmentProblem,
) -> dict[int, float]:
    """
    Allocate factories to angles proportional to lookahead-weighted demand.

    Assigns each angle slot to the geographically closest available factory.
    """
    angle_weights: dict[float, float] = defaultdict(float)
    for term in problem.coverage_terms:
        angle_weights[term.angle] += term.weight

    if not angle_weights:
        return {factory_id: problem.angles[0] for factory_id in problem.factory_ids}

    allocation = integer_allocation(problem.n_factories, dict(angle_weights))

    assignment: dict[int, float] = {}
    available = set(problem.factory_ids)

    for angle in sorted(allocation.keys()):
        count = allocation[angle]
        if count <= 0:
            continue

        qubit_indices = [
            problem.coverage_terms[i].qubit_index
            for i in problem.terms_by_angle.get(angle, [])
        ]

        def factory_score(factory_id: int) -> int:
            fi = problem.factory_index[factory_id]
            return min(problem.distances[fi, qi] for qi in qubit_indices)

        ranked = sorted(available, key=factory_score)
        for factory_id in ranked[:count]:
            assignment[factory_id] = angle
            available.remove(factory_id)

    if available:
        fallback = max(angle_weights, key=angle_weights.get)
        for factory_id in available:
            assignment[factory_id] = fallback

    return assignment


def nearest_qubit_angle_initial(
    problem: AssignmentProblem,
) -> dict[int, float]:
    """Assign each factory to the angle of its nearest logical qubit."""
    assignment: dict[int, float] = {}

    for factory_id in problem.factory_ids:
        fi = problem.factory_index[factory_id]
        best_qi: int | None = None
        best_dist = float("inf")

        for qi, qubit_id in enumerate(problem.qubit_ids):
            dist = int(problem.distances[fi, qi])
            if dist < best_dist or (
                dist == best_dist
                and (best_qi is None or qubit_id < problem.qubit_ids[best_qi])
            ):
                best_dist = dist
                best_qi = qi

        if best_qi is None:
            assignment[factory_id] = problem.angles[0]
            continue

        qubit_id = problem.qubit_ids[best_qi]
        angle = next(
            term.angle for term in problem.coverage_terms if term.qubit_id == qubit_id
        )
        assignment[factory_id] = normalize_angle(angle)

    return assignment
