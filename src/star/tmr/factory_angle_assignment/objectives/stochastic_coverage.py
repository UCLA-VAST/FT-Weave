"""Stochastic coverage objective for factory-to-angle assignment."""

from __future__ import annotations

from collections import defaultdict

from ..problem import AssignmentProblem, normalize_angle


def expected_distance(
    factory_distances: list[int],
    p_theta: float,
    r_max: int,
) -> float:
    """
    E[D_q] = sum_{r=0}^{R_max} (1 - p_theta)^{N_q(r)}.

    N_q(r) counts factories whose Manhattan distance to q is <= r.
    """
    if not factory_distances:
        return float(r_max + 1)

    cost = 0.0
    sorted_dists = sorted(factory_distances)
    n_factories = len(sorted_dists)
    idx = 0
    n_within = 0

    for r in range(r_max + 1):
        while idx < n_factories and sorted_dists[idx] <= r:
            n_within += 1
            idx += 1
        cost += (1.0 - p_theta) ** n_within

    return cost


class StochasticCoverageObjective:
    """Weighted sum of per-qubit expected nearest-successful-factory distances."""

    def __init__(self, problem: AssignmentProblem) -> None:
        self.problem = problem

    def _factories_by_angle(
        self, assignment: dict[int, float]
    ) -> dict[float, list[int]]:
        """Map angle -> list of factory matrix indices assigned to that angle."""
        by_angle: dict[float, list[int]] = defaultdict(list)
        for factory_id, angle in assignment.items():
            by_angle[normalize_angle(angle)].append(
                self.problem.factory_index[factory_id]
            )
        return by_angle

    def _term_contribution(
        self,
        term_index: int,
        factories_by_angle: dict[float, list[int]],
    ) -> float:
        problem = self.problem
        term = problem.coverage_terms[term_index]
        factory_indices = factories_by_angle.get(term.angle, [])
        if not factory_indices:
            return term.weight * float(problem.r_max + 1)

        dists = [int(problem.distances[fi, term.qubit_index]) for fi in factory_indices]
        p_theta = problem.p_theta[term.angle]
        return term.weight * expected_distance(dists, p_theta, problem.r_max)

    def total_cost(self, assignment: dict[int, float]) -> float:
        factories_by_angle = self._factories_by_angle(assignment)
        return sum(
            self._term_contribution(term.term_index, factories_by_angle)
            for term in self.problem.coverage_terms
        )

    def delta_cost(
        self,
        assignment: dict[int, float],
        factory_id: int,
        new_angle: float,
    ) -> float:
        old_angle = normalize_angle(assignment[factory_id])
        new_angle = normalize_angle(new_angle)
        if old_angle == new_angle:
            return 0.0

        factories_by_angle_old = self._factories_by_angle(assignment)
        factories_by_angle_new = {
            angle: list(indices) for angle, indices in factories_by_angle_old.items()
        }

        fi = self.problem.factory_index[factory_id]
        old_list = factories_by_angle_new.get(old_angle, [])
        factories_by_angle_new[old_angle] = [idx for idx in old_list if idx != fi]
        if not factories_by_angle_new[old_angle]:
            del factories_by_angle_new[old_angle]

        factories_by_angle_new.setdefault(new_angle, []).append(fi)

        affected_term_indices: set[int] = set()
        for angle in (old_angle, new_angle):
            affected_term_indices.update(self.problem.terms_by_angle.get(angle, []))

        delta = 0.0
        for term_index in affected_term_indices:
            delta += self._term_contribution(
                term_index, factories_by_angle_new
            ) - self._term_contribution(term_index, factories_by_angle_old)

        return delta
