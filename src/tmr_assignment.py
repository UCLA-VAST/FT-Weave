from typing import List, Dict, Tuple


class MagicStateMatching:
    """
    Three-phase heuristic algorithm for matching required angles to available factories
    for RUS (Repeat-Until-Success) level logical qubits.

    Goal: All required_angles_per_row must be assigned to some available_factories_per_row.
    Each available_factories_per_row can hold multiple required_angles_per_row.
    """

    def __init__(self, required_angles_per_row, available_factories_per_row):
        """
        Initialize the algorithm with required angles and available factories.

        Args:
            required_angles_per_row: List where required_angles_per_row[i] is the number
                                     of required angles at row i (must all become 0)
            available_factories_per_row: List where available_factories_per_row[j] is the
                                         number of available factory slots at row j
                                         (can hold multiple required angles)
        """
        self.required_angles_original = required_angles_per_row[:]
        self.available_factories_original = available_factories_per_row[:]
        self.n_angles = len(required_angles_per_row)
        self.n_factories = len(available_factories_per_row)

        # Working copies
        self.required_angles = required_angles_per_row[:]
        self.available_factories = available_factories_per_row[:]

        # Assignments: assignments[i] = [(factory_row_j, amount, phase_type), ...]
        self.assignments = [[] for _ in range(self.n_angles)]

        # Log for tracking operations
        self.log = []

    def phase1_exact_match(self):
        """
        Phase 1: Exact one-to-one match
        If required_angles[i] = available_factories[j], assign required_angles[i] to available_factories[j]
        """
        phase_log = {
            "phase": 1,
            "description": "Phase 1: Exact one-to-one match",
            "matches": [],
        }

        for i in range(self.n_angles):
            for j in range(self.n_factories):
                if self.required_angles[i] > 0 and self.available_factories[j] > 0:
                    if self.available_factories[j] == self.required_angles[i]:
                        match_amount = self.required_angles[i]

                        # Record assignment
                        self.assignments[i].append((j, match_amount, "exact"))

                        # Update remaining capacity
                        self.required_angles[i] -= match_amount
                        self.available_factories[j] -= match_amount

                        # Log the match
                        phase_log["matches"].append(
                            {
                                "angle_row": i,
                                "factory_row": j,
                                "amount": match_amount,
                                "reason": "required_angles[{}] = available_factories[{}] = {}".format(
                                    i, j, match_amount
                                ),
                            }
                        )

        self.log.append(phase_log)

    def phase2_subset_sum(self):
        """
        Phase 2: Subset-sum matching
        If required_angles[i] + required_angles[j] = available_factories[k],
        assign both to available_factories[k], and vice versa
        """
        phase_log = {
            "phase": 2,
            "description": "Phase 2: Subset-sum matching",
            "matches": [],
        }

        for i in range(self.n_angles):
            for j in range(i + 1, self.n_angles):  # Avoid duplicate pairs
                for k in range(self.n_factories):
                    if (
                        self.required_angles[i] > 0
                        and self.required_angles[j] > 0
                        and self.available_factories[k] > 0
                    ):
                        if (
                            self.required_angles[i] + self.required_angles[j]
                            == self.available_factories[k]
                        ):
                            amount_i = self.required_angles[i]
                            amount_j = self.required_angles[j]

                            # Record assignments
                            self.assignments[i].append((k, amount_i, "subset-sum"))
                            self.assignments[j].append((k, amount_j, "subset-sum"))

                            # Update remaining capacity
                            self.available_factories[k] -= amount_i + amount_j
                            self.required_angles[i] = 0
                            self.required_angles[j] = 0

                            # Log the match
                            phase_log["matches"].append(
                                {
                                    "angle_rows": [i, j],
                                    "factory_row": k,
                                    "amounts": [amount_i, amount_j],
                                    "reason": "required_angles[{}] + required_angles[{}] = available_factories[{}] ({} + {} = {})".format(
                                        i, j, k, amount_i, amount_j, amount_i + amount_j
                                    ),
                                }
                            )

        self.log.append(phase_log)

    def phase3_best_fit(self):
        """
        Phase 3: Best-fit
        For remaining required_angles[i], find available_factories[j] >= required_angles[i]
        which leaves the least leftover space.
        If none fits completely, pick the row with largest capacity and split.

        Continue until all required angles are assigned (all become 0).
        """
        phase_log = {"phase": 3, "description": "Phase 3: Best-fit", "matches": []}

        # Keep assigning until all required angles are satisfied
        for i in range(self.n_angles):
            while self.required_angles[i] > 0:
                best_j = -1
                min_leftover = float("inf")

                # Try to find best fit (least leftover space)
                for j in range(self.n_factories):
                    if self.available_factories[j] >= self.required_angles[i]:
                        leftover = self.available_factories[j] - self.required_angles[i]
                        if leftover < min_leftover:
                            min_leftover = leftover
                            best_j = j

                # If no complete fit, pick largest capacity
                if best_j == -1:
                    max_capacity = 0
                    for j in range(self.n_factories):
                        if self.available_factories[j] > max_capacity:
                            max_capacity = self.available_factories[j]
                            best_j = j

                # Make assignment if valid factory found
                if best_j != -1 and self.available_factories[best_j] > 0:
                    amount = min(
                        self.required_angles[i], self.available_factories[best_j]
                    )

                    # Record assignment
                    self.assignments[i].append((best_j, amount, "best-fit"))

                    # Update remaining capacity
                    self.required_angles[i] -= amount
                    self.available_factories[best_j] -= amount

                    # Log the match
                    if min_leftover == float("inf"):
                        reason = "Split assignment (factory capacity={})".format(
                            self.available_factories[best_j] + amount
                        )
                    else:
                        reason = "Best fit with leftover: {}".format(min_leftover)

                    phase_log["matches"].append(
                        {
                            "angle_row": i,
                            "factory_row": best_j,
                            "amount": amount,
                            "reason": reason,
                        }
                    )
                else:
                    # No more capacity available
                    print(
                        "WARNING: Cannot assign remaining {} angles from row {}".format(
                            self.required_angles[i], i
                        )
                    )
                    break

        self.log.append(phase_log)

    def run(self):
        """Execute all three phases of the algorithm."""
        self.phase1_exact_match()
        self.phase2_subset_sum()
        self.phase3_best_fit()

    def validate_assignment(self):
        """Validate that all required angles have been assigned."""
        unassigned = [i for i, v in enumerate(self.required_angles) if v > 0]
        if unassigned:
            print(
                "\nWARNING: Not all angles assigned! Remaining in rows: {}".format(
                    unassigned
                )
            )
            return False
        return True

    def print_results(self):
        """Print detailed results of the matching algorithm."""
        print("=" * 80)
        print("MAGIC STATE MATCHING ALGORITHM - RESULTS")
        print("=" * 80)
        print("\nInitial Configuration:")
        print("  Required Angles per Row:     {}".format(self.required_angles_original))
        print(
            "  Available Factories per Row: {}".format(
                self.available_factories_original
            )
        )
        print(
            "  Total Required: {}, Total Available: {}".format(
                sum(self.required_angles_original),
                sum(self.available_factories_original),
            )
        )

        if sum(self.required_angles_original) > sum(self.available_factories_original):
            print("  WARNING: Insufficient factory capacity!")
        print()

        # Print phase-by-phase log
        for phase in self.log:
            print("\n{}".format(phase["description"]))
            print("-" * 80)
            if phase["matches"]:
                for match in phase["matches"]:
                    if "angle_rows" in match:
                        # Subset-sum match
                        print(
                            "  Angle rows {} and {} -> Factory row {} ({} + {} = {})".format(
                                match["angle_rows"][0],
                                match["angle_rows"][1],
                                match["factory_row"],
                                match["amounts"][0],
                                match["amounts"][1],
                                sum(match["amounts"]),
                            )
                        )
                        print("    Reason: {}".format(match["reason"]))
                    else:
                        # Exact or best-fit match
                        print(
                            "  Angle row {} -> Factory row {} ({} angles)".format(
                                match["angle_row"],
                                match["factory_row"],
                                match["amount"],
                            )
                        )
                        print("    Reason: {}".format(match["reason"]))
            else:
                print("  No matches in this phase")

        # Print final assignments
        print("\n" + "=" * 80)
        print("FINAL ASSIGNMENTS (Angles -> Factories)")
        print("=" * 80)
        for i in range(self.n_angles):
            total_assigned = sum(amount for _, amount, _ in self.assignments[i])
            status = (
                "OK"
                if total_assigned == self.required_angles_original[i]
                else "INCOMPLETE"
            )
            print(
                "\nAngle Row {} [{}] (required {}, assigned {}):".format(
                    i, status, self.required_angles_original[i], total_assigned
                )
            )
            if self.assignments[i]:
                for factory, amount, phase_type in self.assignments[i]:
                    print(
                        "  -> Factory row {}: {} angles ({})".format(
                            factory, amount, phase_type
                        )
                    )
            else:
                print("  -> No assignments")

        # Print factory utilization
        print("\n" + "=" * 80)
        print("FACTORY UTILIZATION")
        print("=" * 80)
        for j in range(self.n_factories):
            # Find which angle rows are assigned to this factory
            assigned_from = []
            total_used = 0
            for i in range(self.n_angles):
                for factory, amount, _ in self.assignments[i]:
                    if factory == j:
                        assigned_from.append("row{}({})".format(i, amount))
                        total_used += amount

            remaining = self.available_factories_original[j] - total_used
            status = "FULL" if remaining == 0 else "({} remaining)".format(remaining)
            print(
                "Factory row {}: {}/{} used {}".format(
                    j, total_used, self.available_factories_original[j], status
                )
            )
            if assigned_from:
                print("  Holds angles from: {}".format(", ".join(assigned_from)))

        # Validation
        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)
        all_assigned = self.validate_assignment()
        if all_assigned:
            print("SUCCESS: All required angles have been assigned to factories!")
        else:
            print("FAILURE: Some angles remain unassigned!")

        unassigned_angles = sum(self.required_angles)
        unused_factories = sum(self.available_factories)
        print("  Unassigned angles: {}".format(unassigned_angles))
        print("  Unused factory capacity: {}".format(unused_factories))
        print()
