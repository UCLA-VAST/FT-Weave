from typing import Any
from collections import defaultdict


def validate_assignment(result: dict) -> bool:
    """
    Validate that all required angles have been assigned.

    Args:
        result: dictionary returned from run_tmr_row_assignment()
    Returns:
        all_valid: True if all angles properly assigned, False otherwise
    """
    all_valid = True
    for i in result["required_angles_original"].keys():
        total_assigned = sum(amount for _, amount, _ in result["assignments"][i])
        if total_assigned != result["required_angles_original"][i]:
            print(
                "\nWARNING: Angle row {} incomplete! Required: {}, Assigned: {}".format(
                    i, result["required_angles_original"][i], total_assigned
                )
            )
            all_valid = False
    return all_valid


def print_tmr_assignment_results(result: dict):
    """
    Print detailed results of the row matching algorithm for tmr.

    Args:
        result: dictionary returned from run_matching()
    Returns:
        None (prints to console)
    """
    print("=" * 80)
    print("MAGIC STATE MATCHING ALGORITHM - RESULTS")
    print("=" * 80)
    print("\nInitial Configuration:")
    print(
        "  Required Angles per Row:     {}".format(result["required_angles_original"])
    )
    print(
        "  Available Factories per Row: {}".format(
            result["available_factories_original"]
        )
    )
    print(
        "  Total Required: {}, Total Available: {}".format(
            sum(result["required_angles_original"].values()),
            sum(result["available_factories_original"].values()),
        )
    )

    if sum(result["required_angles_original"].values()) > sum(
        result["available_factories_original"].values()
    ):
        print("  WARNING: Insufficient factory capacity!")
    print()

    # Print final assignments
    print("\n" + "=" * 80)
    print("FINAL ASSIGNMENTS (Angles -> Factories)")
    print("=" * 80)
    for i in result["required_angles_original"].keys():
        total_assigned = sum(amount for _, amount, _ in result["assignments"][i])
        status = (
            "OK"
            if total_assigned == result["required_angles_original"][i]
            else "INCOMPLETE"
        )
        print(
            "\nAngle Row {} [{}] (required {}, assigned {}):".format(
                i, status, result["required_angles_original"][i], total_assigned
            )
        )
        if result["assignments"][i]:
            for factory, amount, phase_type in result["assignments"][i]:
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
    for j in result["available_factories_original"].keys():
        # Find which angle rows are assigned to this factory
        assigned_from = []
        total_used = 0
        for i in result["required_angles_original"].keys():
            for factory, amount, _ in result["assignments"][i]:
                if factory == j:
                    assigned_from.append("row{}({})".format(i, amount))
                    total_used += amount

        remaining = result["available_factories_original"][j] - total_used
        status = "FULL" if remaining == 0 else "({} remaining)".format(remaining)
        print(
            "Factory row {}: {}/{} used {}".format(
                j, total_used, result["available_factories_original"][j], status
            )
        )
        if assigned_from:
            print("  Holds angles from: {}".format(", ".join(assigned_from)))

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    all_assigned = validate_assignment(result)
    if all_assigned:
        print("SUCCESS: All required angles have been assigned to factories!")
    else:
        print("FAILURE: Some angles remain unassigned!")

    unused_factories = sum(result["available_factories"].values())
    print("  Unused factory capacity: {}".format(unused_factories))
    print()


def analyze_execution_log(
    execution_log: list[tuple],
    n_factories: int,
) -> dict[str, Any]:
    """
    Analyze an execution log into a plotting-friendly profile.

    The returned profile mirrors information used by `plot_circuit_execution` but
    is compact and numeric so it can be consumed without rendering a large figure.

    Returns a dict with:
      - total_time: overall circuit end time
      - ops: mapping op -> {count, total_time, circuit_time, avg_time, max_time}
        where circuit_time = time occupied by op in circuit (accounting for parallelism)
      - per_factory: mapping factory_id -> {timeline: [entries], busy_time, idle_time, ops}
      - movements: movement-pair stats mapping move_vecs -> {count, total_time, avg, max}
      - failures: overall failure count and per-factory failure counts

    Each timeline entry is (start, end, operation, value, move_vecs_or_None).
    """
    if not execution_log:
        return {
            "total_time": 0,
            "ops": {},
            "per_factory": {},
            "movements": {},
            "failures": {"total": 0, "by_factory": {}},
        }

    # Normalize entries and build per-factory timelines
    per_factory: dict[Any, dict[str, Any]] = {}
    ops: dict[str, dict[str, float]] = {}
    op_intervals: dict[str, list[tuple]] = defaultdict(
        list
    )  # op -> [(start, end), ...]
    movements: dict[tuple, dict[str, float]] = {}
    return_movements: dict[tuple, dict[str, float]] = {}
    rus_failures_total = 0
    rus_failures_by_factory = defaultdict(int)
    tmr_failures_total = 0
    tmr_failures_by_factory = defaultdict(int)
    overall_end = 0
    qubit_cnot_counts = [0 for i in range(n_factories)]
    # Track AOD utilization: aod_idx -> {move_intervals: [...], total_time: 0}
    aod_utilization: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"move_intervals": [], "total_time": 0}
    )

    for e in execution_log:
        # Accept either 5-, 6-, or 7-element tuples (with optional aod_assignment)
        if len(e) == 5:
            start, end, factory_id, operation, value = e
            move_vecs = None
            aod_assignment = 0
        elif len(e) == 6:
            start, end, factory_id, operation, value, move_vecs = e
            aod_assignment = 0
        else:
            start, end, factory_id, operation, value, move_vecs, aod_assignment = e

        overall_end = max(overall_end, end)

        # Record op stats
        dur = max(0, end - start)
        s = ops.setdefault(operation, {"count": 0, "total_time": 0, "max_time": 0})
        s["count"] += 1
        s["total_time"] += dur
        s["max_time"] = max(s["max_time"], dur)

        # Track intervals for circuit_time calculation
        op_intervals[operation].append((start, end))

        # Per-factory timeline
        fid = factory_id
        timeline = per_factory.setdefault(
            fid, {"timeline": [], "busy_time": 0, "ops": {}}
        )
        timeline["timeline"].append((start, end, operation, value, move_vecs))
        timeline["busy_time"] += dur
        opf = timeline["ops"].setdefault(operation, {"count": 0, "total_time": 0})
        opf["count"] += 1
        opf["total_time"] += dur

        if operation == "CNOT":
            qubit_cnot_counts[value] += 1
        # Movements grouping - separate move and return_move
        if operation == "move":
            key = tuple(move_vecs) if move_vecs is not None else ("unknown",)
            mv = movements.setdefault(key, {"count": 0, "total_time": 0, "max_time": 0})
            mv["count"] += 1
            mv["total_time"] += dur
            mv["max_time"] = max(mv["max_time"], dur)
            # Track AOD utilization
            aod_utilization[aod_assignment]["move_intervals"].append((start, end))
            aod_utilization[aod_assignment]["total_time"] += dur
        elif operation == "return_move":
            key = tuple(move_vecs) if move_vecs is not None else ("unknown",)
            mv = return_movements.setdefault(
                key, {"count": 0, "total_time": 0, "max_time": 0}
            )
            mv["count"] += 1
            mv["total_time"] += dur
            mv["max_time"] = max(mv["max_time"], dur)
            # Track AOD utilization
            aod_utilization[aod_assignment]["move_intervals"].append((start, end))
            aod_utilization[aod_assignment]["total_time"] += dur

        # Failures
        if operation == "RUS_fail":
            rus_failures_total += 1
            rus_failures_by_factory[fid] += 1
        elif operation == "TMR_fail":
            tmr_failures_total += 1
            tmr_failures_by_factory[fid] += 1

    # Compute circuit_time for each operation (merge overlapping intervals)
    def merge_intervals(intervals: list[tuple]) -> int:
        """Merge overlapping intervals and return total time covered."""
        if not intervals:
            return 0
        intervals = list(set(intervals))
        intervals.sort()
        total = 0
        for start, end in intervals:
            total += end - start

        # merged_start, merged_end = intervals[0]
        # total = 0
        # for start, end in intervals[1:]:
        #     if start <= merged_end:
        #         merged_end = max(merged_end, end)
        #     else:
        #         total += merged_end - merged_start
        #         merged_start, merged_end = start, end
        # total += merged_end - merged_start
        return total

    for op_name in ops.keys():
        circuit_time = merge_intervals(op_intervals[op_name])
        ops[op_name]["circuit_time"] = circuit_time

    # Finalize averages
    for k, v in ops.items():
        v["avg_time"] = v["total_time"] / v["count"] if v["count"] else 0
    for k, v in movements.items():
        v["avg_time"] = v["total_time"] / v["count"] if v["count"] else 0
    for k, v in return_movements.items():
        v["avg_time"] = v["total_time"] / v["count"] if v["count"] else 0

    # Compute idle times (if n_factories provided use that, else infer from per_factory keys)
    if n_factories is None:
        # infer number of factories as max key + 1 when keys are ints; otherwise skip
        try:
            int_keys = [k for k in per_factory.keys() if isinstance(k, int) and k >= 0]
            n_factories = max(int_keys) + 1 if int_keys else None
        except Exception:
            n_factories = None

    if n_factories is not None:
        # ensure all factories from 0..n_factories-1 are present
        for fid in range(n_factories):
            if fid not in per_factory:
                per_factory[fid] = {"timeline": [], "busy_time": 0, "ops": {}}

    for fid, info in per_factory.items():
        info["timeline"].sort(key=lambda x: x[0])
        info["idle_time"] = overall_end - info.get("busy_time", 0)

    profile = {
        "total_time": overall_end,
        "ops": ops,
        "per_factory": per_factory,
        "movements": movements,
        "return_movements": return_movements,
        "aod_utilization": dict(aod_utilization),
        "failures": {
            "tmr_total": tmr_failures_total,
            "tmr_by_factory": tmr_failures_by_factory,
            "rus_total": rus_failures_total,
            "rus_by_factory": rus_failures_by_factory,
        },
        "qubit_cnot_counts": qubit_cnot_counts,
    }

    return profile


def print_execution_profile(profile: dict[str, Any], top_n_pairs: int = 0) -> None:
    """Print concise execution profile produced by `analyze_execution_log`."""
    total_time = profile.get("total_time", 0)

    print("=" * 70)
    print(f"TOTAL CIRCUIT EXECUTION TIME: {total_time} moments")
    print("=" * 70)
    print("EXECUTION PROFILE SUMMARY")
    print(f"- Total circuit time: {total_time}")

    ops = profile.get("ops", {})
    print(f"- Operations: {len(ops)} types")

    # Define operation order for output
    op_order = [
        "SE",
        "Rz",
        "move",
        "return_move",
        "CNOT",
        "RUS_success",
        "RUS_fail",
        "TMR_fail",
        "Barrier",
    ]

    # Operations with timing details
    timing_ops = {"SE", "Rz", "CNOT"}
    move_ops = {"move", "return_move"}
    # Operations with count only
    count_only_ops = {"RUS_success", "RUS_fail", "TMR_fail"}

    for op_name in op_order:
        if op_name not in ops:
            continue

        s = ops[op_name]

        if op_name in timing_ops:
            circuit_time = s.get("circuit_time", 0)
            print(
                f"  - {op_name}: circuit_time={circuit_time}, total={s['total_time']}"
            )
        elif op_name in move_ops:
            circuit_time = s.get("circuit_time", 0)
            print(
                f"  - {op_name}: circuit_time={circuit_time}, total={s['total_time']}, avg={s.get('avg_time',0):.3f}, max={s['max_time']}"
            )
        elif op_name in count_only_ops:
            print(f"  - {op_name}: count={s['count']}")
