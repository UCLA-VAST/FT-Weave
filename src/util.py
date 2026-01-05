from .config import (
    TMR_P,
    TMR_Q,
    TMR_PREPARATION_TIME,
    CNOT_TIME,
    SE_TIME,
    LOOKAHEAD_THRESHOLD,
    LOOKAHEAD_LEVEL,
)


def write_execution_log(
    execution_log: list,
    start_time: int,
    factory_id: int,
    operation: str,
    qubit: int | None,
    movement_time: int = 0,
):
    if operation == "SE":
        end_time = start_time + SE_TIME
    elif operation == "CNOT":
        end_time = start_time + CNOT_TIME
    elif operation == "Rz":
        end_time = start_time + 1
    elif operation == "move":
        end_time = start_time + movement_time
    else:
        end_time = start_time
    execution_log.append(
        (
            start_time,
            end_time,
            factory_id,
            operation,
            qubit,
        )
    )


def write_injection_log(
    execution_log: list,
    start_time: int,
    factory_id: int,
    result: str,
    qubit: int | None,
):
    write_execution_log(
        execution_log,
        start_time,
        factory_id,
        "CNOT",
        qubit,
    )
    write_execution_log(
        execution_log,
        start_time + SE_TIME,
        factory_id,
        "SE",
        qubit,
    )
    start_time += CNOT_TIME + SE_TIME
    write_execution_log(
        execution_log,
        start_time,
        factory_id,
        result,
        qubit,
    )


def validate_assignment(result: dict) -> bool:
    """
    Validate that all required angles have been assigned.

    Args:
        result: dictionary returned from run_tmr_row_assignment()
    Returns:
        all_valid: True if all angles properly assigned, False otherwise
    """
    all_valid = True
    for i in range(result["n_angles"]):
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
            sum(result["required_angles_original"]),
            sum(result["available_factories_original"]),
        )
    )

    if sum(result["required_angles_original"]) > sum(
        result["available_factories_original"]
    ):
        print("  WARNING: Insufficient factory capacity!")
    print()

    # Print final assignments
    print("\n" + "=" * 80)
    print("FINAL ASSIGNMENTS (Angles -> Factories)")
    print("=" * 80)
    for i in range(result["n_angles"]):
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
    for j in range(result["n_factories"]):
        # Find which angle rows are assigned to this factory
        assigned_from = []
        total_used = 0
        for i in range(result["n_angles"]):
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

    unused_factories = sum(result["available_factories"])
    print("  Unused factory capacity: {}".format(unused_factories))
    print()
