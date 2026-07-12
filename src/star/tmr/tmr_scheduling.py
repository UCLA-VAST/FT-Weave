from .angle_collection import get_angles_for_preparation
from .factory_angle_assignment.integration import optimize_factory_angle_assignment
from .tmr_assignment import assign_factories_for_batch
from src.ds import FactoryPool, QubitAngleTracker, Factory


def schedule_tmr_round(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    code_distance: int,
    column_based_placement: bool = True,
    tmr_assignment_method: str = "matching",
    factories_list: list[int] | None = None,
    prepare_lookahead_angles: bool = True,
    allocation_level_range: range | None = None,
) -> list[Factory]:
    """Schedule TMR preparation for idle factories."""
    if factories_list is None:
        factories_ready_for_rz = factory_pool.get_tmr_before_rz_factories()
    else:
        factories_ready_for_rz = [
            factory_pool.get_factory_by_id(factory_id) for factory_id in factories_list
        ]
    if not factories_ready_for_rz or not qubit_trackers:
        return []

    if tmr_assignment_method == "stochastic_coverage":
        assignment = optimize_factory_angle_assignment(
            factories_ready_for_rz,
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
            code_distance,
            include_lookahead=prepare_lookahead_angles,
            allocation_level_range=allocation_level_range,
        )
        if assignment is None:
            return []
        return factories_ready_for_rz

    if tmr_assignment_method not in ("matching", "naive"):
        raise ValueError(f"Unknown tmr_assignment_method: {tmr_assignment_method}")

    # Get angles to prepare
    batch_angles = get_angles_for_preparation(
        qubit_trackers,
        len(factories_ready_for_rz),
        code_distance=code_distance,
        prepare_lookahead_angles=prepare_lookahead_angles,
        allocation_level_range=allocation_level_range,
    )

    if not batch_angles:
        return []
    # Assign factories
    assign_factories_for_batch(
        factory_pool,
        factories_ready_for_rz,
        qubit_trackers,
        logic_qubit_locations,
        batch_angles,
        code_distance=code_distance,
        column=column_based_placement,
        method=tmr_assignment_method,
    )

    return factories_ready_for_rz
