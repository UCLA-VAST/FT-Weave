from src.tmr.angle_collection import get_angles_for_preparation
from src.tmr.tmr_assignment import assign_factories_for_batch
from src.ds import FactoryPool, QubitAngleTracker, Factory


def schedule_tmr_round(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    column_based_placement: bool = True,
    tmr_assignment_method: str = "matching",
) -> list[Factory]:
    """Schedule TMR preparation for idle factories."""
    idle_factories = factory_pool.get_tmr_before_rz_factories()
    if not idle_factories or not qubit_trackers:
        return []

    # Get angles to prepare
    batch_angles = get_angles_for_preparation(qubit_trackers, len(idle_factories))

    if not batch_angles:
        return []

    # Assign factories
    assign_factories_for_batch(
        factory_pool,
        idle_factories,
        qubit_trackers,
        logic_qubit_locations,
        batch_angles,
        column=column_based_placement,
        method=tmr_assignment_method,
    )

    return idle_factories
