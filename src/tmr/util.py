from src.rus.rus_assignment import AngleFactoryIndex
from src.ds import QubitAngleTracker, Factory, FactoryPool, FactoryState


def build_angle_factory_index_from_tmr_results(
    factory_pool: FactoryPool,
) -> AngleFactoryIndex:
    """
    Build an AngleFactoryIndex directly from TMR results.

    This index maps angles to all factories that successfully completed TMR,
    enabling cross-qubit factory sharing for RUS injection.

    Args:
        qubit_trackers: Dict mapping qubit_id to QubitAngleTracker
        factory_pool: FactoryPool containing Factory objects with TMR results

    Returns:
        AngleFactoryIndex with all angle-factory mappings for TMR-successful factories
    """
    angle_factory_index = AngleFactoryIndex()

    for factory in factory_pool.factories:
        # for factory in factory_list:
        if factory.tmr_state and factory.state == FactoryState.WAIT_FOR_RUS:
            qubit_id = factory.qubit
            assert qubit_id is not None, f"Factory {factory.id} has no qubit assigned"
            assert (
                factory.angle is not None
            ), f"Factory {factory.id} has no angle assigned"
            angle_factory_index.add_factory(factory.id, factory.angle, qubit_id)

    return angle_factory_index


def update_factory_states_post_tmr(
    factory_list: list[Factory],
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
) -> None:
    """
    Args:
        qubit_trackers: Dict mapping qubit_id to QubitAngleTracker
        factory_pool: FactoryPool containing Factory objects with TMR results
    """
    for factory in factory_list:
        if factory.tmr_state is False:
            qubit_id = factory.qubit
            assert (
                qubit_id is not None
            ), f"Factory {factory.id} with angle {factory.angle} has no qubit assigned"
            if qubit_id in qubit_trackers:
                tracker = qubit_trackers[qubit_id]
                tracker.remove_factory(factory.id, factory.angle)
            factory_pool.free_factory(factory.id)
