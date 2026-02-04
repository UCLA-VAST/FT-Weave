from src.rus.angle_factory_index import AngleFactoryIndex
from src.ds.device_state import FactoryPool, QubitAngleTracker


def build_angle_factory_index_from_tmr_results(
    qubit_trackers: dict[int, QubitAngleTracker],
    successful_qubits: set[int],
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

    # Iterate through all qubits and their factories
    for qubit, tracker in qubit_trackers.items():
        if qubit not in successful_qubits:
            removed_factory = []
            for factory_id, angle in tracker.factories:
                # Only include factories that successfully passed TMR
                factory = factory_pool.get_factory_by_id(factory_id)
                if factory is not None:
                    if factory.tmr_state:
                        angle_factory_index.add_factory(factory_id, angle, qubit)
                    else:
                        factory_pool.free_factory(factory_id)
                        removed_factory.append((factory_id, angle))
            for factory_id, angle in removed_factory:
                tracker.remove_factory(factory_id, angle)

            angle_factory_index.add_qubit_for_angle(tracker.target_angle)

    return angle_factory_index
