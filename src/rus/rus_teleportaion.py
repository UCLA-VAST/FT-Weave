from src.rus.util import check_teleportation_worthiness
from src.rus.rus_routing import two_layer_routing
from src.rus.rus_assignment import (
    assign_teleportation_with_sharing,
)
from src.ds import FactoryPool, QubitAngleTracker

from src.tmr.util import (
    build_angle_factory_index_from_tmr_results,
)


def rus_teleportation(
    qubit_trackers: dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
    logic_qubit_locations: list[tuple[int, int]],
    consider_skip_rus: int = 0,
    n_aods: int = 0,
    total_qubits: int = 0,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int, int, int, int, int]]]]:

    # Build angle factory index and schedule RUS teleportation
    angle_factory_index = build_angle_factory_index_from_tmr_results(factory_pool)

    qubit_factory_pairs = assign_teleportation_with_sharing(
        qubit_trackers,
        factory_pool,
        logic_qubit_locations,
        angle_factory_index,
    )
    if not qubit_factory_pairs:
        return [], []
    for qubit, factory_id in qubit_factory_pairs:
        factory = factory_pool.get_factory_by_id(factory_id)
        factory.set_to_rus()
    # routing
    # batch: list of tuple (qubit, x_q, y_q, factory_id, x_f, y_f, reverse)
    routing_batches = two_layer_routing(
        factory_pool, logic_qubit_locations, qubit_factory_pairs
    )
    # check if executing teleporation is worth it.
    if consider_skip_rus:
        routing_batches = check_teleportation_worthiness(
            routing_batches,
            n_aods,
            qubit_factory_pairs,
            total_qubits=total_qubits,
            n_tmr_next_run=len(qubit_trackers),
            parital_skip=(consider_skip_rus == 1),
        )
    return qubit_factory_pairs, routing_batches
