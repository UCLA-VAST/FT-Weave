import logging

import numpy as np

logger = logging.getLogger(__name__)
from .simulation import (
    simulate_TMR_preparation,
    simulate_RUS_injection,
)
from .tmr.tmr_scheduling import schedule_tmr_round
from .tmr.tmr_assignment import reassign_factories, release_useless_factories
from .tmr.util import (
    update_factory_states_post_tmr,
)

from .ds import FactoryPool, QubitAngleTracker
from .rus import (
    update_qubit_state_per_teleportation,
)

from .analog_rotation import (
    execute_movement,
    execute_rus_teleportation,
    execute_tmr_preparation_pre_rz,
    execute_tmr_preparation_rz,
    write_tmr_result_log,
    write_rus_result_log,
)

from .rus.rus_teleportaion import rus_teleportation
from .rus.rus_post_teleportation import rus_post_teleportation

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================


def factory_angle_execution(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    code_distance: int,
    column_based_placement: bool = True,
    n_aods: int = 1,
    consider_skip_rus: int = 0,  # 0: no skip, 1: partial skip, 2: aggressive skip
    tmr_assignment_method: str = "matching",
    trivial_return: bool = True,
    decompose_move: bool = True,
    rng: np.random.Generator | None = None,
    save_log: bool = False,
    log_path: str | None = None,
):
    """
    Execute angle preparation on magic state factories with lookahead optimization.

    Args:
        target_qubits_angles: Dict mapping qubit_id -> target_rotation_angle
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory
        facoty_qubit_map: give (x,y) return 0 if no qubit and factory, 1 if qubit, 2 if factory
        n_aods: Number of AODs available for parallel operations

    Returns:
        circuit_moment: Total circuit execution time
        execution_log: List of (time, factory_id, operation, qubit) for visualization
    """
    if rng is None:
        rng = np.random.default_rng()
    # Map qubit_id to QubitAngleTracker
    qubit_trackers = {
        qubit: QubitAngleTracker(
            qubit_id=qubit,
            target_angle=round(theta, 7),
            factory_limit=len(magic_state_locations),
            code_distance=code_distance,
        )
        for qubit, theta in target_qubits_angles.items()
    }

    # Create FactoryPool with n_factories

    factory_pool.set_locations(magic_state_locations)

    # Log for visualization: (start_time, end_time, factory_id, operation, qubit)
    execution_log = []
    aod_earliest_available_time = [0.0] * n_aods

    circuit_moment = 0

    # ========================================================================
    # MAIN EXECUTION LOOP
    # ========================================================================

    while len(qubit_trackers):
        assert (
            circuit_moment <= 5000
        ), "Circuit execution taking too long, possible infinite loop. #idle factories: {}, Qubit trackers: {}".format(
            len(factory_pool.get_idle_factories()), qubit_trackers
        )
        # print(f"Remaining qubits to prepare: {len(qubit_trackers)}")
        # print("new run")
        # PHASE 1: Assign idle factories to prepare angles
        factory_list = factory_pool.get_idle_factories()
        # print(len(factory_list))
        # input()
        # PHASE 2: Execute TMR preparation
        circuit_moment, execution_log = execute_tmr_preparation_pre_rz(
            factory_list,
            circuit_moment,
            execution_log,
        )
        # print(
        #     "before TMR: factory 15 state: {}".format(
        #         factory_pool.get_factory_by_id(15)
        #     )
        # )
        # print(
        #     "before TMR: qubit 3 state: {}".format(
        #         qubit_trackers[3].factories if 3 in qubit_trackers else "N/A"
        #     )
        # )

        schedule_tmr_round(
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
            code_distance=code_distance,
            column_based_placement=column_based_placement,
            tmr_assignment_method=tmr_assignment_method,
        )
        circuit_moment, execution_log = execute_tmr_preparation_rz(
            factory_list,
            circuit_moment,
            execution_log,
        )
        simulate_TMR_preparation(factory_pool, rng)

        update_factory_states_post_tmr(factory_list, factory_pool, qubit_trackers)

        execution_log = write_tmr_result_log(
            factory_pool,
            circuit_moment,
            execution_log,
        )
        # print(
        #     "after TMR: factory 15 state: {}".format(factory_pool.get_factory_by_id(15))
        # )
        # print(
        #     "after TMR: qubit 3 factories: {}".format(
        #         qubit_trackers[3].factories if 3 in qubit_trackers else "N/A"
        #     )
        # )
        while True:
            # qubit_factory_pairs: list[tuple(qubit, factory_id)]
            # routing batch: list of tuple (qubit, x_q, y_q, factory_id, x_f, y_f)
            qubit_factory_pairs, routing_batches = rus_teleportation(
                qubit_trackers,
                factory_pool,
                logic_qubit_locations,
                consider_skip_rus=consider_skip_rus,
                n_aods=n_aods,
                total_qubits=len(target_qubits_angles),
            )

            if not qubit_factory_pairs or not routing_batches:
                break

            circuit_moment = execute_movement(
                routing_batches,
                execution_log,
                circuit_moment,
                aod_earliest_available_time,
            )
            circuit_moment = execute_rus_teleportation(
                qubit_factory_pairs, circuit_moment, execution_log, aod_id=0
            )

            rus_simulation = simulate_RUS_injection(
                qubit_factory_pairs, factory_pool, rng
            )

            execution_log = write_rus_result_log(
                qubit_factory_pairs, rus_simulation, circuit_moment, execution_log
            )
            circuit_moment = update_qubit_state_per_teleportation(
                qubit_factory_pairs,
                rus_simulation,
                qubit_trackers,
                execution_log,
                circuit_moment,
                aod_id=0,
            )
            # print(
            #     "after RUS: factory 15 state: {}".format(
            #         factory_pool.get_factory_by_id(15)
            #     )
            # )
            # return factories qubit to empty spot
            return_routing_batches = rus_post_teleportation(
                routing_batches,
                factory_pool,
                trivial_return=trivial_return,
                decompose_move=decompose_move,
            )

            circuit_moment = execute_movement(
                return_routing_batches,
                execution_log,
                circuit_moment,
                aod_earliest_available_time,
                move_type="return_move",
            )

            factories = [factory for _, factory in qubit_factory_pairs]
            factory_pool.free_factories(factories)
            # print(
            #     "after free factories: factory 15 state: {}".format(
            #         factory_pool.get_factory_by_id(15)
            #     )
            # )
            if len(qubit_trackers) == 0:
                break

        release_useless_factories(
            factory_pool,
            qubit_trackers,
        )
        # print(
        #     "after release factories: factory 15 state: {}".format(
        #         factory_pool.get_factory_by_id(15)
        #     )
        # )
        reassign_factories(
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
        )
        # print(
        #     "after reassign factories: factory 15 state: {}".format(
        #         factory_pool.get_factory_by_id(15)
        #     )
        # )

        # Add time for injection attempts (CNOT + SE per injection)
        execution_log.append(
            (
                circuit_moment,
                circuit_moment,
                -1,
                "Barrier",
                None,
            )
        )
        # input()
    if save_log and log_path is not None:
        with open(log_path, "w") as f:
            for entry in execution_log:
                f.write(str(entry) + "\n")
    return circuit_moment, execution_log
