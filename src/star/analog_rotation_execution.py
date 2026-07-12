import logging

import numpy as np

# logging.basicConfig(level=logging.DEBUG)
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

from ..ds import FactoryPool, QubitAngleTracker
from .rus import (
    update_qubit_state_per_teleportation,
)

from ..execution_log import (
    execute_movement,
    execute_rus_teleportation,
    execute_tmr_preparation_pre_rz,
    execute_tmr_preparation_rz,
    write_tmr_result_log,
    write_rus_result_log,
    LogicalSEScheduler,
)
from src.execution_log.event_helpers import build_event
from . import config as star_cfg

from .rus.rus_teleportaion import rus_teleportation
from .rus.rus_post_teleportation import rus_post_teleportation


def _run_tmr_round(
    *,
    circuit_moment: float,
    execution_log: list[dict],
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    code_distance: int,
    column_based_placement: bool,
    tmr_assignment_method: str,
    prepare_lookahead_angles: bool,
    allocation_level_range: range | None,
    rng: np.random.Generator,
    logical_se_scheduler: LogicalSEScheduler | None = None,
) -> tuple[float, list[dict]]:
    """Execute one TMR preparation cycle and update factory states."""
    factory_list = factory_pool.get_idle_factories()
    circuit_moment, execution_log = execute_tmr_preparation_pre_rz(
        factory_list,
        circuit_moment,
        execution_log,
        logical_se_scheduler=logical_se_scheduler,
    )
    schedule_tmr_round(
        factory_pool,
        qubit_trackers,
        logic_qubit_locations,
        code_distance=code_distance,
        column_based_placement=column_based_placement,
        tmr_assignment_method=tmr_assignment_method,
        prepare_lookahead_angles=prepare_lookahead_angles,
        allocation_level_range=allocation_level_range,
    )
    circuit_moment, execution_log = execute_tmr_preparation_rz(
        factory_list,
        circuit_moment,
        execution_log,
        logical_se_scheduler=logical_se_scheduler,
    )
    simulate_TMR_preparation(factory_pool, rng)
    update_factory_states_post_tmr(factory_list, factory_pool, qubit_trackers)
    execution_log = write_tmr_result_log(
        factory_pool,
        circuit_moment,
        execution_log,
    )
    return circuit_moment, execution_log


def _execute_rus_until_blocked(
    *,
    circuit_moment: float,
    execution_log: list[dict],
    qubit_trackers: dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
    logic_qubit_locations: list[tuple[int, int]],
    target_qubits_angles: dict[int, float],
    n_aods: int,
    consider_skip_rus: int,
    trivial_return: bool,
    decompose_move: bool,
    aod_earliest_available_time: list[float],
    rng: np.random.Generator,
    logical_se_scheduler: LogicalSEScheduler | None = None,
) -> tuple[float, list[dict]]:
    """Run RUS rounds until no routable factory-qubit pairs remain."""
    while True:
        qubit_factory_pairs, routing_batches = rus_teleportation(
            qubit_trackers,
            factory_pool,
            logic_qubit_locations,
            consider_skip_rus=consider_skip_rus,
            n_aods=n_aods,
            total_qubits=len(target_qubits_angles),
        )
        if not qubit_factory_pairs or not routing_batches:
            return circuit_moment, execution_log

        circuit_moment = execute_movement(
            routing_batches,
            execution_log,
            circuit_moment,
            aod_earliest_available_time,
        )
        circuit_moment = execute_rus_teleportation(
            qubit_factory_pairs,
            circuit_moment,
            execution_log,
            aod_id=0,
            logical_se_scheduler=logical_se_scheduler,
        )
        rus_simulation = simulate_RUS_injection(qubit_factory_pairs, factory_pool, rng)
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
            logical_se_scheduler=logical_se_scheduler,
        )
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
        if logical_se_scheduler is not None:
            logical_se_scheduler.force_due(circuit_moment, 0, execution_log)
        if len(qubit_trackers) == 0:
            return circuit_moment, execution_log


def _append_barrier(execution_log: list[dict], circuit_moment: float) -> None:
    execution_log.append(
        build_event(
            start_time=circuit_moment,
            end_time=circuit_moment,
            factories=-1,
            operation="Barrier",
            aod_assignment=None,
            targets=None,
            move_vecs=None,
        )
    )


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
    logical_se_interval: int | None = None,
    prepare_lookahead_angles: bool = True,
    allocation_level_range: range | None = None,
):
    """
    Execute angle preparation on magic state factories with lookahead optimization.

    Args:
        target_qubits_angles: Dict mapping qubit_id -> target_rotation_angle
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory
        facoty_qubit_map: give (x,y) return 0 if no qubit and factory, 1 if qubit, 2 if factory
        n_aods: Number of AODs available for parallel operations
        tmr_assignment_method: ``"matching"``, ``"naive"``, or ``"stochastic_coverage"``

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

    # Logical-qubit SE scheduler: enabled when interval (override or config) is set.
    se_interval = (
        logical_se_interval
        if logical_se_interval is not None
        else star_cfg.LOGICAL_SE_INTERVAL
    )
    logical_se_scheduler: LogicalSEScheduler | None = None
    if se_interval is not None:
        logical_se_scheduler = LogicalSEScheduler(
            qubit_ids=range(len(logic_qubit_locations)),
            interval=se_interval,
            start_time=circuit_moment,
        )

    # ========================================================================
    # MAIN EXECUTION LOOP
    # ========================================================================

    while len(qubit_trackers):
        # print(
        #     f"Current circuit moment: {circuit_moment:.2f}, Remaining qubits: {len(qubit_trackers)}"
        # )
        if circuit_moment > 3000:
            for log in execution_log[-50:]:
                print(f"Recent log entry: {log}")
        assert (
            circuit_moment <= 3000
        ), "Circuit execution taking too long, possible infinite loop. #idle factories: {}, Qubit trackers: {}".format(
            len(factory_pool.get_idle_factories()), qubit_trackers
        )
        # print(f"Remaining qubits to prepare: {len(qubit_trackers)}")
        # print("new run")
        circuit_moment, execution_log = _run_tmr_round(
            circuit_moment=circuit_moment,
            execution_log=execution_log,
            factory_pool=factory_pool,
            qubit_trackers=qubit_trackers,
            logic_qubit_locations=logic_qubit_locations,
            code_distance=code_distance,
            column_based_placement=column_based_placement,
            tmr_assignment_method=tmr_assignment_method,
            prepare_lookahead_angles=prepare_lookahead_angles,
            allocation_level_range=allocation_level_range,
            rng=rng,
            logical_se_scheduler=logical_se_scheduler,
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

        # print("Updated factory states post TMR completion")
        # for factory in factory_list:
        #     if factory.tmr_state:
        #         print(
        #             f"Factory {factory.id} with angle {factory.angle} success. will be teleported"
        #         )
        # for qubit in qubit_trackers:
        #     print(
        #         f"Qubit {qubit} factories: {qubit_trackers[qubit].factories if qubit in qubit_trackers else 'N/A'}"
        #     )
        # print(
        #     "after TMR: factory 15 state: {}".format(factory_pool.get_factory_by_id(15))
        # )
        # print(
        #     "after TMR: qubit 3 factories: {}".format(
        #         qubit_trackers[3].factories if 3 in qubit_trackers else "N/A"
        #     )
        # )
        circuit_moment, execution_log = _execute_rus_until_blocked(
            circuit_moment=circuit_moment,
            execution_log=execution_log,
            qubit_trackers=qubit_trackers,
            factory_pool=factory_pool,
            logic_qubit_locations=logic_qubit_locations,
            target_qubits_angles=target_qubits_angles,
            n_aods=n_aods,
            consider_skip_rus=consider_skip_rus,
            trivial_return=trivial_return,
            decompose_move=decompose_move,
            aod_earliest_available_time=aod_earliest_available_time,
            rng=rng,
            logical_se_scheduler=logical_se_scheduler,
        )

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
        _append_barrier(execution_log, circuit_moment)
        # input()
    if save_log and log_path is not None:
        with open(log_path, "w") as f:
            for entry in execution_log:
                f.write(str(entry) + "\n")
    return circuit_moment, execution_log
