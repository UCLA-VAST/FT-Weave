import logging
from typing import Any, Optional

import heapq
from collections import deque
from itertools import count

from . import config as tcfg
from .util import (
    build_circuit_dag,
    compute_longest_path_to_sink,
    expand_multi_target_layers,
)
from src.ds import FactoryStateT, TFactoryPool, move_duration
from src.execution_log import (
    write_execution_log,
    execute_movement,
    execute_rus_teleportation,
    write_rus_result_log,
    clean_up_execution_log,
    LogicalSEScheduler,
)
from src.t_cultivation.rus import rus_teleportation
from src.t_cultivation.simulation import (
    simulate_RUS_teleportation,
    simulate_stage1_preparation,
    simulate_stage2_preparation,
)
from src.star.rus.rus_post_teleportation import rus_post_teleportation
from src.t_cultivation.util import (
    complete_stage1_preparation,
    count_t_gates_in_instructions,
    t_gate_count_per_unique_rz_angle,
)
from src.util import analyze_execution_log, print_execution_profile

import numpy as np

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def t_cultivation_execution(
    circuit: list[dict],
    n_factories: int,
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    rng: np.random.Generator,
    n_aods: int = 2,
    to_decompose: bool = False,
    epsilon: float = 1e-4,
    num_subfactories: Optional[int] = None,
    synchronize_factory_execution: Optional[bool] = None,
    print_profile: bool = False,
    logical_se_interval: Optional[int] = None,
    code_distance: Optional[int] = None,
    trivial_return: bool = False,
    decompose_move: bool = False,
    redistribute_stage1_success: bool = True,
) -> list[dict[str, Any]]:
    """
    Execute a quantum circuit using T cultivation.

    Args:
        circuit: List of instruction dictionaries, where each instruction has the format:
                 {"gate": str, "targets": list[int] | list[tuple[int, int]], "params": dict}
        n_aods: Number of AODs. If ``1``, factory, gates, moves, and RUS all use AOD 0
            (fully serialized). If ``>= 2``, **only** the T-factory stages (SE) use AOD 0,
            with start times taken after ``aod_earliest_available_time[0]``; Clifford gates
            use AODs ``1 .. n_aods-1``; movement and RUS use ``pick_any_aod`` / ``execute_movement``.
        synchronize_factory_execution: If True, stage-1 restarts are delayed until all
            currently running stage-2 factories complete.
        code_distance: Surface-code distance used to choose the stage-2 SE duration.
        trivial_return / decompose_move: Passed to :func:`rus_post_teleportation` for
            return routing after RUS (aligned with STAR).
        redistribute_stage1_success: If True (default), after stage-1 simulation spare
            successes may be moved between factories via
            :func:`redistribute_stage1_successes`. If False, raw subfactory outcomes are
            kept and no patch-redistribution moves occur.

    Returns:
        execution_log: List of execution event dictionaries with timestamps and details.
    """
    _cfg = tcfg.get_config()
    logger.info("t_cultivation config (current global):")
    for _k in sorted(_cfg.keys()):
        logger.info("  %s: %s", _k, _cfg[_k])

    logger.info(
        "t_cultivation_execution: n_instr=%d n_factories=%d n_aods=%d to_decompose=%s "
        "epsilon=%g trivial_return=%s decompose_move=%s redistribute_stage1_success=%s",
        len(circuit),
        n_factories,
        n_aods,
        to_decompose,
        epsilon,
        trivial_return,
        decompose_move,
        redistribute_stage1_success,
    )
    se_stage_2_duration = tcfg.stage2_duration_from_code_distance(code_distance)

    if n_factories < 0:
        raise ValueError("n_factories must be non-negative")
    if n_aods < 1:
        raise ValueError("n_aods must be at least 1")

    if not circuit:
        return []

    expanded_circuit = expand_multi_target_layers(circuit, epsilon, to_decompose)
    for inst in expanded_circuit:
        logger.debug("Expanded instruction: %s", inst)

    logger.info(
        "Expanded circuit: %d instructions (from %d original)",
        len(expanded_circuit),
        len(circuit),
    )

    rz_t_by_angle = t_gate_count_per_unique_rz_angle(circuit, epsilon)
    total_t_tdg_gates = count_t_gates_in_instructions(expanded_circuit)
    if rz_t_by_angle:
        logger.info(
            "T+Tdg count per distinct Rz angle (one gridsynth decomposition per angle):"
        )
        for theta_key in sorted(rz_t_by_angle.keys()):
            logger.info(
                "  theta=%g: T+Tdg count = %s",
                theta_key,
                rz_t_by_angle[theta_key],
            )
    logger.info("Total T+Tdg gates in expanded circuit: %d", total_t_tdg_gates)

    predecessors, successors = build_circuit_dag(expanded_circuit)
    longest_path_to_sink = compute_longest_path_to_sink(
        successors, len(expanded_circuit)
    )

    remaining_deps = {node: len(deps) for node, deps in predecessors.items()}
    logger.debug("Built DAG: nodes=%d", len(predecessors))

    ready_clifford: deque[int] = deque()
    ready_t = set()
    for node, dep_count in remaining_deps.items():
        if dep_count == 0:
            if expanded_circuit[node]["gate"] in {"T", "Tdg"}:
                ready_t.add(node)
            else:
                ready_clifford.append(node)
    logger.debug(
        "Initial ready queues: clifford=%d t=%d", len(ready_clifford), len(ready_t)
    )

    k_sub = (
        max(1, num_subfactories)
        if num_subfactories is not None
        else tcfg.compute_num_subfactories(
            tcfg.FACTORY_PHYSICAL_SIZE, tcfg.STAGE_1_RESOURCE_UNITS
        )
    )
    factory_pool = TFactoryPool(n_factories, num_subfactories=k_sub)
    factory_pool.set_locations(magic_state_locations)
    sync_stages = (
        tcfg.SYNCHRONIZE_FACTORY_EXECUTION
        if synchronize_factory_execution is None
        else synchronize_factory_execution
    )
    if not sync_stages:
        raise NotImplementedError(
            "Non-synchronized factory execution is not implemented yet because "
            "AOD assignment for asynchronous factory progression is not handled."
        )
    events: list[tuple[float, int, dict[str, Any]]] = []
    event_counter = count()
    execution_log: list[dict] = []
    current_time = 0.0
    completed_nodes: set[int] = set()
    t_tdg_implemented_count = 0
    aod_earliest_available_time = [0.0] * n_aods

    # Logical-qubit SE scheduler: piggy-back slightly early when possible, but
    # keep x as a hard maximum idle gap.
    se_interval = (
        logical_se_interval
        if logical_se_interval is not None
        else tcfg.LOGICAL_SE_INTERVAL
    )
    logical_se_scheduler: LogicalSEScheduler | None = None
    if se_interval is not None:
        logical_se_scheduler = LogicalSEScheduler(
            qubit_ids=range(len(logic_qubit_locations)),
            interval=se_interval,
            start_time=current_time,
        )

    def pick_gate_aod(earliest_start: float) -> tuple[int, float]:
        """Pick AOD for Clifford gates.

        With ``n_aods == 1``, all operations share AOD 0 (fully serialized). With
        ``n_aods >= 2``, gates use AOD indices ``1 .. n_aods-1`` (AOD 0 reserved for
        the factory pipeline when parallel).
        """
        if n_aods == 1:
            gate_aod_id = 0
        else:
            gate_aod_id = min(
                range(1, n_aods), key=lambda idx: aod_earliest_available_time[idx]
            )
        start_time = max(earliest_start, aod_earliest_available_time[gate_aod_id])
        return gate_aod_id, start_time

    def pick_any_aod(earliest_start: float) -> tuple[int, float]:
        """Pick earliest available AOD from all ids (including AOD 0)."""
        aod_id = min(range(n_aods), key=lambda idx: aod_earliest_available_time[idx])
        start_time = max(earliest_start, aod_earliest_available_time[aod_id])
        return aod_id, start_time

    def pick_factory_aod(earliest_start: float) -> tuple[int, float]:
        """T-factory pipeline always uses AOD 0; start time follows its availability."""
        factory_aod_id = 0
        start_time = max(earliest_start, aod_earliest_available_time[factory_aod_id])
        return factory_aod_id, start_time

    def mark_node_completed(node: int):
        nonlocal t_tdg_implemented_count
        if node in completed_nodes:
            return
        completed_nodes.add(node)
        if expanded_circuit[node]["gate"] in {"T", "Tdg"}:
            t_tdg_implemented_count += 1
        logger.debug("Completed node=%d gate=%s", node, expanded_circuit[node]["gate"])
        for successor in successors[node]:
            remaining_deps[successor] -= 1
            if remaining_deps[successor] == 0:
                if expanded_circuit[successor]["gate"] in {"T", "Tdg"}:
                    ready_t.add(successor)
                else:
                    ready_clifford.append(successor)
        logger.debug(
            "Post-complete queues: clifford=%d t=%d completed=%d/%d",
            len(ready_clifford),
            len(ready_t),
            len(completed_nodes),
            len(expanded_circuit),
        )

    def should_schedule_t_preparation_for_remaining_demand() -> bool:
        """
        If enough factories already hold magic for all remaining T/Tdg work, skip a new
        SE round (idle factories stay idle until RUS consumes states).

        Remaining demand uses ``total_t_tdg_gates`` (fixed at expand time) minus
        ``t_tdg_implemented_count`` (incremented when each T/Tdg node completes).
        """
        need = total_t_tdg_gates - t_tdg_implemented_count
        if need <= 0:
            return False
        ready_magic = len(factory_pool.get_wait_for_rus_factories())
        if ready_magic >= need:
            logger.debug(
                "Skip T prep: WAIT_FOR_RUS=%d >= remaining T/Tdg=%d (implemented %d/%d)",
                ready_magic,
                need,
                t_tdg_implemented_count,
                total_t_tdg_gates,
            )
            return False
        return True

    def maybe_enqueue_rus_start(at_time: float) -> None:
        """If T gates are ready and factories hold magic, schedule RUS (one pending max)."""
        if not ready_t:
            return
        wait_for_rus_ids = [f.id for f in factory_pool.get_wait_for_rus_factories()]
        if not wait_for_rus_ids:
            return
        pending_rus_start = any(e.get("type") == "rus_start" for _, _, e in events)
        if pending_rus_start:
            return
        heapq.heappush(
            events,
            (
                at_time,
                next(event_counter),
                {
                    "type": "rus_start",
                    "factory_list": wait_for_rus_ids,
                    "aod_id": 0,
                },
            ),
        )
        logger.debug(
            "Enqueued rus_start at t=%.3f wait_for_rus=%s ready_t_nodes=%d",
            at_time,
            wait_for_rus_ids,
            len(ready_t),
        )

    def schedule_t_preparation(at_time: float):
        if sync_stages and any(
            f.state
            in (FactoryStateT.STAGE_2, FactoryStateT.WAIT_FOR_RUS, FactoryStateT.RUS)
            for f in factory_pool.get_busy_factories()
        ):
            logger.debug(
                "Delayed stage-1 restart at t=%.3f due to synchronized stage/RUS barrier",
                at_time,
            )
            return
        idle_factories = deque(factory_pool.get_idle_factories())
        factory_ids = [f.id for f in idle_factories]
        if not factory_ids:
            return
        aod_id, start_time = pick_factory_aod(at_time)
        logger.debug(
            "Schedule T prep at t=%.3f on idle factories=%s aod_id=%d",
            start_time,
            factory_ids,
            aod_id,
        )
        for factory in idle_factories:
            factory.restart()

        stage_1_end = start_time + tcfg.SE_STAGE_1
        aod_earliest_available_time[aod_id] = stage_1_end

        # execution_log.append(
        #     (
        #         start_time,
        #         stage_1_end,
        #         factory_ids,
        #         "SE_stage_1",
        #         None,
        #         [],
        #     )
        # )
        write_execution_log(
            execution_log,
            start_time=start_time,
            factories=factory_ids,
            operation="SE_stage_1",
            aod_assignment=aod_id,
            targets=None,
        )
        if logical_se_scheduler is not None:
            for k in range(int(tcfg.SE_STAGE_1)):
                logical_se_scheduler.piggyback(
                    start_time + k, aod_id, execution_log
                )
        heapq.heappush(
            events,
            (
                stage_1_end,
                next(event_counter),
                {
                    "type": "stage_1_completion",
                    "factory_list": factory_ids,
                    "aod_id": aod_id,
                },
            ),
        )

    def schedule_ready_clifford_operations(opt_time: float):
        while ready_clifford:
            node = ready_clifford.popleft()
            instr = expanded_circuit[node]
            gate = instr["gate"]
            aod_id, gate_start_time = pick_gate_aod(opt_time)
            logger.debug(
                "Schedule Clifford node=%d gate=%s at t=%.3f targets=%s aod_id=%d",
                node,
                gate,
                gate_start_time,
                instr.get("targets", []),
                aod_id,
            )
            duration = tcfg.CNOT_TIME if gate in {"CNOT", "CZ"} else tcfg.SE_TIME
            finish_time = gate_start_time + duration
            targets = instr.get("targets", [])
            if gate in {"CNOT", "CZ"}:
                assert all(
                    len(t) == 2 for t in targets
                ), "CNOT/CZ targets must be pairs of qubits"
                c_qubits = [t[0] for t in targets]
                t_qubits = [t[1] for t in targets]
                init_locs = [logic_qubit_locations[q] for q in c_qubits]
                end_locs = [logic_qubit_locations[q] for q in t_qubits]
                move_vecs = []
                reverse_move_vecs = []
                move_dur = 0
                for init_loc, end_loc in zip(init_locs, end_locs):
                    move_vecs.append(
                        [
                            f"({init_loc[0]},{init_loc[1]})",
                            f"({end_loc[0]},{end_loc[1]})",
                        ]
                    )
                    reverse_move_vecs.append(
                        [
                            f"({end_loc[0]},{end_loc[1]})",
                            f"({init_loc[0]},{init_loc[1]})",
                        ]
                    )
                    move_dur = max(
                        move_dur,
                        move_duration(init_loc[0], init_loc[1], end_loc[0], end_loc[1]),
                    )
                write_execution_log(
                    execution_log,
                    start_time=gate_start_time,
                    factories=[],
                    operation=gate,
                    aod_assignment=aod_id,
                    targets=c_qubits,
                    move_vecs=move_vecs,
                    movement_time=move_dur,
                )
                cnot_start_time = gate_start_time + move_dur
                if logical_se_scheduler is not None:
                    # Catch up any missed hard SE deadlines before the CNOT
                    # starts. Qubits whose deadline is exactly the CNOT start
                    # are covered by the CNOT itself, so they are reset below
                    # rather than duplicated in an overlapping SE_q entry.
                    logical_se_scheduler.force_due(
                        cnot_start_time,
                        aod_id,
                        execution_log,
                        include_current=False,
                    )
                write_execution_log(
                    execution_log,
                    start_time=cnot_start_time,
                    factories=[],
                    operation=gate,
                    aod_assignment=aod_id,
                    targets=targets,
                )
                write_execution_log(
                    execution_log,
                    start_time=gate_start_time + move_dur + duration,
                    factories=[],
                    operation=gate,
                    aod_assignment=aod_id,
                    targets=c_qubits,
                    move_vecs=reverse_move_vecs,
                    movement_time=move_dur,
                )
                finish_time += 2 * move_dur
                if logical_se_scheduler is not None:
                    # CNOT/CZ already performs an SE round on its participants;
                    # reset their clocks first, then piggy-back any other
                    # still-idle logical qubits onto the same CNOT moment.
                    logical_se_scheduler.reset(
                        c_qubits + t_qubits, finish_time
                    )
                    logical_se_scheduler.piggyback(
                        gate_start_time + move_dur, aod_id, execution_log
                    )
            else:
                write_execution_log(
                    execution_log,
                    start_time=gate_start_time,
                    factories=[],
                    operation=gate,
                    aod_assignment=aod_id,
                    targets=targets,
                )
                if logical_se_scheduler is not None:
                    flat_targets: list[int] = []
                    for tgt in targets:
                        if isinstance(tgt, (list, tuple)):
                            flat_targets.extend(tgt)
                        elif tgt is not None:
                            flat_targets.append(tgt)
                    logical_se_scheduler.reset(flat_targets, finish_time)
            aod_earliest_available_time[aod_id] = finish_time
            heapq.heappush(
                events,
                (
                    finish_time,
                    next(event_counter),
                    {
                        "type": "op_complete",
                        "node": node,
                        "factory_id": None,
                        "aod_id": aod_id,
                    },
                ),
            )
            logger.debug("Enqueued op_complete node=%d at t=%.3f", node, finish_time)

    schedule_ready_clifford_operations(current_time)
    schedule_t_preparation(current_time)

    while len(completed_nodes) < len(expanded_circuit):
        if not events:
            remaining_nodes = [
                index
                for index in range(len(expanded_circuit))
                if index not in completed_nodes
            ]
            for log in execution_log:
                print(log)
            raise RuntimeError(
                "No schedulable events remain. Circuit may contain a dependency cycle or unsatisfiable resource constraints. "
                f"Remaining operation count: {len(remaining_nodes)}"
            )

        current_time, _, first_event = heapq.heappop(events)
        # if current_time > 1000:
        #     raise RuntimeError(f"Current time is {current_time}, which is too large")
        logger.debug(
            "Processing events at t=%.3f first_event=%s pending_events=%d",
            current_time,
            first_event["type"],
            len(events),
        )
        same_time_events = {
            "op_complete": [],
            "rus_teleportation": [],
            "rus_completion": [],
            "rus_start": [],
            "stage_1_completion": [],
            "stage_2_completion": [],
        }
        same_time_events[first_event["type"]].append(first_event)
        while events and events[0][0] == current_time:
            _, _, event = heapq.heappop(events)
            same_time_events[event["type"]].append(event)

        if same_time_events["stage_1_completion"]:
            factory_list = []
            local_time = current_time
            for event in same_time_events["stage_1_completion"]:
                factory_list.extend(event["factory_list"])
            factory_list = list(dict.fromkeys(factory_list))
            # Stage 1: simulate per-subfactory outcomes, then redistribute and advance.
            simulate_stage1_preparation(factory_pool, rng, factory_list)
            success_factory_ids, redistribution_delay = complete_stage1_preparation(
                factory_pool,
                factory_list,
                execution_log,
                local_time,
                same_time_events["stage_1_completion"][0]["aod_id"],
                redistribute_stage1_success=redistribute_stage1_success,
            )
            logger.debug(
                "Stage 1 complete at t=%.3f success=%s failed=%d",
                local_time,
                success_factory_ids,
                len(factory_list) - len(success_factory_ids),
            )

            if len(success_factory_ids) < len(factory_list):
                schedule_t_preparation(local_time)
            if success_factory_ids:
                stage_2_start_time = local_time + redistribution_delay
                aod_s2 = same_time_events["stage_1_completion"][0]["aod_id"]
                stage_2_end = stage_2_start_time + se_stage_2_duration
                aod_earliest_available_time[aod_s2] = max(
                    aod_earliest_available_time[aod_s2], stage_2_end
                )
                write_execution_log(
                    execution_log,
                    start_time=stage_2_start_time,
                    factories=success_factory_ids,
                    operation="SE_stage_2",
                    aod_assignment=aod_s2,
                    targets=None,
                    movement_time=se_stage_2_duration,
                )
                if logical_se_scheduler is not None:
                    for k in range(int(se_stage_2_duration)):
                        logical_se_scheduler.piggyback(
                            stage_2_start_time + k, aod_s2, execution_log
                        )
                heapq.heappush(
                    events,
                    (
                        stage_2_end,
                        next(event_counter),
                        {
                            "type": "stage_2_completion",
                            "factory_list": success_factory_ids,
                            "aod_id": aod_s2,
                        },
                    ),
                )
        if same_time_events["stage_2_completion"]:
            factory_list = []
            local_time = current_time
            for event in same_time_events["stage_2_completion"]:
                factory_list.extend(event["factory_list"])
            success_factory_ids = simulate_stage2_preparation(
                factory_pool, rng, factory_list
            )
            logger.debug(
                "Stage 2 complete at t=%.3f success=%s failed=%d",
                local_time,
                success_factory_ids,
                len(factory_list) - len(success_factory_ids),
            )

            if len(success_factory_ids) < len(factory_list):
                schedule_t_preparation(local_time)

            if success_factory_ids:
                heapq.heappush(
                    events,
                    (
                        local_time,
                        next(event_counter),
                        {
                            "type": "rus_start",
                            "factory_list": success_factory_ids,
                            "aod_id": same_time_events["stage_2_completion"][0][
                                "aod_id"
                            ],
                        },
                    ),
                )
        if same_time_events["rus_start"]:
            wave_factory_ids: list[int] = []
            for event in same_time_events["rus_start"]:
                wave_factory_ids.extend(event["factory_list"])
            wave_factory_ids = list(dict.fromkeys(wave_factory_ids))
            # Magic ready at *any* factory in WAIT_FOR_RUS, including leftovers from a
            # previous wave when num_factories > num_qubits (Hungarian used only one
            # factory per qubit; others stayed WAIT_FOR_RUS and must be eligible next).
            wait_for_rus_ids = [f.id for f in factory_pool.get_wait_for_rus_factories()]
            logger.debug(
                "rus_start at t=%.3f wave_factories=%s wait_for_rus=%s",
                current_time,
                wave_factory_ids,
                wait_for_rus_ids,
            )
            local_time = current_time
            qubits_to_teleport = []
            qubit_idx_to_node = {}
            if not ready_t:
                if not wait_for_rus_ids:
                    logger.debug(
                        "rus_start at t=%.3f: no T gates ready and no WAIT_FOR_RUS factories",
                        local_time,
                    )
                    continue
                heapq.heappush(
                    events,
                    (
                        local_time + 1,
                        next(event_counter),
                        {
                            "type": "rus_start",
                            "factory_list": wait_for_rus_ids,
                            "aod_id": same_time_events["rus_start"][0]["aod_id"],
                        },
                    ),
                )
                continue
            if not wait_for_rus_ids:
                logger.warning(
                    "rus_start at t=%.3f but no factories in WAIT_FOR_RUS; skipping batch",
                    current_time,
                )
                continue
            for ready_node in ready_t:
                instr = expanded_circuit[ready_node]
                qubits_to_teleport.extend(instr.get("targets", []))
                for q in instr.get("targets", []):
                    qubit_idx_to_node[q] = ready_node

            qubit_factory_pairs, routing_batches = rus_teleportation(
                qubits_to_teleport,
                logic_qubit_locations,
                factory_pool,
                wait_for_rus_ids,
                qubit_to_node=qubit_idx_to_node,
                longest_path_to_sink=longest_path_to_sink,
                critical_path_weight=tcfg.RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT,
            )
            logger.debug(
                "RUS start at t=%.3f ready_t=%d assigned_pairs=%d",
                local_time,
                len(ready_t),
                len(qubit_factory_pairs),
            )
            if qubit_factory_pairs and routing_batches:
                # Execute movement to factories
                local_time = execute_movement(
                    routing_batches,
                    execution_log,
                    local_time,
                    aod_earliest_available_time,
                )
                # find earliest available AOD for teleportation (including AOD 0)
                aod_idx, rus_start_time = pick_any_aod(local_time)
                local_time = execute_rus_teleportation(
                    qubit_factory_pairs,
                    rus_start_time,
                    execution_log,
                    aod_idx,
                    logical_se_scheduler=logical_se_scheduler,
                )
                aod_earliest_available_time[aod_idx] = local_time
                logger.debug(
                    f"RUS teleportation executed, update AOD available time: {aod_earliest_available_time}"
                )
                # Schedule teleportation event
                heapq.heappush(
                    events,
                    (
                        local_time,
                        next(event_counter),
                        {
                            "type": "rus_teleportation",
                            "qubit_factory_pairs": qubit_factory_pairs,
                            "qubit_idx_to_node": qubit_idx_to_node,
                            "routing_batches": routing_batches,
                            "aod_id": aod_idx,
                        },
                    ),
                )

        for event in same_time_events["rus_teleportation"]:
            local_time = current_time
            qubit_idx_to_node = event["qubit_idx_to_node"]
            qubit_factory_pairs = event["qubit_factory_pairs"]
            factory_list = [f for _, f in qubit_factory_pairs]

            # Simulate RUS teleportation results
            rus_simulation = simulate_RUS_teleportation(
                event["qubit_factory_pairs"], factory_pool, rng
            )
            logger.debug(
                "RUS teleportation at t=%.3f pairs=%d successes=%d",
                local_time,
                len(qubit_factory_pairs),
                sum(rus_simulation),
            )

            execution_log = write_rus_result_log(
                event["qubit_factory_pairs"],
                rus_simulation,
                local_time,
                execution_log,
            )

            # Schedule return movement
            return_routing_batches = rus_post_teleportation(
                event["routing_batches"],
                factory_pool,
                trivial_return=trivial_return,
                decompose_move=decompose_move,
            )

            local_time = execute_movement(
                return_routing_batches,
                execution_log,
                local_time,
                aod_earliest_available_time,
                move_type="return_move",
            )

            # ! Update qubit states based on teleportation results
            insert_s = False
            for (qubit, factory_id), rus_success in zip(
                qubit_factory_pairs, rus_simulation
            ):
                gate_name = expanded_circuit[qubit_idx_to_node[qubit]]["gate"]
                if (not rus_success and gate_name == "T") or (
                    rus_success and gate_name == "Tdg"
                ):
                    write_execution_log(
                        execution_log,
                        start_time=local_time,
                        factories=[],
                        operation="S",
                        aod_assignment=event["aod_id"],
                        targets=[qubit],
                    )
                    if logical_se_scheduler is not None:
                        logical_se_scheduler.reset(
                            [qubit], local_time + tcfg.SE_TIME
                        )
                    insert_s = True
                ready_t.remove(qubit_idx_to_node[qubit])
                node = qubit_idx_to_node[qubit]
                mark_node_completed(node)
            if insert_s:
                aod_s = event["aod_id"]
                aod_earliest_available_time[aod_s] = max(
                    aod_earliest_available_time[aod_s],
                    local_time + tcfg.SE_TIME,
                )
                local_time += 1
                logger.debug("Inserted corrective S gate at t=%.3f", local_time)
            heapq.heappush(
                events,
                (
                    local_time,
                    next(event_counter),
                    {
                        "type": "rus_completion",
                        "factory_list": factory_list,
                        "aod_id": event["aod_id"],
                    },
                ),
            )

            schedule_ready_clifford_operations(local_time)

        for event in same_time_events["rus_completion"]:
            factory_pool.free_factories(event["factory_list"])
            logger.debug("Freed factories after RUS: %s", event["factory_list"])

            maybe_enqueue_rus_start(current_time)

            if should_schedule_t_preparation_for_remaining_demand():
                schedule_t_preparation(current_time)

        for event in same_time_events["op_complete"]:
            node = event["node"]
            mark_node_completed(node)
            schedule_ready_clifford_operations(current_time)
            maybe_enqueue_rus_start(current_time)

        if logical_se_scheduler is not None:
            logical_se_scheduler.force_due(current_time, None, execution_log)

    execution_log = clean_up_execution_log(execution_log, current_time)
    for log in execution_log:
        logger.debug("Execution log: %s", log)
        # logger.info("Execution log: %s", log)

    logger.info(
        "t_cultivation_execution finished: end_time=%.3f log_entries=%d "
        "t_tdg_implemented=%d/%d",
        current_time,
        len(execution_log),
        t_tdg_implemented_count,
        total_t_tdg_gates,
    )
    if print_profile:
        profiling = analyze_execution_log(execution_log, n_factories=n_factories)
        print_execution_profile(profiling)
    return execution_log
