import logging
from typing import Any, Optional

import heapq
from collections import deque
from itertools import count

from .config import (
    CNOT_TIME,
    FACTORY_PHYSICAL_SIZE,
    SE_STAGE_1,
    SE_STAGE_2,
    SE_TIME,
    STAGE_1_RESOURCE_UNITS,
    compute_num_subfactories,
)
from .util import expand_multi_target_layers, build_circuit_dag
from src.ds import TFactoryPool, move_duration
from src.execution_log import (
    write_execution_log,
    execute_movement,
    execute_rus_teleportation,
    write_rus_result_log,
    clean_up_execution_log,
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
) -> list[tuple]:
    """
    Execute a quantum circuit using T cultivation.

    Args:
        circuit: List of instruction dictionaries, where each instruction has the format:
                 {"gate": str, "targets": list[int] | list[tuple[int, int]], "params": dict}

    Returns:
        execution_log: List of execution tuples with timestamps and details.
    """
    logger.info(
        "t_cultivation_execution: n_instr=%d n_factories=%d n_aods=%d to_decompose=%s epsilon=%g",
        len(circuit),
        n_factories,
        n_aods,
        to_decompose,
        epsilon,
    )

    if n_factories < 0:
        raise ValueError("n_factories must be non-negative")
    if n_aods < 2:
        raise ValueError(
            "n_aods must be at least 2: AOD 0 for T factory, AOD > 0 for gates/RUS"
        )

    if not circuit:
        return []

    expanded_circuit = expand_multi_target_layers(circuit, epsilon, to_decompose)[:4]
    for inst in expanded_circuit:
        logger.debug("Expanded instruction: %s", inst)

    logger.info(
        "Expanded circuit: %d instructions (from %d original)",
        len(expanded_circuit),
        len(circuit),
    )

    rz_t_by_angle = t_gate_count_per_unique_rz_angle(circuit, epsilon)
    total_t_expanded = count_t_gates_in_instructions(expanded_circuit)
    if rz_t_by_angle:
        print(
            "T+Tdg count per distinct Rz angle (one gridsynth decomposition per angle):"
        )
        for theta_key in sorted(rz_t_by_angle.keys()):
            print(f"  theta={theta_key:g}: T+Tdg count = {rz_t_by_angle[theta_key]}")
    print(f"Total T+Tdg gates in expanded circuit: {total_t_expanded}")

    predecessors, successors = build_circuit_dag(expanded_circuit)

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
        else compute_num_subfactories(FACTORY_PHYSICAL_SIZE, STAGE_1_RESOURCE_UNITS)
    )
    factory_pool = TFactoryPool(n_factories, num_subfactories=k_sub)
    factory_pool.set_locations(magic_state_locations)
    events: list[tuple[float, int, dict[str, Any]]] = []
    event_counter = count()
    execution_log: list[tuple] = []
    current_time = 0.0
    completed_nodes: set[int] = set()
    aod_earliest_available_time = [0.0] * n_aods

    def pick_gate_aod(earliest_start: float) -> tuple[int, float]:
        """Pick earliest available gate/RUS AOD from ids > 0."""
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

    def mark_node_completed(node: int):
        if node in completed_nodes:
            return
        completed_nodes.add(node)
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

    def schedule_t_preparation(at_time: float):
        idle_factories = deque(factory_pool.get_idle_factories())
        factory_ids = [f.id for f in idle_factories]
        aod_id = 0
        start_time = max(at_time, aod_earliest_available_time[aod_id])
        logger.debug(
            "Schedule T prep at t=%.3f on idle factories=%s aod_id=%d",
            start_time,
            factory_ids,
            aod_id,
        )
        for factory in idle_factories:
            factory.restart()

        stage_1_end = start_time + SE_STAGE_1
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
            duration = CNOT_TIME if gate in {"CNOT", "CZ"} else SE_TIME
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
                write_execution_log(
                    execution_log,
                    start_time=gate_start_time + move_dur,
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
            else:
                write_execution_log(
                    execution_log,
                    start_time=gate_start_time,
                    factories=[],
                    operation=gate,
                    aod_assignment=aod_id,
                    targets=targets,
                )
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
            raise RuntimeError(
                "No schedulable events remain. Circuit may contain a dependency cycle or unsatisfiable resource constraints. "
                f"Remaining operation indices: {remaining_nodes}"
            )

        current_time, _, first_event = heapq.heappop(events)
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
            # Stage 1: simulate per-subfactory outcomes, then redistribute and advance.
            simulate_stage1_preparation(factory_pool, rng, factory_list)
            success_factory_ids = complete_stage1_preparation(
                factory_pool,
                factory_list,
                execution_log,
                local_time,
                same_time_events["stage_1_completion"][0]["aod_id"],
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
                write_execution_log(
                    execution_log,
                    start_time=local_time,
                    factories=success_factory_ids,
                    operation="SE_stage_2",
                    aod_assignment=same_time_events["stage_1_completion"][0]["aod_id"],
                    targets=None,
                )
                heapq.heappush(
                    events,
                    (
                        local_time + SE_STAGE_2,
                        next(event_counter),
                        {
                            "type": "stage_2_completion",
                            "factory_list": success_factory_ids,
                            "aod_id": same_time_events["stage_1_completion"][0][
                                "aod_id"
                            ],
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
            factory_list = []
            for event in same_time_events["rus_start"]:
                factory_list.extend(event["factory_list"])
            # find rus assignments for each factory ready for rus
            local_time = current_time
            qubits_to_teleport = []
            qubit_idx_to_node = {}
            if not ready_t:
                heapq.heappush(
                    events,
                    (
                        local_time + 1,
                        next(event_counter),
                        {
                            "type": "rus_start",
                            "factory_list": factory_list,
                            "aod_id": same_time_events["rus_start"][0]["aod_id"],
                        },
                    ),
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
                factory_list,
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
                    qubit_factory_pairs, rus_start_time, execution_log, aod_idx
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
                trivial_return=False,
                decompose_move=False,
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
                    insert_s = True
                ready_t.remove(qubit_idx_to_node[qubit])
                node = qubit_idx_to_node[qubit]
                mark_node_completed(node)
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
            if insert_s:
                local_time += 1
                logger.debug("Inserted corrective S gate at t=%.3f", local_time)
            schedule_ready_clifford_operations(local_time)

        for event in same_time_events["rus_completion"]:
            factory_pool.free_factories(event["factory_list"])
            logger.debug("Freed factories after RUS: %s", event["factory_list"])
            schedule_t_preparation(current_time)

        for event in same_time_events["op_complete"]:
            node = event["node"]
            mark_node_completed(node)
            schedule_ready_clifford_operations(current_time)

    execution_log = clean_up_execution_log(execution_log, current_time)
    for log in execution_log:
        # logger.debug("Execution log: %s", log)
        logger.info("Execution log: %s", log)

    logger.info(
        "t_cultivation_execution finished: end_time=%.3f log_entries=%d",
        current_time,
        len(execution_log),
    )
    profiling = analyze_execution_log(execution_log, n_factories=n_factories)
    print_execution_profile(profiling)
    return execution_log
