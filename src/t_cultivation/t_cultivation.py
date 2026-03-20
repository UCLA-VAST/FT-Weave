import logging
from typing import Any

import heapq
from collections import deque
from itertools import count

from .config import (
    CNOT_TIME,
    SE_STAGE_1,
    SE_STAGE_2,
    SE_TIME,
)
from .util import expand_multi_target_layers, build_circuit_dag
from src.ds import TFactoryPool, move_duration
from src.execution_log import (
    write_execution_log,
    execute_movement,
    execute_rus_teleportation,
    write_rus_result_log,
)
from src.t_cultivation.rus import rus_teleportation
from src.t_cultivation.simulation import (
    simulate_RUS_teleportation,
    simulate_stage1_preparation,
    simulate_stage2_preparation,
)
from src.star.rus.rus_post_teleportation import rus_post_teleportation

import numpy as np

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def t_cultivation_execution(
    circuit: list[dict],
    n_factories: int,
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    rng: np.random.Generator,
    n_aods: int = 1,
    to_decompose: bool = False,
) -> list[tuple]:
    """
    Execute a quantum circuit using T cultivation.

    Args:
        circuit: List of instruction dictionaries, where each instruction has the format:
                 {"gate": str, "targets": list[int] | list[tuple[int, int]], "params": dict}

    Returns:
        execution_log: List of execution tuples with timestamps and details.
    """
    if n_factories < 0:
        raise ValueError("n_factories must be non-negative")

    if not circuit:
        return []

    expanded_circuit = expand_multi_target_layers(circuit, to_decompose)

    predecessors, successors = build_circuit_dag(expanded_circuit)
    remaining_deps = {node: len(deps) for node, deps in predecessors.items()}

    ready_clifford: deque[int] = deque()
    ready_t = set()
    for node, dep_count in remaining_deps.items():
        if dep_count == 0:
            if expanded_circuit[node]["gate"] == "T":
                ready_t.add(node)
            else:
                ready_clifford.append(node)

    factory_pool = TFactoryPool(n_factories)
    factory_pool.set_locations(magic_state_locations)
    events: list[tuple[float, int, dict[str, Any]]] = []
    event_counter = count()
    execution_log: list[tuple] = []
    current_time = 0.0
    completed_nodes: set[int] = set()
    aod_earliest_available_time = [0.0] * n_aods

    def mark_node_completed(node: int):
        if node in completed_nodes:
            return
        completed_nodes.add(node)
        for successor in successors[node]:
            remaining_deps[successor] -= 1
            if remaining_deps[successor] == 0:
                if expanded_circuit[successor]["gate"] == "T":
                    ready_t.add(successor)
                else:
                    ready_clifford.append(successor)

    def schedule_t_preparation(at_time: float):
        idle_factories = deque(factory_pool.get_idle_factories())
        factory_ids = [f.id for f in idle_factories]
        for factory in idle_factories:
            factory.restart()

        stage_1_end = at_time + SE_STAGE_1

        execution_log.append(
            (
                at_time,
                stage_1_end,
                factory_ids,
                "SE_stage_1",
                None,
                [],
            )
        )
        heapq.heappush(
            events,
            (
                stage_1_end,
                next(event_counter),
                {
                    "type": "stage_1_completion",
                    "factory_list": factory_ids,
                },
            ),
        )

    def schedule_ready_clifford_operations(opt_time: float):
        while ready_clifford:
            node = ready_clifford.popleft()
            instr = expanded_circuit[node]
            gate = instr["gate"]
            duration = CNOT_TIME if gate in {"CNOT", "CZ"} else SE_TIME
            finish_time = opt_time + duration
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
                    start_time=opt_time,
                    factories=[],
                    operation=gate,
                    aod_assignment=None,
                    targets=c_qubits,
                    move_vecs=move_vecs,
                    movement_time=move_dur,
                )
                write_execution_log(
                    execution_log,
                    start_time=opt_time + move_dur,
                    factories=[],
                    operation=gate,
                    aod_assignment=None,
                    targets=targets,
                )
                write_execution_log(
                    execution_log,
                    start_time=opt_time + move_dur + duration,
                    factories=[],
                    operation=gate,
                    aod_assignment=None,
                    targets=c_qubits,
                    move_vecs=reverse_move_vecs,
                    movement_time=move_dur,
                )
                finish_time += 2 * move_dur
            else:
                write_execution_log(
                    execution_log,
                    start_time=opt_time,
                    factories=[],
                    operation=gate,
                    aod_assignment=None,
                    targets=targets,
                )
            heapq.heappush(
                events,
                (
                    finish_time,
                    next(event_counter),
                    {
                        "type": "op_complete",
                        "node": node,
                        "factory_id": None,
                    },
                ),
            )

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

        for event in same_time_events["stage_1_completion"]:
            local_time = current_time
            factory_list = event["factory_list"]
            # simulate stage 1 success/failure for each factory
            success_factory_ids = simulate_stage1_preparation(
                factory_pool, rng, factory_list
            )

            if len(success_factory_ids) < len(factory_list):
                schedule_t_preparation(local_time)

            heapq.heappush(
                events,
                (
                    local_time + SE_STAGE_2,
                    next(event_counter),
                    {"type": "stage_2_completion", "factory_list": success_factory_ids},
                ),
            )

        for event in same_time_events["stage_2_completion"]:
            local_time = current_time
            factory_list = event["factory_list"]
            success_factory_ids = simulate_stage2_preparation(
                factory_pool, rng, factory_list
            )

            if len(success_factory_ids) < len(factory_list):
                schedule_t_preparation(local_time)

            heapq.heappush(
                events,
                (
                    local_time,
                    next(event_counter),
                    {"type": "rus_start", "factory_list": success_factory_ids},
                ),
            )

        for event in same_time_events["rus_start"]:
            # find rus assignments for each factory ready for rus
            local_time = current_time
            factory_list = event["factory_list"]
            qubits_to_teleport = []
            qubit_idx_to_node = {}
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
            if qubit_factory_pairs and routing_batches:
                # Execute movement to factories
                local_time = execute_movement(
                    routing_batches,
                    execution_log,
                    local_time,
                    aod_earliest_available_time,
                )
                # find earliest available AOD for teleportation
                aod_idx = aod_earliest_available_time.index(
                    min(aod_earliest_available_time)
                )
                local_time = execute_rus_teleportation(
                    qubit_factory_pairs, local_time, execution_log, aod_idx
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
                            "type": "RUS_teleportation",
                            "qubit_factory_pairs": qubit_factory_pairs,
                            "qubit_idx_to_node": qubit_idx_to_node,
                            "routing_batches": routing_batches,
                            "aod_id": event["aod_id"],
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
                if not rus_success:
                    write_execution_log(
                        execution_log,
                        start_time=local_time,
                        factories=[],
                        operation="S",
                        aod_assignment=None,
                        targets=[qubit],
                    )
                    insert_s = True
                ready_t.remove(qubit_idx_to_node[qubit])
                node = qubit_idx_to_node[qubit]
                mark_node_completed(node)
            if insert_s:
                local_time += 1

            heapq.heappush(
                events,
                (
                    local_time,
                    next(event_counter),
                    {
                        "type": "rus_completion",
                        "factory_list": factory_list,
                    },
                ),
            )

        for event in same_time_events["rus_completion"]:
            factory_pool.free_factories(event["factory_list"])

        for event in same_time_events["op_complete"]:
            node = event["node"]
            mark_node_completed(node)

        schedule_ready_clifford_operations(current_time)

    return execution_log
