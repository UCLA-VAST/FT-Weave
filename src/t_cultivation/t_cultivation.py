import logging

import heapq
from collections import deque
from itertools import count

from .config import CNOT_TIME, SE_STAGE_1, SE_STAGE_2, SE_TIME
from .util import _build_circuit_dag, _expand_multi_target_layers
from src.ds.device_state_t import FactoryPool

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def t_cultivation_execution(
    circuit: list[dict], n_factories: int, to_decompose: bool = False
) -> list[dict]:
    """
    Execute a quantum circuit using T cultivation.

    Args:
        circuit: List of instruction dictionaries, where each instruction has the format:
                 {"gate": str, "targets": list[int] | list[tuple[int, int]], "params": dict}

    Returns:
        execution_log: List of executed instructions with timestamps and details.
    """
    if n_factories < 0:
        raise ValueError("n_factories must be non-negative")

    if not circuit:
        return []

    if to_decompose:
        expanded_circuit = _expand_multi_target_layers(circuit)
    else:
        expanded_circuit = circuit

    predecessors, successors = _build_circuit_dag(expanded_circuit)
    remaining_deps = {node: len(deps) for node, deps in predecessors.items()}

    ready_clifford: deque[int] = deque()
    ready_t: deque[int] = deque()
    for node, dep_count in remaining_deps.items():
        if dep_count == 0:
            if expanded_circuit[node]["gate"] == "T":
                ready_t.append(node)
            else:
                ready_clifford.append(node)

    factory_pool = FactoryPool(n_factories)
    events: list[tuple[int, int, dict]] = []
    event_counter = count()
    execution_log: list[dict] = []

    completed = 0
    current_time = 0

    def schedule_ready_operations(at_time: int):
        while ready_clifford:
            node = ready_clifford.popleft()
            instr = expanded_circuit[node]
            gate = instr["gate"]
            duration = CNOT_TIME if gate in {"CNOT", "CZ"} else SE_TIME
            finish_time = at_time + duration
            execution_log.append(
                {
                    "time": at_time,
                    "action": f"start_{gate}",
                    "targets": instr.get("targets", []),
                    "details": {
                        "kind": "Clifford",
                        "duration": duration,
                        "params": instr.get("params", {}),
                        "op_index": node,
                    },
                }
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

        while ready_t and available_factories:
            node = ready_t.popleft()
            factory_id = available_factories.pop()
            instr = expanded_circuit[node]
            duration = _gate_duration("T")
            finish_time = at_time + duration
            execution_log.append(
                {
                    "time": at_time,
                    "action": "start_T",
                    "targets": instr.get("targets", []),
                    "details": {
                        "kind": "T",
                        "duration": duration,
                        "params": instr.get("params", {}),
                        "factory_id": factory_id,
                        "op_index": node,
                    },
                }
            )
            heapq.heappush(
                events,
                (
                    finish_time,
                    next(event_counter),
                    {
                        "type": "op_complete",
                        "node": node,
                        "factory_id": factory_id,
                    },
                ),
            )

    schedule_ready_operations(current_time)

    while completed < len(expanded_circuit):
        if not events:
            remaining_nodes = [
                index
                for index in range(len(expanded_circuit))
                if remaining_deps[index] >= 0
                and not any(
                    log["details"].get("op_index") == index
                    and log["action"].startswith("finish_")
                    for log in execution_log
                )
            ]
            raise RuntimeError(
                "No schedulable events remain. Circuit may contain a dependency cycle or unsatisfiable resource constraints. "
                f"Remaining operation indices: {remaining_nodes}"
            )

        current_time, _, first_event = heapq.heappop(events)
        same_time_events = [first_event]
        while events and events[0][0] == current_time:
            _, _, event = heapq.heappop(events)
            same_time_events.append(event)

        for event in same_time_events:
            node = event["node"]
            instr = expanded_circuit[node]
            gate = instr["gate"]
            gate_kind = _gate_kind(gate)

            if event["factory_id"] is not None:
                available_factories.append(event["factory_id"])

            execution_log.append(
                {
                    "time": current_time,
                    "action": f"finish_{gate}",
                    "targets": instr.get("targets", []),
                    "details": {
                        "kind": gate_kind,
                        "params": instr.get("params", {}),
                        "factory_id": event["factory_id"],
                        "op_index": node,
                    },
                }
            )
            completed += 1

            for successor in successors[node]:
                remaining_deps[successor] -= 1
                if remaining_deps[successor] == 0:
                    successor_kind = _gate_kind(expanded_circuit[successor]["gate"])
                    if successor_kind == "T":
                        ready_t.append(successor)
                    else:
                        ready_clifford.append(successor)

        schedule_ready_operations(current_time)

    return execution_log
