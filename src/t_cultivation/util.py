from __future__ import annotations

from numbers import Real
from typing import Optional

import numpy as np
from qiskit.synthesis import gridsynth_rz
from scipy.optimize import linear_sum_assignment

from src.ds import TFactoryPool, move_duration
from src.execution_log import write_execution_log


_GRIDSYNTH_GATE_MAP = {
    "h": "H",
    "s": "S",
    "sdg": "Sdg",
    "t": "T",
    "tdg": "Tdg",
    "x": "X",
    "y": "Y",
    "z": "Z",
}


def is_rz_gate(gate_name: str) -> bool:
    return gate_name.lower() == "rz"


def count_t_gates_in_instructions(instructions: list[dict]) -> int:
    """Count T and Tdg gates in a flat instruction list (e.g. expanded circuit)."""
    return sum(1 for inst in instructions if inst.get("gate") in ("T", "Tdg"))


def t_gate_count_per_unique_rz_angle(
    circuit: list[dict], epsilon: float, angle_digits: int = 12
) -> dict[float, int]:
    """For each distinct Rz theta in *circuit*, T+Tdg count for one Rz(theta) via gridsynth."""
    per_angle: dict[float, int] = {}
    for instr in circuit:
        if not is_rz_gate(instr.get("gate", "")):
            continue
        theta_raw = instr.get("params", {}).get("theta")
        if not isinstance(theta_raw, Real):
            raise ValueError("Rz instruction requires numeric params['theta']")
        theta = float(theta_raw)
        key = round(theta, angle_digits)
        if key in per_angle:
            continue
        templates = gridsynth_rz_templates(theta, epsilon=epsilon)
        per_angle[key] = sum(1 for t in templates if t["gate"] in ("T", "Tdg"))
    return per_angle


def gridsynth_rz_templates(theta: float, epsilon: float) -> list[dict]:
    """Return a single-qubit gate template list from qiskit gridsynth for Rz(theta)."""
    synthesized_circuit = gridsynth_rz(theta, epsilon=epsilon)
    templates: list[dict] = []

    for circuit_instruction in synthesized_circuit.data:
        qiskit_gate_name = circuit_instruction.operation.name.lower()
        mapped_gate_name = _GRIDSYNTH_GATE_MAP.get(qiskit_gate_name)
        if mapped_gate_name is None:
            raise ValueError(
                f"Unsupported gridsynth gate '{qiskit_gate_name}' for Rz decomposition"
            )
        templates.append({"gate": mapped_gate_name, "params": {}})

    if not templates:
        return [{"gate": "Rz", "params": {"theta": theta}}]

    return templates


def extract_target_qubits(targets: list) -> list[int]:
    """Extract qubit ids from targets that may contain ints or 2-qubit tuples/lists."""
    qubits: list[int] = []
    for target in targets:
        if isinstance(target, int):
            qubits.append(target)
        elif isinstance(target, (tuple, list)) and len(target) == 2:
            q0, q1 = target
            if not isinstance(q0, int) or not isinstance(q1, int):
                raise ValueError(
                    f"Invalid two-qubit target {target}; qubit ids must be ints"
                )
            qubits.extend([q0, q1])
        else:
            raise ValueError(
                f"Invalid target {target}; expected int or tuple/list with two ints"
            )
    return qubits


def expand_multi_target_layers(
    circuit: list[dict], epsilon: float, to_decompose: bool
) -> list[dict]:
    """Expand layer instructions into sequential individual instructions.

    Supported expansions:
    - Rz layers are decomposed via qiskit gridsynth, regardless of `to_decompose`.
        For **multiple qubits sharing the same Rz angle**, the gridsynth template is
        identical on every qubit: non-injection gates (**H**, **S**, **Sdg**, **X**,
        **Y**, **Z**, …) are emitted as **one multi-target instruction per template
        step** (same layer on all those qubits). **T** / **Tdg** steps stay **one
        instruction per qubit** so factory RUS injection remains per-T.
    - Bare T/Tdg layers are always expanded to single-target instructions, regardless
        of `to_decompose`.
    - For non-Rz gates, expansion occurs only when `to_decompose=True`:
        - Single-qubit layer: [q0, q1, q2] -> one instruction per qubit.
        - Two-qubit layer: [(q0, q1), (q2, q3)] -> one instruction per pair.

    """
    expanded_circuit: list[dict] = []

    rz_templates_by_instruction: dict[int, list[dict]] = {}
    for original_index, instr in enumerate(circuit):
        targets = instr.get("targets", [])
        gate_name = instr.get("gate", "")

        if not isinstance(targets, list):
            raise ValueError("Instruction targets must be a list")

        if is_rz_gate(gate_name):
            if not all(isinstance(target, int) for target in targets):
                raise ValueError("Rz targets must be a list of qubit indices")
            theta = instr.get("params", {}).get("theta")
            if not isinstance(theta, Real):
                raise ValueError("Rz instruction requires numeric params['theta']")
            rz_templates = gridsynth_rz_templates(float(theta), epsilon=epsilon)
            rz_templates_by_instruction[original_index] = rz_templates

    for original_index, instr in enumerate(circuit):
        targets = instr.get("targets", [])
        gate_name = instr.get("gate", "")

        if is_rz_gate(gate_name):
            if not all(isinstance(target, int) for target in targets):
                raise ValueError("Rz targets must be a list of qubit indices")

            rz_templates = rz_templates_by_instruction.get(original_index)
            if rz_templates is None:
                theta = instr.get("params", {}).get("theta")
                if not isinstance(theta, Real):
                    raise ValueError("Rz instruction requires numeric params['theta']")
                rz_templates = gridsynth_rz_templates(float(theta), epsilon=epsilon)

            # Template-major order: for each gridsynth step, either one parallel
            # layer on all qubits (H/S/…) or individual T/Tdg per qubit (injection).
            for template in rz_templates:
                g = template["gate"]
                params = dict(template.get("params", {}))
                if g in ("T", "Tdg"):
                    for target in targets:
                        expanded_circuit.append(
                            {
                                "gate": g,
                                "targets": [target],
                                "params": params,
                            }
                        )
                elif g == "Rz":
                    # Rare gridsynth fallback; keep per-qubit Rz for scheduling parity.
                    for target in targets:
                        expanded_circuit.append(
                            {
                                "gate": g,
                                "targets": [target],
                                "params": params,
                            }
                        )
                else:
                    expanded_circuit.append(
                        {
                            "gate": g,
                            "targets": list(targets),
                            "params": params,
                        }
                    )
            continue

        if gate_name in {"T", "Tdg"}:
            if not all(isinstance(target, int) for target in targets):
                raise ValueError("T/Tdg targets must be a list of qubit indices")
            for target in targets:
                gate_instr = {
                    "gate": instr["gate"],
                    "targets": [target],
                    "params": instr.get("params", {}),
                }
                expanded_circuit.append(gate_instr)
            continue

        if not to_decompose:
            single_instr = dict(instr)
            if "depends_on" in single_instr:
                single_instr.pop("depends_on")
            expanded_circuit.append(single_instr)
            continue

        has_single_qubit_targets = all(isinstance(target, int) for target in targets)
        has_pair_targets = any(
            isinstance(target, (tuple, list)) and len(target) == 2 for target in targets
        )

        if has_single_qubit_targets and len(targets) > 1:
            for target in targets:
                gate_instr = {
                    "gate": instr["gate"],
                    "targets": [target],
                    "params": instr.get("params", {}),
                }

                expanded_circuit.append(gate_instr)
            continue

        if not has_pair_targets:
            single_instr = dict(instr)
            if "depends_on" in single_instr:
                single_instr.pop("depends_on")
            expanded_circuit.append(single_instr)
            continue

        for target in targets:
            if not (isinstance(target, (tuple, list)) and len(target) == 2):
                raise ValueError(
                    "Mixed single- and two-qubit targets in one instruction are unsupported. "
                    "Please split them into separate instructions."
                )

        for target in targets:
            gate_instr = {
                "gate": instr["gate"],
                "targets": list(target),
                "params": instr.get("params", {}),
            }

            expanded_circuit.append(gate_instr)

    return expanded_circuit


def build_circuit_dag(
    circuit: list[dict],
) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    """Build DAG dependencies from explicit deps and qubit ordering constraints."""
    predecessors: dict[int, set[int]] = {index: set() for index in range(len(circuit))}
    successors: dict[int, list[int]] = {index: [] for index in range(len(circuit))}

    # Implicit per-qubit dependencies from operation order in circuit list.
    last_op_on_qubit: dict[int, int] = {}
    for index, instr in enumerate(circuit):
        targets = instr.get("targets", [])
        for qubit in extract_target_qubits(targets):
            if qubit in last_op_on_qubit:
                predecessors[index].add(last_op_on_qubit[qubit])
            last_op_on_qubit[qubit] = index

    # Materialize successor lists.
    for node, deps in predecessors.items():
        for dep in deps:
            successors[dep].append(node)

    return predecessors, successors


def compute_longest_path_to_sink(
    successors: dict[int, list[int]],
    num_nodes: int,
) -> list[int]:
    """
    For each DAG node, length of the longest path starting at that node (inclusive)
    following successor edges toward sinks. T/Tdg nodes with more downstream work
    get larger values — used to prioritize RUS assignment toward the critical path.

    Implemented with a single backward pass (no recursion). Edges from
    ``build_circuit_dag`` always go from a smaller instruction index to a larger
    one, so processing ``num_nodes-1 .. 0`` suffices.
    """
    dp = [1] * num_nodes
    for u in range(num_nodes - 1, -1, -1):
        outs = successors.get(u, [])
        if outs:
            dp[u] = 1 + max(dp[v] for v in outs)
    return dp


# ---------------------------------------------------------------------------
# T-cultivation stage 1: redistribution and factory state (not random draws)
# ---------------------------------------------------------------------------


def raw_stage1_counts(
    factory_pool: TFactoryPool, factory_ids: list[int]
) -> dict[int, int]:
    """Count True stage_1_success flags per factory."""
    out: dict[int, int] = {}
    for fid in factory_ids:
        f = factory_pool.get_factory_by_id(fid)
        out[fid] = sum(1 for sf in f.subfactories if sf.stage_1_success)
    return out


def redistribute_stage1_successes(
    factory_pool: TFactoryPool,
    factory_ids: list[int],
    counts: dict[int, int],
    execution_log: Optional[list],
    log_time: float,
    aod_id: int,
) -> list[tuple[int, int]]:
    """
    Move spare stage-1 successes from donors (count >= 2) to receivers (count == 0)
    using minimum total move cost (Hungarian / linear sum assignment).
    Mutates ``counts`` in place. Returns list of (donor_factory_id, receiver_factory_id).
    """
    donor_units: list[int] = []
    for fid in factory_ids:
        c = counts[fid]
        donor_units.extend([fid] * max(0, c - 1))

    receivers = [fid for fid in factory_ids if counts[fid] == 0]
    transfers: list[tuple[int, int]] = []

    if not receivers or not donor_units:
        return transfers

    n_r, n_d = len(receivers), len(donor_units)
    cost = np.zeros((n_r, n_d))
    for i, r_fid in enumerate(receivers):
        r_loc = factory_pool.get_factory_by_id(r_fid).location
        for j, d_fid in enumerate(donor_units):
            d_loc = factory_pool.get_factory_by_id(d_fid).location
            cost[i, j] = move_duration(r_loc[0], r_loc[1], d_loc[0], d_loc[1])

    row_ind, col_ind = linear_sum_assignment(cost)

    for ri, dj in zip(row_ind, col_ind):
        recv_fid = receivers[ri]
        don_fid = donor_units[dj]
        counts[recv_fid] += 1
        counts[don_fid] -= 1
        transfers.append((don_fid, recv_fid))

    if execution_log is not None:
        for don_fid, recv_fid in transfers:
            dloc = factory_pool.get_factory_by_id(don_fid).location
            rloc = factory_pool.get_factory_by_id(recv_fid).location
            write_execution_log(
                execution_log,
                start_time=log_time,
                factories=[don_fid, recv_fid],
                operation="move",
                aod_assignment=aod_id,
                targets=None,
                movement_time=move_duration(dloc[0], dloc[1], rloc[0], rloc[1]),
                move_vecs=[
                    [
                        f"({dloc[0]},{dloc[1]})",
                        f"({rloc[0]},{rloc[1]})",
                    ]
                ],
            )

    return transfers


def finalize_stage1_outcomes(
    factory_pool: TFactoryPool,
    factory_ids: list[int],
    counts: dict[int, int],
) -> list[int]:
    """
    Apply post-redistribution counts to subfactories, then move each factory to stage 2
    or idle. Returns factory ids that pass stage 1.
    """
    success_factory_ids: list[int] = []
    for factory_id in factory_ids:
        factory = factory_pool.get_factory_by_id(factory_id)
        # Synchronize factory subfactory states with post-redistribution counts.
        n_success = max(0, min(counts.get(factory_id, 0), len(factory.subfactories)))
        for i, sf in enumerate(factory.subfactories):
            sf.stage_1_success = i < n_success
        if factory.stage_1_passed():
            factory.set_stage_2_state()
            success_factory_ids.append(factory_id)
        else:
            factory.free()
    return success_factory_ids


def complete_stage1_preparation(
    factory_pool: TFactoryPool,
    factory_ids: list[int],
    execution_log: Optional[list],
    log_time: float,
    aod_id: int,
) -> tuple[list[int], float]:
    """
    After ``simulate_stage1_preparation``: redistribute spare successes, sync subfactories,
    and advance factories to stage 2 or free.
    """
    counts = raw_stage1_counts(factory_pool, factory_ids)
    transfers = redistribute_stage1_successes(
        factory_pool,
        factory_ids,
        counts,
        execution_log,
        log_time,
        aod_id,
    )
    redistribution_delay = 0.0
    for donor_id, receiver_id in transfers:
        donor = factory_pool.get_factory_by_id(donor_id).location
        receiver = factory_pool.get_factory_by_id(receiver_id).location
        redistribution_delay = max(
            redistribution_delay,
            move_duration(donor[0], donor[1], receiver[0], receiver[1]),
        )

    success_factory_ids = finalize_stage1_outcomes(factory_pool, factory_ids, counts)
    return success_factory_ids, redistribution_delay
