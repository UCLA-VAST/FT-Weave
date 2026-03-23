from qiskit.synthesis import gridsynth_rz
from numbers import Real


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


def gridsynth_rz_templates(theta: float) -> list[dict]:
    """Return a single-qubit gate template list from qiskit gridsynth for Rz(theta)."""
    synthesized_circuit = gridsynth_rz(theta)
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


def expand_multi_target_layers(circuit: list[dict], to_decompose: bool) -> list[dict]:
    """Expand layer instructions into sequential individual instructions.

        Supported expansions:
        - Rz layers are always expanded to single-target instructions and decomposed
            via qiskit gridsynth, regardless of `to_decompose`.
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
            rz_templates = gridsynth_rz_templates(float(theta))
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
                rz_templates = gridsynth_rz_templates(float(theta))

            for target in targets:
                for template in rz_templates:
                    gate_instr = {
                        "gate": template["gate"],
                        "targets": [target],
                        "params": dict(template["params"]),
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
