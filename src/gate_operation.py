from src.ds.logical_grid import LogicalGridManager


def h_gate(
    begin_inst_id: int,
    qubits: list[int],
    locations: list[tuple[int, int, int]],
    logical_grid: LogicalGridManager,
) -> list[dict]:
    """
    apply h gate to a list of qubits

    Args:
        qubits: a list of qubits to perfrom H gates
        locations: (x, y) location for each qubit

    Returns:
        inst: A sequence of instruction to perform h gates
    """
    return []


def s_gate(
    begin_inst_id: int,
    qubits: list[int],
    locations: list[tuple[int, int, int]],
    logical_grid: LogicalGridManager,
) -> list[dict]:
    """
    apply s gate to a list of qubits

    Args:
        qubits: a list of qubits to perfrom s gates
        locations: (x, y) location for each qubit

    Returns:
        inst: A sequence of instruction to perform s gates
    """
    return []


def rz_gate(
    begin_inst_id: int,
    qubits: list[int],
    angles: list[float],
    logic_qubit_locations: list[tuple[int, int, int]],
    magic_state_locations: list[tuple[int, int, int]],
    logical_grid: LogicalGridManager,
) -> list[dict]:
    """
    apply rz gate to a list of qubits with target angles

    Args:
        qubits: a list of qubits to perfrom rz gates
        angles: the target angles for qubits
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory

    Returns:
        inst: A sequence of instruction to perform rz gates
    """
    assert len(qubits) == len(angles)
    return []


def cx_gate(
    begin_inst_id: int,
    mobile_logical_qubits: list[int],
    pivot_logical_qubits: list[int],
    locations: list[tuple[int, int, int]],
    logical_grid: LogicalGridManager,
) -> list[dict]:
    """
    Apply cz gate between mobile_qubits and pivot_qubits.
    Assume the movement patterns are the same for all qubits

    Args:
        mobile_qubits: qubit to be moved
        pivot_qubits: target location
        locations: (x, y) location for each qubit

    Returns:
        inst: A sequence of instruction to perform cz gates
    """
    assert len(mobile_logical_qubits) == len(pivot_logical_qubits)
    mobile_qubits = []
    begin_location = []
    for qubit in mobile_logical_qubits:
        physical_qubit_indices, physical_qubit_locations = (
            logical_grid.physical_indices_location_for_logical_qubit(
                qubit, locations[qubit]
            )
        )
        mobile_qubits += physical_qubit_indices
        begin_location += physical_qubit_locations

    end_location = []
    for qubit in mobile_logical_qubits:
        physical_qubit_indices, physical_qubit_locations = (
            logical_grid.physical_indices_location_for_logical_qubit(
                qubit, locations[qubit]
            )
        )
        end_location += physical_qubit_locations

    inst_idx = begin_inst_id
    rearrange_prompt = {
        "type": "rearrangeJob",
        "id": inst_idx,
        "aod_id": 0,
        "aod_qubits": mobile_qubits,
        "begin_locs": begin_location,
        "end_locs": end_location,
        "dependency": {},
    }
    inst_idx += 1
    rydberg_prompt = {
        "type": "rydberg",
        "id": inst_idx,
        "zone_id": 0,
        "gates": [],
        "dependency": {},
    }
    inst_idx += 1
    rearrange_prompt_reverse = {
        "type": "rearrangeJob",
        "id": inst_idx,
        "aod_id": 0,
        "aod_qubits": mobile_qubits,
        "begin_locs": end_location,
        "end_locs": begin_location,
        "dependency": {},
    }

    insts = [rearrange_prompt, rydberg_prompt, rearrange_prompt_reverse]
    return insts
