def h_gate(qubits: list[int], locations: list[tuple[int, int]]):
    """
    apply h gate to a list of qubits

    Args:
        qubits: a list of qubits to perfrom H gates
        locations: (x, y) location for each qubit

    Returns:
        inst: A sequence of instruction to perform h gates
    """
    pass


def s_gate(qubits: list[int], locations: list[tuple[int, int]]):
    """
    apply s gate to a list of qubits

    Args:
        qubits: a list of qubits to perfrom s gates
        locations: (x, y) location for each qubit

    Returns:
        inst: A sequence of instruction to perform s gates
    """
    pass


def rz_gate(
    qubits: list[int],
    angles: list[float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
):
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
    pass


def cx_gate(
    mobile_qubits: list[int], pivot_qubits: list[int], locations: list[tuple[int, int]]
):
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
    assert len(mobile_qubits) == len(pivot_qubits)
    pass
