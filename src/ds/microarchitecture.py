from itertools import product


def get_microarchitecture(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple[int, int],
    placement: str = "seperate_region",
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Generate microarchitecture layout for logic qubits and magic state factories.
    Args:
        n_qubits: Total number of logical qubits
        n_factories: Total number of magic state factories
        qubit_layout: (n_columns, n_rows) layout of the qubit grid
        placement: Placement method for factories ("seperate_region", "col_based", "checkerboard")
    Returns:
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory
    """
    n_columns, n_rows = qubit_layout
    if placement == "seperate_region_row":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        magic_state_locations = [
            (x, n_rows + y) for x, y in product(range(n_columns), range(n_rows))
        ]
    elif placement == "seperate_region_col":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        magic_state_locations = [
            (x + n_columns, y) for x, y in product(range(n_columns), range(n_rows))
        ]
    elif placement == "row_based":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(0, 2 * n_rows, 2))
        ]
        magic_state_locations = [
            (x, y) for x, y in product(range(n_columns), range(1, 2 * n_rows + 1, 2))
        ]
    elif placement == "col_based":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(0, 2 * n_columns, 2), range(n_rows))
        ]
        magic_state_locations = [
            (x, y) for x, y in product(range(1, 2 * n_columns + 1, 2), range(n_rows))
        ]
    elif placement == "checkerboard":
        logic_qubit_locations = []
        magic_state_locations = []
        for x, y in product(range(n_columns * 2), range(n_rows)):
            if (x + y) % 2 == 0:
                logic_qubit_locations.append((x, y))
            else:
                magic_state_locations.append((x, y))
    else:
        raise ValueError(f"Unknown placement strategy: {placement}")
    return logic_qubit_locations[:n_qubits], magic_state_locations[:n_factories]
