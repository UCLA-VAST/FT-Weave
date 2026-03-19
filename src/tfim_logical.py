def lattice_index(x, y, L):
    return x + y * L


def add_zz_layer(qc: list, pairs, theta, logical=False):
    """
    Add ZZ layer using CZ and H decomposition of CX.

    CX-RZ-CX  →  H CZ RX CZ H
    """
    targets = [j for i, j in pairs]
    if logical:
        qc.append(
            {
                "gate": "CNOT",
                "targets": pairs,
                "params": {},
            }
        )
        qc.append(
            {
                "gate": "Rz",
                "targets": targets,
                "params": {"theta": 2 * theta},
            }
        )
        qc.append(
            {
                "gate": "CNOT",
                "targets": pairs,
                "params": {},
            }
        )
    else:
        qc.append(
            {
                "gate": "H",
                "targets": targets,
                "params": {},
            }
        )
        qc.append(
            {
                "gate": "CZ",
                "targets": pairs,
                "params": {},
            }
        )
        qc.append(
            {
                "gate": "Rx",
                "targets": targets,
                "params": {"theta": 2 * theta},
            }
        )
        qc.append(
            {
                "gate": "CZ",
                "targets": pairs,
                "params": {},
            }
        )
        qc.append(
            {
                "gate": "H",
                "targets": targets,
                "params": {},
            }
        )


def add_transverse_field_layer(qc: list, n_qubits: int, theta: float, logical=False):
    """
    Add transverse field (X) layer.

    For logical: H-Rz-H decomposition
    For physical: Rx gate
    """
    targets = [q for q in range(n_qubits)]
    if logical:
        qc.append(
            {
                "gate": "H",
                "targets": targets,
                "params": {},
            }
        )
        qc.append(
            {
                "gate": "Rz",
                "targets": targets,
                "params": {"theta": 2 * theta},
            }
        )
        qc.append(
            {
                "gate": "H",
                "targets": targets,
                "params": {},
            }
        )
    else:
        qc.append(
            {
                "gate": "Rx",
                "targets": targets,
                "params": {"theta": 2 * theta},
            }
        )


def cancel_hadamard(qc: list) -> list:
    """
    Merge consecutive H instructions and cancel out pairs on the same qubits.
    Two H gates applied to the same qubit cancel out.

    Example:
        {H, targets:[0, 1, 2]}, {H, targets:[0, 1, 3]}, {H, targets:[3, 4]}
        -> {H, targets:[2, 4]}

    Args:
        qc: List of instruction dictionaries

    Returns:
        New list of instructions with consecutive H gates merged and cancelled
    """
    new_qc = []
    i = 0

    while i < len(qc):
        if qc[i]["gate"] != "H":
            new_qc.append(qc[i])
            i += 1
        else:
            # Collect all consecutive H gates
            h_targets = set(qc[i]["targets"])
            j = i + 1

            while j < len(qc) and qc[j]["gate"] == "H":
                # XOR operation: add new targets, remove duplicates (cancellation)
                current_targets = set(qc[j]["targets"])
                h_targets = h_targets.symmetric_difference(current_targets)
                j += 1

            # Add merged H instruction if there are remaining targets
            if h_targets:
                new_qc.append(
                    {
                        "gate": "H",
                        "targets": sorted(list(h_targets)),
                        "params": {},
                    }
                )

            i = j

    return new_qc


def generate_one_layer_2d_tfim_circuit_cz(
    n_qubits: int, qubit_layout: tuple, J, h, dt, logical=False, periodic=True, order=1
):
    """
    2D TFIM circuit using CZ-native interactions.
    Internal H gates from CX decomposition are cancelled.

    With periodic=True, couplings wrap around both lattice dimensions.
    For the even/odd ZZ matching schedule to keep each Rz sublayer at N/2
    targets, the lattice must be even-by-even.

    Args:
        n_qubits: Number of qubits
        qubit_layout: Tuple of (rows, cols) for the qubit lattice
        J: ZZ coupling strength
        h: Transverse field strength
        dt: Time step
        logical: Whether to use logical (CNOT-based) gates
        periodic: Whether to use periodic boundary conditions
        order: Trotter order (1 for first-order, 2 for second-order)
    """

    qc: list[dict] = []

    theta_zz = J * dt
    theta_x = h * dt

    horizontal_even = []
    horizontal_odd = []
    vertical_even = []
    vertical_odd = []
    row, col = qubit_layout

    if periodic and (row % 2 != 0 or col % 2 != 0):
        raise ValueError(
            "Periodic scheduling requires an even-by-even lattice so each ZZ Rz layer acts on half the qubits."
        )

    for x in range(col):
        for y in range(row):

            if periodic or y < row - 1:
                q1 = lattice_index(x, y, col)
                q2 = lattice_index(x, (y + 1) % row, col)
                if y % 2 == 0:
                    vertical_even.append((q1, q2))
                else:
                    vertical_odd.append((q1, q2))

            if periodic or x < col - 1:
                q1 = lattice_index(x, y, col)
                q2 = lattice_index((x + 1) % col, y, col)
                if x % 2 == 0:
                    horizontal_even.append((q1, q2))
                else:
                    horizontal_odd.append((q1, q2))

    if order == 2:
        # Second-order Trotter: symmetric decomposition
        # exp(-i*dt*H) ≈ exp(-i*dt/2*H_zz) exp(-i*dt*H_x) exp(-i*dt/2*H_zz)
        theta_zz_half = theta_zz / 2

        # First half of ZZ layers
        add_zz_layer(qc, horizontal_even, theta_zz_half, logical)
        add_zz_layer(qc, horizontal_odd, theta_zz_half, logical)
        add_zz_layer(qc, vertical_even, theta_zz_half, logical)
        add_zz_layer(qc, vertical_odd, theta_zz_half, logical)

        # Full transverse field layer
        add_transverse_field_layer(qc, n_qubits, theta_x, logical)

        # Second half of ZZ layers
        add_zz_layer(qc, horizontal_even, theta_zz_half, logical)
        add_zz_layer(qc, horizontal_odd, theta_zz_half, logical)
        add_zz_layer(qc, vertical_even, theta_zz_half, logical)
        add_zz_layer(qc, vertical_odd, theta_zz_half, logical)

    else:
        # First-order Trotter (default)
        # ZZ layers (CZ-native)
        add_zz_layer(qc, horizontal_even, theta_zz, logical)
        add_zz_layer(qc, horizontal_odd, theta_zz, logical)
        add_zz_layer(qc, vertical_even, theta_zz, logical)
        add_zz_layer(qc, vertical_odd, theta_zz, logical)

        # Transverse field layer
        add_transverse_field_layer(qc, n_qubits, theta_x, logical)

    qc = cancel_hadamard(qc)
    return qc
