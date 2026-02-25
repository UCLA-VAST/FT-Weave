def lattice_index(x, y, L):
    return x + y * L


def add_zz_layer_cz_native(qc: list, pairs, theta, use_rz=False):
    """
    Add ZZ layer using CZ and H decomposition of CX.

    CX-RZ-CX  →  H CZ RX CZ H
    """
    targets = [j for i, j in pairs]
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
    if use_rz:
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
    n_qubits: int, qubit_layout: tuple, J, h, dt, use_rz=False
):
    """
    2D TFIM circuit using CZ-native interactions.
    Internal H gates from CX decomposition are cancelled.
    """

    qc: list[dict] = []

    theta_zz = J * dt
    theta_x = h * dt

    horizontal_even = []
    horizontal_odd = []
    vertical_even = []
    vertical_odd = []
    row, col = qubit_layout
    for x in range(col):
        for y in range(row):

            if y < row - 1:
                q1 = lattice_index(x, y, col)
                q2 = lattice_index(x, y + 1, col)
                if y % 2 == 0:
                    vertical_even.append((q1, q2))
                else:
                    vertical_odd.append((q1, q2))

            if x < col - 1:
                q1 = lattice_index(x, y, col)
                q2 = lattice_index(x + 1, y, col)
                if x % 2 == 0:
                    horizontal_even.append((q1, q2))
                else:
                    horizontal_odd.append((q1, q2))
    # ZZ layers (CZ-native)
    add_zz_layer_cz_native(qc, horizontal_even, theta_zz, use_rz)
    add_zz_layer_cz_native(qc, horizontal_odd, theta_zz, use_rz)
    add_zz_layer_cz_native(qc, vertical_even, theta_zz, use_rz)
    add_zz_layer_cz_native(qc, vertical_odd, theta_zz, use_rz)

    # Transverse field layer
    targets = ([q for q in range(n_qubits)],)
    if use_rz:
        qc.append(
            {
                "gate": "H",
                "targets": targets,
                "params": {},
            }
        )
        qc.append(
            {
                "gate": "RZ",
                "targets": targets,
                "params": {"theta": 2 * theta_x},
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
                "gate": "RX",
                "targets": targets,
                "params": {"theta": 2 * theta_x},
            }
        )

    qc = cancel_hadamard(qc)
    return qc
