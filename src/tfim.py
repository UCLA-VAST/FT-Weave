from .gate_operation import cx_gate, h_gate, s_gate, rz_gate


# Plaquette-structured Trotter steps for 2D TFIM (Qiskit)
def lattice_index(x, y, Lx, Ly):
    return y * Lx + x


def plaquette_origins(Lx, Ly, parity):
    """
    Return (x,y) origins (top-left corner) of 2x2 plaquettes for given parity.
    parity=0 -> origins at (0,0), (2,0), (0,2), ...
    parity=1 -> origins at (1,0), (3,0), ...
    Ensures x+1 < Lx and y+1 < Ly (so plaquette fits).
    """
    xs = range(parity, Lx - 1, 2)
    ys = range(0, Ly - 1, 2)
    return [(x, y) for y in ys for x in xs]


def apply_edges_batch(
    circuit_instructions,
    edges,
    alpha,
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
):
    """
    edges: list of (i,j) pairs (disjoint pairs required for true parallelism).
    Implements each exp(-i alpha Z_i Z_j) for all edges in 'edges' in parallel:
      - do CX on all (i->j)
      - do RZ on all j (angles 2*alpha)
      - do CX on all (i->j)
    (Qiskit will still serialize gates, but this is the intended parallel schedule.)
    """
    # first CX round
    mobile_qubits, pivot_qubits = map(list, zip(*edges))
    circuit_instructions.append(
        cx_gate(mobile_qubits, pivot_qubits, logic_qubit_locations)
    )
    # RZ on targets
    angles = [
        alpha for i in range(len(pivot_qubits))
    ]  # todo: derive physical angles from the logical angles
    circuit_instructions.append(
        rz_gate(pivot_qubits, angles, logic_qubit_locations, magic_state_locations)
    )
    # second CX round
    circuit_instructions.append(
        cx_gate(mobile_qubits, pivot_qubits, logic_qubit_locations)
    )


def plaquette_edges_for_origin(x, y, Lx, Ly):
    """
    For plaquette with origin (x,y) (top-left),
    returns two sublists of edges: subA (top,bottom) and subB (left,right).
    Each edge is (i,j) where i is control, j is target (arbitrary choice).
    """
    a = lattice_index(x, y, Lx, Ly)
    b = lattice_index(x + 1, y, Lx, Ly)
    c = lattice_index(x, y + 1, Lx, Ly)
    d = lattice_index(x + 1, y + 1, Lx, Ly)
    # sublayer A: (a,b) and (c,d)
    subA = [(a, b), (c, d)]
    # sublayer B: (a,c) and (b,d)
    subB = [(a, c), (b, d)]
    return subA, subB


def tfim_trotter_plaquette(
    Lx,
    Ly,
    J,
    h,
    t,
    n_steps,
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    periodic=False,
):
    """
    Build Trotterized circuit using plaquettization scheduling.
    periodic flag ignored for plaquette tiling here (open BC assumed).
    """
    N = Lx * Ly
    circuit_instructions = []

    dt = t / n_steps
    alpha = -J * dt  # as in the earlier derivation: exp(-i alpha Z_iZ_j)

    for step in range(n_steps):
        # For each parity (0 then 1)
        for parity in (0, 1):
            origins = plaquette_origins(Lx, Ly, parity)
            # gather all edges for sublayer A and B across all plaquettes of this parity
            edges_A = []
            edges_B = []
            for x, y in origins:
                subA, subB = plaquette_edges_for_origin(x, y, Lx, Ly)
                edges_A.extend(subA)
                edges_B.extend(subB)
            # apply sublayer A (all disjoint edges)
            if edges_A:
                apply_edges_batch(
                    circuit_instructions,
                    edges_A,
                    alpha,
                    logic_qubit_locations,
                    magic_state_locations,
                )
            # apply sublayer B
            if edges_B:
                apply_edges_batch(
                    circuit_instructions,
                    edges_B,
                    alpha,
                    logic_qubit_locations,
                    magic_state_locations,
                )

        # X-field layer (apply RX on every qubit)
        phi = -2.0 * h * dt  # RX(phi) = exp(-i phi/2 X) implements exp(i h dt X)
        all_qubits = [i for i in range(N)]
        circuit_instructions.append(h_gate(all_qubits, logic_qubit_locations))
        angles = [
            phi for i in range(N)
        ]  # todo: derive physical angles from the logical angles
        circuit_instructions.append(
            rz_gate(all_qubits, angles, logic_qubit_locations, magic_state_locations)
        )
        circuit_instructions.append(h_gate(all_qubits, logic_qubit_locations))

    return circuit_instructions
