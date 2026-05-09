"""Logical fidelity estimation for T-cultivation execution of TFIM layers.

Mirrors ``star_fiedlity_simulation.py``. ``execution_logs`` is a list whose
entries are per-instruction logs (one inner list per gate in
``qc_one_layer``: Rz rounds plus interleaved Clifford layers). The simulator
flattens the full list and walks every entry.

The execution log contains two distinct kinds of CNOT entries that we must
keep separate when accounting for fidelity:

1. Circuit-level Clifford CNOT - logical CNOT between two logical qubits
   emitted by the compiled TFIM circuit. These entries have an empty
   ``factories`` list. T-cultivation writes three entries per circuit CNOT
   (a move-only one, the actual gate, and a return-move-only one), all with
   ``operation="CNOT"``; only the actual-gate entry has ``move_vecs is None``
   and ``targets`` shaped as ``list[tuple[int, int]]``. We count only the
   actual-gate entries into ``n_cnot`` / ``fidelity_cnot``.

2. Teleportation/RUS injection CNOT - logical CNOT between a T-factory and
   its target logical qubit. These entries have a non-empty ``factories``
   list and ``targets`` is a parallel list of bare qubit ids. They contribute
   to ``n_cnot_teleportation`` / ``fidelity_of_rz_teleportaion`` and each
   such CNOT also consumes one logical T magic state, which contributes to
   ``fidelity_of_t_gate``.

The discriminator is ``factories``: empty -> circuit CNOT, non-empty ->
teleportation CNOT.

Idle error: each ``SE_q`` log entry covers one syndrome-extraction round on
every qubit in its ``targets`` list. We sum the per-entry target counts to
get ``n_se_q`` total qubit-SE rounds, then ``fidelity_idle = (1 - p_I) ^
n_se_q`` where ``p_I = get_logical_error_rate('I')``.

T-decomposition (gridsynth) error: every per-qubit ``Rz(theta)`` in the
logical circuit is approximated by a Clifford+T template whose operator
distance from the exact rotation is at most ``synthesis_epsilon``. The
resulting state infidelity per Rz is ~``synthesis_epsilon ** 2``, so the
total synthesis fidelity is
``fidelity_synthesis = (1 - synthesis_epsilon ** 2) ** n_rz``. When
``synthesis_epsilon`` is ``None`` (the default), this term is set to 1.0
so older callers see no behavioural change.
"""

from __future__ import annotations

from src.error_model import LogicalErrorModel


def simluate_trotter_2d_tfim_fidelity(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple,
    n_trotter_steps: int,
    qc_one_layer: list[dict],
    execution_logs: list[list[dict]],
    logical_error_model: LogicalErrorModel,
    synthesis_epsilon: float | None = None,
) -> dict:
    """
    Estimate logical fidelity for TFIM evolution under T-cultivation.

    H=-J sum_{i,j} Z_iZ_j - h sum_i X_i. One Trotter step uses the same logical
    layer as STAR (CNOT / Rz / H in ``qc_one_layer``); Rz are executed as T/Tdg
    via RUS in the simulator.

    Args:
        n_qubits: Number of logical qubits.
        n_factories: Number of T factories (used to size factory state table).
        qubit_layout: (rows, cols) layout (same convention as elsewhere).
        n_trotter_steps: Number of Trotter steps; scales the per-layer counts.
        qc_one_layer: One logical TFIM layer (``logical=True`` from
            ``generate_one_layer_2d_tfim_circuit_cz``). Used for layer
            metadata only - the per-CNOT / per-H counts are taken from the
            execution log so the same factories/targets discriminator works.
        execution_logs: Output of ``t_cultivation_execution`` - a list with
            one inner log per gate in ``qc_one_layer`` (Rz rounds plus
            Clifford layers, in order).
        logical_error_model: Surface-code logical error model. For T-cultivation,
            call ``logical_error_model.integrate_t_cultivation_fidelity_target(p)``
            so ``get_logical_fidelity('T')`` uses logical T fidelity ``1 - p`` (default).
        synthesis_epsilon: Gridsynth approximation accuracy used when
            decomposing each per-qubit ``Rz(theta)`` into a Clifford+T
            template (the same value passed to ``t_cultivation_execution``
            via ``config['epsilon']``). Per-Rz state infidelity is taken as
            ``epsilon ** 2``, so total decomposition fidelity is
            ``(1 - epsilon ** 2) ** n_rz``. ``None`` disables this term.

    Returns:
        Fidelity profile dict aligned with STAR naming where possible.
    """
    _row, _col = qubit_layout

    n_cnot = 0
    n_cnot_teleportation = 0
    n_h_in_rz = 0
    n_s = 0
    n_t = 0
    n_se_q = 0
    for log in execution_logs:
        if not isinstance(log, list):
            continue
        for entry in log:
            factories = entry.get("factories")
            operation = entry.get("operation")
            targets = entry.get("targets")
            move_vecs = entry.get("move_vecs")

            # Movement / barrier / outcome entries do not contribute to the
            # analytic fidelity here.
            if operation in (
                "move",
                "return_move",
                "Barrier",
                "RUS_success",
                "RUS_fail",
                # Pauli frame updates - tracked classically, no fidelity hit.
                "X",
                "Z",
                "Y",
            ):
                continue
            # Factory-side syndrome extraction - deferred from this analytic
            # model (factory fidelity is folded into the T-cultivation target
            # via ``integrate_t_cultivation_fidelity_target``).
            if operation in ("SE", "SE_stage_1", "SE_stage_2"):
                continue

            if operation == "SE_q":
                # Logical-qubit syndrome extraction: each target qubit
                # contributes one idle-error round. SE_q entries are
                # coalesced (one entry per moment with multi-qubit targets),
                # so total qubit-rounds = sum of |targets|.
                tlist = targets if isinstance(targets, list) else []
                n_se_q += len(tlist)
                continue

            if operation == "CNOT":
                fac_list = (
                    list(factories)
                    if isinstance(factories, (list, tuple))
                    else ([factories] if factories is not None else [])
                )
                tgt_list = (
                    list(targets) if isinstance(targets, (list, tuple)) else []
                )
                if not fac_list:
                    # Empty factories => circuit-level Clifford CNOT.
                    # T-cultivation writes 3 entries per circuit CNOT (move,
                    # actual CNOT, return-move), all tagged "CNOT". The
                    # actual-gate entry has ``move_vecs is None`` and
                    # ``targets`` is a list of (control, target) pairs;
                    # the two movement entries have ``move_vecs`` set and
                    # ``targets`` as a list of ints. We count only the
                    # actual-gate entry to avoid triple-counting.
                    if move_vecs is None:
                        n_cnot += len(tgt_list)
                else:
                    # Non-empty factories => teleportation/RUS CNOT.
                    # ``factories[i]`` is paired with ``targets[i]``; each
                    # (factory, qubit) pair is one logical CNOT that consumes
                    # one prepared T magic state.
                    for factory_id, _qubit in zip(fac_list, tgt_list):
                        if not isinstance(factory_id, int):
                            continue
                        if not (0 <= factory_id < n_factories):
                            continue
                        n_cnot_teleportation += 1
                        n_t += 1
                continue

            if operation in ("T", "Tdg"):
                assert(0), "Direct T/Tdg gates are not supported in T-cultivation. Should be realized via teleportation CNOTs"

            if operation == "S":
                n_s += 1
                continue

            if operation == "H":
                tlist = targets if isinstance(targets, list) else []
                n_h_in_rz += len(tlist) if tlist else 1
                continue

            raise ValueError(f"Unexpected operation in execution log: {operation}")

    # Scale Trotter-layer counts to the requested number of Trotter steps.
    n_cnot *= n_trotter_steps
    n_cnot_teleportation *= n_trotter_steps
    n_t *= n_trotter_steps
    n_h_in_rz *= n_trotter_steps
    n_s *= n_trotter_steps
    n_se_q *= n_trotter_steps

    # Total per-qubit Rz gates that go through gridsynth decomposition: each
    # ``Rz`` instruction in the layer fans out to ``len(targets)`` per-qubit
    # rotations, repeated for every Trotter step.
    n_rz_decomposition = (
        sum(
            len(inst.get("targets", []))
            for inst in qc_one_layer
            if inst.get("gate") == "Rz"
        )
        * n_trotter_steps
    )

    cnot_fid = logical_error_model.get_logical_fidelity("CNOT")
    fidelity_cnot = cnot_fid**n_cnot
    fidelity_of_rz_teleportaion = cnot_fid**n_cnot_teleportation
    fidelity_of_rz_s = logical_error_model.get_logical_fidelity("S") ** n_s
    fidelity_h = logical_error_model.get_logical_fidelity("H") ** n_h_in_rz
    # Idle-error contribution: each qubit-SE round costs one factor of the
    # idle (identity) logical fidelity.
    fidelity_idle = logical_error_model.get_logical_fidelity("I") ** n_se_q
    # Gridsynth approximation contribution: per-Rz state infidelity ~ eps**2.
    if synthesis_epsilon is None:
        fidelity_synthesis = 1.0
    else:
        eps = float(synthesis_epsilon)
        if eps < 0:
            raise ValueError(f"synthesis_epsilon must be non-negative, got {eps}")
        fidelity_synthesis = (1.0 - eps * eps) ** n_rz_decomposition

    # ``fidelity_of_rz_h`` is kept for output-compat: in the previous code it
    # tracked H gates emitted *inside* the Rz rounds, which is exactly
    # ``n_h_in_rz`` here, so it equals ``fidelity_h``.
    fidelity_of_rz_h = fidelity_h
    fidelity_of_rz_layer = (
        fidelity_of_rz_teleportaion
        * fidelity_of_rz_s
        * fidelity_of_rz_h
    )
    fidelity = (
        fidelity_of_rz_layer
        * fidelity_cnot
        * fidelity_idle
        * fidelity_synthesis
    )

    return {
        "fidelity": fidelity,
        "fidelity_of_rz_layer": fidelity_of_rz_layer,
        "fidelity_of_rz_teleportaion": fidelity_of_rz_teleportaion,
        "fidelity_of_rz_s": fidelity_of_rz_s,
        "fidelity_of_rz_h": fidelity_of_rz_h,
        "fidelity_cnot": fidelity_cnot,
        "fidelity_h": fidelity_h,
        "fidelity_idle": fidelity_idle,
        "fidelity_synthesis": fidelity_synthesis,
        "synthesis_epsilon": synthesis_epsilon,
        "n_cnot": n_cnot,
        "n_cnot_teleportation": n_cnot_teleportation,
        "n_h": n_h_in_rz,
        "n_s": n_s,
        "n_t": n_t,
        "n_se_q": n_se_q,
        "n_rz_decomposition": n_rz_decomposition,
    }
