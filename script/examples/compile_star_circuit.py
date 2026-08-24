#!/usr/bin/env python3
"""Minimal example: compile a general circuit on the STAR architecture.

STAR injects arbitrary-angle rotations directly, so the input is a
**Clifford+Rz** circuit (no bare ``T``/``Tdg``). Two equivalent entry points are
shown:

  1. OpenQASM file  -> internal instruction IR
  2. Hand-built list[dict] IR (same format as the TFIM generators)

Run from the repo root::

    uv run script/examples/compile_star_circuit.py

With the bundled sample circuit::

    uv run script/examples/compile_star_circuit.py \\
        --qasm script/examples/minimal_clifford_rz.qasm

See ``compile_t_cultivation_circuit.py`` for the T-cultivation counterpart.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.animator import plot_circuit_execution
from src.animator.pipeline_utils import merge_layer_logs_with_global_timeline
from src.circuit.circuit_info import count_qubits
from src.circuit.gate_set import validate_circuit_for_backend
from src.circuit.layer_partition import partition_into_layers
from src.circuit.placement import resolve_qubit_layout
from src.circuit.qasm_loader import load_qasm_file
from src.star.general_circuit_star import compile_circuit_star
from src.util import execution_log_wall_time

_DEFAULT_QASM = os.path.join(os.path.dirname(__file__), "minimal_clifford_rz.qasm")


def build_hand_circuit() -> list[dict]:
    """Same circuit as minimal_clifford_rz.qasm, built programmatically."""

    return [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "Rz", "targets": [0], "params": {"angles": {0: 0.31}}},
        {"gate": "Rz", "targets": [1], "params": {"angles": {1: 0.17}}},
        {"gate": "CNOT", "targets": [(0, 1)], "params": {}},
        {"gate": "CNOT", "targets": [(1, 2)], "params": {}},
        {"gate": "CNOT", "targets": [(0, 3)], "params": {}},
        {"gate": "H", "targets": [3], "params": {}},
        {"gate": "Rz", "targets": [1], "params": {"angles": {1: 0.42}}},
        {"gate": "Rz", "targets": [3], "params": {"angles": {3: 0.08}}},
    ]


def compile_for_star(
    flat_circuit: list[dict],
    *,
    code_distance: int = 7,
    n_aods: int = 2,
    placement: str = "col_based",
    parallel_execution: bool = False,
    seed: int = 0,
) -> tuple[list[dict], list[list[dict]], int]:
    """Compile *flat_circuit* and return (layered circuit, execution logs, n_qubits)."""
    validate_circuit_for_backend(flat_circuit, backend="star")

    n_qubits = count_qubits(flat_circuit)
    qubit_layout = resolve_qubit_layout(n_qubits)

    # Consecutive Rz rotations between Clifford blocks are merged into one
    # layer, so a whole layer of rotations is injected in a single STAR round.
    circuit = partition_into_layers(flat_circuit)

    # See ``SETTINGS`` in script/evaluation_fidelity_star.py for what these mean.
    config = {
        "n_aods": n_aods,
        "rng": np.random.default_rng(seed),
        "consider_skip_rus": 2,
        "trivial_return": False,
        "decompose_move": True,
        "tmr_assignment_method": "matching",
        "prepare_lookahead_angles": True,
        "save_log": False,
    }

    layered_circuit, execution_logs, _profiling = compile_circuit_star(
        circuit,
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
        code_distance=code_distance,
        config=config,
        parallel_execution=parallel_execution,
        analyze_result=False,
    )
    return layered_circuit, execution_logs, n_qubits


def plot_execution(
    execution_logs: list[list[dict]],
    *,
    n_qubits: int,
    n_factories: int | None = None,
    save_path: str,
) -> None:
    """Plot a timeline with logical-qubit rows (q*) and factory rows (f*)."""
    if n_factories is None:
        n_factories = n_qubits

    log = merge_layer_logs_with_global_timeline(execution_logs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_circuit_execution(
        log,
        n_factories=n_factories,
        n_logical_qubits=n_qubits,
        show_logical_qubits=True,
        save_path=save_path,
    )


def _summarize(layered_circuit: list[dict], execution_logs: list[list[dict]]) -> None:
    print("Layered circuit:")
    for i, instr in enumerate(layered_circuit):
        print(f"  [{i}] {instr['gate']:5s}  targets={instr['targets']}")

    print("\nExecution logs:")
    for i, log in enumerate(execution_logs):
        wall = execution_log_wall_time(log)
        ops: dict = {}
        for entry in log:
            op = entry.get("operation")
            ops[op] = ops.get(op, 0) + 1
        print(f"  log[{i}]  events={len(log)}  wall_time={wall:.2f}  ops={ops}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qasm",
        default=_DEFAULT_QASM,
        help="OpenQASM 2.0 file (default: bundled minimal_clifford_rz.qasm)",
    )
    parser.add_argument(
        "--hand-built",
        action="store_true",
        help="Use the programmatic hand-built circuit instead of QASM",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--code-distance", type=int, default=7)
    parser.add_argument("--n-aods", type=int, default=2)
    parser.add_argument(
        "--placement",
        default="col_based",
        help="Magic-state microarchitecture (col_based, seperate_region_row, ...)",
    )
    parser.add_argument(
        "--parallel-execution",
        action="store_true",
        help="Asynchronous (per-factory) execution instead of synchronous",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save execution timeline PDF (logical qubits + factory lanes)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            _REPO_ROOT, "output/circuit_execution/examples/minimal_clifford_rz.pdf"
        ),
        help="PDF path when --plot is set",
    )
    args = parser.parse_args()

    if args.hand_built:
        print("Input: hand-built Clifford+Rz circuit\n")
        flat = build_hand_circuit()
    else:
        print(f"Input: {args.qasm}\n")
        flat = load_qasm_file(args.qasm)

    layered, logs, n_qubits = compile_for_star(
        flat,
        code_distance=args.code_distance,
        n_aods=args.n_aods,
        placement=args.placement,
        parallel_execution=args.parallel_execution,
        seed=args.seed,
    )

    mode = "asynchronous" if args.parallel_execution else "synchronous"
    print(f"Mode: {mode} execution, placement={args.placement}, d={args.code_distance}")
    print(f"Qubits: {n_qubits}\n")
    _summarize(layered, logs)

    if args.plot:
        plot_execution(logs, n_qubits=n_qubits, save_path=args.output)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
