#!/usr/bin/env python3
"""Minimal example: compile a general circuit on the T-cultivation architecture.

Two equivalent entry points are shown:
  1. OpenQASM file  -> internal instruction IR
  2. Hand-built list[dict] IR (same format as TFIM generators)

Run from the repo root:

    uv run script/examples/compile_t_cultivation_circuit.py

With the bundled sample circuit:

    uv run script/examples/compile_t_cultivation_circuit.py \\
        --qasm script/examples/minimal_clifford_t.qasm

Use --no-split-layers when the circuit has Clifford gates (e.g. CNOT) that must
go through the real scheduler instead of the animation stub.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.animator import plot_t_cultivation_execution
from src.animator.pipeline_utils import merge_layer_logs_with_global_timeline
from src.circuit.circuit_info import count_qubits
from src.circuit.gate_set import validate_circuit_for_backend
from src.circuit.layer_partition import partition_into_layers
from src.circuit.placement import resolve_qubit_layout
from src.circuit.qasm_loader import load_qasm_file
from src.t_cultivation.config import update_config
from src.t_cultivation.general_circuit_t import compile_circuit_t_cultivation
from src.util import execution_log_wall_time

_DEFAULT_QASM = os.path.join(os.path.dirname(__file__), "minimal_clifford_t.qasm")


def build_hand_circuit() -> list[dict]:
    """Same circuit as minimal_clifford_t.qasm, built programmatically."""

    return [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [1], "params": {}},
        {"gate": "CNOT", "targets": [(0, 1)], "params": {}},
        {"gate": "CNOT", "targets": [(1, 2)], "params": {}},
        {"gate": "CNOT", "targets": [(0, 3)], "params": {}},
        {"gate": "H", "targets": [3], "params": {}},
        {"gate": "T", "targets": [1], "params": {}},
        {"gate": "T", "targets": [3], "params": {}},
    ]


def compile_for_t_cultivation(
    flat_circuit: list[dict],
    *,
    split_layers: bool,
    code_distance: int = 7,
    n_aods: int = 2,
    placement: str = "col_based",
    seed: int = 0,
) -> tuple[list[dict], list[list[dict]], int]:
    """Compile *flat_circuit* and return (layered circuit, execution logs, n_qubits)."""
    validate_circuit_for_backend(flat_circuit, backend="t_cultivation")

    n_qubits = count_qubits(flat_circuit)
    qubit_layout = resolve_qubit_layout(n_qubits)

    circuit = partition_into_layers(flat_circuit) if split_layers else flat_circuit

    update_config(
        STAGE_2_FIDELITY_TARGET=1e-8,
        FACTORY_PHYSICAL_SIZE=2,
        STAGE_1_RESOURCE_UNITS=1,
    )
    config = {
        "n_aods": n_aods,
        "rng": np.random.default_rng(seed),
        "epsilon": 1e-4,
        "to_decompose": False,
        "trivial_return": False,
        "decompose_move": True,
        "redistribute_stage1_success": True,
    }

    layered_circuit, execution_logs, _profiling = compile_circuit_t_cultivation(
        circuit,
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
        code_distance=code_distance,
        config=config,
        split_layers=split_layers,
        analyze_result=False,
    )
    return layered_circuit, execution_logs, n_qubits


def plot_execution(
    execution_logs: list[list[dict]],
    *,
    n_qubits: int,
    n_factories: int | None = None,
    save_path: str,
    split_layers: bool,
) -> None:
    """Plot a timeline with logical-qubit rows (q*) and factory rows (f*)."""
    if n_factories is None:
        n_factories = n_qubits

    if split_layers and len(execution_logs) > 1:
        log = merge_layer_logs_with_global_timeline(execution_logs)
    elif len(execution_logs) == 1:
        log = execution_logs[0]
    else:
        log = merge_layer_logs_with_global_timeline(execution_logs)

    plot_t_cultivation_execution(
        execution_log=log,
        n_qubits=n_qubits,
        n_factories=n_factories,
        save_path=save_path,
    )


def _summarize(layered_circuit: list[dict], execution_logs: list[list[dict]]) -> None:
    print("Layered circuit:")
    for i, instr in enumerate(layered_circuit):
        print(f"  [{i}] {instr['gate']:5s}  targets={instr['targets']}")

    print("\nExecution logs:")
    for i, log in enumerate(execution_logs):
        wall = execution_log_wall_time(log)
        ops = {}
        for entry in log:
            op = entry.get("operation")
            ops[op] = ops.get(op, 0) + 1
        print(f"  log[{i}]  events={len(log)}  wall_time={wall:.2f}  ops={ops}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qasm",
        default=_DEFAULT_QASM,
        help="OpenQASM 2.0 file (default: bundled minimal_clifford_t.qasm)",
    )
    parser.add_argument(
        "--hand-built",
        action="store_true",
        help="Use the programmatic hand-built circuit instead of QASM",
    )
    split = parser.add_mutually_exclusive_group()
    split.add_argument(
        "--no-split-layers",
        action="store_true",
        help="One scheduler pass for the whole circuit (needed for real CNOT)",
    )
    split.add_argument(
        "--split-layers",
        action="store_true",
        help="Per non-Clifford layer (default); Clifford layers are stub logs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save execution timeline PDF (logical qubits + factory lanes)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            _REPO_ROOT, "output/circuit_execution/examples/minimal_clifford_t.pdf"
        ),
        help="PDF path when --plot is set",
    )
    args = parser.parse_args()

    split_layers = args.split_layers

    if args.hand_built:
        print("Input: hand-built Clifford+T circuit\n")
        flat = build_hand_circuit()
    else:
        print(f"Input: {args.qasm}\n")
        flat = load_qasm_file(args.qasm)

    layered, logs, n_qubits = compile_for_t_cultivation(
        flat, split_layers=split_layers, seed=args.seed
    )

    print(f"Mode: {'split_layers' if split_layers else 'no_split (full scheduler)'}")
    print(f"Qubits: {n_qubits}\n")
    _summarize(layered, logs)
    # for l in logs:
    #     for entry in l:
    #         print(entry)
    if args.plot:
        plot_execution(
            logs,
            n_qubits=n_qubits,
            save_path=args.output,
            split_layers=split_layers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
