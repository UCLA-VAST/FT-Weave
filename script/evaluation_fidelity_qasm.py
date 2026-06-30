#!/usr/bin/env python3
"""Evaluate logical fidelity for a general OpenQASM circuit via STAR or T-cultivation."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from script.script_utils import ensure_csv_writer

from src.circuit.circuit_info import count_qubits
from src.circuit.gate_set import GateSetKind, validate_circuit_for_backend
from src.circuit.layer_partition import partition_into_layers
from src.circuit.placement import resolve_qubit_layout
from src.circuit.qasm_loader import load_qasm_file
from src.error_model import LogicalErrorModel
from src.fidelity_simulation.circuit_fidelity import (
    simulate_star_circuit_fidelity,
    simulate_t_cultivation_circuit_fidelity,
)
from src.star.config import get_config as get_star_config, update_config as update_star_config
from src.star.general_circuit_star import compile_circuit_star
from src.t_cultivation.config import update_config as update_t_config
from src.t_cultivation.general_circuit_t import compile_circuit_t_cultivation


STAR_RESULT_FIELDS = [
    "qasm_path",
    "backend",
    "n_qubits",
    "layout_cols",
    "layout_rows",
    "code_distance",
    "n_aods",
    "placement",
    "split_layers",
    "to_decompose",
    "trivial_return",
    "decompose_move",
    "parallel_execution",
    "prepare_lookahead_angles",
    "consider_skip_rus",
    "logical_se_interval",
    "trial_seed",
    "fidelity",
    "fidelity_of_rz_layer",
    "fidelity_of_rz_injection",
    "fidelity_of_rz_teleportaion",
    "fidelity_of_rz_s",
    "fidelity_cnot",
    "fidelity_1q",
    "fidelity_idle",
    "n_cnot",
    "n_cnot_teleportation",
    "n_h",
    "n_s",
    "n_se_q",
]

T_RESULT_FIELDS = [
    "qasm_path",
    "backend",
    "n_qubits",
    "layout_cols",
    "layout_rows",
    "code_distance",
    "n_aods",
    "placement",
    "split_layers",
    "to_decompose",
    "trivial_return",
    "decompose_move",
    "redistribute_stage1_success",
    "logical_se_interval",
    "epsilon",
    "trial_seed",
    "fidelity",
    "fidelity_of_rz_layer",
    "fidelity_of_rz_teleportaion",
    "fidelity_of_rz_s",
    "fidelity_of_rz_h",
    "fidelity_of_t_gate",
    "fidelity_cnot",
    "fidelity_h",
    "fidelity_idle",
    "fidelity_synthesis",
    "n_cnot",
    "n_cnot_teleportation",
    "n_h",
    "n_s",
    "n_t",
    "n_se_q",
    "n_rz_decomposition",
]


def _resolve_backend(
    flat_circuit: list[dict],
    backend: str,
) -> tuple[str, GateSetKind]:
    if backend == "auto":
        kind = validate_circuit_for_backend(flat_circuit, backend="auto")
        if kind == GateSetKind.CLIFFORD_RZ:
            return "star", kind
        return "t_cultivation", kind
    kind = validate_circuit_for_backend(flat_circuit, backend=backend)
    return backend, kind


def run_evaluation_qasm(params: dict) -> dict:
    """Compile and evaluate one QASM circuit. Returns fidelity profile dict."""
    qasm_path = params["qasm_path"]
    flat_circuit = load_qasm_file(qasm_path)
    n_qubits = count_qubits(flat_circuit)
    layout_override = params.get("layout_override")
    cols, rows = resolve_qubit_layout(n_qubits, layout_override)
    qubit_layout = (cols, rows)

    backend, gate_kind = _resolve_backend(flat_circuit, params.get("backend", "auto"))

    split_layers = params.get("split_layers", True)
    if backend == "star":
        circuit = partition_into_layers(flat_circuit)
        split_layers = True
    else:
        circuit = (
            partition_into_layers(flat_circuit) if split_layers else flat_circuit
        )

    code_distance = params["code_distance"]
    logical_error_model = LogicalErrorModel(code_distance)
    trial_seed = params.get("trial_seed", 0)
    rng = np.random.default_rng(trial_seed)
    n_aods = params["n_aods"]
    placement = params.get("placement", "col_based")
    analyze_result = params.get("profile", False)
    logical_se_interval = params.get("logical_se_interval")

    if backend == "star":
        original_se = get_star_config().get("LOGICAL_SE_INTERVAL")
        update_star_config(LOGICAL_SE_INTERVAL=logical_se_interval)
        try:
            config = {
                "n_aods": n_aods,
                "consider_skip_rus": params.get("consider_skip_rus", 0),
                "trivial_return": params.get("trivial_return", False),
                "decompose_move": params.get("decompose_move", True),
                "tmr_assignment_method": "matching",
                "prepare_lookahead_angles": params.get(
                    "prepare_lookahead_angles", True
                ),
                "rng": rng,
                "save_log": False,
            }
            circuit, full_logs, profiling = compile_circuit_star(
                circuit,
                n_qubits=n_qubits,
                qubit_layout=qubit_layout,
                placement=placement,
                code_distance=code_distance,
                config=config,
                parallel_execution=params.get("parallel_execution", False),
                analyze_result=analyze_result,
            )
            result = simulate_star_circuit_fidelity(
                n_factories=n_qubits,
                circuit=circuit,
                execution_logs=full_logs,
                logical_error_model=logical_error_model,
            )
        finally:
            update_star_config(LOGICAL_SE_INTERVAL=original_se)

        row = {
            "qasm_path": qasm_path,
            "backend": backend,
            "n_qubits": n_qubits,
            "layout_cols": cols,
            "layout_rows": rows,
            "code_distance": code_distance,
            "n_aods": n_aods,
            "placement": placement,
            "split_layers": split_layers,
            "to_decompose": False,
            "trivial_return": config["trivial_return"],
            "decompose_move": config["decompose_move"],
            "parallel_execution": params.get("parallel_execution", False),
            "prepare_lookahead_angles": config["prepare_lookahead_angles"],
            "consider_skip_rus": config["consider_skip_rus"],
            "logical_se_interval": logical_se_interval,
            "trial_seed": trial_seed,
            **result,
        }
        if analyze_result and profiling:
            row["total_depth"] = sum(float(r.get("total_time", 0.0)) for r in profiling)
        return row

    # T-cultivation path
    update_t_config(
        STAGE_2_FIDELITY_TARGET=params.get("fidelity_target", 1e-8),
        FACTORY_PHYSICAL_SIZE=params.get("factory_physical_size", 2),
        STAGE_1_RESOURCE_UNITS=1,
        LOGICAL_SE_INTERVAL=logical_se_interval,
    )
    logical_error_model.integrate_t_cultivation_fidelity_target(
        params.get("fidelity_target", 1e-8)
    )
    epsilon = params.get("epsilon", 1e-4)
    config = {
        "n_aods": n_aods,
        "rng": rng,
        "epsilon": epsilon,
        "to_decompose": params.get("to_decompose", False),
        "trivial_return": params.get("trivial_return", False),
        "decompose_move": params.get("decompose_move", True),
        "redistribute_stage1_success": params.get(
            "redistribute_stage1_success", True
        ),
    }
    circuit, full_logs, profiling = compile_circuit_t_cultivation(
        circuit,
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
        code_distance=code_distance,
        config=config,
        split_layers=split_layers,
        analyze_result=analyze_result,
    )
    result = simulate_t_cultivation_circuit_fidelity(
        n_factories=n_qubits,
        circuit=circuit,
        execution_logs=full_logs,
        logical_error_model=logical_error_model,
        synthesis_epsilon=epsilon,
        n_trotter_steps=1,
    )
    row = {
        "qasm_path": qasm_path,
        "backend": backend,
        "gate_kind": gate_kind.value,
        "n_qubits": n_qubits,
        "layout_cols": cols,
        "layout_rows": rows,
        "code_distance": code_distance,
        "n_aods": n_aods,
        "placement": placement,
        "split_layers": split_layers,
        "to_decompose": config["to_decompose"],
        "trivial_return": config["trivial_return"],
        "decompose_move": config["decompose_move"],
        "redistribute_stage1_success": config["redistribute_stage1_success"],
        "logical_se_interval": logical_se_interval,
        "epsilon": epsilon,
        "trial_seed": trial_seed,
        **result,
    }
    if analyze_result and profiling:
        row["total_depth"] = sum(float(r.get("total_time", 0.0)) for r in profiling)
    return row


def _write_csv_row(output_dir: str, backend: str, row: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = (
        "qasm_star_fidelity_results.csv"
        if backend == "star"
        else "qasm_t_cultivation_fidelity_results.csv"
    )
    csv_path = os.path.join(output_dir, filename)
    fieldnames = STAR_RESULT_FIELDS if backend == "star" else T_RESULT_FIELDS
    extra_keys = [k for k in row if k not in fieldnames]
    all_fields = fieldnames + [k for k in extra_keys if k not in fieldnames]
    csv_file, writer = ensure_csv_writer(csv_path, all_fields)
    try:
        writer.writerow({k: row.get(k) for k in all_fields})
    finally:
        csv_file.close()
    return csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate logical fidelity for a general OpenQASM circuit."
    )
    parser.add_argument("--qasm", required=True, help="Path to OpenQASM 2.0 file")
    parser.add_argument(
        "--backend",
        choices=["auto", "star", "t_cultivation"],
        default="auto",
        help="Compilation backend (default: auto from gate set)",
    )
    parser.add_argument("--code-distance", type=int, default=9)
    parser.add_argument("--n-aods", type=int, default=3)
    parser.add_argument("--placement", default="col_based")
    parser.add_argument(
        "--layout",
        nargs=2,
        type=int,
        metavar=("COLS", "ROWS"),
        help="Override qubit grid layout (default: nearest square)",
    )
    split = parser.add_mutually_exclusive_group()
    split.add_argument(
        "--split-layers",
        action="store_true",
        default=None,
        help="Compile non-Clifford layers separately (default for T-cultivation)",
    )
    split.add_argument(
        "--no-split-layers",
        action="store_true",
        help="Run one T-cultivation scheduler pass (enables circuit CNOT scheduling)",
    )
    parser.add_argument(
        "--decompose-clifford-layers",
        action="store_true",
        help="Split multi-target Clifford layers (T-cultivation only)",
    )
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="output/evaluation/qasm",
        help="Directory for CSV results",
    )
    parser.add_argument("--profile", action="store_true", help="Collect timing profile")
    parser.add_argument("--logical-se-interval", type=int, default=None)
    parser.add_argument("--trivial-return", action="store_true")
    parser.add_argument("--decompose-move", action="store_true", default=True)
    parser.add_argument("--parallel-execution", action="store_true")
    parser.add_argument("--consider-skip-rus", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.no_split_layers:
        split_layers = False
    elif args.split_layers:
        split_layers = True
    else:
        split_layers = True

    params = {
        "qasm_path": os.path.abspath(args.qasm),
        "backend": args.backend,
        "code_distance": args.code_distance,
        "n_aods": args.n_aods,
        "placement": args.placement,
        "layout_override": tuple(args.layout) if args.layout else None,
        "split_layers": split_layers,
        "to_decompose": args.decompose_clifford_layers,
        "epsilon": args.epsilon,
        "trial_seed": args.trial_seed,
        "output_dir": args.output_dir,
        "profile": args.profile,
        "logical_se_interval": args.logical_se_interval,
        "trivial_return": args.trivial_return,
        "decompose_move": args.decompose_move,
        "parallel_execution": args.parallel_execution,
        "consider_skip_rus": args.consider_skip_rus,
    }

    row = run_evaluation_qasm(params)
    backend = row["backend"]
    csv_path = _write_csv_row(args.output_dir, backend, row)

    print("=" * 72)
    print("QASM FIDELITY EVALUATION")
    print("=" * 72)
    print(f"  qasm:      {params['qasm_path']}")
    print(f"  backend:   {backend}")
    print(f"  n_qubits:  {row['n_qubits']}")
    print(f"  layout:    {row['layout_cols']} x {row['layout_rows']}")
    print(f"  fidelity:  {row['fidelity']:.6e}")
    print(f"  csv:       {csv_path}")
    if not split_layers and backend == "t_cultivation":
        print(
            "  note: --no-split-layers routes Clifford gates (incl. CNOT) "
            "through the T-cultivation scheduler."
        )
    elif split_layers and backend == "t_cultivation":
        print(
            "  note: --split-layers stubs Clifford layers; only Rz/T rounds "
            "use the scheduler (TFIM-style)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
