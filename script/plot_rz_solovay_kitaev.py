#!/usr/bin/env python3
"""Plot an RZ gate realized via GridSynth decomposition.

Example:
    python script/plot_rz_solovay_kitaev.py --angle 0.7 --epsilon 1e-10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _import_qiskit_gridsynth():
    """Import GridSynth utility across Qiskit versions."""
    try:
        from qiskit.synthesis import gridsynth_rz  # type: ignore[import-not-found]
    except Exception:
        try:
            from qiskit.synthesis.discrete_basis import gridsynth_rz  # type: ignore[import-not-found]
        except Exception:
            from qiskit.transpiler.passes.synthesis.gridsynth import gridsynth_rz  # type: ignore

    from qiskit import QuantumCircuit  # type: ignore[import-not-found]
    from qiskit.quantum_info import Operator  # type: ignore[import-not-found]

    return (
        QuantumCircuit,
        Operator,
        gridsynth_rz,
    )


def build_approximation(
    angle: float,
    epsilon: float,
):
    (
        QuantumCircuit,
        Operator,
        gridsynth_rz,
    ) = _import_qiskit_gridsynth()

    target = QuantumCircuit(1, name="target")
    target.rz(angle, 0)

    approx = QuantumCircuit(1, name="gridsynth_decomp")
    approx.compose(gridsynth_rz(angle, epsilon=epsilon), inplace=True)

    target_u = np.asarray(Operator(target).data)
    approx_u = np.asarray(Operator(approx).data)
    phase = np.angle(np.vdot(target_u.flatten(), approx_u.flatten()))
    fro_err = float(np.linalg.norm(target_u - np.exp(1j * phase) * approx_u, ord="fro"))

    return target, approx, fro_err


def count_t_gates(circuit) -> int:
    return sum(1 for operation, _, _ in circuit.data if operation.name in {"t", "tdg"})


def make_plots(
    target,
    approx,
    angle: float,
    fro_err: float,
    output_target: str,
    output_decomp: str,
    font_size: float,
):
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "figure.titlesize": font_size,
        }
    )
    t_count = count_t_gates(approx)

    fig_target = plt.figure(figsize=(14, 2.8))
    ax_target = fig_target.add_subplot(1, 1, 1)
    target.draw("mpl", ax=ax_target, fold=-1)
    ax_target.set_title(f"Target circuit: Rz({angle:.6f})")
    fig_target.tight_layout()

    fig_decomp = plt.figure(figsize=(14, 2.8))
    ax_decomp = fig_decomp.add_subplot(1, 1, 1)
    approx.draw("mpl", ax=ax_decomp, fold=-1)
    ax_decomp.set_title(
        "GridSynth decomposition "
        f"(T-count: {t_count}, phase-aligned Frobenius error: {fro_err:.3e})"
    )
    fig_decomp.tight_layout()

    if output_target:
        Path(output_target).parent.mkdir(parents=True, exist_ok=True)
        fig_target.savefig(output_target, dpi=200, bbox_inches="tight")
    if output_decomp:
        Path(output_decomp).parent.mkdir(parents=True, exist_ok=True)
        fig_decomp.savefig(output_decomp, dpi=200, bbox_inches="tight")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angle", type=float, default=0.0029, help="RZ angle in radians"
    )
    parser.add_argument("--epsilon", type=float, default=1e-4, help="GridSynth epsilon")
    parser.add_argument(
        "--output-target",
        type=str,
        default="script/rz_target.png",
        help="Optional output path for target circuit figure",
    )
    parser.add_argument(
        "--output-decomp",
        type=str,
        default="output/analysis_plots/rz_solovay_kitaev_decomposition.png",
        help="Optional output path for decomposition circuit figure",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=13,
        help="Font size used in both figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        target, approx, fro_err = build_approximation(
            angle=args.angle,
            epsilon=args.epsilon,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "qiskit"
        raise SystemExit(
            f"Missing dependency '{missing}'. Install with: pip install qiskit pylatexenc"
        )

    make_plots(
        target,
        approx,
        angle=args.angle,
        fro_err=fro_err,
        output_target=args.output_target,
        output_decomp=args.output_decomp,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
