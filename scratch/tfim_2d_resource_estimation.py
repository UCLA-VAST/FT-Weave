#!/usr/bin/env python3
"""Resource estimation for 2D TFIM with second-order (Strang) Trotter, Qualtran.

Follows the pattern in
https://qualtran.readthedocs.io/en/latest/bloqs/chemistry/resource_estimation.html
using QECGatesCost, QubitCount, and optional T-count classification.

Hamiltonian (open boundary  Lx × Ly  grid):
  H = -J * sum_{<i,j>} Z_i Z_j - Gamma * sum_i X_i

ZZ terms are grouped into four mutually commuting layers (even/odd horizontal and
vertical bonds), then a symmetric (Strang) second-order product formula in five
groups: ZZ_he, ZZ_ho, ZZ_ve, ZZ_vo, X.

Run with the Qualtran venv (Python 3.12):
  .venv-qualtran/bin/python scratch/tfim_2d_resource_estimation.py
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional, Tuple

import attrs
from qualtran import Bloq, BloqBuilder, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import CNOT, Rz
from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary
from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary
from qualtran.drawing import Text, WireSymbol
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
from qualtran.resource_counting.classify_bloqs import classify_t_count_by_bloq_type
from qualtran.resource_counting.generalizers import generalize_cswap_approx
from qualtran.surface_code import (
    AlgorithmSummary,
    CCZ2TFactory,
    LogicalErrorModel,
    MagicStateFactory,
    MultiFactory,
    PhysicalCostModel,
    PhysicalParameters,
    QECScheme,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
    iter_simple_data_blocks,
)
from qualtran.surface_code.data_block import SimpleDataBlock


@attrs.frozen
class GridZZCommutingLayer(Bloq):
    """Product of exp(-i angle Z_p Z_q) over disjoint pairs (p, q), via CNOT–Rz–CNOT."""

    nsites: int
    bonds: tuple[tuple[int, int], ...]
    angle: float
    eps: float = 1e-10

    @property
    def signature(self) -> Signature:
        return Signature.build(system=self.nsites)

    def wire_symbol(
        self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()
    ) -> WireSymbol:
        if reg is None:
            return Text("ZZ_layer")
        return super().wire_symbol(reg, idx)

    def build_composite_bloq(
        self, bb: BloqBuilder, system: Soquet
    ) -> Dict[str, Soquet]:
        system = bb.split(system)
        for i, j in self.bonds:
            system[i], system[j] = bb.add(CNOT(), ctrl=system[i], target=system[j])
            system[j] = bb.add(Rz(self.angle, self.eps), q=system[j])
            system[i], system[j] = bb.add(CNOT(), ctrl=system[i], target=system[j])
        return {"system": bb.join(system)}


def _grid_indices(width: int, height: int) -> int:
    return width * height


def _horizontal_bonds_even(width: int, height: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for r in range(height):
        base = r * width
        for c in range(0, width - 1, 2):
            out.append((base + c, base + c + 1))
    return tuple(out)


def _horizontal_bonds_odd(width: int, height: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for r in range(height):
        base = r * width
        for c in range(1, width - 1, 2):
            out.append((base + c, base + c + 1))
    return tuple(out)


def _vertical_bonds_even(width: int, height: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for c in range(width):
        for r in range(0, height - 1, 2):
            out.append((r * width + c, (r + 1) * width + c))
    return tuple(out)


def _vertical_bonds_odd(width: int, height: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for c in range(width):
        for r in range(1, height - 1, 2):
            out.append((r * width + c, (r + 1) * width + c))
    return tuple(out)


def build_second_order_tfim_trotter_step(
    width: int,
    height: int,
    j_coupling: float,
    gamma: float,
    dt: float,
    eps: float = 1e-10,
) -> TrotterizedUnitary:
    """One second-order Trotter step (Strang) for 2D TFIM."""
    n = _grid_indices(width, height)
    he = _horizontal_bonds_even(width, height)
    ho = _horizontal_bonds_odd(width, height)
    ve = _vertical_bonds_even(width, height)
    vo = _vertical_bonds_odd(width, height)

    zz_he = GridZZCommutingLayer(nsites=n, bonds=he, angle=0.0, eps=eps)
    zz_ho = GridZZCommutingLayer(nsites=n, bonds=ho, angle=0.0, eps=eps)
    zz_ve = GridZZCommutingLayer(nsites=n, bonds=ve, angle=0.0, eps=eps)
    zz_vo = GridZZCommutingLayer(nsites=n, bonds=vo, angle=0.0, eps=eps)
    x_bloq = IsingXUnitary(nsites=n, angle=0.0, eps=eps)

    # Strang: A/2 B/2 C/2 D/2 E D/2 C/2 B/2 A/2  with A..D = ZZ layers, E = X field.
    half_j = 0.5 * j_coupling
    indices = (0, 1, 2, 3, 4, 3, 2, 1, 0)
    coeffs = (
        half_j,
        half_j,
        half_j,
        half_j,
        gamma,
        half_j,
        half_j,
        half_j,
        half_j,
    )
    bloqs = (zz_he, zz_ho, zz_ve, zz_vo, x_bloq)
    return TrotterizedUnitary(bloqs=bloqs, indices=indices, coeffs=coeffs, timestep=dt)


def _factory_distances(factory: MagicStateFactory) -> Tuple[int, int, int]:
    """Return (l1_d, l2_d, n_parallel_factories) for CCZ2T or MultiFactory(CCZ2T)."""
    n_par = 1
    inner: MagicStateFactory = factory
    if isinstance(factory, MultiFactory):
        n_par = factory.n_factories
        inner = factory.base_factory
    if isinstance(inner, CCZ2TFactory):
        return inner.distillation_l1_d, inner.distillation_l2_d, n_par
    return -1, -1, n_par


def estimate_gidney_fowler_ccz2t(
    algo: AlgorithmSummary,
    *,
    phys_err: float = 1e-3,
    error_budget: float = 0.01,
    cycle_time_us: float = 1.0,
    routing_overhead: float = 0.5,
    factory: Optional[MagicStateFactory] = None,
) -> Tuple[PhysicalCostModel, SimpleDataBlock, MagicStateFactory]:
    """Pick data-block distance from remaining error budget (Gidney–Fowler CCZ2T model).

    Mirrors ``get_ccz2t_costs_from_error_budget`` but returns the model and blocks so we
    can report factory vs data footprint and distillation distances explicitly.
    """
    fac: MagicStateFactory = CCZ2TFactory() if factory is None else factory

    qec_scheme = QECScheme.make_gidney_fowler()
    err_model = LogicalErrorModel(qec_scheme=qec_scheme, physical_error=phys_err)
    factory_error = fac.factory_error(
        n_logical_gates=algo.n_logical_gates, logical_error_model=err_model
    )
    n_cycles_magic = fac.n_cycles(
        n_logical_gates=algo.n_logical_gates, logical_error_model=err_model
    )
    err_budget = error_budget - factory_error
    if err_budget < 0:
        raise ValueError(
            f"Factory error {factory_error:g} exceeds total error budget {error_budget:g}"
        )
    n_logical_qubits = math.ceil((1 + routing_overhead) * algo.n_algo_qubits)
    data_unit_cells = n_logical_qubits * n_cycles_magic
    target_err_per_round = err_budget / data_unit_cells
    data_d = qec_scheme.code_distance_from_budget(
        physical_error=phys_err, budget=target_err_per_round
    )
    data_block = SimpleDataBlock(data_d=data_d, routing_overhead=routing_overhead)
    model = PhysicalCostModel(
        physical_params=PhysicalParameters(
            physical_error=phys_err, cycle_time_us=cycle_time_us
        ),
        data_block=data_block,
        factory=fac,
        qec_scheme=qec_scheme,
    )
    return model, data_block, fac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--j", type=float, default=1.0, dest="j_coupling")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--trotter-steps",
        type=int,
        default=1,
        help="Scale logical GateCounts by this many steps for FT costing.",
    )
    parser.add_argument("--phys-err", type=float, default=1e-3)
    parser.add_argument("--error-budget", type=float, default=0.01)
    parser.add_argument("--cycle-us", type=float, default=1.0)
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Minimize qubit-hours over a small factory×data_block grid (slower).",
    )
    args = parser.parse_args()

    width, height = args.width, args.height
    j_coupling = args.j_coupling
    gamma = args.gamma
    dt = args.dt

    step = build_second_order_tfim_trotter_step(
        width, height, j_coupling=j_coupling, gamma=gamma, dt=dt
    )

    gate_counts = get_cost_value(step, QECGatesCost(), generalizer=generalize_cswap_approx)
    logical_qubits = get_cost_value(step, QubitCount())
    gate_counts_scaled = gate_counts * args.trotter_steps

    ts_rot = 11  # default synthesis cost per rotation (mixed fallback, ~1e-3 error)
    gc_ft = gate_counts_scaled
    t_equiv = gc_ft.total_t_count(ts_per_rotation=ts_rot)
    t_ccz = gc_ft.total_t_and_ccz_count(ts_per_rotation=ts_rot)

    print("2D TFIM resource estimate (one second-order Trotter step)")
    print(f"  Grid: {width}×{height} = {width * height} sites, open boundary")
    print(f"  J = {j_coupling}, Γ = {gamma}, Δt = {dt}")
    print(f"  Trotter steps (FT scaling): {args.trotter_steps}")
    print()
    print("Logical qubits (QubitCount, sequential layout):")
    print(f"  {logical_qubits}")
    print()
    print("GateCounts per step (QECGatesCost, generalized CSWAP):")
    for k, v in gate_counts.asdict().items():
        print(f"  {k}: {v}")
    if args.trotter_steps != 1:
        print(f"  (×{args.trotter_steps} steps — scaled FT inputs use totals below)")
        print("  Scaled totals:")
        for k, v in gate_counts_scaled.asdict().items():
            print(f"    {k}: {v}")
    print()
    print(
        "T-style budget for FT costing "
        f"(ts_per_rotation={ts_rot}; {'scaled' if args.trotter_steps != 1 else 'per-step'} totals):"
    )
    print(f"  n_T (incl. rotations as T): {t_ccz['n_t']}")
    print(f"  n_Toffoli/CSwap/And (as CCZ-like): {t_ccz['n_ccz']}")
    print(f"  total T equivalents: {t_equiv}")
    print()
    print("Clifford vs non‑Clifford (logical gate counts, same scope as FT totals above):")
    print(
        "  Clifford ops (surface-code-friendly count): "
        f"{gc_ft.clifford}"
    )
    print(
        "  Single-qubit rotations (non‑Clifford, need synthesis): "
        f"{gc_ft.rotation}"
    )
    print(f"  Native T gates: {gc_ft.t}")
    print(
        "  Non‑Clifford T budget (rotations→T at ts_per_rotation + native T): "
        f"{gc_ft.t + ts_rot * gc_ft.rotation}"
    )
    print()
    print("T-count mix by bloq module (classify_t_count_by_bloq_type):")
    classified = classify_t_count_by_bloq_type(step, generalizer=generalize_cswap_approx)
    for k in sorted(classified.keys(), key=lambda x: -classified[x]):
        print(f"  {k}: {classified[k]}")

    print()
    print("— Surface code (Gidney–Fowler / CCZ2T, Qualtran surface_code) —")
    algo = AlgorithmSummary(n_algo_qubits=int(logical_qubits), n_logical_gates=gc_ft)

    if args.grid_search:
        fact_iter = iter_ccz2t_factories(l1_start=9, l1_stop=21, l2_stop=35)
        d_iter = iter_simple_data_blocks(d_start=7, d_stop=29)
        try:
            best, best_factory, best_db = get_ccz2t_costs_from_grid_search(
                n_logical_gates=algo.n_logical_gates,
                n_algo_qubits=algo.n_algo_qubits,
                phys_err=args.phys_err,
                error_budget=args.error_budget,
                cycle_time_us=args.cycle_us,
                factory_iter=tuple(fact_iter),
                data_block_iter=tuple(d_iter),
            )
        except ValueError as e:
            print(f"  Grid search failed: {e}")
            return
        l1, l2, n_par = _factory_distances(best_factory)
        n_fact = best_factory.n_physical_qubits()
        n_dat = best_db.n_physical_qubits(algo.n_algo_qubits)
        print("  Mode: grid search minimizing qubit-hours (narrow ranges; edit script to expand).")
        print(f"  Data-block code distance d (logical algorithm storage): {best_db.data_d}")
        print(
            "  Distillation code distances (CCZ2T factory): "
            f"L1={l1}, L2={l2}, parallel factories={n_par}"
        )
        print(f"  Physical qubits — magic-state factory: {n_fact:,}")
        print(f"  Physical qubits — data block: {n_dat:,}")
        print(f"  Physical qubits — total footprint: {best.footprint:,}")
        model_gs = PhysicalCostModel(
            physical_params=PhysicalParameters(
                physical_error=args.phys_err, cycle_time_us=args.cycle_us
            ),
            data_block=best_db,
            factory=best_factory,
            qec_scheme=QECScheme.make_gidney_fowler(),
        )
        print(f"  EC cycles (model): {model_gs.n_cycles(algo):,}")
        print(f"  Wall-clock time (hours): {best.duration_hr:g}")
        print(f"  Failure probability (model): {best.failure_prob:g}")
        print(
            "  Space–time volume (Qualtran metric qubit-hours = footprint × duration): "
            f"{best.qubit_hours:g}"
        )
    else:
        try:
            model, data_block, factory = estimate_gidney_fowler_ccz2t(
                algo,
                phys_err=args.phys_err,
                error_budget=args.error_budget,
                cycle_time_us=args.cycle_us,
            )
        except ValueError as e:
            print(f"  Error-budget estimate failed: {e}")
            return
        l1, l2, n_par = _factory_distances(factory)
        n_fact = factory.n_physical_qubits()
        n_dat = data_block.n_physical_qubits(algo.n_algo_qubits)
        n_cyc = model.n_cycles(algo)
        fail = model.error(algo)
        footprint = model.n_phys_qubits(algo)
        dur = model.duration_hr(algo)
        qh = footprint * dur
        print(
            "  Mode: error-budget allocation (factory error fixed by CCZ2T; "
            "remaining budget → data-block distance)."
        )
        print(f"  Data-block code distance d (logical algorithm storage): {data_block.data_d}")
        print(
            "  Distillation code distances (CCZ2T factory): "
            f"L1={l1}, L2={l2}, parallel factories={n_par}"
        )
        print(f"  Physical qubits — magic-state factory: {n_fact:,}")
        print(f"  Physical qubits — data block: {n_dat:,}")
        print(f"  Physical qubits — total footprint: {footprint:,}")
        print(f"  EC cycles (model): {n_cyc:,}")
        print(f"  Wall-clock time (hours): {dur:g}")
        print(f"  Failure probability (model): {fail:g}")
        print(
            "  Space–time volume (Qualtran qubit-hours = footprint × duration): "
            f"{qh:g}"
        )


if __name__ == "__main__":
    main()
