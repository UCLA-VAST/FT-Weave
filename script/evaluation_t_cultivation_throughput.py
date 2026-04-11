import os
import sys
import logging
import argparse
import csv
from dataclasses import dataclass

import numpy as np

# Ensure repository root is on sys.path so `src` is importable when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animator.circuit_execution_visualization import plot_t_cultivation_execution
from src.t_cultivation.t_cultivation import t_cultivation_execution
from src.t_cultivation.config import get_config, update_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@dataclass(frozen=True)
class SweepSpec:
    """Definition of a T-cultivation circuit sweep."""

    n_qubits: int
    k_t_per_qubit: int
    # If True, build K layers where each layer applies T to all qubits in parallel.
    # If False, build K sequential T gates on each qubit (we only use this for n_qubits=1).
    layers_parallel: bool


@dataclass(frozen=True)
class TSetting:
    """Evaluation setting for stage-2 fidelity and factory physical size."""

    fidelity_target: float
    factory_physical_size: int
    distance: int


def build_locations(
    n_qubits: int, n_factories: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Build logic/factory locations with two placement strategies:

    1) Sparse factories (n_factories < n_qubits):
       Place factories approximately uniformly along the qubit line.

    2) Dense factories (n_factories >= n_qubits):
       Place factories in rows close to qubits (above/below y=0), filling full
       rows of width n_qubits and then a partial row if needed.
    """
    logic_qubit_locations = [(x, 0) for x in range(n_qubits)]
    if n_factories <= 0:
        return logic_qubit_locations, []
    if n_qubits <= 0:
        return [], []

    magic_state_locations: list[tuple[int, int]] = []

    if n_factories < n_qubits:
        # Uniformly spread factories along x in [0, n_qubits-1].
        # np.linspace gives stable spacing for all nf<nq.
        xs = np.linspace(0, n_qubits - 1, n_factories + 2)[1:-1]
        for x in xs:
            magic_state_locations.append((int(round(float(x))), 1))
        return logic_qubit_locations, magic_state_locations

    # Dense regime:
    # choose integer lattice points around the qubit line y=0 that minimize the
    # maximum nearest-qubit Manhattan distance; then balance assignments so
    # factories are spread uniformly around qubits and around +/- rows.
    def nearest_qubit_x(x: int) -> int:
        return min(max(x, 0), n_qubits - 1)

    def nearest_line_distance(x: int, y: int) -> int:
        # Distance to nearest qubit on segment x in [0, n_qubits-1], y=0.
        x_extra = 0
        if x < 0:
            x_extra = -x
        elif x > n_qubits - 1:
            x_extra = x - (n_qubits - 1)
        return abs(y) + x_extra

    def candidate_points_within_radius(radius: int) -> list[tuple[int, int, int, int]]:
        # tuple: (x, y, distance, nearest_qx)
        pts: list[tuple[int, int, int, int]] = []
        for y in range(-radius, radius + 1):
            if y == 0:
                continue
            for x in range(-radius, (n_qubits - 1) + radius + 1):
                d = nearest_line_distance(x, y)
                if d <= radius:
                    pts.append((x, y, d, nearest_qubit_x(x)))
        return pts

    radius = 1
    while True:
        candidates = candidate_points_within_radius(radius)
        if len(candidates) >= n_factories:
            break
        radius += 1

    # Greedy balancing across nearest qubits and upper/lower rows.
    qubit_load = [0 for _ in range(n_qubits)]
    side_load = {1: 0, -1: 0}  # +1 for y>0, -1 for y<0
    selected: list[tuple[int, int]] = []
    remaining = candidates.copy()
    center_x = (n_qubits - 1) / 2.0
    for _ in range(n_factories):
        best_i = -1
        best_key = None
        for i, (x, y, d, qx) in enumerate(remaining):
            side = 1 if y > 0 else -1
            key = (
                d,  # prioritize minimum travel distance
                qubit_load[qx],  # spread around logical qubits
                side_load[side],  # balance upper/lower rows
                abs(y),  # prefer nearer rows first
                abs(x - qx),  # prefer being near nearest qubit's x
                abs(x - center_x),  # gentle center preference for symmetry
                x,
                y,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        x, y, _d, qx = remaining.pop(best_i)
        selected.append((x, y))
        qubit_load[qx] += 1
        side_load[1 if y > 0 else -1] += 1

    magic_state_locations.extend(selected)
    return logic_qubit_locations, magic_state_locations


def build_circuit_t_layers(spec: SweepSpec) -> list[dict]:
    """Build a circuit containing only T gates."""
    n_qubits = spec.n_qubits
    k = spec.k_t_per_qubit
    if n_qubits <= 0 or k <= 0:
        return []

    if spec.layers_parallel:
        # K layers; each layer applies T to all qubits in one moment.
        return [
            {"gate": "T", "targets": list(range(n_qubits)), "params": {}}
            for _ in range(k)
        ]

    # Sequential T gates on each qubit (currently used only for n_qubits==1).
    if n_qubits != 1:
        raise ValueError(
            "Sequential mode is only supported for n_qubits=1 in this evaluator."
        )
    return [{"gate": "T", "targets": [0], "params": {}} for _ in range(k)]


def total_t_from_spec(spec: SweepSpec) -> int:
    return spec.n_qubits * spec.k_t_per_qubit


def run_t_cultivation_metrics(
    *,
    spec: SweepSpec,
    n_factories: int,
    seed: int,
    n_aods: int = 2,
    to_decompose: bool = False,
    epsilon: float = 1e-4,
    print_profile: bool = True,
) -> tuple[float, float, list]:
    """
    Return single-factory throughput, makespan, and raw execution_log.

    Throughput is 0.0 when the log is empty or makespan is non-positive.
    """
    circuit = build_circuit_t_layers(spec)
    logic_qubit_locations, magic_state_locations = build_locations(
        spec.n_qubits, n_factories
    )

    execution_log = t_cultivation_execution(
        circuit=circuit,
        n_factories=n_factories,
        logic_qubit_locations=logic_qubit_locations,
        magic_state_locations=magic_state_locations,
        rng=np.random.default_rng(seed),
        n_aods=n_aods,
        to_decompose=to_decompose,
        epsilon=epsilon,
        print_profile=print_profile,
    )
    if not execution_log:
        return 0.0, 0.0, []
    makespan = max(entry[1] for entry in execution_log)
    if makespan <= 0:
        return 0.0, 0.0, execution_log
    single_factory_throughput = total_t_from_spec(spec) / makespan / max(1, n_factories)
    return single_factory_throughput, makespan, execution_log


def metrics_for_seed(
    *,
    spec: SweepSpec,
    n_factories: int,
    seed: int,
    n_aods: int = 2,
    to_decompose: bool = False,
    epsilon: float = 1e-4,
    print_profile: bool = True,
) -> tuple[float, float]:
    """Return throughput and makespan only (used in Monte Carlo trials)."""
    tp, ms, _ = run_t_cultivation_metrics(
        spec=spec,
        n_factories=n_factories,
        seed=seed,
        n_aods=n_aods,
        to_decompose=to_decompose,
        epsilon=epsilon,
        print_profile=print_profile,
    )
    return tp, ms


def metric_from_makespan(
    *,
    spec: SweepSpec,
    n_factories: int,
    makespan: float,
    metric_mode: str,
) -> float:
    """Return per-point metric value from makespan."""
    total_t = total_t_from_spec(spec)
    if makespan <= 0 or total_t <= 0:
        return 0.0
    if metric_mode == "throughput":
        return total_t / makespan / max(1, n_factories)
    if metric_mode == "cycles_per_t":
        return makespan * max(1, n_factories) / total_t
    raise ValueError(f"Unsupported metric_mode: {metric_mode}")


def mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def mean_min_max(xs: list[float]) -> tuple[float, float, float]:
    """Return mean, min, max for asymmetric error bars."""
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(arr.mean()), float(arr.min()), float(arr.max())


def _setting_subtitle(setting: TSetting) -> str:
    return f"distance-{setting.distance}, LER: {setting.fidelity_target:g}"


def _setting_filename_slug(setting: TSetting) -> str:
    """Compact, filesystem-safe tag for a TSetting (used in PDF names)."""
    return f"d{setting.distance}_ler{setting.fidelity_target:g}".replace(".", "p")


def _first_trial_timeline_path(
    *,
    base_dir: str,
    fig_tag: str,
    setting: TSetting,
    seed: int,
    nf: int | None = None,
    n_qubits: int | None = None,
    ratio: int | None = None,
) -> str:
    """
    Build a PDF path that identifies figure type, physics setting, sweep point, and trial 0.

    fig_tag: "fig1" | "fig2" | "fig3"
    """
    slug = _setting_filename_slug(setting)
    if fig_tag == "fig1":
        assert nf is not None
        name = f"{fig_tag}_{slug}_nf{nf}_trial0_seed{seed}.pdf"
    elif fig_tag == "fig2":
        assert nf is not None and n_qubits is not None and ratio is not None
        name = f"{fig_tag}_{slug}_fq{ratio}_nq{n_qubits}_nf{nf}_trial0_seed{seed}.pdf"
    elif fig_tag == "fig3":
        assert nf is not None
        name = f"{fig_tag}_{slug}_nq10_nf{nf}_trial0_seed{seed}.pdf"
    else:
        raise ValueError(f"Unknown fig_tag: {fig_tag}")
    return os.path.join(base_dir, name)


def apply_setting(setting: TSetting) -> None:
    """Apply one evaluation setting to global T-cultivation config."""
    update_config(
        STAGE_2_FIDELITY_TARGET=setting.fidelity_target,
        FACTORY_PHYSICAL_SIZE=setting.factory_physical_size,
        STAGE_1_RESOURCE_UNITS=1,
    )


def main(
    factory_counts: list[int],
    metric_mode: str = "cycles_per_t",
    print_profile: bool = False,
    num_trials: int = 10,
    output_csv: str = "output/t_cultivation/factory_throughput/t_cultivation_throughput_trials.csv",
):
    # In all sweeps: fixed K = 10 T gates per qubit.
    k_per_qubit = 10
    settings = [
        TSetting(fidelity_target=1e-8, factory_physical_size=2, distance=7),
        TSetting(fidelity_target=1e-8, factory_physical_size=4, distance=13),
        TSetting(fidelity_target=1e-10, factory_physical_size=4, distance=13),
    ]

    original_cfg = get_config().copy()
    output_dir = "output/t_cultivation/factory_throughput"
    os.makedirs(output_dir, exist_ok=True)
    first_trial_dir = os.path.join(output_dir, "first_trial_timelines")
    os.makedirs(first_trial_dir, exist_ok=True)
    csv_dir = os.path.dirname(output_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    first_trial_jobs: list[tuple[TSetting, str, SweepSpec, int, int]] = []
    trial_rows: list[dict] = []
    try:
        # ------------------------------------------------------------------
        # Type 1: 1 qubit, fixed K=10, sweep number of factories
        # ------------------------------------------------------------------
        spec_fig1 = SweepSpec(
            n_qubits=1, k_t_per_qubit=k_per_qubit, layers_parallel=False
        )
        for setting_idx, setting in enumerate(settings):
            apply_setting(setting)
            for nf in factory_counts:
                logging.info(
                    "Fig1 [distance-%d, LER=%g] nf=%d (%d trials)",
                    setting.distance,
                    setting.fidelity_target,
                    nf,
                    num_trials,
                )
                for t in range(num_trials):
                    seed = 1000000 * setting_idx + 1000 * nf + t
                    try:
                        _, makespan = metrics_for_seed(
                            spec=spec_fig1,
                            n_factories=nf,
                            seed=seed,
                        )
                        metric_value = metric_from_makespan(
                            spec=spec_fig1,
                            n_factories=nf,
                            makespan=makespan,
                            metric_mode=metric_mode,
                        )
                    except RuntimeError as e:
                        logging.warning(
                            "Fig1 failed [distance-%d, LER=%g] nf=%d: %s",
                            setting.distance,
                            setting.fidelity_target,
                            nf,
                            e,
                        )
                        makespan = 0.0
                        metric_value = 0.0

                    trial_rows.append(
                        {
                            "figure": "fig1",
                            "setting_index": setting_idx,
                            "distance": setting.distance,
                            "fidelity_target": setting.fidelity_target,
                            "factory_physical_size": setting.factory_physical_size,
                            "metric_mode": metric_mode,
                            "trial": t,
                            "seed": seed,
                            "n_qubits": spec_fig1.n_qubits,
                            "k_t_per_qubit": spec_fig1.k_t_per_qubit,
                            "layers_parallel": spec_fig1.layers_parallel,
                            "ratio": "",
                            "n_factories": nf,
                            "makespan": makespan,
                            "metric_value": metric_value,
                        }
                    )

                seed0 = 1000000 * setting_idx + 1000 * nf
                first_trial_jobs.append(
                    (
                        setting,
                        _first_trial_timeline_path(
                            base_dir=first_trial_dir,
                            fig_tag="fig1",
                            setting=setting,
                            seed=seed0,
                            nf=nf,
                        ),
                        spec_fig1,
                        nf,
                        seed0,
                    )
                )

        # ------------------------------------------------------------------
        # Type 2: qubit count 1..10, factory/qubit ratio in {1,2,4,8}
        # ------------------------------------------------------------------
        qubit_counts = list(range(1, 11))
        ratios = [1, 2, 4, 8]
        for setting_idx, setting in enumerate(settings):
            apply_setting(setting)
            for ratio in ratios:
                for n_qubits in qubit_counts:
                    n_factories = ratio * n_qubits
                    spec = SweepSpec(
                        n_qubits=n_qubits,
                        k_t_per_qubit=k_per_qubit,
                        layers_parallel=True,
                    )
                    for t in range(num_trials):
                        seed = (
                            2000000 * setting_idx
                            + 100000 * ratio
                            + 1000 * n_factories
                            + 13 * n_qubits
                            + t
                        )
                        try:
                            _, makespan = metrics_for_seed(
                                spec=spec,
                                n_factories=n_factories,
                                seed=seed,
                                print_profile=print_profile,
                            )
                            metric_value = metric_from_makespan(
                                spec=spec,
                                n_factories=n_factories,
                                makespan=makespan,
                                metric_mode=metric_mode,
                            )
                        except RuntimeError as e:
                            logging.warning(
                                "Type2 failed [%s] ratio=%d nq=%d nf=%d: %s",
                                _setting_subtitle(setting),
                                ratio,
                                n_qubits,
                                n_factories,
                                e,
                            )
                            makespan = 0.0
                            metric_value = 0.0

                        trial_rows.append(
                            {
                                "figure": "fig2",
                                "setting_index": setting_idx,
                                "distance": setting.distance,
                                "fidelity_target": setting.fidelity_target,
                                "factory_physical_size": setting.factory_physical_size,
                                "metric_mode": metric_mode,
                                "trial": t,
                                "seed": seed,
                                "n_qubits": spec.n_qubits,
                                "k_t_per_qubit": spec.k_t_per_qubit,
                                "layers_parallel": spec.layers_parallel,
                                "ratio": ratio,
                                "n_factories": n_factories,
                                "makespan": makespan,
                                "metric_value": metric_value,
                            }
                        )

                    seed0 = (
                        2000000 * setting_idx
                        + 100000 * ratio
                        + 1000 * n_factories
                        + 13 * n_qubits
                    )
                    first_trial_jobs.append(
                        (
                            setting,
                            _first_trial_timeline_path(
                                base_dir=first_trial_dir,
                                fig_tag="fig2",
                                setting=setting,
                                seed=seed0,
                                nf=n_factories,
                                n_qubits=n_qubits,
                                ratio=ratio,
                            ),
                            spec,
                            n_factories,
                            seed0,
                        )
                    )

        # ------------------------------------------------------------------
        # Type 3: fixed 10 qubits, factories 10..1
        # ------------------------------------------------------------------
        n_qubits_fixed = 10
        factory_counts_fig3 = list(range(10, 0, -1))
        spec_fig3 = SweepSpec(
            n_qubits=n_qubits_fixed, k_t_per_qubit=k_per_qubit, layers_parallel=True
        )
        for setting_idx, setting in enumerate(settings):
            apply_setting(setting)
            for n_factories in factory_counts_fig3:
                for t in range(num_trials):
                    seed = 3000000 * setting_idx + 1000 * n_factories + t
                    try:
                        _, makespan = metrics_for_seed(
                            spec=spec_fig3,
                            n_factories=n_factories,
                            seed=seed,
                            print_profile=print_profile,
                        )
                        metric_value = metric_from_makespan(
                            spec=spec_fig3,
                            n_factories=n_factories,
                            makespan=makespan,
                            metric_mode=metric_mode,
                        )
                    except RuntimeError as e:
                        logging.warning(
                            "Type3 failed [distance-%d, LER=%g] nf=%d: %s",
                            setting.distance,
                            setting.fidelity_target,
                            n_factories,
                            e,
                        )
                        makespan = 0.0
                        metric_value = 0.0

                    trial_rows.append(
                        {
                            "figure": "fig3",
                            "setting_index": setting_idx,
                            "distance": setting.distance,
                            "fidelity_target": setting.fidelity_target,
                            "factory_physical_size": setting.factory_physical_size,
                            "metric_mode": metric_mode,
                            "trial": t,
                            "seed": seed,
                            "n_qubits": spec_fig3.n_qubits,
                            "k_t_per_qubit": spec_fig3.k_t_per_qubit,
                            "layers_parallel": spec_fig3.layers_parallel,
                            "ratio": "",
                            "n_factories": n_factories,
                            "makespan": makespan,
                            "metric_value": metric_value,
                        }
                    )

                seed0 = 3000000 * setting_idx + 1000 * n_factories
                first_trial_jobs.append(
                    (
                        setting,
                        _first_trial_timeline_path(
                            base_dir=first_trial_dir,
                            fig_tag="fig3",
                            setting=setting,
                            seed=seed0,
                            nf=n_factories,
                        ),
                        spec_fig3,
                        n_factories,
                        seed0,
                    )
                )

        fieldnames = [
            "figure",
            "setting_index",
            "distance",
            "fidelity_target",
            "factory_physical_size",
            "metric_mode",
            "trial",
            "seed",
            "n_qubits",
            "k_t_per_qubit",
            "layers_parallel",
            "ratio",
            "n_factories",
            "makespan",
            "metric_value",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_rows)
        logging.info("Saved CSV: %s (%d rows)", output_csv, len(trial_rows))

        for setting, path, spec, n_factories, seed in first_trial_jobs:
            # Only plot d13, ler1e-08
            if setting.distance != 13 or setting.fidelity_target != 1e-8:
                continue
            apply_setting(setting)
            try:
                _, _, log = run_t_cultivation_metrics(
                    spec=spec,
                    n_factories=n_factories,
                    seed=seed,
                    print_profile=print_profile,
                )
            except RuntimeError as e:
                logging.warning(
                    "First-trial timeline skipped (run failed): %s: %s", path, e
                )
                continue
            if not log:
                logging.warning("First-trial timeline skipped (empty log): %s", path)
                continue
            if print_profile:
                plot_t_cultivation_execution(log, spec.n_qubits, n_factories, path)
        plotted_jobs = [
            j
            for j in first_trial_jobs
            if j[0].distance == 13 and j[0].fidelity_target == 1e-8
        ]
        if plotted_jobs:
            logging.info(
                "Wrote %d first-trial execution timeline PDFs under %s",
                len(plotted_jobs),
                first_trial_dir,
            )

    finally:
        # Restore original config so this script does not leak global settings.
        update_config(
            SE_STAGE_1=original_cfg["SE_STAGE_1"],
            SE_STAGE_2=original_cfg["SE_STAGE_2"],
            CNOT_TIME=original_cfg["CNOT_TIME"],
            SE_TIME=original_cfg["SE_TIME"],
            TELEPORTATION_SUCCESS_RATE=original_cfg["TELEPORTATION_SUCCESS_RATE"],
            STAGE_1_SUCCESS_RATE=original_cfg["STAGE_1_SUCCESS_RATE"],
            STAGE_2_SUCCESS_RATE=original_cfg["STAGE_2_SUCCESS_RATE"],
            STAGE_2_FIDELITY_TARGET=original_cfg["STAGE_2_FIDELITY_TARGET"],
            ANGLE_S=original_cfg["ANGLE_S"],
            FACTORY_PHYSICAL_SIZE=original_cfg["FACTORY_PHYSICAL_SIZE"],
            STAGE_1_RESOURCE_UNITS=original_cfg["STAGE_1_RESOURCE_UNITS"],
            SYNCHRONIZE_FACTORY_EXECUTION=original_cfg["SYNCHRONIZE_FACTORY_EXECUTION"],
            RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT=original_cfg[
                "RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT"
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate T-cultivation duration + throughput/cycles-per-T sweeps."
    )

    filename = (
        "output/t_cultivation/factory_throughput/t_cultivation_throughput_trials.csv"
    )

    args = parser.parse_args()
    num_trials = 10
    metric_mode = "cycles_per_t"
    factory_counts = list(range(1, 21))
    main(
        factory_counts,
        metric_mode=metric_mode,
        print_profile=True,
        num_trials=num_trials,
        output_csv=filename,
    )
