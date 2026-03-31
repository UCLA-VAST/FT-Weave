import os
import sys
import logging
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

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
):
    # Sweep settings (keep defaults modest so the script runs in reasonable time).
    # For better stochastic smoothing, increase NUM_TRIALS.
    num_trials = 10

    # In all sweeps: fixed K = 10 T gates per qubit.
    k_per_qubit = 10
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
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
    first_trial_jobs: list[tuple[TSetting, str, SweepSpec, int, int]] = []
    try:
        # ------------------------------------------------------------------
        # Type 1: 1 qubit, fixed K=10, sweep number of factories
        #   Row 1: circuit duration, Row 2: throughput
        # ------------------------------------------------------------------
        fig1, axes1 = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
        fig1_duration_ymax = 0.0
        fig1_metric_ymax = 0.0
        spec_fig1 = SweepSpec(
            n_qubits=1, k_t_per_qubit=k_per_qubit, layers_parallel=False
        )
        ax_time = axes1[0]
        ax_thr = axes1[1]
        for idx, setting in enumerate(settings):
            apply_setting(setting)
            means_fig1: list[float] = []
            lower_err_fig1: list[float] = []
            upper_err_fig1: list[float] = []
            means_time_fig1: list[float] = []
            lower_err_time_fig1: list[float] = []
            upper_err_time_fig1: list[float] = []
            for nf in factory_counts:
                trial_values_fig1: list[float] = []
                trial_times_fig1: list[float] = []
                logging.info(
                    "Fig1 [distance-%d, LER=%g] nf=%d (%d trials)",
                    setting.distance,
                    setting.fidelity_target,
                    nf,
                    num_trials,
                )
                for t in range(num_trials):
                    seed = 1000000 * settings.index(setting) + 1000 * nf + t
                    try:
                        _, makespan = metrics_for_seed(
                            spec=spec_fig1,
                            n_factories=nf,
                            seed=seed,
                        )
                        trial_values_fig1.append(
                            metric_from_makespan(
                                spec=spec_fig1,
                                n_factories=nf,
                                makespan=makespan,
                                metric_mode=metric_mode,
                            )
                        )
                        trial_times_fig1.append(makespan)
                    except RuntimeError as e:
                        logging.warning(
                            "Fig1 failed [distance-%d, LER=%g] nf=%d: %s",
                            setting.distance,
                            setting.fidelity_target,
                            nf,
                            e,
                        )
                        trial_values_fig1.append(0.0)
                        trial_times_fig1.append(0.0)
                m, vmin, vmax = mean_min_max(trial_values_fig1)
                means_fig1.append(m)
                lower_err_fig1.append(max(0.0, m - vmin))
                upper_err_fig1.append(max(0.0, vmax - m))
                mt, tmin, tmax = mean_min_max(trial_times_fig1)
                means_time_fig1.append(mt)
                lower_err_time_fig1.append(max(0.0, mt - tmin))
                upper_err_time_fig1.append(max(0.0, tmax - mt))
                seed0 = 1000000 * settings.index(setting) + 1000 * nf
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
            fig1_duration_ymax = max(
                fig1_duration_ymax,
                max(
                    (m + e for m, e in zip(means_time_fig1, upper_err_time_fig1)),
                    default=0.0,
                ),
            )
            fig1_metric_ymax = max(
                fig1_metric_ymax,
                max((m + e for m, e in zip(means_fig1, upper_err_fig1)), default=0.0),
            )
            ax_thr.errorbar(
                factory_counts,
                means_fig1,
                yerr=[lower_err_fig1, upper_err_fig1],
                capsize=3,
                marker="o",
                label=_setting_subtitle(setting),
            )
            ax_time.errorbar(
                factory_counts,
                means_time_fig1,
                yerr=[lower_err_time_fig1, upper_err_time_fig1],
                capsize=3,
                marker="o",
                label=_setting_subtitle(setting),
            )
        ax_thr.set_xlabel("Number of factories")
        ax_thr.grid(True, alpha=0.3)
        ax_thr.tick_params(axis="both", labelsize=13)
        ax_thr.legend(fontsize=11)

        ax_time.set_xlabel("Number of factories")
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(axis="both", labelsize=13)
        ax_time.legend(fontsize=11)

        axes1[0].set_ylabel("Circuit duration")
        if metric_mode == "throughput":
            axes1[1].set_ylabel("Single-factory throughput (T / time / factory)")
        else:
            axes1[1].set_ylabel("Cycles per T per factory")
        fig1.suptitle("1 qubit, K=10 T gates, factories sweep", fontsize=18)
        fig1.tight_layout()
        fig1_duration_ylim_top = max(1.0, fig1_duration_ymax * 1.08)
        fig1_metric_ylim_top = max(1.0, fig1_metric_ymax * 1.08)
        axes1[0].set_ylim(0.0, fig1_duration_ylim_top)
        axes1[1].set_ylim(0.0, fig1_metric_ylim_top)

        # ------------------------------------------------------------------
        # Type 2: qubit count 1..10, factory/qubit ratio in {1,2,4,8}
        #   Row 1: circuit duration, Row 2: throughput
        # ------------------------------------------------------------------
        qubit_counts = list(range(1, 11))
        ratios = [1, 2, 4, 8]
        fig2, axes2 = plt.subplots(
            2, len(settings), figsize=(6 * len(settings), 8.0), sharex="col"
        )
        if len(settings) == 1:
            axes2 = np.asarray(axes2).reshape(2, 1)
        fig2_duration_ymax = 0.0
        fig2_metric_ymax = 0.0
        for idx, setting in enumerate(settings):
            ax_time = axes2[0, idx]
            ax_thr = axes2[1, idx]
            apply_setting(setting)
            for ratio in ratios:
                means_fig2: list[float] = []
                lower_err_fig2: list[float] = []
                upper_err_fig2: list[float] = []
                means_time_fig2: list[float] = []
                lower_err_time_fig2: list[float] = []
                upper_err_time_fig2: list[float] = []
                for n_qubits in qubit_counts:
                    n_factories = ratio * n_qubits
                    spec = SweepSpec(
                        n_qubits=n_qubits,
                        k_t_per_qubit=k_per_qubit,
                        layers_parallel=True,
                    )
                    trial_values_fig2: list[float] = []
                    trial_times_fig2: list[float] = []
                    for t in range(num_trials):
                        seed = (
                            2000000 * settings.index(setting)
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
                            trial_values_fig2.append(
                                metric_from_makespan(
                                    spec=spec,
                                    n_factories=n_factories,
                                    makespan=makespan,
                                    metric_mode=metric_mode,
                                )
                            )
                            trial_times_fig2.append(makespan)
                        except RuntimeError as e:
                            logging.warning(
                                "Type2 failed [%s] ratio=%d nq=%d nf=%d: %s",
                                _setting_subtitle(setting),
                                ratio,
                                n_qubits,
                                n_factories,
                                e,
                            )
                            trial_values_fig2.append(0.0)
                            trial_times_fig2.append(0.0)
                    m, vmin, vmax = mean_min_max(trial_values_fig2)
                    means_fig2.append(m)
                    lower_err_fig2.append(max(0.0, m - vmin))
                    upper_err_fig2.append(max(0.0, vmax - m))
                    mt, tmin, tmax = mean_min_max(trial_times_fig2)
                    means_time_fig2.append(mt)
                    lower_err_time_fig2.append(max(0.0, mt - tmin))
                    upper_err_time_fig2.append(max(0.0, tmax - mt))
                    seed0 = (
                        2000000 * settings.index(setting)
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
                ax_thr.errorbar(
                    qubit_counts,
                    means_fig2,
                    yerr=[lower_err_fig2, upper_err_fig2],
                    capsize=3,
                    marker="o",
                    label=f"f/q={ratio}",
                )
                ax_time.errorbar(
                    qubit_counts,
                    means_time_fig2,
                    yerr=[lower_err_time_fig2, upper_err_time_fig2],
                    capsize=3,
                    marker="o",
                    label=f"f/q={ratio}",
                )
                fig2_duration_ymax = max(
                    fig2_duration_ymax,
                    max(
                        (m + e for m, e in zip(means_time_fig2, upper_err_time_fig2)),
                        default=0.0,
                    ),
                )
                fig2_metric_ymax = max(
                    fig2_metric_ymax,
                    max(
                        (m + e for m, e in zip(means_fig2, upper_err_fig2)), default=0.0
                    ),
                )
            ax_thr.set_xlabel("Number of qubits")
            ax_thr.grid(True, alpha=0.3)
            ax_thr.legend(fontsize=12)
            ax_thr.tick_params(axis="both", labelsize=13)

            ax_time.set_xlabel("Number of qubits")
            ax_time.grid(True, alpha=0.3)
            ax_time.set_title(_setting_subtitle(setting), fontsize=14)
            ax_time.legend(fontsize=12)
            ax_time.tick_params(axis="both", labelsize=13)

        axes2[0, 0].set_ylabel("Circuit duration")
        if metric_mode == "throughput":
            axes2[1, 0].set_ylabel("Single-factory throughput (T / time / factory)")
        else:
            axes2[1, 0].set_ylabel("Cycles per T per factory")
        fig2.suptitle("Qubits 1..10, K=10, ratio sweep (factories/qubits)", fontsize=18)
        fig2.tight_layout()
        fig2_duration_ylim_top = max(1.0, fig2_duration_ymax * 1.08)
        fig2_metric_ylim_top = max(1.0, fig2_metric_ymax * 1.08)
        for _ax in axes2[0, :]:
            _ax.set_ylim(0.0, fig2_duration_ylim_top)
        for _ax in axes2[1, :]:
            _ax.set_ylim(0.0, fig2_metric_ylim_top)

        # ------------------------------------------------------------------
        # Type 3: fixed 10 qubits, factories 10..1
        #   Row 1: circuit duration, Row 2: throughput
        # ------------------------------------------------------------------
        n_qubits_fixed = 10
        factory_counts_fig3 = list(range(10, 0, -1))
        spec_fig3 = SweepSpec(
            n_qubits=n_qubits_fixed, k_t_per_qubit=k_per_qubit, layers_parallel=True
        )
        fig3, axes3 = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
        fig3_duration_ymax = 0.0
        fig3_metric_ymax = 0.0
        ax_time = axes3[0]
        ax_thr = axes3[1]
        for idx, setting in enumerate(settings):
            apply_setting(setting)
            means: list[float] = []
            lower_err_fig3: list[float] = []
            upper_err_fig3: list[float] = []
            means_time_fig3: list[float] = []
            lower_err_time_fig3: list[float] = []
            upper_err_time_fig3: list[float] = []
            for n_factories in factory_counts_fig3:
                trial_values: list[float] = []
                trial_times: list[float] = []
                for t in range(num_trials):
                    seed = 3000000 * settings.index(setting) + 1000 * n_factories + t
                    try:
                        _, makespan = metrics_for_seed(
                            spec=spec_fig3,
                            n_factories=n_factories,
                            seed=seed,
                            print_profile=print_profile,
                        )
                        trial_values.append(
                            metric_from_makespan(
                                spec=spec_fig3,
                                n_factories=n_factories,
                                makespan=makespan,
                                metric_mode=metric_mode,
                            )
                        )
                        trial_times.append(makespan)
                    except RuntimeError as e:
                        logging.warning(
                            "Type3 failed [distance-%d, LER=%g] nf=%d: %s",
                            setting.distance,
                            setting.fidelity_target,
                            n_factories,
                            e,
                        )
                        trial_values.append(0.0)
                        trial_times.append(0.0)
                m, vmin, vmax = mean_min_max(trial_values)
                means.append(m)
                lower_err_fig3.append(max(0.0, m - vmin))
                upper_err_fig3.append(max(0.0, vmax - m))
                mt, tmin, tmax = mean_min_max(trial_times)
                means_time_fig3.append(mt)
                lower_err_time_fig3.append(max(0.0, mt - tmin))
                upper_err_time_fig3.append(max(0.0, tmax - mt))
                seed0 = 3000000 * settings.index(setting) + 1000 * n_factories
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
            fig3_duration_ymax = max(
                fig3_duration_ymax,
                max(
                    (m + e for m, e in zip(means_time_fig3, upper_err_time_fig3)),
                    default=0.0,
                ),
            )
            fig3_metric_ymax = max(
                fig3_metric_ymax,
                max((m + e for m, e in zip(means, upper_err_fig3)), default=0.0),
            )

            ax_thr.errorbar(
                factory_counts_fig3,
                means,
                yerr=[lower_err_fig3, upper_err_fig3],
                capsize=3,
                marker="o",
                label=_setting_subtitle(setting),
            )
            ax_time.errorbar(
                factory_counts_fig3,
                means_time_fig3,
                yerr=[lower_err_time_fig3, upper_err_time_fig3],
                capsize=3,
                marker="o",
                label=_setting_subtitle(setting),
            )
        ax_thr.set_xlabel("Number of factories")
        ax_thr.grid(True, alpha=0.3)
        ax_thr.tick_params(axis="both", labelsize=13)
        ax_thr.legend(fontsize=11)

        ax_time.set_xlabel("Number of factories")
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(axis="both", labelsize=13)
        ax_time.legend(fontsize=11)

        axes3[0].set_ylabel("Circuit duration")
        if metric_mode == "throughput":
            axes3[1].set_ylabel("Single-factory throughput (T / time / factory)")
        else:
            axes3[1].set_ylabel("Cycles per T per factory")
        fig3.suptitle("10 qubits, K=10, factories sweep 10->1", fontsize=18)
        fig3.tight_layout()
        fig3_duration_ylim_top = max(1.0, fig3_duration_ymax * 1.08)
        fig3_metric_ylim_top = max(1.0, fig3_metric_ymax * 1.08)
        axes3[0].set_ylim(0.0, fig3_duration_ylim_top)
        axes3[1].set_ylim(0.0, fig3_metric_ylim_top)

        fig1.savefig(
            f"{output_dir}/fig_1_t_cultivation_one_qubit_vs_factories.pdf",
            dpi=200,
        )
        logging.info(
            "Saved plot: %s",
            f"{output_dir}/fig_1_t_cultivation_one_qubit_vs_factories.pdf",
        )
        fig2.savefig(f"{output_dir}/fig_2_t_cultivation_ratio_sweep.pdf", dpi=200)
        logging.info(
            "Saved plot: %s",
            f"{output_dir}/fig_2_t_cultivation_ratio_sweep.pdf",
        )
        fig3.savefig(
            f"{output_dir}/fig_3_t_cultivation_10q_factory_sweep.pdf",
            dpi=200,
        )
        logging.info(
            "Saved plot: %s",
            f"{output_dir}/fig_3_t_cultivation_10q_factory_sweep.pdf",
        )

        for setting, path, spec, n_factories, seed in first_trial_jobs:
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
            plt.close()
        if first_trial_jobs:
            logging.info(
                "Wrote %d first-trial execution timeline PDFs under %s",
                len(first_trial_jobs),
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
    parser.add_argument(
        "--metric-mode",
        choices=["throughput", "cycles_per_t"],
        default="cycles_per_t",
        help="Bottom-row metric to plot (default: cycles_per_t).",
    )
    args = parser.parse_args()

    factory_counts = list(range(1, 10))
    main(
        factory_counts,
        metric_mode=args.metric_mode,
        print_profile=True,
    )
