import os
import sys
import logging
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
    """Use a simple grid: logic qubits on y=0, magic states on y=1."""
    logic_qubit_locations = [(x, 0) for x in range(n_qubits)]
    magic_state_locations = [(x, 1) for x in range(n_factories)]
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
) -> tuple[float, float]:
    """Return throughput and makespan only (used in Monte Carlo trials)."""
    tp, ms, _ = run_t_cultivation_metrics(
        spec=spec,
        n_factories=n_factories,
        seed=seed,
        n_aods=n_aods,
        to_decompose=to_decompose,
        epsilon=epsilon,
    )
    return tp, ms


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


def apply_setting(setting: TSetting) -> None:
    """Apply one evaluation setting to global T-cultivation config."""
    update_config(
        STAGE_2_FIDELITY_TARGET=setting.fidelity_target,
        FACTORY_PHYSICAL_SIZE=setting.factory_physical_size,
        STAGE_1_RESOURCE_UNITS=1,
    )


def main(factory_counts: list[int]):
    # Sweep settings (keep defaults modest so the script runs in reasonable time).
    # For better stochastic smoothing, increase NUM_TRIALS.
    num_trials = 10

    # In all sweeps: fixed K = 10 T gates per qubit.
    k_per_qubit = 10
    # When mean circuit duration is below this (and > 0), emit a lane timeline PDF.
    execution_detail_threshold = 20.0

    settings = [
        TSetting(
            fidelity_target=1e-8, factory_physical_size=2, distance=7
        ),
        TSetting(
            fidelity_target=1e-8, factory_physical_size=4, distance=13
        ),
        TSetting(fidelity_target=1e-10, factory_physical_size=4, distance=13),
    ]

    original_cfg = get_config().copy()
    output_dir = "output/t_cultivation/factory_throughput"
    os.makedirs(output_dir, exist_ok=True)
    detail_dir = os.path.join(output_dir, "execution_detail")
    os.makedirs(detail_dir, exist_ok=True)
    detail_jobs: list[tuple[TSetting, str, SweepSpec, int, int]] = []
    try:
        # ------------------------------------------------------------------
        # Type 1: 1 qubit, fixed K=10, sweep number of factories
        #   Row 1: circuit duration, Row 2: throughput
        # ------------------------------------------------------------------
        fig1, axes1 = plt.subplots(
            2, len(settings), figsize=(6 * len(settings), 8.0), sharex="col"
        )
        if len(settings) == 1:
            axes1 = np.asarray(axes1).reshape(2, 1)
        spec_fig1 = SweepSpec(
            n_qubits=1, k_t_per_qubit=k_per_qubit, layers_parallel=False
        )
        for idx, setting in enumerate(settings):
            ax_time = axes1[0, idx]
            ax_thr = axes1[1, idx]
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
                        throughput_pf, makespan = metrics_for_seed(
                            spec=spec_fig1, n_factories=nf, seed=seed
                        )
                        trial_values_fig1.append(throughput_pf)
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
                if 0 < mt < execution_detail_threshold:
                    detail_jobs.append(
                        (
                            setting,
                            os.path.join(
                                detail_dir,
                                f"type1_set{idx}_nf{nf}.pdf",
                            ),
                            spec_fig1,
                            nf,
                            seed0,
                        )
                    )
            ax_thr.errorbar(
                factory_counts,
                means_fig1,
                yerr=[lower_err_fig1, upper_err_fig1],
                capsize=3,
                marker="o",
            )
            ax_thr.set_xlabel("Number of factories")
            ax_thr.grid(True, alpha=0.3)
            ax_thr.set_ylim(bottom=0.0)

            ax_time.errorbar(
                factory_counts,
                means_time_fig1,
                yerr=[lower_err_time_fig1, upper_err_time_fig1],
                capsize=3,
                marker="o",
            )
            ax_time.set_xlabel("Number of factories")
            ax_time.grid(True, alpha=0.3)
            ax_time.set_title(_setting_subtitle(setting), fontsize=10)
            ax_time.set_ylim(bottom=0.0)

        axes1[0, 0].set_ylabel("Circuit duration")
        axes1[1, 0].set_ylabel("Single-factory throughput (T / time / factory)")
        fig1.suptitle("1 qubit, K=10 T gates, factories sweep", fontsize=12)
        fig1.tight_layout()
        for _ax in axes1[0, :]:
            _ax.set_ylim(bottom=0.0)
        fig1.savefig(
            f"{output_dir}/t_cultivation_one_qubit_vs_factories.pdf",
            dpi=200,
        )
        logging.info(
            "Saved plot: %s",
            f"{output_dir}/t_cultivation_one_qubit_vs_factories.pdf",
        )

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
                            throughput_pf, makespan = metrics_for_seed(
                                spec=spec, n_factories=n_factories, seed=seed
                            )
                            trial_values_fig2.append(throughput_pf)
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
                    if 0 < mt < execution_detail_threshold:
                        detail_jobs.append(
                            (
                                setting,
                                os.path.join(
                                    detail_dir,
                                    f"type2_set{idx}_r{ratio}_nq{n_qubits}.pdf",
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
            ax_thr.set_xlabel("Number of qubits")
            ax_thr.grid(True, alpha=0.3)
            ax_thr.set_ylim(bottom=0.0)
            ax_thr.legend(fontsize=8)

            ax_time.set_xlabel("Number of qubits")
            ax_time.grid(True, alpha=0.3)
            ax_time.set_title(_setting_subtitle(setting), fontsize=10)
            ax_time.set_ylim(bottom=0.0)
            ax_time.legend(fontsize=8)

        axes2[0, 0].set_ylabel("Circuit duration")
        axes2[1, 0].set_ylabel("Single-factory throughput (T / time / factory)")
        fig2.suptitle("Qubits 1..10, K=10, ratio sweep (factories/qubits)", fontsize=12)
        fig2.tight_layout()
        for _ax in axes2[0, :]:
            _ax.set_ylim(bottom=0.0)
        fig2.savefig(f"{output_dir}/t_cultivation_ratio_sweep.pdf", dpi=200)
        logging.info(
            "Saved plot: %s",
            f"{output_dir}/t_cultivation_ratio_sweep.pdf",
        )

        # ------------------------------------------------------------------
        # Type 3: fixed 10 qubits, factories 10..1
        #   Row 1: circuit duration, Row 2: throughput
        # ------------------------------------------------------------------
        n_qubits_fixed = 10
        factory_counts_fig3 = list(range(10, 0, -1))
        spec_fig3 = SweepSpec(
            n_qubits=n_qubits_fixed, k_t_per_qubit=k_per_qubit, layers_parallel=True
        )
        fig3, axes3 = plt.subplots(
            2, len(settings), figsize=(6 * len(settings), 8.0), sharex="col"
        )
        if len(settings) == 1:
            axes3 = np.asarray(axes3).reshape(2, 1)
        for idx, setting in enumerate(settings):
            ax_time = axes3[0, idx]
            ax_thr = axes3[1, idx]
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
                        throughput_pf, makespan = metrics_for_seed(
                            spec=spec_fig3, n_factories=n_factories, seed=seed
                        )
                        trial_values.append(throughput_pf)
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
                if 0 < mt < execution_detail_threshold:
                    detail_jobs.append(
                        (
                            setting,
                            os.path.join(
                                detail_dir,
                                f"type3_set{idx}_nf{n_factories}.pdf",
                            ),
                            spec_fig3,
                            n_factories,
                            seed0,
                        )
                    )

            ax_thr.errorbar(
                factory_counts_fig3,
                means,
                yerr=[lower_err_fig3, upper_err_fig3],
                capsize=3,
                marker="o",
            )
            ax_thr.set_xlabel("Number of factories")
            ax_thr.grid(True, alpha=0.3)
            ax_thr.set_ylim(bottom=0.0)

            ax_time.errorbar(
                factory_counts_fig3,
                means_time_fig3,
                yerr=[lower_err_time_fig3, upper_err_time_fig3],
                capsize=3,
                marker="o",
            )
            ax_time.set_xlabel("Number of factories")
            ax_time.grid(True, alpha=0.3)
            ax_time.set_title(_setting_subtitle(setting), fontsize=10)
            ax_time.set_ylim(bottom=0.0)

        axes3[0, 0].set_ylabel("Circuit duration")
        axes3[1, 0].set_ylabel("Single-factory throughput (T / time / factory)")
        fig3.suptitle("10 qubits, K=10, factories sweep 10->1", fontsize=12)
        fig3.tight_layout()
        for _ax in axes3[0, :]:
            _ax.set_ylim(bottom=0.0)
        fig3.savefig(
            f"{output_dir}/t_cultivation_10q_factory_sweep.pdf",
            dpi=200,
        )
        logging.info(
            "Saved plot: %s",
            f"{output_dir}/t_cultivation_10q_factory_sweep.pdf",
        )

        for setting, path, spec, n_factories, seed in detail_jobs:
            apply_setting(setting)
            try:
                _, _, log = run_t_cultivation_metrics(
                    spec=spec, n_factories=n_factories, seed=seed
                )
            except RuntimeError as e:
                logging.warning("Detail plot skipped (run failed): %s: %s", path, e)
                continue
            if not log:
                logging.warning("Detail plot skipped (empty log): %s", path)
                continue
            plot_t_cultivation_execution(log, spec.n_qubits, n_factories, path)
            plt.close()
        if detail_jobs:
            logging.info(
                "Wrote %d execution detail PDFs under %s",
                len(detail_jobs),
                detail_dir,
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
    factory_counts = list(range(1, 10))
    main(factory_counts)
