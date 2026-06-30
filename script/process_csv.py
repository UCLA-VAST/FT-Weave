# analyze_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import math
import os
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
from src.t_cultivation.config import STAGE_1_SUCCESS_RATE

_FIG_FONT_SIZE = 18

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": _FIG_FONT_SIZE,
        "axes.titlesize": _FIG_FONT_SIZE,
        "axes.labelsize": _FIG_FONT_SIZE,
        "xtick.labelsize": _FIG_FONT_SIZE,
        "ytick.labelsize": _FIG_FONT_SIZE,
        "legend.fontsize": _FIG_FONT_SIZE,
        "figure.titlesize": _FIG_FONT_SIZE,
    }
)


# (placement, prepare_lookahead_angles, trivial_return, consider_skip_rus,
#  decompose_move, parallel_execution) — keep in sync with evaluation_fidelity_star.py
SETTINGS = [
    # ("seperate_region_row", False, True, 0, False, False),  # vanilla
    # ("seperate_region_row", False, False, 0, True, False),  # optimized return
    # (
    #     "seperate_region_row",
    #     False,
    #     False,
    #     2,
    #     True,
    #     False,
    # ),  # optimized return + skip whole RUS
    # ("seperate_region_row", True, True, 0, False, False),  # lookahead angles
    # (
    #     "seperate_region_row",
    #     True,
    #     False,
    #     0,
    #     True,
    #     False,
    # ),  # lookahead + optimized return
    # (
    #     "seperate_region_row",
    #     True,
    #     False,
    #     2,
    #     True,
    #     False,
    # ),  # lookahead + optimized return + skip whole RUS
    ("col_based", False, True, 0, False, False),  # vanilla
    # ("col_based", True, True, 0, False, False),  # lookahead angles
    # ("col_based", True, False, 0, True, False),  # lookahead + opt. move
    # (
    #     "col_based",
    #     True,
    #     False,
    #     2,
    #     True,
    #     False,
    # ),  # lookahead + opt. move + skip RUS
    ("col_based", False, False, 0, True, False),  # opt. move
    ("col_based", False, False, 2, True, False),  # opt. move + skip RUS
    (
        "col_based",
        True,
        False,
        2,
        True,
        False,
    ),  # lookahead + opt. move + skip RUS
    (
        "col_based",
        False,
        False,
        2,
        False,
        True,
    ),  # opt. return + skip whole RUS + async. RUS
]

MAIN_SETTING: tuple = ("col_based", False, False, 2, True, False)

# Cross-placement microarch figures: compile tuple from ``TEMP_SETTINGS`` in
# ``evaluation_fidelity_star.py``; the placement field is ignored when plotting.
MICROARCH_COMPARE_SETTING: tuple = (
    "col_based",
    False,
    False,
    2,
    False,
    True,
)

# ``MAIN_SETTINGS`` in evaluation_fidelity_star.py (col_based, sync. execution).
STAR_VS_T_RUNTIME_SETTING: tuple = ("col_based", True, False, 2, True, False)
STAR_COL_BASED_VANILLA_SETTING: tuple = ("col_based", False, True, 0, False, False)

STAR_SETTING_LABELS: list[str] = [
    # "Vanilla (sep. region)",
    # "Opt. return (sep. region)",
    # "Opt. return + skip whole RUS (sep. region)",
    # "Lookahead (sep. region)",
    # "Lookahead + opt. return (sep. region)",
    # "Lookahead + opt. return + skip whole RUS (sep. region)",
    "Vanilla",
    "Opt. move",
    "Opt. move + skip RUS",
    "Opt. move + skip RUS + lookahead angles",
    "Opt. move + skip RUS + async. exec.",
]

# Configuration columns (experimental settings)
CONFIG_COLS = [
    "n_qubits",
    "qubit_cols",
    "qubit_rows",
    "placement",
    "n_aods",
    "consider_skip_rus",
    "tmr_assignment_method",
    "prepare_lookahead_angles",
    "trivial_return",
    "decompose_move",
    "parallel_execution",
]

# Result columns
RESULT_COLS = [
    "total_time",
    "movement_time",
    "return_movement_time",
    # "TMR_round",
    # "RUS_round",
    # "max_rus_per_qubit",
    # "avg_rus_per_qubit",
]

# ----- IMPORTANT SETTINGS -----
SWEEP_COL = "n_qubits"  # x-axis
EXPECTED_TRIALS_PER_CONFIG = 10

_RZ_ANGLE_COUNT_CACHE: dict[tuple[int, int, int], list[int]] = {}

_LOWER_BOUND_COMPONENTS: dict[str, tuple[int, int]] | None = None

MERGED_ROUND_NAME = "merged_rounds_except_4"
INDIVIDUAL_ROUND = 4

# STAR profiling CSV distances from ``evaluation_fidelity_star.py`` logical models.
STAR_PROFILING_CODE_DISTANCES: tuple[int, ...] = (7, 9, 13)
# STAR curves on STAR vs T-cultivation combined runtime PDFs (omit d=13 for readability).
STAR_VS_T_RUNTIME_STAR_DISTANCES: tuple[int, ...] = (7, 9)
# Console runtime summary: one readable block (full_trotter, col_based for cross-arch).
RUNTIME_SUMMARY_OVERALL_AODS: tuple[int, ...] = (1, 5)
RUNTIME_SUMMARY_STAR_CODE_DISTANCES: tuple[int, ...] = (7, 9)
RUNTIME_SUMMARY_T_CODE_DISTANCES: tuple[int, ...] = (9, 13)
RUNTIME_SUMMARY_T_LER = 1e-8
# T-cultivation profiling placements for architecture-specific runtime figures.
T_CULTIVATION_ARCHITECTURE_PLACEMENTS: tuple[str, ...] = (
    "seperate_region_row",
    "col_based",
    "checkerboard",
)
# Default compile tuple for T-cultivation architecture placement figures (sync with
# ``MAIN_COMPILE_SETTING`` in ``evaluation_fidelity_t_cultivation.py``).
T_CULTIVATION_MAIN_COMPILE_SETTING: tuple[bool, bool, bool] = (False, True, True)


def _setting_filter(
    df: pd.DataFrame, setting: tuple, *, match_placement: bool = True
) -> pd.Series:
    placement, prepare_lookahead, trivial_return, skip_rus, decompose_move, parallel = (
        setting
    )
    mask = df["trivial_return"] == trivial_return
    if match_placement:
        mask &= df["placement"].astype(str).str.strip() == str(placement).strip()
    mask &= df["consider_skip_rus"] == skip_rus
    mask &= df["decompose_move"] == decompose_move
    mask &= df["parallel_execution"] == parallel
    if "prepare_lookahead_angles" in df.columns:
        mask &= df["prepare_lookahead_angles"] == prepare_lookahead
    else:
        mask &= prepare_lookahead is True
    return mask


def _filter_col_based(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "placement" not in df.columns:
        return df
    return df[df["placement"] == "col_based"].copy()


def _get_available_settings(df: pd.DataFrame, max_settings: int = 2) -> list[tuple]:
    """Return the first ``max_settings`` SETTINGS tuples that have rows in ``df``."""
    available: list[tuple] = []
    for setting in SETTINGS:
        if df.empty:
            break
        if df.loc[_setting_filter(df, setting)].empty:
            continue
        available.append(setting)
        if len(available) >= max_settings:
            break
    return available


def _build_aod_color_map(aod_values: list[int]) -> dict[int, tuple]:
    """Return a consistent color map keyed by AOD count."""
    unique_aods = sorted({int(v) for v in aod_values})
    cmap = plt.get_cmap("tab10")
    return {aod: cmap(i % 10) for i, aod in enumerate(unique_aods)}


# ------------------------------
# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def _get_theoretical_lower_bound(
    n_values: list,
) -> dict[str, list]:
    """Calculate theoretical lower bounds for circuit execution time by layer type.

    Args:
        n_values: List of qubit counts to evaluate

    Returns:
        Dict with keys 'x_layer', 'zz_layer', 'full_trotter' mapping to lists of time bounds
    """
    constant_tmr_half_overhead = 6 + 2 + 1
    constant_tmr_overhead = 12 + 2 + 1

    x_layer_bounds = [
        constant_tmr_overhead * int(math.log2(float(n)) + 1) for n in n_values
    ]
    zz_layer_bounds = [
        constant_tmr_half_overhead * int(math.log2(float(n / 2)) + 1) for n in n_values
    ]
    full_trotter_bounds = [
        2 * x + zz * 4 for x, zz in zip(x_layer_bounds, zz_layer_bounds)
    ]

    return {
        "x_layer": x_layer_bounds,
        "zz_layer": zz_layer_bounds,
        "full_trotter": full_trotter_bounds,
    }


def _get_theoretical_lower_bound_t_cultivation(
    n_values: list,
    code_distance: int | None = None,
    fidelity_target: float | None = None,
) -> dict[str, list]:
    """Calculate theoretical lower bounds for T-cultivation execution time.

    Uses the per-T-gate expectation model from the in-code comments:
    - stage-1 factory attempts: d=7 -> 2*10, d=13 -> 10
    - stage-2 factory attempts: LER=1e-8 -> 1.1*4, LER=1e-10 -> 2.5*4
    - RUS expected retries: 50% success -> factor 2
    - S-gate overhead: +1

    The returned bounds reuse STAR's layer-shape and scale it by the expected
    T-cultivation work factor for the selected setting.
    """

    n_t_per_qubit = 40

    # Stage-1: success-rate-dependent expected attempts.
    stage_1_parallelization = 4
    if code_distance is not None and int(code_distance) < 13:
        stage_1_parallelization = 2

    effective_stage_1_success_rate = (
        1 - (1 - STAGE_1_SUCCESS_RATE) ** stage_1_parallelization
    )

    # stage1_factor = 12.5 * effective_stage_1_success_rate / (1 - effective_stage_1_success_rate ** 2)

    # Stage-2: depends on LER target used by T-cultivation setting.
    if code_distance is not None and int(code_distance) <= 13:
        expected_attempt = 1 / (effective_stage_1_success_rate * 0.66)
        overhead = 12.5 + 0.5 * effective_stage_1_success_rate
    else:
        expected_attempt = 1 / (effective_stage_1_success_rate * 0.66)
        overhead = 12.5 + 6 * effective_stage_1_success_rate

    rus_factor = 2.0 + 1.0
    s_gate_overhead = 1.0
    expected_time_for_one_resource = (
        (overhead) * expected_attempt + rus_factor + s_gate_overhead
    )
    # print(f"expected_time_for_one_resource: {expected_time_for_one_resource}")
    # input()
    # for x layer bound, we can assume some parallelization and only count half the T gates per qubit
    x_layer_bounds = [expected_time_for_one_resource * n_t_per_qubit for _ in n_values]

    expected_time_for_one_resource = (
        (overhead) * expected_attempt / 2 + rus_factor + s_gate_overhead
    )
    # for zz layer bound, we can assume more parallelization and only count a quarter of the T gates per qubit
    zz_layer_bounds = [expected_time_for_one_resource * n_t_per_qubit for _ in n_values]

    full_trotter_bounds = [
        2 * x + 4 * zz for x, zz in zip(x_layer_bounds, zz_layer_bounds)
    ]
    return {
        "x_layer": x_layer_bounds,
        "zz_layer": zz_layer_bounds,
        "full_trotter": full_trotter_bounds,
    }


def _plot_total_time(df, line_col, prefix, output_dir):
    lines = sorted(df[line_col].unique())

    plt.figure(figsize=(7, 5))

    for line_name in lines:
        sub = df[df[line_col] == line_name].sort_values(SWEEP_COL)

        plt.errorbar(
            sub[SWEEP_COL],
            sub["total_time_mean"],
            yerr=sub["total_time_std"],
            marker="o",
            capsize=4,
            label=str(line_name),
        )

    plt.xlabel(SWEEP_COL)
    plt.ylabel("Execution Time")
    plt.legend(title="microarchitecture")
    plt.title("Total Execution Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_total_time.pdf"))
    plt.close()


def _aggregate_microarch_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate microarch metrics across trials and keep sample counts for validation."""
    g = (
        df.groupby(["n_aods", "placement", "n_qubits"])
        .agg(
            total_time_mean=("total_time", "mean"),
            total_time_std=("total_time", "std"),
            total_time_min=("total_time", "min"),
            total_time_max=("total_time", "max"),
            total_time_count=("total_time", "count"),
            movement_time_mean=("movement_time", "mean"),
            movement_time_std=("movement_time", "std"),
            movement_time_min=("movement_time", "min"),
            movement_time_max=("movement_time", "max"),
            movement_time_count=("movement_time", "count"),
            return_movement_time_mean=("return_movement_time", "mean"),
            return_movement_time_std=("return_movement_time", "std"),
            return_movement_time_min=("return_movement_time", "min"),
            return_movement_time_max=("return_movement_time", "max"),
            return_movement_time_count=("return_movement_time", "count"),
        )
        .reset_index()
    )

    std_cols = [
        "total_time_std",
        "movement_time_std",
        "return_movement_time_std",
    ]
    for col in std_cols:
        if col in g.columns:
            g[col] = g[col].fillna(0.0)

    if "total_time_count" in g.columns:
        unique_counts = sorted(g["total_time_count"].dropna().astype(int).unique())
        if len(unique_counts) > 0 and any(
            c != EXPECTED_TRIALS_PER_CONFIG for c in unique_counts
        ):
            print(
                "[microarch] Warning: trial counts per point are",
                unique_counts,
                f"(expected {EXPECTED_TRIALS_PER_CONFIG}).",
            )

    return g


def _get_round_angle_counts(
    n_qubits: int, qubit_rows: int, qubit_cols: int
) -> list[int]:
    key = (int(n_qubits), int(qubit_rows), int(qubit_cols))
    if key in _RZ_ANGLE_COUNT_CACHE:
        return _RZ_ANGLE_COUNT_CACHE[key]

    qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
        n_qubits=int(n_qubits),
        qubit_layout=(int(qubit_rows), int(qubit_cols)),
        J=1.0,
        h=1.0,
        dt=1.0,
        logical=True,
        order=2,
    )
    counts = [len(inst["targets"]) for inst in qc_one_layer if inst.get("gate") == "Rz"]
    _RZ_ANGLE_COUNT_CACHE[key] = counts
    return counts


def _build_round_angle_lookup(
    dfs_dict: dict[str, pd.DataFrame],
) -> dict[str, dict[int, int]]:
    round_angle_lookup: dict[str, dict[int, int]] = {}

    for round_name, df in dfs_dict.items():
        if df.empty:
            round_angle_lookup[round_name] = {}
            continue

        layout_rows = (
            df[["n_qubits", "qubit_rows", "qubit_cols"]]
            .dropna()
            .drop_duplicates()
            .sort_values("n_qubits")
        )
        if layout_rows.empty:
            round_angle_lookup[round_name] = {}
            continue

        per_qubit_angles: dict[int, int] = {}
        if str(round_name) == MERGED_ROUND_NAME:
            for _, row in layout_rows.iterrows():
                n_qubits = int(row["n_qubits"])
                counts = _get_round_angle_counts(
                    n_qubits,
                    int(row["qubit_rows"]),
                    int(row["qubit_cols"]),
                )
                if len(counts) > 0:
                    # For ZZ layers, show per-layer angle count using ZZ-only angles.
                    zz_layer_count = max(1, len(counts) - 1)
                    zz_total_angles = sum(
                        count
                        for round_idx, count in enumerate(counts)
                        if round_idx != INDIVIDUAL_ROUND
                    )
                    merged_angle_count = int(zz_total_angles / zz_layer_count)
                    per_qubit_angles[n_qubits] = int(merged_angle_count)
        elif str(round_name).startswith("round_"):
            try:
                round_idx = int(str(round_name).split("_")[-1])
            except (TypeError, ValueError):
                round_idx = None

            if round_idx is not None:
                for _, row in layout_rows.iterrows():
                    n_qubits = int(row["n_qubits"])
                    counts = _get_round_angle_counts(
                        n_qubits,
                        int(row["qubit_rows"]),
                        int(row["qubit_cols"]),
                    )
                    if 0 <= round_idx < len(counts):
                        per_qubit_angles[n_qubits] = int(counts[round_idx])
        elif str(round_name) == "full_trotter":
            for _, row in layout_rows.iterrows():
                n_qubits = int(row["n_qubits"])
                counts = _get_round_angle_counts(
                    n_qubits,
                    int(row["qubit_rows"]),
                    int(row["qubit_cols"]),
                )
                per_qubit_angles[n_qubits] = int(sum(counts))

        round_angle_lookup[round_name] = per_qubit_angles

    return round_angle_lookup


def _build_round_plot_dicts(df: pd.DataFrame) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Build plotting dictionaries with all rounds except round_4 merged, and round_4 separate."""
    df_full = aggregate_full_trotter(df)

    dfs_dict_micro = {"full_trotter": df_full}
    dfs_dict_ablation = {"full_trotter": df_full}
    dfs_dict_aod = {"full_trotter": df_full}

    round_values = sorted(
        pd.to_numeric(df["round"], errors="coerce").dropna().astype(int).unique()
    )

    merged_rounds = [
        round_idx for round_idx in round_values if round_idx != INDIVIDUAL_ROUND
    ]
    if merged_rounds:
        merged_df = df[df["round"].isin(merged_rounds)].copy()
        dfs_dict_micro[MERGED_ROUND_NAME] = merged_df
        dfs_dict_ablation[MERGED_ROUND_NAME] = merged_df
        dfs_dict_aod[MERGED_ROUND_NAME] = merged_df

    if INDIVIDUAL_ROUND in round_values:
        df_round = df[df["round"] == INDIVIDUAL_ROUND].copy()
        round_name = f"round_{INDIVIDUAL_ROUND}"
        dfs_dict_micro[round_name] = df_round
        dfs_dict_ablation[round_name] = df_round
        dfs_dict_aod[round_name] = df_round

    return dfs_dict_micro, dfs_dict_ablation, dfs_dict_aod


def _format_round_title(round_name: str) -> str:
    if round_name == "full_trotter":
        return "Full Trotter"
    if round_name == MERGED_ROUND_NAME:
        return "ZZ layers"
    if round_name == f"round_{INDIVIDUAL_ROUND}":
        return "X layer"
    return round_name


def _format_setting_title(setting_label: str | None, setting_idx: int) -> str:
    """Convert internal setting tags to presentation labels for figure titles."""
    if setting_label is not None:
        label = str(setting_label).strip().lower().replace(" ", "_")
        if label in {
            "setting_4",
            "setting_9",
            "sync",
            "sync_exec",
            "sync._exec.",
        }:
            return "Sync. Execution"
        if label in {
            "setting_6",
            "setting_12",
            "async",
            "async_exec",
            "async._exec.",
            "microarch_compare",
        }:
            return "Async. Execution"
        return str(setting_label)

    if setting_idx in {4, 9}:
        return "Sync. Execution"
    if setting_idx in {6, 12}:
        return "Async. Execution"
    return "Execution"


def _format_microarch_placement_label(placement: str) -> str:
    normalized = str(placement).strip().lower()
    if normalized == "col_based":
        return "Column-Based"
    if normalized in {"seperate_region_row", "separate_region_row"}:
        return "Seperate Row Region"
    if normalized == "checkerboard":
        return "Checkerboard"
    return str(placement).replace("_", " ").title()


def _format_nqubit_ticklabels(
    x_vals: list, round_name: str, round_angle_lookup: dict[str, dict[int, int]] | None
) -> list[str]:
    if round_angle_lookup is None:
        return [str(int(x)) if pd.notna(x) else "" for x in x_vals]

    angle_map = round_angle_lookup.get(round_name, {})
    labels = []
    for x in x_vals:
        if pd.isna(x):
            labels.append("")
            continue
        n_qubits = int(x)
        if n_qubits in angle_map:
            labels.append(f"{n_qubits} ({angle_map[n_qubits]})")
        else:
            labels.append(str(n_qubits))
    return labels


def _add_star_reference_tick(ax, star_values: np.ndarray) -> None:
    """Add a readable y-tick near STAR duration in mixed STAR vs T plots."""
    if star_values.size == 0:
        return
    star_values = star_values[np.isfinite(star_values)]
    if star_values.size == 0:
        return
    star_tick = float(np.median(star_values))
    if star_tick <= 0:
        return

    y_min, y_max = ax.get_ylim()
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return

    tick_step = 5000
    base_ticks = []
    tick = max(tick_step, int(np.floor(y_min / tick_step)) * tick_step)
    if tick == 0:
        tick = tick_step
    while tick <= y_max:
        if tick >= y_min:
            base_ticks.append(tick)
        tick += tick_step

    y_ticks = sorted(set(float(v) for v in [*base_ticks, star_tick] if np.isfinite(v)))
    ax.set_yticks(y_ticks)


def _add_standard_y_ticks_with_star_reference(ax, star_values: np.ndarray) -> None:
    """Use regular y-ticks and keep the STAR reference tick as an extra marker."""
    if star_values.size == 0:
        return
    star_values = star_values[np.isfinite(star_values)]
    if star_values.size == 0:
        return
    star_tick = float(np.median(star_values))
    if star_tick <= 0:
        return

    locator = mticker.MaxNLocator(nbins=4, prune="lower")
    ax.yaxis.set_major_locator(locator)
    y_ticks = list(ax.get_yticks())
    y_ticks.append(star_tick)
    y_ticks = sorted(set(float(v) for v in y_ticks if np.isfinite(v)))
    ax.set_yticks(y_ticks)


def _apply_shared_bottom_legend(fig, ncol: int = 4, anchor_y: float = 0.01) -> None:
    """Collapse subplot legends into one shared figure legend at the bottom."""
    handles: list = []
    labels: list[str] = []
    for ax in fig.axes:
        handles_local, labels_local = ax.get_legend_handles_labels()
        for handle, label in zip(handles_local, labels_local):
            if not label or label.startswith("_") or label in labels:
                continue
            handles.append(handle)
            labels.append(label)
        ax_legend = ax.get_legend()
        if ax_legend is not None:
            ax_legend.remove()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, anchor_y),
            ncol=ncol,
            fontsize=_FIG_FONT_SIZE,
            frameon=True,
        )


def _plot_movement_bar(df, line_col, prefix, output_dir):
    lines = sorted(df[line_col].unique())
    x_vals = sorted(df[SWEEP_COL].unique())
    width = 0.8 / len(lines)

    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i) for i in range(10)]

    plt.figure(figsize=(7, 5))
    handles = []
    for i, line_name in enumerate(lines):
        sub = df[df[line_col] == line_name].sort_values(SWEEP_COL)

        x = [v + i * width for v in range(len(x_vals))]

        move = sub["movement_time_mean"].values
        ret = sub["return_movement_time_mean"].values

        base_color = base_colors[i % len(base_colors)]

        # lighter shade for return
        lighter = mcolors.to_rgba(base_color, alpha=0.35)

        plt.bar(
            x,
            move,
            width=width,
            color=base_color,
            # label=f"{l} move",
        )

        plt.bar(
            x,
            ret,
            width=width,
            bottom=move,
            color=lighter,
            # label=f"{l} return",
        )
        patch = mpatches.Patch(color=base_color, label=line_name)
        handles.append(patch)

    plt.xticks([r + width for r in range(len(x_vals))], x_vals)
    plt.ylim(bottom=0, top=1100)
    plt.xlabel(SWEEP_COL)
    plt.ylabel("movement time")
    plt.title("Movement Time Breakdown")
    move_patch = mpatches.Patch(color="black", label="Move")
    return_patch = mpatches.Patch(color="grey", label="Return")
    handles.append(move_patch)
    handles.append(return_patch)
    plt.legend(title="microarchitecture/move type", handles=handles)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_movement.pdf"))
    plt.close()


def plot_nAOD_placement_lines(
    df,
    output_dir,
    sweep_col="n_qubits",
    setting_idx=3,
    skip_placements=True,
    tag="",
):
    """
    df: full dataframe
    output_dir: folder to save plots
    sweep_col: x-axis
    """
    os.makedirs(output_dir, exist_ok=True)
    setting = SETTINGS[setting_idx]
    df_col = df.loc[_setting_filter(df, setting)].copy()

    # Normalize n_aods to int
    df_col["n_aods"] = df_col["n_aods"].astype(int)

    # Aggregate mean/std
    grouped = (
        df_col.groupby(["placement", "n_aods", sweep_col])[["total_time"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]

    placements = sorted(grouped["placement"].unique())
    min_aods = grouped["n_aods"].min()
    max_aods = grouped["n_aods"].max()

    # Create a color map for each placement
    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i) for i in range(10)]  # up to 10 placements

    plt.figure(figsize=(7, 5))

    for p_idx, placement in enumerate(placements):
        if skip_placements and placement not in ["col_based", "checkerboard"]:
            continue
        sub = grouped[grouped["placement"] == placement]

        aods_values = sorted(sub["n_aods"].unique())
        # Map n_aods to light -> dark shades
        cmap = plt.get_cmap("Greys")  # light -> dark

        for aod in aods_values:
            sub_aod = sub[sub["n_aods"] == aod].sort_values(sweep_col)
            plt.errorbar(
                sub_aod[sweep_col],
                sub_aod["total_time_mean"],
                yerr=sub_aod["total_time_std"],
                marker="o",
                capsize=3,
                color=base_colors[p_idx % len(base_colors)],
                alpha=0.3 + 0.7 * (aod - min_aods) / (max_aods - min_aods + 1e-5),
                label=f"{placement}, #aod={aod}",
            )

    plt.xlabel(sweep_col)
    plt.ylabel("Execution Time")
    plt.ylim(bottom=20, top=200)
    plt.title("Total time for placement × #aod")
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(fontsize=_FIG_FONT_SIZE, ncol=2)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    plt.savefig(
        os.path.join(
            output_dir,
            f"total_time_placement_nAOD_setting_{setting_idx}_skip_placements-{skip_placements}{suffix}.pdf",
        )
    )
    plt.close()


# ------------------------------------------------------------
# Fig 1 : placement comparison (average over settings)
# ------------------------------------------------------------
def plot_microarch_comp_average_all(df, output_dir):
    raise NotImplementedError("Average across all elemtns may not work.")
    os.makedirs(output_dir, exist_ok=True)
    g = (
        df.groupby(["n_aods", "placement", "n_qubits"])[RESULT_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )

    g.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in g.columns
    ]

    aods = sorted(g["n_aods"].unique())
    for a in [aods[0], aods[-1]]:  # only plot for lowest and highest AODs
        _plot_total_time(g[g["n_aods"] == a], "placement", f"fig1_nAOD{a}", output_dir)
        _plot_movement_bar(
            g[g["n_aods"] == a], "placement", f"fig1_nAOD{a}", output_dir
        )


# ------------------------------------------------------------
# Fig 2 : controlled placement comparison
# ------------------------------------------------------------
def plot_microarch_comp_setting(df, output_dir, setting_idx=1, tag=""):
    os.makedirs(output_dir, exist_ok=True)
    setting = SETTINGS[setting_idx]
    df_ctrl = df.loc[_setting_filter(df, setting, match_placement=False)]

    if df_ctrl.empty:
        print(f"No data for microarch setting_idx={setting_idx} (tag={tag}), skipping.")
        return

    g = _aggregate_microarch_trials(df_ctrl)
    aods = sorted(g["n_aods"].unique())
    if len(aods) == 0:
        print(
            f"No AOD values for microarch setting_idx={setting_idx} (tag={tag}), skipping."
        )
        return
    suffix = f"_{tag}" if tag else ""
    for a in [aods[0], aods[-1]]:  # only plot for lowest and highest AODs
        _plot_total_time(
            g[g["n_aods"] == a],
            "placement",
            f"fig2_nAOD{a}_setting_{setting_idx}{suffix}",
            output_dir,
        )
        _plot_movement_bar(
            g[g["n_aods"] == a],
            "placement",
            f"fig2_nAOD{a}_setting_{setting_idx}{suffix}",
            output_dir,
        )


# ------------------------------------------------------------
# Ablation study (col-based placement)


def plot_ablation_combined(dfs_dict, output_dir, placement, round_angle_lookup=None):
    """Create ablation figures with rounds as columns and settings as different lines per panel.

    Generates separate figures for each code distance if code_distance column exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    combined_df = pd.concat(list(dfs_dict.values()), ignore_index=True)
    if combined_df.empty:
        print(f"No data for ablation ({placement}), skipping.")
        return

    df_col = combined_df[combined_df["placement"] == placement].copy()
    if df_col.empty:
        print(f"No data for ablation ({placement}), skipping.")
        return

    ablation_labels = STAR_SETTING_LABELS

    # Slightly larger typography for ablation figures only.
    ablation_font_size = _FIG_FONT_SIZE + 2

    # Aggregate per round (use dfs_dict keys to determine available rounds)
    round_order = ["full_trotter", MERGED_ROUND_NAME, f"round_{INDIVIDUAL_ROUND}"]
    round_names = [r for r in round_order if r in dfs_dict]
    if not round_names:
        round_names = list(dfs_dict.keys())

    # Check if code_distance column exists and get available distances
    if "code_distance" in df_col.columns:
        df_col["code_distance"] = pd.to_numeric(
            df_col["code_distance"], errors="coerce"
        )
        distance_values = sorted(df_col["code_distance"].dropna().astype(int).unique())
    else:
        distance_values = [None]

    agg_data_by_distance = {}
    for distance in distance_values:
        # Filter by distance if present
        if distance is None:
            df_distance = df_col.copy()
        else:
            df_distance = df_col[df_col["code_distance"] == distance].copy()

        agg_data = {}
        for label_idx, (label, setting) in enumerate(zip(ablation_labels, SETTINGS)):
            subdf = df_distance.loc[_setting_filter(df_distance, setting)]
            if subdf.empty:
                continue

            round_data = {}
            for round_name in round_names:
                # Get the appropriate dataframe for this round from dfs_dict
                if round_name in dfs_dict:
                    df_round = dfs_dict[round_name]
                    # Filter by placement, distance, and setting
                    sub_round = df_round[df_round["placement"] == placement].copy()
                    if distance is not None:
                        sub_round = sub_round[
                            sub_round["code_distance"] == distance
                        ].copy()
                    sub_round = sub_round.loc[_setting_filter(sub_round, setting)]
                else:
                    continue

                if sub_round.empty:
                    continue

                g = (
                    sub_round.groupby(["n_aods", "n_qubits"])[RESULT_COLS]
                    .agg(["mean", "std", "min", "max"])
                    .reset_index()
                )
                g.columns = [
                    "_".join(c).strip("_") if isinstance(c, tuple) else c
                    for c in g.columns
                ]
                if not g.empty:
                    round_data[round_name] = g

            if round_data:
                agg_data[label_idx] = (label, round_data)

        if agg_data:
            agg_data_by_distance[distance] = agg_data

    if not agg_data_by_distance:
        print(f"No data for ablation ({placement}), skipping.")
        return

    # Generate figures for each distance
    for distance in distance_values:
        agg_data = agg_data_by_distance.get(distance, {})
        if not agg_data:
            print(f"No data for ablation ({placement}, d={distance}), skipping.")
            continue

        all_aods = sorted(
            {
                int(aod)
                for label_idx, (label, round_data) in agg_data.items()
                for g in round_data.values()
                for aod in g["n_aods"].dropna().unique()
            }
        )
        aods_to_plot = [aod for aod in [1, 5] if aod in all_aods]
        if not aods_to_plot:
            aods_to_plot = all_aods[:2]
        if not aods_to_plot:
            print(f"No AOD values for ablation ({placement}, d={distance}), skipping.")
            continue

        setting_indices = sorted(agg_data.keys())
        cmap = plt.get_cmap("tab10")
        base_colors = [cmap(i) for i in range(10)]
        setting_color_map = {
            idx: base_colors[i % len(base_colors)]
            for i, idx in enumerate(setting_indices)
        }

        def _build_shared_ylim_ablation(y_vals, force_zero_bottom=False):
            finite_vals = [float(v) for v in y_vals if pd.notna(v) and np.isfinite(v)]
            if not finite_vals:
                return (0.0, 1.0)
            y_min, y_max = min(finite_vals), max(finite_vals)
            if force_zero_bottom:
                y_min = min(y_min, 0.0)
            if y_max <= y_min:
                pad = max(abs(y_max), 1.0) * 0.1
                return (y_min - pad, y_max + pad)
            pad = (y_max - y_min) * 0.05
            if force_zero_bottom:
                return (y_min, y_max + pad)
            return (y_min - pad, y_max + pad)

        execution_ylim_by_round = {}
        movement_ylim_by_round = {}
        for round_name in round_names:
            y_vals_exec = []
            y_vals_move = []
            for label_idx, (label, round_data) in agg_data.items():
                if round_name not in round_data:
                    continue
                g_round = round_data[round_name]
                for aod_val in aods_to_plot:
                    g_aod = g_round[g_round["n_aods"] == aod_val]
                    if not g_aod.empty:
                        total_min = pd.to_numeric(
                            g_aod["total_time_min"], errors="coerce"
                        )
                        total_max = pd.to_numeric(
                            g_aod["total_time_max"], errors="coerce"
                        )
                        y_vals_exec.extend(total_min.tolist())
                        y_vals_exec.extend(total_max.tolist())
                        move_min = pd.to_numeric(
                            g_aod["movement_time_min"], errors="coerce"
                        )
                        move_max = pd.to_numeric(
                            g_aod["movement_time_max"], errors="coerce"
                        )
                        ret_min = pd.to_numeric(
                            g_aod["return_movement_time_min"], errors="coerce"
                        )
                        ret_max = pd.to_numeric(
                            g_aod["return_movement_time_max"], errors="coerce"
                        )
                        stacked_min = move_min + ret_min
                        stacked_max = move_max + ret_max
                        y_vals_move.extend(stacked_min.tolist())
                        y_vals_move.extend(stacked_max.tolist())
            execution_ylim_by_round[round_name] = _build_shared_ylim_ablation(
                y_vals_exec, force_zero_bottom=False
            )
            movement_ylim_by_round[round_name] = _build_shared_ylim_ablation(
                y_vals_move, force_zero_bottom=True
            )

        def _draw_execution_panel(ax, round_name, aod_value, show_title):
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)
            x_vals = []
            for setting_idx in setting_indices:
                label, round_data = agg_data[setting_idx]
                if round_name not in round_data:
                    continue
                g_round = round_data[round_name]
                g_aod = g_round[g_round["n_aods"] == aod_value]
                if g_aod.empty:
                    continue
                x_vals = sorted(set(x_vals).union(g_aod[SWEEP_COL].dropna().unique()))
                mean_vals = g_aod["total_time_mean"]
                min_vals = g_aod["total_time_min"]
                max_vals = g_aod["total_time_max"]
                err_lower = mean_vals - min_vals
                err_upper = max_vals - mean_vals
                print(
                    f"    [Ablation execution] distance={distance}, round={_format_round_title(round_name)}, "
                    f"AOD={int(aod_value)}, setting={label}: {mean_vals.tolist()}"
                )
                ax.errorbar(
                    x_vals,
                    mean_vals,
                    yerr=[err_lower, err_upper],
                    marker="o",
                    capsize=4,
                    linewidth=1.8,
                    color=setting_color_map[setting_idx],
                    label=label,
                )
            ax.set_xticks(x_vals)
            ax.set_xticklabels(
                _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
                rotation=45,
                ha="right",
            )
            ax.tick_params(labelsize=ablation_font_size)
            # Don't set ylim for ablation execution time - let it auto-scale
            if show_title:
                ax.set_title(
                    _format_round_title(round_name),
                    fontsize=ablation_font_size,
                    pad=10,
                )

        def _draw_movement_panel(ax, round_name, aod_value, show_title):
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)
            x_vals_all = []
            for label_idx in setting_indices:
                label, round_data = agg_data[label_idx]
                if round_name not in round_data:
                    continue
                g_round = round_data[round_name]
                g_aod = g_round[g_round["n_aods"] == aod_value]
                if not g_aod.empty:
                    x_vals_all.extend(g_aod[SWEEP_COL].dropna().unique())

            if not x_vals_all:
                ax.set_xticks([])
                ax.set_ylim(*movement_ylim_by_round[round_name])
                return

            x_vals = sorted(set(x_vals_all))
            x_index = {x: idx for idx, x in enumerate(x_vals)}
            width = 0.8 / max(1, len(setting_indices))

            for i, setting_idx in enumerate(setting_indices):
                label, round_data = agg_data[setting_idx]
                if round_name not in round_data:
                    continue
                g_round = round_data[round_name]
                g_aod = g_round[g_round["n_aods"] == aod_value]
                if g_aod.empty:
                    continue

                positions = [
                    x_index[x] + i * width
                    for x in g_aod[SWEEP_COL].tolist()
                    if x in x_index
                ]
                move = g_aod.loc[
                    g_aod[SWEEP_COL].isin(x_index), "movement_time_mean"
                ].values
                ret = g_aod.loc[
                    g_aod[SWEEP_COL].isin(x_index), "return_movement_time_mean"
                ].values

                base_color = setting_color_map[setting_idx]
                lighter = mcolors.to_rgba(base_color, alpha=0.35)

                ax.bar(positions, move, width=width, color=base_color, label=label)
                ax.bar(positions, ret, width=width, bottom=move, color=lighter)

            ax.set_xticks(
                [
                    idx + width * (len(setting_indices) - 1) / 2
                    for idx in range(len(x_vals))
                ]
            )
            ax.set_xticklabels(
                _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
                rotation=45,
                ha="right",
            )
            ax.tick_params(labelsize=ablation_font_size)
            ax.set_ylim(*movement_ylim_by_round[round_name])
            if show_title:
                ax.set_title(
                    _format_round_title(round_name),
                    fontsize=ablation_font_size,
                    pad=10,
                )

        for metric_name, draw_panel, file_suffix in [
            ("Execution Time", _draw_execution_panel, "execution"),
            ("Movement Time", _draw_movement_panel, "movement"),
        ]:
            # Use taller subplots for execution time to prevent line crowding
            height_multiplier = 6.8 if metric_name == "Execution Time" else 5.6
            fig, axes = plt.subplots(
                len(aods_to_plot),
                len(round_names),
                figsize=(6.9 * len(round_names), height_multiplier * len(aods_to_plot)),
                squeeze=False,
            )

            for row_idx, aod_value in enumerate(aods_to_plot):
                for col_idx, round_name in enumerate(round_names):
                    ax = axes[row_idx, col_idx]
                    draw_panel(ax, round_name, aod_value, show_title=(row_idx == 0))
                    if col_idx == 0:
                        ax.set_ylabel(
                            f"AOD = {aod_value}\n{metric_name}",
                            fontsize=ablation_font_size,
                        )
                    else:
                        ax.set_ylabel("")
                    if row_idx == len(aods_to_plot) - 1:
                        ax.set_xlabel("Number of Qubits", fontsize=ablation_font_size)
                    else:
                        ax.set_xlabel("")

            fig.suptitle(
                f"Ablation Study, {metric_name}, distance={distance}",
                fontsize=ablation_font_size,
                y=0.995,
            )
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=setting_color_map[idx],
                    linewidth=2,
                    label=agg_data[idx][0],
                )
                for idx in setting_indices
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.035),
                ncol=4,
                fontsize=ablation_font_size,
                frameon=True,
            )
            # fig.tight_layout(rect=(0.03, 0.15, 0.98, 0.99))
            # fig.subplots_adjust(top=0.92, bottom=0.15, hspace=0.15, wspace=0.25)

            fig.tight_layout(rect=(0, 0.29, 1, 1))
            fig.subplots_adjust(bottom=0.18, hspace=0.32, wspace=0.32)

            distance_suffix = f"_distance_{distance}" if distance is not None else ""
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"ablation_{placement}{distance_suffix}_{file_suffix}.pdf",
                ),
                bbox_inches="tight",
                pad_inches=0.10,
            )
            plt.close(fig)


def aggregate_full_trotter(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5 consecutive Rz rounds (0-4) into one full-trotter-step row."""
    if "round" not in df.columns:
        raise ValueError(
            "Input dataframe must contain 'round' column for full trotter."
        )

    config_cols = [
        "n_qubits",
        "qubit_cols",
        "qubit_rows",
        "code_distance",
        "placement",
        "n_aods",
        "consider_skip_rus",
        "tmr_assignment_method",
        "prepare_lookahead_angles",
        "trivial_return",
        "decompose_move",
        "parallel_execution",
    ]
    # T-cultivation profiling rows include trial + factory/fidelity settings; must stay in
    # the groupby so full-trotter aggregation does not merge distinct settings.
    for optional in (
        "trial",
        "fidelity_target",
        "factory_physical_size",
        "redistribute_stage1_success",
    ):
        if optional in df.columns and optional not in config_cols:
            config_cols.append(optional)

    work = df.copy().reset_index(drop=False).rename(columns={"index": "_row_order"})

    # each new full trotter step starts when round == 0 within the same configuration
    work["trotter_id"] = work.groupby(config_cols)["round"].transform(
        lambda s: (s == 0).cumsum() - 1
    )

    value_cols_sum = [
        "total_time",
        "movement_time",
        "return_movement_time",
        "TMR_round",
        "RUS_round",
    ]
    value_cols_avg = ["max_rus_per_qubit", "avg_rus_per_qubit"]

    agg_spec = {col: "sum" for col in value_cols_sum if col in work.columns}
    agg_spec.update({col: "mean" for col in value_cols_avg if col in work.columns})

    full = (
        work.groupby(config_cols + ["trotter_id"], as_index=False)
        .agg(agg_spec)
        .sort_values(config_cols + ["trotter_id"])
    )
    full["round"] = "full"
    return full


def plot_microarch_comp_setting_combined(
    dfs_dict,
    output_dir,
    setting_idx=1,
    round_angle_lookup=None,
    setting_override=None,
    setting_label=None,
):
    """Create separate execution-time and movement-time microarch figures."""
    os.makedirs(output_dir, exist_ok=True)
    if setting_override is None:
        setting = SETTINGS[setting_idx]
    else:
        setting = setting_override

    agg_data = {}
    for round_name, df in dfs_dict.items():
        df_ctrl = df.loc[_setting_filter(df, setting, match_placement=False)]
        if df_ctrl.empty:
            continue
        agg_data[round_name] = _aggregate_microarch_trials(df_ctrl)

    if not agg_data:
        print(
            f"No data for microarch setting_idx={setting_idx}, skipping combined plot."
        )
        return

    preferred_round_order = [
        "full_trotter",
        MERGED_ROUND_NAME,
        f"round_{INDIVIDUAL_ROUND}",
    ]
    round_names = [
        round_name for round_name in preferred_round_order if round_name in agg_data
    ]
    if not round_names:
        round_names = list(agg_data.keys())

    all_aods = sorted(
        {int(aod) for g in agg_data.values() for aod in g["n_aods"].dropna().unique()}
    )
    aods_to_plot = [aod for aod in [1, 5] if aod in all_aods]
    if not aods_to_plot:
        aods_to_plot = all_aods[:2]
    if not aods_to_plot:
        print(
            f"No AOD values for microarch setting_idx={setting_idx}, skipping combined plot."
        )
        return

    placements = sorted(
        {
            str(placement_name)
            for grouped in agg_data.values()
            for placement_name in grouped["placement"].dropna().unique()
        }
    )
    if not placements:
        print(
            f"No placement values for microarch setting_idx={setting_idx}, skipping combined plot."
        )
        return

    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i) for i in range(10)]
    placement_color_map = {
        placement_name: base_colors[i % len(base_colors)]
        for i, placement_name in enumerate(placements)
    }

    setting_title = _format_setting_title(setting_label, setting_idx)
    file_setting_tag = (
        str(setting_idx)
        if setting_override is None
        else (setting_label or "auto").replace(" ", "_").replace(",", "")
    )

    def _build_shared_ylim(y_mins, y_maxs, force_zero_bottom=False):
        finite_mins = [float(v) for v in y_mins if pd.notna(v) and np.isfinite(v)]
        finite_maxs = [float(v) for v in y_maxs if pd.notna(v) and np.isfinite(v)]
        if not finite_mins or not finite_maxs:
            return (0.0, 1.0)

        y_min = min(finite_mins)
        y_max = max(finite_maxs)
        if force_zero_bottom:
            y_min = min(y_min, 0.0)

        if y_max <= y_min:
            pad = max(abs(y_max), 1.0) * 0.1
            return (y_min - pad, y_max + pad)

        pad = (y_max - y_min) * 0.05
        if force_zero_bottom:
            return (y_min, y_max + pad)
        return (y_min - pad, y_max + pad)

    def _theoretical_bound_values(round_name, x_vals):
        if not x_vals:
            return []
        bounds = _get_theoretical_lower_bound(x_vals)
        if round_name == "full_trotter":
            return bounds["full_trotter"]
        if round_name == MERGED_ROUND_NAME:
            return bounds["zz_layer"]
        if round_name == f"round_{INDIVIDUAL_ROUND}":
            return bounds["x_layer"]
        return []

    execution_ylim_by_round = {}
    movement_ylim_by_round = {}
    for round_name in round_names:
        y_mins_exec = []
        y_maxs_exec = []
        y_mins_move = []
        y_maxs_move = []

        g_round = agg_data[round_name]
        for aod_value in aods_to_plot:
            g_aod = g_round[g_round["n_aods"] == aod_value]
            if g_aod.empty:
                continue

            total_min = pd.to_numeric(
                g_aod.get("total_time_min", g_aod["total_time_mean"]),
                errors="coerce",
            )
            total_max = pd.to_numeric(
                g_aod.get("total_time_max", g_aod["total_time_mean"]),
                errors="coerce",
            )
            y_mins_exec.extend(total_min.tolist())
            y_maxs_exec.extend(total_max.tolist())

            x_vals = sorted(g_aod[SWEEP_COL].dropna().unique())
            theo_vals = _theoretical_bound_values(round_name, x_vals)
            y_mins_exec.extend(theo_vals)
            y_maxs_exec.extend(theo_vals)

            move_mean = pd.to_numeric(g_aod["movement_time_mean"], errors="coerce")
            ret_mean = pd.to_numeric(
                g_aod["return_movement_time_mean"], errors="coerce"
            )
            stacked_total = move_mean + ret_mean
            y_mins_move.extend(stacked_total.tolist())
            y_maxs_move.extend(stacked_total.tolist())

        execution_ylim_by_round[round_name] = _build_shared_ylim(
            y_mins_exec,
            y_maxs_exec,
            force_zero_bottom=False,
        )
        movement_ylim_by_round[round_name] = _build_shared_ylim(
            y_mins_move,
            y_maxs_move,
            force_zero_bottom=True,
        )

    def _draw_execution_panel(ax, round_name, g_aod, show_title):
        x_vals = sorted(g_aod[SWEEP_COL].dropna().unique())
        for placement_name in placements:
            sub = g_aod[g_aod["placement"] == placement_name].sort_values(SWEEP_COL)
            if sub.empty:
                continue
            mean_vals = sub["total_time_mean"]
            min_vals = sub["total_time_min"]
            max_vals = sub["total_time_max"]
            err_lower = mean_vals - min_vals
            err_upper = max_vals - mean_vals
            ax.errorbar(
                sub[SWEEP_COL],
                mean_vals,
                yerr=[err_lower, err_upper],
                marker="o",
                capsize=4,
                linewidth=1.8,
                color=placement_color_map[placement_name],
            )

        if x_vals:
            bounds = _get_theoretical_lower_bound(x_vals)
            if round_name == "full_trotter":
                ax.plot(x_vals, bounds["full_trotter"], "k--", linewidth=1.5)
            elif round_name == MERGED_ROUND_NAME:
                ax.plot(x_vals, bounds["zz_layer"], "k--", linewidth=1.5)
            elif round_name == f"round_{INDIVIDUAL_ROUND}":
                ax.plot(x_vals, bounds["x_layer"], "k--", linewidth=1.5)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(
            _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
            rotation=45,
            ha="right",
        )
        if show_title:
            ax.set_title(
                _format_round_title(round_name), fontsize=_FIG_FONT_SIZE, pad=10
            )

    def _draw_movement_panel(ax, round_name, g_aod, show_title):
        x_vals = sorted(g_aod[SWEEP_COL].dropna().unique())
        x_index = {x: idx for idx, x in enumerate(x_vals)}
        width = 0.8 / max(1, len(placements))

        for i, placement_name in enumerate(placements):
            sub = g_aod[g_aod["placement"] == placement_name].sort_values(SWEEP_COL)
            if sub.empty:
                continue
            positions = [
                x_index[x] + i * width for x in sub[SWEEP_COL].tolist() if x in x_index
            ]
            move = sub.loc[sub[SWEEP_COL].isin(x_index), "movement_time_mean"].values
            ret = sub.loc[
                sub[SWEEP_COL].isin(x_index), "return_movement_time_mean"
            ].values
            base_color = placement_color_map[placement_name]
            lighter = mcolors.to_rgba(base_color, alpha=0.35)
            ax.bar(positions, move, width=width, color=base_color)
            ax.bar(positions, ret, width=width, bottom=move, color=lighter)

        ax.set_xticks(
            [idx + width * (len(placements) - 1) / 2 for idx in range(len(x_vals))]
        )
        ax.set_xticklabels(
            _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
            rotation=45,
            ha="right",
        )
        if show_title:
            ax.set_title(
                _format_round_title(round_name), fontsize=_FIG_FONT_SIZE, pad=10
            )

    for metric_name, draw_panel, file_suffix, legend_handles in [
        (
            "Execution Time",
            _draw_execution_panel,
            "execution",
            [
                Line2D(
                    [0],
                    [0],
                    color=placement_color_map[p],
                    marker="o",
                    linestyle="-",
                    label=_format_microarch_placement_label(p),
                )
                for p in placements
            ]
            + [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="--",
                    label="Expected time",
                )
            ],
        ),
        (
            "Movement Time",
            _draw_movement_panel,
            "movement",
            [
                mpatches.Patch(
                    color=placement_color_map[p],
                    label=_format_microarch_placement_label(p),
                )
                for p in placements
            ]
            + [
                mpatches.Patch(color="black", label="Move"),
                mpatches.Patch(color="grey", label="Return"),
            ],
        ),
    ]:
        ylim_by_round = (
            execution_ylim_by_round
            if metric_name == "Execution Time"
            else movement_ylim_by_round
        )

        fig, axes = plt.subplots(
            len(aods_to_plot),
            len(round_names),
            figsize=(5.8 * len(round_names), 4.4 * len(aods_to_plot)),
            squeeze=False,
        )

        for row_idx, aod_value in enumerate(aods_to_plot):
            for col_idx, round_name in enumerate(round_names):
                g = agg_data[round_name]
                g_aod = g[g["n_aods"] == aod_value]
                ax = axes[row_idx, col_idx]
                if g_aod.empty:
                    ax.axis("off")
                    continue
                ax.set_axisbelow(True)
                ax.grid(True, alpha=0.3)
                draw_panel(ax, round_name, g_aod, show_title=(row_idx == 0))
                ax.set_ylim(*ylim_by_round[round_name])
                ax.tick_params(axis="both", which="major", pad=1)
                if col_idx == 0:
                    ax.set_ylabel(f"AOD = {aod_value}\n{metric_name}")
                else:
                    ax.set_ylabel("", labelpad=0)
                if row_idx == len(aods_to_plot) - 1:
                    ax.set_xlabel("Number of Qubits (angles)", labelpad=0)
                else:
                    ax.set_xlabel("", labelpad=0)

        fig.suptitle(
            f"Microarchitecture evaluation - {setting_title} - {metric_name}",
            fontsize=_FIG_FONT_SIZE,
            y=0.995,
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=(6 if metric_name == "Movement Time" else min(4, len(legend_handles))),
            fontsize=_FIG_FONT_SIZE,
            frameon=True,
        )
        fig.tight_layout(
            rect=(0.04, 0.20, 0.98, 0.90), pad=0.10, w_pad=0.03, h_pad=0.03
        )
        fig.subplots_adjust(
            top=0.90,
            bottom=0.22,
            hspace=0.40,
            wspace=0.25,
            left=0.04,
            right=0.98,
        )
        fig.savefig(
            os.path.join(
                output_dir, f"fig2_setting_{file_setting_tag}_{file_suffix}.pdf"
            ),
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)


def plot_nAOD_placement_lines_combined(
    dfs_dict,
    output_dir,
    setting_idx=4,
    round_angle_lookup=None,
    setting_override=None,
    setting_label=None,
    placement="col_based",
):
    """Create combined AOD study plots with rows for each round and one figure per code distance.

    Args:
        placement: Which placement to plot. Can be "col_based", "checkerboard", or None for all.
                   Defaults to "col_based".
    """
    os.makedirs(output_dir, exist_ok=True)

    agg_data = {}
    if setting_override is None:
        setting = SETTINGS[setting_idx]
    else:
        setting = setting_override

    for round_name, df in dfs_dict.items():
        df_col = df[_setting_filter(df, setting)].copy()

        if df_col.empty:
            continue

        df_col["n_aods"] = df_col["n_aods"].astype(int)
        if "code_distance" in df_col.columns:
            df_col["code_distance"] = pd.to_numeric(
                df_col["code_distance"], errors="coerce"
            )

        group_cols = ["placement", "n_aods", "n_qubits"]
        if "code_distance" in df_col.columns:
            group_cols.insert(0, "code_distance")

        grouped = (
            df_col.groupby(group_cols)[["total_time"]]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        grouped.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in grouped.columns
        ]

        if not grouped.empty:
            agg_data[round_name] = grouped

    if not agg_data:
        print(
            f"No data for AOD study setting_idx={setting_idx}, skipping combined plot."
        )
        return

    # Get plotted AODs and distances
    first_grouped = list(agg_data.values())[0]
    if "code_distance" in first_grouped.columns:
        distance_values = sorted(
            first_grouped["code_distance"].dropna().astype(int).unique()
        )
    else:
        distance_values = [None]
    all_aods = sorted(
        {
            int(aod)
            for grouped in agg_data.values()
            for aod in grouped["n_aods"].dropna().astype(int).unique()
        }
    )
    aod_color_map = _build_aod_color_map(all_aods)
    execution_mode = _format_setting_title(setting_label, setting_idx)

    figure_title = f"AOD number comparison, {execution_mode}"
    file_setting_tag = (
        str(setting_idx)
        if setting_override is None
        else (setting_label or "auto").replace(" ", "_").replace(",", "")
    )

    round_items = list(agg_data.items())

    def _draw_aod_round(ax, round_name, grouped):
        placements_to_plot = [placement] if placement else ["col_based", "checkerboard"]
        for p in placements_to_plot:
            if p not in grouped["placement"].unique():
                continue
            sub = grouped[grouped["placement"] == p]

            aods_values = sorted(sub["n_aods"].unique())
            for aod in aods_values:
                sub_aod = sub[sub["n_aods"] == aod].sort_values("n_qubits")
                mean_vals = sub_aod["total_time_mean"]
                min_vals = sub_aod["total_time_min"]
                max_vals = sub_aod["total_time_max"]
                err_lower = mean_vals - min_vals
                err_upper = max_vals - mean_vals
                ax.errorbar(
                    sub_aod["n_qubits"],
                    mean_vals,
                    yerr=[err_lower, err_upper],
                    marker="o",
                    capsize=3,
                    color=aod_color_map.get(int(aod)),
                    alpha=1.0,
                    label=f"#AOD={aod}",
                )

        x_vals = sorted(grouped["n_qubits"].unique())
        bounds = _get_theoretical_lower_bound(x_vals)
        if round_name == "full_trotter":
            ideal_rounds = bounds["full_trotter"]
            ax.plot(
                x_vals,
                ideal_rounds,
                "k--",
                linewidth=1.5,
                label="Expected time",
            )
        elif round_name == MERGED_ROUND_NAME:
            zz_bounds = bounds["zz_layer"]
            ax.plot(
                x_vals,
                zz_bounds,
                "k--",
                linewidth=1.5,
                label="Expected time",
            )
        elif round_name == f"round_{INDIVIDUAL_ROUND}":
            x_bounds = bounds["x_layer"]
            ax.plot(
                x_vals,
                x_bounds,
                "k--",
                linewidth=1.5,
                label="Expected time",
            )
        ax.set_xticks(x_vals)
        ax.set_xticklabels(
            _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
            rotation=45,
            ha="right",
        )
        ax.set_xlabel("Number of Qubits (angles)")
        ax.set_ylabel("Execution Time")
        # Apply y-limit only for single trotter rounds, not for full_trotter
        if round_name != "full_trotter":
            ax.set_ylim(0, 400)
        ax.set_title(f"{_format_round_title(round_name)}")
        ax.legend(fontsize=_FIG_FONT_SIZE, ncol=2)

    for distance in distance_values:
        if distance is None:
            distance_round_items = round_items
            distance_suffix = ""
            distance_title = figure_title
        else:
            distance_round_items = []
            for round_name, grouped in round_items:
                grouped_distance = grouped[grouped["code_distance"] == distance].copy()
                if not grouped_distance.empty:
                    distance_round_items.append((round_name, grouped_distance))

            if not distance_round_items:
                continue

            distance_suffix = f"_distance_{distance}"
            distance_title = f"{figure_title}, distance = {distance}"

        fig, axes = plt.subplots(
            len(distance_round_items),
            1,
            figsize=(8.6, 5.4 * len(distance_round_items)),
        )
        if len(distance_round_items) == 1:
            axes = [axes]

        for row_idx, (round_name, grouped) in enumerate(distance_round_items):
            _draw_aod_round(axes[row_idx], round_name, grouped)

        fig.suptitle(distance_title, y=0.98, fontsize=_FIG_FONT_SIZE)
        _apply_shared_bottom_legend(fig, ncol=6, anchor_y=-0.02)
        fig.tight_layout(rect=(0, 0.28, 1, 1))
        fig.subplots_adjust(bottom=0.33, wspace=0.32)
        placement_tag = f"_{placement}" if placement else ""
        fig.savefig(
            os.path.join(
                output_dir,
                f"total_time_placement_nAOD_setting_{file_setting_tag}_skip_placements-True_combined_vertical{placement_tag}{distance_suffix}.pdf",
            )
        )
        plt.close(fig)

        fig, axes = plt.subplots(
            1,
            len(distance_round_items),
            figsize=(5.4 * len(distance_round_items), 5.6),
        )
        if len(distance_round_items) == 1:
            axes = [axes]

        for col_idx, (round_name, grouped) in enumerate(distance_round_items):
            _draw_aod_round(axes[col_idx], round_name, grouped)

        fig.suptitle(distance_title, y=0.98, fontsize=_FIG_FONT_SIZE)
        _apply_shared_bottom_legend(fig, ncol=6, anchor_y=-0.02)
        fig.tight_layout(rect=(0, 0.28, 1, 1))
        fig.subplots_adjust(bottom=0.33, wspace=0.32)
        placement_tag = f"_{placement}" if placement else ""
        fig.savefig(
            os.path.join(
                output_dir,
                f"total_time_placement_nAOD_setting_{file_setting_tag}_skip_placements-True_combined_horizontal{placement_tag}{distance_suffix}.pdf",
            )
        )
        plt.close(fig)


def _print_aod_vs_theoretical_improvement(
    dfs_dict: dict[str, pd.DataFrame],
) -> None:
    """Print improvement (overhead) of each AOD value versus theoretical lower bound."""
    print("\n  [AOD vs Theoretical Lower Bound]")
    eps = 1e-15

    for round_name, df in dfs_dict.items():
        if df.empty:
            continue

        x_vals = sorted(
            pd.to_numeric(df["n_qubits"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if not x_vals:
            continue

        bounds = _get_theoretical_lower_bound(x_vals)
        if round_name == "full_trotter":
            theo_vals = bounds["full_trotter"]
        elif round_name == "merged_rounds_except_4":
            theo_vals = bounds["zz_layer"]
        elif round_name == "round_4":
            theo_vals = bounds["x_layer"]
        else:
            continue

        aod_values = sorted(
            pd.to_numeric(df["n_aods"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        for aod in aod_values:
            sub_aod = df[df["n_aods"] == aod]
            if sub_aod.empty:
                continue

            mean_times = (
                pd.to_numeric(sub_aod["total_time"], errors="coerce").dropna().tolist()
            )
            if not mean_times:
                continue

            geomean_actual = float(np.exp(np.mean(np.log(np.maximum(mean_times, eps)))))
            geomean_theo = float(np.exp(np.mean(np.log(np.maximum(theo_vals, eps)))))
            ratio = geomean_actual / (geomean_theo + eps)
            print(
                f"    {_format_round_title(round_name)} AOD={aod} / theoretical: {ratio:.4f}x"
            )


def process_full_trotter_csv(
    csv_file: str,
    output_dir: str,
    *,
    only_distances: list[int] | None = None,
):
    """Generate full-trotter and per-round analyses for microarchitecture, ablation, and AOD studies.

    Args:
        only_distances: If provided, restrict processing to these ``code_distance``
            values. Useful to incrementally regenerate a single distance's plots
            without redoing already-complete distances.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file, engine="python", on_bad_lines="skip")
    df = _coerce_result_cols_numeric(df)
    df = _normalize_config_types(df)

    if "round" not in df.columns:
        raise ValueError("CSV must include 'round' column for full-trotter processing")

    if "code_distance" in df.columns:
        available_distances = set(
            pd.to_numeric(df["code_distance"], errors="coerce").dropna().astype(int)
        )
        preferred = [
            d for d in STAR_PROFILING_CODE_DISTANCES if d in available_distances
        ]
        if not preferred:
            distances_to_plot = sorted(available_distances)
        else:
            # Keep evaluation order (7, 9, 13), then any extra distances in the CSV.
            distances_to_plot = preferred + sorted(available_distances - set(preferred))
        if only_distances is not None:
            requested = {int(d) for d in only_distances}
            distances_to_plot = [d for d in distances_to_plot if int(d) in requested]
            if not distances_to_plot:
                print(f"No data for requested distances {sorted(requested)}; skipping.")
                return
    else:
        distances_to_plot = [None]

    for distance in distances_to_plot:
        if distance is None:
            df_distance = df
            distance_output_dir = output_dir
        else:
            df_distance = df[df["code_distance"] == distance].copy()
            if df_distance.empty:
                continue
            distance_output_dir = os.path.join(output_dir, f"distance_{distance}")

        dfs_dict_micro, dfs_dict_ablation, dfs_dict_aod = _build_round_plot_dicts(
            df_distance
        )
        round_angle_lookup = _build_round_angle_lookup(dfs_dict_micro)

        preferred_indices = [9, 12]
        settings_to_plot: list[tuple[str, tuple]] = []
        for idx in preferred_indices:
            setting = MAIN_SETTING
            if not df_distance[_setting_filter(df_distance, setting)].empty:
                settings_to_plot.append((f"setting_{idx}", setting))

        if len(settings_to_plot) == 0:
            auto_settings = _get_available_settings(df_distance, max_settings=2)
            settings_to_plot = [
                (f"auto_{i+1}", setting) for i, setting in enumerate(auto_settings)
            ]

        micro_dir = os.path.join(distance_output_dir, "microarch_setting")
        if not df_distance.loc[
            _setting_filter(df_distance, MICROARCH_COMPARE_SETTING, match_placement=False)
        ].empty:
            plot_microarch_comp_setting_combined(
                dfs_dict_micro,
                micro_dir,
                round_angle_lookup=round_angle_lookup,
                setting_override=MICROARCH_COMPARE_SETTING,
                setting_label="microarch_compare",
            )
        else:
            print(
                "No cross-placement microarch data for MICROARCH_COMPARE_SETTING; "
                "skipping microarch_setting figures."
            )

        ablation_dir = os.path.join(distance_output_dir, "ablation")
        # STAR ablation study: column-based placement only (no checkerboard).
        plot_ablation_combined(
            dfs_dict_ablation,
            os.path.join(ablation_dir, "col_based"),
            "col_based",
            round_angle_lookup=round_angle_lookup,
        )

        aod_dir = os.path.join(distance_output_dir, "aod_study")
        for setting_label, setting in settings_to_plot:
            plot_nAOD_placement_lines_combined(
                dfs_dict_aod,
                aod_dir,
                round_angle_lookup=round_angle_lookup,
                setting_override=setting,
                setting_label=setting_label,
            )

        # Print AOD improvements vs theoretical bound for this distance
        if distance is not None:
            print(f"\n  AOD vs theoretical bound (distance={distance}):")
            for round_name, aod_df in dfs_dict_aod.items():
                _print_aod_vs_theoretical_improvement({round_name: aod_df})
        else:
            print("\n  AOD vs theoretical bound (all distances):")
            _print_aod_vs_theoretical_improvement(dfs_dict_aod)

    print("Saved full-trotter plots to", output_dir)


def _coerce_result_cols_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = RESULT_COLS + [
        "TMR_round",
        "RUS_round",
        "max_rus_per_qubit",
        "avg_rus_per_qubit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_config_types(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [
        "trivial_return",
        "decompose_move",
        "parallel_execution",
        "redistribute_stage1_success",
        "prepare_lookahead_angles",
    ]
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "false": False})
                )
            else:
                df[col] = df[col].astype(bool)

    int_cols = [
        "n_qubits",
        "qubit_cols",
        "qubit_rows",
        "n_aods",
        "consider_skip_rus",
        "round",
        "code_distance",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _summarize_star_full_trotter_execution(star_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize STAR execution time per full trotter step."""
    star_df = star_df.copy()
    star_df = _coerce_result_cols_numeric(star_df)
    star_df = _normalize_config_types(star_df)
    full = aggregate_full_trotter(star_df)
    summary = (
        full.groupby(["n_qubits", "code_distance"], as_index=False)
        .agg(
            execution_time_mean=("total_time", "mean"),
            execution_time_std=("total_time", "std"),
            n_samples=("total_time", "count"),
        )
        .assign(method="star")
    )
    summary["execution_time_std"] = summary["execution_time_std"].fillna(0.0)
    return summary


def _summarize_t_cultivation_execution(t_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize T-cultivation execution time using available timing columns."""
    t_df = t_df.copy()
    t_df = _coerce_result_cols_numeric(t_df)
    t_df = _normalize_config_types(t_df)

    time_col = (
        "step_execution_time" if "step_execution_time" in t_df.columns else "total_time"
    )
    group_cols = ["n_qubits", "code_distance"]
    for optional_col in ["n_aods", "factory_physical_size", "fidelity_target"]:
        if optional_col in t_df.columns:
            group_cols.append(optional_col)

    summary = (
        t_df.groupby(group_cols, as_index=False)
        .agg(
            execution_time_mean=(time_col, "mean"),
            execution_time_std=(time_col, "std"),
            n_samples=(time_col, "count"),
        )
        .assign(method="t_cultivation")
    )
    summary["execution_time_std"] = summary["execution_time_std"].fillna(0.0)
    return summary


def _build_layer_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build dataframes for full trotter, ZZ layers, and X layer."""
    work = df.copy()
    default_cols = {
        "consider_skip_rus": 0,
        "tmr_assignment_method": "unknown",
        "prepare_lookahead_angles": False,
        "trivial_return": False,
        "decompose_move": False,
        "parallel_execution": False,
    }
    for col, default_val in default_cols.items():
        if col not in work.columns:
            work[col] = default_val

    frames = {"full_trotter": aggregate_full_trotter(work)}

    round_values = pd.to_numeric(work["round"], errors="coerce")
    zz_df = work[round_values != INDIVIDUAL_ROUND].copy()
    x_df = work[round_values == INDIVIDUAL_ROUND].copy()
    if not zz_df.empty:
        frames["zz_layers"] = zz_df
    if not x_df.empty:
        frames["x_layer"] = x_df
    return frames


def _summarize_execution_by_qubits(df: pd.DataFrame, method: str) -> pd.DataFrame:
    group_cols = ["n_qubits", "n_aods"]
    if "code_distance" in df.columns:
        group_cols.insert(1, "code_distance")

    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            execution_time_mean=("total_time", "mean"),
            execution_time_std=("total_time", "std"),
            execution_time_min=("total_time", "min"),
            execution_time_max=("total_time", "max"),
            n_samples=("total_time", "count"),
        )
        .assign(method=method)
    )
    summary["execution_time_std"] = summary["execution_time_std"].fillna(0.0)
    return summary


def _format_t_cultivation_setting_inline(
    code_distance: float | int,
    fidelity_target: float,
    factory_physical_size: float | int,
) -> str:
    """Single-line setting for subplot titles (below layer name)."""
    return f"d={int(code_distance)}"


def _format_t_cultivation_setting_label(
    code_distance: float | int,
    fidelity_target: float,
    factory_physical_size: float | int,
) -> str:
    """Multi-line setting label for row titles in the T-cultivation AOD plot."""
    return f"d={int(code_distance)}"


def _dedupe_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep first occurrence when CSV has duplicated column names (e.g. two ``code_distance``)."""
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


# T-cultivation runtime lines vs STAR (matches fidelity evaluation triples).
# Each tuple is (code_distance, fidelity_target, factory_physical_size).
SHOW_T_CULTIVATION_D13 = False
T_CULTIVATION_RUNTIME_LINE_SETTINGS: list[tuple[int, float, int]] = [
    (9, 1e-8, 2),
    (13, 1e-8, 4),
]


def _t_cultivation_runtime_line_settings() -> list[tuple[int, float, int]]:
    if SHOW_T_CULTIVATION_D13:
        return T_CULTIVATION_RUNTIME_LINE_SETTINGS
    return [
        setting
        for setting in T_CULTIVATION_RUNTIME_LINE_SETTINGS
        if int(setting[0]) != 13
    ]


def _runtime_summary_t_triples() -> list[tuple[int, float, int]]:
    """T-cultivation rows included in the runtime text summary (d=9,13 at LER=1e-8)."""
    out: list[tuple[int, float, int]] = []
    for cd, ft, fps in _t_cultivation_runtime_line_settings():
        if int(cd) not in RUNTIME_SUMMARY_T_CODE_DISTANCES:
            continue
        if not np.isclose(float(ft), float(RUNTIME_SUMMARY_T_LER), rtol=0.0, atol=0.0):
            continue
        out.append((int(cd), float(ft), int(fps)))
    return sorted(out, key=lambda t: int(t[0]))


# Compile ablation grid—keep in sync with ``COMPILE_SETTINGS`` in
# ``evaluation_fidelity_t_cultivation.py``: (trivial_return, decompose_move,
# redistribute_stage1_success).
_T_COMPILE_ABLATION_GRID: list[tuple[bool, bool, bool]] = [
    (True, False, False),
    (False, False, False),
    (False, True, False),
    (False, True, True),
]
_T_COMPILE_ABLATION_LABELS: list[str] = [
    "Vanilla",
    "Opt. return",
    "Opt. Decomp. return",
    "Opt. Decomp. return + S1 redist.",
]


def _filter_t_cultivation_d13_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    if SHOW_T_CULTIVATION_D13 or "code_distance" not in df.columns:
        return df
    out = df.copy()
    cd = pd.to_numeric(out["code_distance"], errors="coerce")
    return out[cd != 13].copy()


def _print_runtime_star_vs_t_all_in_one_improvements(
    star_layers: dict[str, pd.DataFrame],
    t_layers: dict[str, pd.DataFrame],
    layer_order: list[str],
    star_distances: list[int],
    target_aods: list[int],
    effective_t_placement: str,
    setting_4: tuple,
    filename_suffix: str,
) -> None:
    """Log T/STAR execution metrics for the all-in-one STAR vs T figure (matched ``n_qubits``)."""
    eps = 1e-15

    def _geomean_ratio(num: np.ndarray, den: np.ndarray) -> float:
        r = np.clip(num, eps, None) / np.clip(den, eps, None)
        return float(np.exp(np.mean(np.log(np.clip(r, eps, None)))))

    def _star_dist_for_t(t_cd: int) -> int | None:
        ti = int(t_cd)
        if ti in star_distances:
            return ti
        if 9 in star_distances:
            return 9
        if 7 in star_distances:
            return 7
        return int(star_distances[0]) if star_distances else None

    pretty_layer = {
        "full_trotter": "Full Trotter",
        "zz_layers": "ZZ layers",
        "x_layer": "X layer",
    }
    t_line_settings = _t_cultivation_runtime_line_settings()
    if not t_line_settings:
        return

    pdf_tail = (
        f"runtime_star_vs_t_cultivation_all_in_one_aod_"
        f"{'-'.join(str(a) for a in target_aods)}{filename_suffix}.pdf"
    )
    print("\n  [STAR vs T cultivation all-in-one figure — execution, matched n_qubits]")
    print(
        f"      figure: {pdf_tail} · AODs={list(target_aods)}, "
        f"T placement={effective_t_placement}, STAR=main compile setting · "
        "avg_improvement_vs_STAR = mean((STAR-T)/STAR)*100 (positive => T faster)"
    )

    # STAR: same compile/setting, compare code distances (same curve families as the figure).
    sds_sorted = sorted(int(d) for d in star_distances)
    if len(sds_sorted) >= 2:
        print("    [STAR — distance ratio at matched n_qubits, main compile setting]")
        for layer_name in layer_order:
            star_filtered = star_layers[layer_name][
                _setting_filter(star_layers[layer_name], setting_4)
            ].copy()
            layer_title = pretty_layer.get(layer_name, layer_name)
            for aod in target_aods:
                for d_lo, d_hi in zip(sds_sorted, sds_sorted[1:]):
                    star_lo = star_filtered[
                        pd.to_numeric(
                            star_filtered["code_distance"], errors="coerce"
                        ).astype(int)
                        == int(d_lo)
                    ].copy()
                    star_hi = star_filtered[
                        pd.to_numeric(
                            star_filtered["code_distance"], errors="coerce"
                        ).astype(int)
                        == int(d_hi)
                    ].copy()
                    sum_lo = _summarize_execution_by_qubits(star_lo, method="STAR")
                    sum_hi = _summarize_execution_by_qubits(star_hi, method="STAR")
                    row_lo = sum_lo[sum_lo["n_aods"] == int(aod)][
                        ["n_qubits", "execution_time_mean"]
                    ].rename(columns={"execution_time_mean": "rt_lo"})
                    row_hi = sum_hi[sum_hi["n_aods"] == int(aod)][
                        ["n_qubits", "execution_time_mean"]
                    ].rename(columns={"execution_time_mean": "rt_hi"})
                    m_sd = row_hi.merge(row_lo, on="n_qubits", how="inner")
                    if m_sd.empty:
                        continue
                    g_sd = _geomean_ratio(
                        m_sd["rt_hi"].to_numpy(dtype=float),
                        m_sd["rt_lo"].to_numpy(dtype=float),
                    )
                    print(
                        f"      {layer_title}, AOD={int(aod)}: "
                        f"STAR d={int(d_hi)}/d={int(d_lo)}: avg_ratio={g_sd:.4f}x"
                    )

    # T cultivation: same placement, compare runtime line distances (each line’s triple).
    t_pairs_sorted = sorted(t_line_settings, key=lambda s: int(s[0]))
    if len(t_pairs_sorted) >= 2:
        print(
            "    [T cultivation — distance ratio at matched n_qubits, "
            f"placement={effective_t_placement}]"
        )

        def _t_triple_aod_subdf(
            work: pd.DataFrame, cd: int, ft: float, fps: int, aod_val: int
        ) -> pd.DataFrame:
            sub = work[work["n_aods"] == int(aod_val)].copy()
            mask = (
                (
                    pd.to_numeric(sub["code_distance"], errors="coerce").astype(int)
                    == int(cd)
                )
                & np.isclose(
                    pd.to_numeric(sub["fidelity_target"], errors="coerce"),
                    float(ft),
                    rtol=0.0,
                    atol=0.0,
                )
                & (sub["factory_physical_size"] == fps)
            )
            return sub.loc[mask].copy()

        for layer_name in layer_order:
            t_work = t_layers[layer_name]
            if "placement" in t_work.columns:
                t_work = t_work[
                    t_work["placement"].astype(str).str.strip() == effective_t_placement
                ].copy()
            if not all(
                c in t_work.columns
                for c in ("code_distance", "fidelity_target", "factory_physical_size")
            ):
                continue
            layer_title = pretty_layer.get(layer_name, layer_name)
            for aod in target_aods:
                for (cd_lo, ft_lo, fps_lo), (cd_hi, ft_hi, fps_hi) in zip(
                    t_pairs_sorted, t_pairs_sorted[1:]
                ):
                    t_lo = _t_triple_aod_subdf(t_work, cd_lo, ft_lo, fps_lo, aod)
                    t_hi = _t_triple_aod_subdf(t_work, cd_hi, ft_hi, fps_hi, aod)
                    sum_lo = _summarize_execution_by_qubits(
                        t_lo, method="T cultivation"
                    )
                    sum_hi = _summarize_execution_by_qubits(
                        t_hi, method="T cultivation"
                    )
                    row_lo = sum_lo[["n_qubits", "execution_time_mean"]].rename(
                        columns={"execution_time_mean": "rt_lo"}
                    )
                    row_hi = sum_hi[["n_qubits", "execution_time_mean"]].rename(
                        columns={"execution_time_mean": "rt_hi"}
                    )
                    m_td = row_hi.merge(row_lo, on="n_qubits", how="inner")
                    if m_td.empty:
                        continue
                    g_td = _geomean_ratio(
                        m_td["rt_hi"].to_numpy(dtype=float),
                        m_td["rt_lo"].to_numpy(dtype=float),
                    )
                    print(
                        f"      {layer_title}, AOD={int(aod)}: "
                        f"T d={int(cd_hi)}/d={int(cd_lo)} "
                        f"(d{int(cd_hi)}: nf={int(fps_hi)}, LER={ft_hi:.0e}; "
                        f"d{int(cd_lo)}: nf={int(fps_lo)}, LER={ft_lo:.0e}): "
                        f"avg_ratio={g_td:.4f}x"
                    )

    print("    [T vs STAR — cross-method, matched n_qubits]")
    printed_any = False
    for layer_name in layer_order:
        star_filtered = star_layers[layer_name][
            _setting_filter(star_layers[layer_name], setting_4)
        ].copy()
        t_work = t_layers[layer_name]
        if "placement" in t_work.columns:
            t_work = t_work[
                t_work["placement"].astype(str).str.strip() == effective_t_placement
            ].copy()
        if not all(
            c in t_work.columns
            for c in ("code_distance", "fidelity_target", "factory_physical_size")
        ):
            continue
        layer_title = pretty_layer.get(layer_name, layer_name)
        for aod in target_aods:
            for cd, ft, fps in t_line_settings:
                sd = _star_dist_for_t(int(cd))
                if sd is None:
                    continue
                star_d = star_filtered[
                    pd.to_numeric(
                        star_filtered["code_distance"], errors="coerce"
                    ).astype(int)
                    == int(sd)
                ].copy()
                star_summary = _summarize_execution_by_qubits(star_d, method="STAR")
                star_a = star_summary[star_summary["n_aods"] == int(aod)]
                if star_a.empty:
                    continue
                t_candidates = t_work[t_work["n_aods"] == int(aod)].copy()
                cd_match = pd.to_numeric(
                    t_candidates["code_distance"], errors="coerce"
                ).astype(int) == int(cd)
                ft_match = np.isclose(
                    pd.to_numeric(t_candidates["fidelity_target"], errors="coerce"),
                    float(ft),
                    rtol=0.0,
                    atol=0.0,
                )
                mask = (
                    cd_match & ft_match & (t_candidates["factory_physical_size"] == fps)
                )
                t_setting = t_candidates[mask]
                t_summary = _summarize_execution_by_qubits(
                    t_setting, method="T cultivation"
                )
                if t_summary.empty:
                    continue
                left = star_a[["n_qubits", "execution_time_mean"]].rename(
                    columns={"execution_time_mean": "star_t"}
                )
                right = t_summary[["n_qubits", "execution_time_mean"]].rename(
                    columns={"execution_time_mean": "t_t"}
                )
                m = left.merge(right, on="n_qubits", how="inner")
                if m.empty:
                    continue
                star_arr = np.clip(m["star_t"].to_numpy(dtype=float), eps, None)
                t_arr = np.clip(m["t_t"].to_numpy(dtype=float), eps, None)
                g = _geomean_ratio(t_arr, star_arr)
                arith_imp = float(np.mean((star_arr - t_arr) / star_arr) * 100.0)
                if int(sd) == int(cd):
                    pair = f"T d={int(cd)} / STAR d={int(cd)}"
                else:
                    pair = f"T d={int(cd)} / STAR d={int(sd)}"
                print(
                    f"      {layer_title}, AOD={int(aod)}: {pair} "
                    f"(nf={int(fps)}, LER={ft:.0e}): avg_ratio={g:.4f}x, "
                    f"avg_improvement_vs_STAR={arith_imp:+.2f}%"
                )
                printed_any = True
    if not printed_any:
        print("      ([T vs STAR] no overlapping T/STAR points for configured lines)")


def _plot_star_vs_t_cultivation_best(
    star_layers: dict[str, pd.DataFrame],
    t_layers: dict[str, pd.DataFrame],
    output_dir: str,
    target_aods: list[int] | None = None,
    *,
    t_placement: str | None = None,
    log_console: bool = True,
):
    """Compare STAR (d=7, 9 when present) and T-cultivation runtime in one 2x3 figure.

    STAR d=13 is intentionally omitted from this figure; T-cultivation may still show
    d=13 when enabled via ``_t_cultivation_runtime_line_settings``.

    T-cultivation curves use a single ``placement`` slice (e.g. ``col_based``). When
    ``t_placement`` is omitted, T rows default to ``col_based`` so one mixed-placement
    CSV does not merge distinct architectures.
    """
    if target_aods is None:
        target_aods = [1, 5]

    effective_t_placement = t_placement if t_placement is not None else "col_based"
    placement_title = _format_microarch_placement_label(effective_t_placement)
    filename_suffix = (
        "" if effective_t_placement == "col_based" else f"_{effective_t_placement}"
    )

    os.makedirs(output_dir, exist_ok=True)
    pretty_name = {
        "full_trotter": "Full Trotter",
        "zz_layers": "ZZ layers",
        "x_layer": "X layer",
    }
    layer_order = [
        k
        for k in ["full_trotter", "zz_layers", "x_layer"]
        if k in star_layers and k in t_layers
    ]
    if len(layer_order) == 0:
        return

    star_runtime_setting = STAR_VS_T_RUNTIME_SETTING

    star_dist_all: set[int] = set()
    for layer_name in layer_order:
        star_filtered = star_layers[layer_name][
            _setting_filter(star_layers[layer_name], star_runtime_setting)
        ].copy()
        if not star_filtered.empty and "code_distance" in star_filtered.columns:
            star_dist_all.update(
                pd.to_numeric(star_filtered["code_distance"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

    star_distances = [d for d in STAR_VS_T_RUNTIME_STAR_DISTANCES if d in star_dist_all]
    if len(star_distances) == 0:
        return

    fig, axes = plt.subplots(len(target_aods), len(layer_order), figsize=(16, 8.5))
    if len(target_aods) == 1:
        axes = [axes]
    if len(layer_order) == 1:
        axes = [[ax] for ax in axes]

    star_style = {
        7: {"color": "tab:blue", "marker": "o"},
        9: {"color": "tab:red", "marker": "D"},
    }
    t_style_colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]
    t_line_settings = _t_cultivation_runtime_line_settings()

    for row_idx, aod in enumerate(target_aods):
        for col_idx, layer_name in enumerate(layer_order):
            ax = axes[row_idx][col_idx]
            star_filtered = star_layers[layer_name][
                _setting_filter(star_layers[layer_name], star_runtime_setting)
            ].copy()
            t_work = t_layers[layer_name]
            if "placement" in t_work.columns:
                t_work = t_work[
                    t_work["placement"].astype(str).str.strip() == effective_t_placement
                ].copy()

            star_ref_values: list[float] = []
            star_x_values: list[int] = []
            plotted_star = 0
            for sd in star_distances:
                star_d = star_filtered[
                    pd.to_numeric(
                        star_filtered["code_distance"], errors="coerce"
                    ).astype(int)
                    == int(sd)
                ].copy()
                star_summary = _summarize_execution_by_qubits(star_d, method="STAR")
                star_a = star_summary[star_summary["n_aods"] == aod].sort_values(
                    "n_qubits"
                )
                if star_a.empty:
                    continue
                plotted_star += 1
                style = star_style.get(sd, {"color": "black", "marker": "o"})
                mean_vals = star_a["execution_time_mean"]
                min_vals = star_a["execution_time_min"]
                max_vals = star_a["execution_time_max"]
                err_lower = mean_vals - min_vals
                err_upper = max_vals - mean_vals
                ax.errorbar(
                    star_a["n_qubits"],
                    mean_vals,
                    yerr=[err_lower, err_upper],
                    marker=style["marker"],
                    capsize=3,
                    linewidth=2.0,
                    color=style["color"],
                    label=f"STAR, d={int(sd)}",
                )
                star_x_values.extend(star_a["n_qubits"].astype(int).tolist())
                star_ref_values.extend(
                    star_a["execution_time_mean"].to_numpy(dtype=float).tolist()
                )

            if len(star_x_values) > 0:
                x_star = sorted(set(int(v) for v in star_x_values))
                star_bounds = _get_theoretical_lower_bound(x_star)
                if layer_name == "full_trotter":
                    y_star_exp = star_bounds["full_trotter"]
                elif layer_name == "zz_layers":
                    y_star_exp = star_bounds["zz_layer"]
                else:
                    y_star_exp = star_bounds["x_layer"]
                ax.plot(
                    x_star,
                    y_star_exp,
                    color="black",
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.9,
                    label="Expected time (STAR)",
                )

            plotted_t = 0
            if all(
                c in t_work.columns
                for c in (
                    "code_distance",
                    "fidelity_target",
                    "factory_physical_size",
                )
            ):
                t_candidates = t_work[t_work["n_aods"] == aod].copy()
                for cd, ft, fps in t_line_settings:
                    cd_match = pd.to_numeric(
                        t_candidates["code_distance"], errors="coerce"
                    ).astype(int) == int(cd)
                    ft_match = np.isclose(
                        pd.to_numeric(t_candidates["fidelity_target"], errors="coerce"),
                        float(ft),
                        rtol=0.0,
                        atol=0.0,
                    )
                    mask = (
                        cd_match
                        & ft_match
                        & (t_candidates["factory_physical_size"] == fps)
                    )
                    t_setting = t_candidates[mask]
                    t_summary = _summarize_execution_by_qubits(
                        t_setting, method="T cultivation"
                    ).sort_values("n_qubits")
                    if t_summary.empty:
                        continue
                    plotted_t += 1
                    t_color = t_style_colors[
                        t_line_settings.index((cd, ft, fps)) % len(t_style_colors)
                    ]
                    mean_vals = t_summary["execution_time_mean"]
                    min_vals = t_summary["execution_time_min"]
                    max_vals = t_summary["execution_time_max"]
                    err_lower = mean_vals - min_vals
                    err_upper = max_vals - mean_vals
                    ax.errorbar(
                        t_summary["n_qubits"],
                        mean_vals,
                        yerr=[err_lower, err_upper],
                        marker="s",
                        capsize=3,
                        linewidth=1.8,
                        color=t_color,
                        label=f"T cultivation, d={int(cd)}",
                    )
                    x_t = sorted(
                        set(t_summary["n_qubits"].dropna().astype(int).tolist())
                    )
                    if len(x_t) > 0:
                        t_bounds = _get_theoretical_lower_bound_t_cultivation(
                            x_t,
                            code_distance=cd,
                            fidelity_target=ft,
                        )
                        if layer_name == "full_trotter":
                            y_t_exp = t_bounds["full_trotter"]
                        elif layer_name == "zz_layers":
                            y_t_exp = t_bounds["zz_layer"]
                        else:
                            y_t_exp = t_bounds["x_layer"]
                        ax.plot(
                            x_t,
                            y_t_exp,
                            color=t_color,
                            linestyle="-.",
                            linewidth=1.4,
                            alpha=0.9,
                            label=f"Expected time (T, d={int(cd)})",
                        )

            if row_idx == 0:
                ax.set_title(pretty_name[layer_name], fontsize=_FIG_FONT_SIZE, pad=10)
            ax.tick_params(axis="both", which="major", pad=1)
            ax.set_xlabel("Number of Qubits", labelpad=0)
            if col_idx == 0:
                ax.set_ylabel(f"AOD {aod}\nExecution Time", fontsize=_FIG_FONT_SIZE)
            else:
                ax.set_ylabel("Execution Time", labelpad=0)
            ax.set_ylim(bottom=0)
            if layer_name in {"zz_layers", "x_layer"}:
                if len(star_ref_values) > 0:
                    _add_standard_y_ticks_with_star_reference(
                        ax, np.asarray(star_ref_values, dtype=float)
                    )
            else:
                if len(star_ref_values) > 0:
                    _add_star_reference_tick(
                        ax, np.asarray(star_ref_values, dtype=float)
                    )
            ax.grid(True, alpha=0.3)
            if plotted_star > 0 or plotted_t > 0:
                ax.legend(fontsize=_FIG_FONT_SIZE)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=_FIG_FONT_SIZE,
                )

    if log_console:
        _print_runtime_star_vs_t_all_in_one_improvements(
            star_layers,
            t_layers,
            layer_order,
            star_distances,
            target_aods,
            effective_t_placement,
            star_runtime_setting,
            filename_suffix,
        )

    fig.suptitle(
        "STAR vs T cultivation runtime · "
        "Columns: Full Trotter | ZZ layers | X layer · "
        f"Rows: AOD 1 (top), AOD 5 (bottom) · T: {placement_title}",
        fontsize=_FIG_FONT_SIZE,
        y=0.99,
    )
    _apply_shared_bottom_legend(fig, ncol=3)
    fig.tight_layout(rect=(0.04, 0.22, 0.98, 0.90), pad=0.10, w_pad=0.03, h_pad=0.03)
    fig.subplots_adjust(
        top=0.90,
        hspace=0.42,
        wspace=0.28,
        bottom=0.22,
        left=0.04,
        right=0.98,
    )
    filename = os.path.join(
        output_dir,
        f"runtime_star_vs_t_cultivation_all_in_one_aod_"
        f"{'-'.join(str(a) for a in target_aods)}{filename_suffix}.pdf",
    )
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _mask_t_cultivation_triple(
    df: pd.DataFrame, cd: int, ft: float, fps: int
) -> pd.Series:
    """Boolean mask for one (code_distance, fidelity_target, factory_physical_size) setting."""
    cd_m = pd.to_numeric(df["code_distance"], errors="coerce").astype(int) == int(cd)
    ft_series = pd.to_numeric(df["fidelity_target"], errors="coerce")
    ft_m = np.isclose(ft_series, float(ft), rtol=1e-12, atol=1e-20)
    return cd_m & ft_m & (df["factory_physical_size"] == fps)


def _mask_t_cultivation_compile(
    df: pd.DataFrame,
    trivial_return: bool,
    decompose_move: bool,
    redistribute_stage1_success: bool,
) -> pd.Series:
    """Boolean mask for one T-cultivation compile tuple."""
    return (
        (df["trivial_return"] == trivial_return)
        & (df["decompose_move"] == decompose_move)
        & (df["redistribute_stage1_success"] == redistribute_stage1_success)
    )


def _filter_t_cultivation_main_compile_setting(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows matching ``T_CULTIVATION_MAIN_COMPILE_SETTING`` when compile columns exist.

    If ``trivial_return`` / ``decompose_move`` / ``redistribute_stage1_success`` are
    absent (legacy CSV), returns ``df`` unchanged.
    """
    cols = ("trivial_return", "decompose_move", "redistribute_stage1_success")
    if df.empty or not all(c in df.columns for c in cols):
        return df
    tr, dm, rs = T_CULTIVATION_MAIN_COMPILE_SETTING
    return df.loc[_mask_t_cultivation_compile(df, tr, dm, rs)].copy()


def _plot_t_cultivation_multi_aod(
    t_layers: dict[str, pd.DataFrame],
    output_dir: str,
    *,
    verbose: bool = True,
):
    """T-cultivation AOD comparison: one row per setting, three columns for layers.

    Uses **column-based** placement only (same microarchitecture as the main STAR
    comparison CSV slice).
    """
    os.makedirs(output_dir, exist_ok=True)
    pretty_name = {
        "full_trotter": "Full Trotter",
        "zz_layers": "ZZ layers",
        "x_layer": "X layer",
    }
    layer_order = [k for k in ["full_trotter", "zz_layers", "x_layer"] if k in t_layers]
    if len(layer_order) == 0:
        return

    required = ("code_distance", "fidelity_target", "factory_physical_size")
    if not all(c in t_layers[layer_order[0]].columns for c in required):
        if verbose:
            print("Skipping T-cultivation multi-AOD plot: missing setting columns.")
        return

    ref_layer = t_layers[layer_order[0]]
    if "placement" in ref_layer.columns:
        ref_layer = _filter_col_based(ref_layer)
    all_t_aods = sorted(
        pd.to_numeric(ref_layer["n_aods"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    aod_color_map = _build_aod_color_map(all_t_aods)

    t_line_settings = _t_cultivation_runtime_line_settings()
    n_settings = len(t_line_settings)
    if n_settings == 0:
        if verbose:
            print("Skipping T-cultivation multi-AOD plot: no selected settings.")
        return
    fig, axes = plt.subplots(
        n_settings,
        len(layer_order),
        figsize=(5.6 * len(layer_order), 4.8 * n_settings),
        squeeze=False,
    )

    plotted_any = False
    for row_idx, (cd, ft, fps) in enumerate(t_line_settings):
        row_label = _format_t_cultivation_setting_label(cd, ft, fps)
        for col_idx, layer_name in enumerate(layer_order):
            ax = axes[row_idx][col_idx]
            layer_df = t_layers[layer_name]
            if "placement" in layer_df.columns:
                layer_df = _filter_col_based(layer_df)
            mask = _mask_t_cultivation_triple(layer_df, cd, ft, fps)
            sub_df = layer_df.loc[mask].copy()

            if sub_df.empty:
                if row_idx == 0:
                    ax.set_title(
                        pretty_name[layer_name], fontsize=_FIG_FONT_SIZE, pad=10
                    )
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=_FIG_FONT_SIZE,
                )
                ax.set_xlabel("Number of Qubits")
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{row_label}\nExecution Time", fontsize=_FIG_FONT_SIZE
                    )
                else:
                    ax.set_ylabel("Execution Time")
                ax.grid(True, alpha=0.3)
                continue

            summary = _summarize_execution_by_qubits(sub_df, method="T cultivation")
            aod_values = sorted(summary["n_aods"].dropna().astype(int).unique())
            for aod in aod_values:
                sub = summary[summary["n_aods"] == aod].sort_values("n_qubits")
                if sub.empty:
                    continue
                plotted_any = True
                mean_vals = sub["execution_time_mean"]
                min_vals = sub["execution_time_min"]
                max_vals = sub["execution_time_max"]
                err_lower = mean_vals - min_vals
                err_upper = max_vals - mean_vals
                ax.errorbar(
                    sub["n_qubits"],
                    mean_vals,
                    yerr=[err_lower, err_upper],
                    marker="o",
                    capsize=3,
                    linewidth=1.8,
                    color=aod_color_map.get(int(aod)),
                    label=f"AOD={aod}",
                )

            # Overlay theoretical lower bound for this layer as a dotted line.
            x_vals = sorted(summary["n_qubits"].dropna().astype(int).unique())
            if len(x_vals) > 0:
                bounds = _get_theoretical_lower_bound_t_cultivation(
                    x_vals,
                    code_distance=cd,
                    fidelity_target=ft,
                )
                if layer_name == "full_trotter":
                    y_bound = bounds["full_trotter"]
                elif layer_name == "zz_layers":
                    y_bound = bounds["zz_layer"]
                else:
                    y_bound = bounds["x_layer"]
                ax.plot(
                    x_vals,
                    y_bound,
                    color="black",
                    linestyle=":",
                    linewidth=1.6,
                    label="Expected time",
                )

            if row_idx == 0:
                ax.set_title(pretty_name[layer_name], fontsize=_FIG_FONT_SIZE, pad=12)
            ax.tick_params(axis="both", which="major", pad=1)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nExecution Time", fontsize=_FIG_FONT_SIZE)
            else:
                ax.set_ylabel("Execution Time")
            ax.set_xlabel("Number of Qubits")
            ax.grid(True, alpha=0.3)
            if len(aod_values) > 0:
                ax.legend(fontsize=_FIG_FONT_SIZE)

    if not plotted_any:
        plt.close(fig)
        return

    fig.suptitle("T-cultivation runtime vs AOD", fontsize=_FIG_FONT_SIZE, y=0.995)
    _apply_shared_bottom_legend(fig, ncol=6, anchor_y=0.03)
    fig.tight_layout(rect=(0.04, 0.13, 0.98, 0.93), pad=0.10, w_pad=0.03, h_pad=0.03)
    fig.subplots_adjust(
        top=0.93,
        hspace=0.35,
        wspace=0.32,
        left=0.04,
        right=0.98,
        bottom=0.15,
    )
    fig.savefig(
        os.path.join(output_dir, "runtime_t_cultivation_multi_aod.pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def _plot_t_cultivation_compile_ablation(
    t_layers: dict[str, pd.DataFrame],
    output_dir: str,
    *,
    placement: str = "col_based",
    verbose: bool = True,
):
    """Ablation over T-cultivation compile options (trivial return, decompose move, S1 patch).

    Layout: AOD rows × layer columns (Full Trotter / ZZ / X) with separate execution and
    movement figures. Each line is one tuple in ``_T_COMPILE_ABLATION_GRID`` (aligned with
    ``evaluation_fidelity_t_cultivation.COMPILE_SETTINGS``). For each code distance, the
    **first** ``(code_distance, fidelity_target, factory_physical_size)`` in
    :func:`_t_cultivation_runtime_line_settings` for that distance is held fixed.
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(_T_COMPILE_ABLATION_GRID) != len(_T_COMPILE_ABLATION_LABELS):
        raise ValueError("T-cultivation compile ablation grid/label length mismatch.")

    pretty_name = {
        "full_trotter": "Full Trotter",
        "zz_layers": "ZZ layers",
        "x_layer": "X layer",
    }
    layer_order = [k for k in ["full_trotter", "zz_layers", "x_layer"] if k in t_layers]
    if not layer_order:
        return

    compile_cols = {
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
    }
    if not compile_cols.issubset(t_layers[layer_order[0]].columns):
        if verbose:
            print(
                "Skipping T-cultivation compile ablation: CSV missing compile columns "
                f"{sorted(compile_cols)}."
            )
        return

    required = ("code_distance", "fidelity_target", "factory_physical_size")
    if not all(c in t_layers[layer_order[0]].columns for c in required):
        if verbose:
            print(
                "Skipping T-cultivation compile ablation: missing fidelity triple columns."
            )
        return

    settings = _t_cultivation_runtime_line_settings()
    if not settings:
        if verbose:
            print(
                "Skipping T-cultivation compile ablation: no selected runtime line settings."
            )
        return

    distances = sorted({int(cd) for cd, _ft, _fps in settings})

    all_aods: set[int] = set()
    for layer_df in t_layers.values():
        layer_work = layer_df.copy()
        if "placement" in layer_work.columns:
            layer_work = layer_work[
                layer_work["placement"].astype(str).str.strip() == placement
            ].copy()
        if "n_aods" in layer_work.columns:
            all_aods.update(
                pd.to_numeric(layer_work["n_aods"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
    aods_to_plot = [aod for aod in [1, 5] if aod in all_aods]
    if not aods_to_plot:
        aods_to_plot = sorted(all_aods)[:2]
    if not aods_to_plot:
        if verbose:
            print("Skipping T-cultivation compile ablation: no AOD values.")
        return

    def _filtered_compile_grouped(
        layer_df: pd.DataFrame,
        triple: tuple[int, float, int],
        compile_tuple: tuple[bool, bool, bool],
    ) -> pd.DataFrame:
        work = layer_df.copy()
        if "placement" in work.columns:
            work = work[work["placement"].astype(str).str.strip() == placement].copy()
        cd, ft, fps = triple
        tr, dm, rs = compile_tuple
        mask = _mask_t_cultivation_triple(
            work, cd, ft, fps
        ) & _mask_t_cultivation_compile(work, tr, dm, rs)
        work = work[mask].copy()
        if work.empty:
            return work
        return (
            work.groupby(["n_aods", "n_qubits"])[RESULT_COLS]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .pipe(
                lambda g: g.set_axis(
                    [
                        "_".join(c).strip("_") if isinstance(c, tuple) else c
                        for c in g.columns
                    ],
                    axis=1,
                )
            )
        )

    ablation_font_size = _FIG_FONT_SIZE + 2
    n_compile = len(_T_COMPILE_ABLATION_GRID)

    for distance in distances:
        distance_settings = [s for s in settings if int(s[0]) == distance]
        if not distance_settings:
            continue
        primary = distance_settings[0]
        cd, ft, fps = primary
        triple_note = f"d={int(cd)}"

        compile_colors = {
            idx: plt.get_cmap("tab10")(idx % 10) for idx in range(n_compile)
        }

        data_by_layer: dict[str, dict[int, pd.DataFrame]] = {}
        for layer_name in layer_order:
            layer_data: dict[int, pd.DataFrame] = {}
            for idx, compile_tuple in enumerate(_T_COMPILE_ABLATION_GRID):
                grouped = _filtered_compile_grouped(
                    t_layers[layer_name], primary, compile_tuple
                )
                if not grouped.empty:
                    layer_data[idx] = grouped
            if layer_data:
                data_by_layer[layer_name] = layer_data

        if not data_by_layer:
            if verbose:
                print(
                    f"No data for T-cultivation compile ablation ({placement}, d={distance}), skipping."
                )
            continue

        def _draw_execution_panel(ax, layer_name: str, aod: int, show_title: bool):
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)
            x_vals: list[int] = []
            for idx in range(n_compile):
                grouped = data_by_layer.get(layer_name, {}).get(idx)
                if grouped is None:
                    continue
                sub = grouped[grouped["n_aods"] == aod].sort_values("n_qubits")
                if sub.empty:
                    continue
                x_vals = sorted(set(x_vals).union(sub["n_qubits"].dropna().astype(int)))
                mean_vals = sub["total_time_mean"]
                min_vals = sub["total_time_min"]
                max_vals = sub["total_time_max"]
                ax.errorbar(
                    sub["n_qubits"],
                    mean_vals,
                    yerr=[mean_vals - min_vals, max_vals - mean_vals],
                    marker="o",
                    capsize=4,
                    linewidth=1.8,
                    color=compile_colors[idx],
                    label=_T_COMPILE_ABLATION_LABELS[idx],
                )
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(int(v)) for v in x_vals], rotation=45, ha="right")
            ax.tick_params(labelsize=ablation_font_size)
            if show_title:
                ax.set_title(
                    pretty_name[layer_name], fontsize=ablation_font_size, pad=10
                )

        def _draw_movement_panel(ax, layer_name: str, aod: int, show_title: bool):
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)
            x_vals: list[int] = []
            for idx in range(n_compile):
                grouped = data_by_layer.get(layer_name, {}).get(idx)
                if grouped is None:
                    continue
                sub = grouped[grouped["n_aods"] == aod].sort_values("n_qubits")
                if not sub.empty:
                    x_vals = sorted(
                        set(x_vals).union(sub["n_qubits"].dropna().astype(int))
                    )
            if not x_vals:
                ax.set_xticks([])
                return

            x_index = {x: i for i, x in enumerate(x_vals)}
            width = 0.8 / max(1, n_compile)
            for idx in range(n_compile):
                grouped = data_by_layer.get(layer_name, {}).get(idx)
                if grouped is None:
                    continue
                sub = grouped[grouped["n_aods"] == aod].sort_values("n_qubits")
                if sub.empty:
                    continue
                positions = [x_index[int(x)] + idx * width for x in sub["n_qubits"]]
                move = sub["movement_time_mean"].to_numpy(dtype=float)
                ret = sub["return_movement_time_mean"].to_numpy(dtype=float)
                base_color = compile_colors[idx]
                ax.bar(
                    positions,
                    move,
                    width=width,
                    color=base_color,
                    label=_T_COMPILE_ABLATION_LABELS[idx],
                )
                ax.bar(
                    positions,
                    ret,
                    width=width,
                    bottom=move,
                    color=mcolors.to_rgba(base_color, alpha=0.35),
                )
            ax.set_xticks([i + width * (n_compile - 1) / 2 for i in range(len(x_vals))])
            ax.set_xticklabels([str(int(v)) for v in x_vals], rotation=45, ha="right")
            ax.tick_params(labelsize=ablation_font_size)
            if show_title:
                ax.set_title(
                    pretty_name[layer_name], fontsize=ablation_font_size, pad=10
                )

        for metric_name, draw_panel, file_suffix in [
            ("Execution Time", _draw_execution_panel, "execution"),
            ("Movement Time", _draw_movement_panel, "movement"),
        ]:
            fig, axes = plt.subplots(
                len(aods_to_plot),
                len(layer_order),
                figsize=(6.9 * len(layer_order), 6.4 * len(aods_to_plot)),
                squeeze=False,
            )
            for row_idx, aod in enumerate(aods_to_plot):
                for col_idx, layer_name in enumerate(layer_order):
                    ax = axes[row_idx, col_idx]
                    draw_panel(ax, layer_name, aod, show_title=(row_idx == 0))
                    if col_idx == 0:
                        ax.set_ylabel(
                            f"AOD = {aod}\n{metric_name}",
                            fontsize=ablation_font_size,
                        )
                    else:
                        ax.set_ylabel("")
                    if row_idx == len(aods_to_plot) - 1:
                        ax.set_xlabel("Number of Qubits", fontsize=ablation_font_size)
                    else:
                        ax.set_xlabel("")

            handles = [
                Line2D(
                    [0],
                    [0],
                    color=compile_colors[idx],
                    linewidth=2,
                    label=_T_COMPILE_ABLATION_LABELS[idx],
                )
                for idx in range(n_compile)
                if any(idx in layer_data for layer_data in data_by_layer.values())
            ]
            fig.suptitle(
                f"T-cultivation compile ablation (d={distance}), {metric_name}\n"
                f"({triple_note})",
                fontsize=ablation_font_size,
                y=0.995,
            )
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.035),
                ncol=max(1, len(handles)),
                fontsize=ablation_font_size,
                frameon=True,
            )
            fig.tight_layout(rect=(0, 0.25, 1, 1))
            fig.subplots_adjust(bottom=0.16, hspace=0.32, wspace=0.32)
            output_path = os.path.join(
                output_dir,
                f"t_cultivation_compile_ablation_{placement}_d{distance}_{file_suffix}.pdf",
            )
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0.10)
            plt.close(fig)
            if verbose:
                print(f"Saved: {output_path}")


def _plot_t_cultivation_architecture_placement_execution(
    t_layers: dict[str, pd.DataFrame],
    output_dir: str,
    *,
    verbose: bool = True,
) -> None:
    """Compare T-cultivation placements at ``T_CULTIVATION_MAIN_COMPILE_SETTING`` (STAR-style).

    One PDF per code distance from ``_t_cultivation_runtime_line_settings()`` (same
    primary triple as compile ablation): rows are AODs (1 and 5 when present), columns
    are layers. Each panel plots execution time vs qubit count with one curve per
    placement in ``T_CULTIVATION_ARCHITECTURE_PLACEMENTS``, plus a dashed T theoretical
    lower bound for the selected fidelity triple.
    """
    os.makedirs(output_dir, exist_ok=True)
    pretty_round = {
        "full_trotter": "Full Trotter",
        "zz_layers": "ZZ layers",
        "x_layer": "X layer",
    }
    layer_order = [k for k in ["full_trotter", "zz_layers", "x_layer"] if k in t_layers]
    if not layer_order:
        return

    compile_cols = {
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
    }
    if not compile_cols.issubset(t_layers[layer_order[0]].columns):
        if verbose:
            print(
                "Skipping T-cultivation architecture placement plot: CSV missing compile columns."
            )
        return

    required = (
        "code_distance",
        "fidelity_target",
        "factory_physical_size",
        "placement",
    )
    if not all(c in t_layers[layer_order[0]].columns for c in required):
        if verbose:
            print(
                "Skipping T-cultivation architecture placement plot: missing fidelity/placement columns."
            )
        return

    settings = _t_cultivation_runtime_line_settings()
    if not settings:
        return

    distances = sorted({int(s[0]) for s in settings})
    tr_m, dm_m, rs_m = T_CULTIVATION_MAIN_COMPILE_SETTING
    fs = _FIG_FONT_SIZE + 2

    for distance in distances:
        distance_settings = [s for s in settings if int(s[0]) == distance]
        if not distance_settings:
            continue
        primary = distance_settings[0]
        cd, ft, fps = primary
        triple_note = f"d={int(cd)}"

        agg_data: dict[str, pd.DataFrame] = {}
        for layer_name in layer_order:
            work = t_layers[layer_name].copy()
            mask = _mask_t_cultivation_compile(
                work, tr_m, dm_m, rs_m
            ) & _mask_t_cultivation_triple(work, cd, ft, fps)
            work = work.loc[mask].copy()
            if work.empty or "placement" not in work.columns:
                continue
            work["placement"] = work["placement"].astype(str).str.strip()
            work = work[
                work["placement"].isin(T_CULTIVATION_ARCHITECTURE_PLACEMENTS)
            ].copy()
            if work.empty:
                continue
            agg_data[layer_name] = _aggregate_microarch_trials(work)

        if not agg_data:
            if verbose:
                print(
                    f"No data for T-cultivation architecture placement plot (d={distance}), skipping."
                )
            continue

        round_names = [r for r in layer_order if r in agg_data]
        present: set[str] = set()
        for g in agg_data.values():
            for p in g["placement"].astype(str).str.strip().unique():
                present.add(str(p))
        placements = [p for p in T_CULTIVATION_ARCHITECTURE_PLACEMENTS if p in present]
        if not placements:
            continue

        all_aods = sorted(
            {
                int(a)
                for g in agg_data.values()
                for a in pd.to_numeric(g["n_aods"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            }
        )
        aods_to_plot = [a for a in [1, 5] if a in all_aods]
        if not aods_to_plot:
            aods_to_plot = all_aods[:2]
        if not aods_to_plot:
            continue

        cmap = plt.get_cmap("tab10")
        placement_color = {p: cmap(i % 10) for i, p in enumerate(placements)}

        execution_ylim: dict[str, tuple[float, float]] = {}
        for round_name in round_names:
            g_round = agg_data[round_name]
            y_mins: list[float] = []
            y_maxs: list[float] = []
            for aod_value in aods_to_plot:
                g_aod = g_round[g_round["n_aods"] == aod_value]
                if g_aod.empty:
                    continue
                for pname in placements:
                    sub = g_aod[g_aod["placement"] == pname]
                    if sub.empty:
                        continue
                    total_min = pd.to_numeric(
                        sub.get("total_time_min", sub["total_time_mean"]),
                        errors="coerce",
                    )
                    total_max = pd.to_numeric(
                        sub.get("total_time_max", sub["total_time_mean"]),
                        errors="coerce",
                    )
                    y_mins.extend(total_min.tolist())
                    y_maxs.extend(total_max.tolist())
                x_vals_lim = sorted(g_aod[SWEEP_COL].dropna().astype(int).unique())
                if x_vals_lim:
                    tb = _get_theoretical_lower_bound_t_cultivation(
                        x_vals_lim,
                        code_distance=cd,
                        fidelity_target=ft,
                    )
                    if round_name == "full_trotter":
                        yexp = tb["full_trotter"]
                    elif round_name == "zz_layers":
                        yexp = tb["zz_layer"]
                    else:
                        yexp = tb["x_layer"]
                    y_mins.extend(yexp)
                    y_maxs.extend(yexp)
            if y_mins and y_maxs:
                pad = (max(y_maxs) - min(y_mins)) * 0.05
                execution_ylim[round_name] = (min(y_mins) - pad, max(y_maxs) + pad)
            else:
                execution_ylim[round_name] = (0.0, 1.0)

        fig, axes = plt.subplots(
            len(aods_to_plot),
            len(round_names),
            figsize=(5.8 * len(round_names), 4.4 * len(aods_to_plot)),
            squeeze=False,
        )

        for row_idx, aod_value in enumerate(aods_to_plot):
            for col_idx, round_name in enumerate(round_names):
                ax = axes[row_idx, col_idx]
                g_round = agg_data[round_name]
                g_aod = g_round[g_round["n_aods"] == aod_value]
                if g_aod.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=fs,
                    )
                    ax.set_axis_off()
                    continue

                x_vals: list[int] = []
                for pname in placements:
                    sub = g_aod[g_aod["placement"] == pname].sort_values(SWEEP_COL)
                    if sub.empty:
                        continue
                    x_vals = sorted(
                        set(x_vals).union(sub[SWEEP_COL].dropna().astype(int).tolist())
                    )
                    mean_vals = sub["total_time_mean"]
                    min_vals = sub["total_time_min"]
                    max_vals = sub["total_time_max"]
                    err_lower = mean_vals - min_vals
                    err_upper = max_vals - mean_vals
                    ax.errorbar(
                        sub[SWEEP_COL],
                        mean_vals,
                        yerr=[err_lower, err_upper],
                        marker="o",
                        capsize=4,
                        linewidth=1.8,
                        color=placement_color[pname],
                    )

                if not x_vals:
                    x_vals = sorted(
                        g_aod[SWEEP_COL].dropna().astype(int).unique().tolist()
                    )

                if x_vals:
                    tb = _get_theoretical_lower_bound_t_cultivation(
                        x_vals,
                        code_distance=cd,
                        fidelity_target=ft,
                    )
                    if round_name == "full_trotter":
                        y_t = tb["full_trotter"]
                    elif round_name == "zz_layers":
                        y_t = tb["zz_layer"]
                    else:
                        y_t = tb["x_layer"]
                    ax.plot(
                        x_vals,
                        y_t,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                    )

                ax.set_axisbelow(True)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(*execution_ylim[round_name])
                if x_vals:
                    ax.set_xticks(x_vals)
                    ax.set_xticklabels(
                        [str(int(v)) for v in x_vals], rotation=45, ha="right"
                    )
                else:
                    ax.set_xticks([])
                ax.tick_params(labelsize=fs)
                if row_idx == 0:
                    ax.set_title(pretty_round[round_name], fontsize=fs, pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"AOD = {aod_value}\nExecution Time", fontsize=fs)
                else:
                    ax.set_ylabel("Execution Time", fontsize=fs)
                if row_idx == len(aods_to_plot) - 1:
                    ax.set_xlabel("Number of Qubits", fontsize=fs)
                else:
                    ax.set_xlabel("")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=placement_color[p],
                marker="o",
                linestyle="-",
                label=_format_microarch_placement_label(p),
            )
            for p in placements
        ] + [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Expected time",
            ),
        ]
        fig.suptitle(
            f"T-cultivation placement comparison (main compile) · {triple_note}",
            fontsize=fs,
            y=0.99,
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(legend_handles)),
            fontsize=fs,
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 0.92), pad=0.10, w_pad=0.03, h_pad=0.03)
        fig.subplots_adjust(bottom=0.18, hspace=0.35, wspace=0.28)
        out_path = os.path.join(
            output_dir,
            f"t_cultivation_architecture_placement_d{distance}_execution.pdf",
        )
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)
        if verbose:
            print(f"Saved: {out_path}")


def _print_runtime_speedup_by_setting(
    star_layers: dict[str, pd.DataFrame],
    t_layers: dict[str, pd.DataFrame],
    *,
    target_aods: list[int] | None = None,
) -> None:
    """Print runtime speedup = STAR execution time / T-cultivation execution time.

    Comparison is done on matched ``n_qubits`` points for each:
    - layer (full_trotter / zz_layers / x_layer)
    - AOD value
    - STAR code distance (7, 9, 13 when present in profiling CSV)
    - T-cultivation setting (code_distance, fidelity_target, factory_physical_size)
    """
    if target_aods is None:
        target_aods = [1, 5]

    star_runtime_setting = STAR_VS_T_RUNTIME_SETTING
    layer_order = [
        k
        for k in ["full_trotter", "zz_layers", "x_layer"]
        if k in star_layers and k in t_layers
    ]
    if len(layer_order) == 0:
        print("Runtime speedup summary skipped: no overlapping STAR/T layer frames.")
        return

    print("\nRuntime ratio summary (ratio = T runtime / STAR runtime):")
    printed_any = False
    eps = 1e-15

    for layer_name in layer_order:
        star_filtered = star_layers[layer_name][
            _setting_filter(star_layers[layer_name], star_runtime_setting)
        ].copy()
        t_work = t_layers[layer_name].copy()

        if star_filtered.empty or t_work.empty:
            continue

        if "code_distance" not in star_filtered.columns:
            continue

        star_filtered["code_distance"] = pd.to_numeric(
            star_filtered["code_distance"], errors="coerce"
        )
        star_distances = sorted(
            star_filtered["code_distance"].dropna().astype(int).unique().tolist()
        )
        star_distances = [
            d for d in STAR_PROFILING_CODE_DISTANCES if d in star_distances
        ] or star_distances
        if len(star_distances) == 0:
            continue

        if not all(
            c in t_work.columns
            for c in ("code_distance", "fidelity_target", "factory_physical_size")
        ):
            continue

        for aod in target_aods:
            t_aod = t_work[pd.to_numeric(t_work["n_aods"], errors="coerce") == int(aod)]
            if t_aod.empty:
                continue

            for sd in star_distances:
                star_sd = star_filtered[
                    pd.to_numeric(star_filtered["code_distance"], errors="coerce")
                    == int(sd)
                ].copy()
                star_summary = _summarize_execution_by_qubits(star_sd, method="STAR")
                star_summary = star_summary[
                    pd.to_numeric(star_summary["n_aods"], errors="coerce") == int(aod)
                ].copy()
                if star_summary.empty:
                    continue
                star_summary = star_summary[["n_qubits", "execution_time_mean"]].rename(
                    columns={"execution_time_mean": "star_runtime"}
                )

                for cd, ft, fps in _t_cultivation_runtime_line_settings():
                    mask = _mask_t_cultivation_triple(t_aod, cd, ft, fps)
                    t_setting = t_aod.loc[mask].copy()
                    t_summary = _summarize_execution_by_qubits(
                        t_setting, method="T cultivation"
                    )
                    if t_summary.empty:
                        continue
                    t_summary = t_summary[["n_qubits", "execution_time_mean"]].rename(
                        columns={"execution_time_mean": "t_runtime"}
                    )

                    merged = (
                        star_summary.merge(t_summary, on="n_qubits", how="inner")
                        .sort_values("n_qubits")
                        .reset_index(drop=True)
                    )
                    if merged.empty:
                        continue

                    merged["ratio_t_over_star"] = np.clip(
                        merged["t_runtime"], eps, None
                    ) / np.clip(merged["star_runtime"], eps, None)

                    ratios = merged["ratio_t_over_star"].to_numpy(dtype=float)
                    geomean_ratio = float(
                        np.exp(np.mean(np.log(np.clip(ratios, eps, None))))
                    )
                    mean_ratio = float(np.mean(ratios))

                    print(
                        "  "
                        f"layer={layer_name}, AOD={int(aod)}, STAR_d={int(sd)} vs "
                        f"T(d={int(cd)}): "
                        f"n_points={len(merged)}, "
                        f"geomean_ratio_t_over_star={geomean_ratio:.4f}x, "
                        f"mean_ratio_t_over_star={mean_ratio:.4f}x"
                    )
                    printed_any = True

    if not printed_any:
        print("  No matched points found for runtime speedup computation.")


def _print_requested_runtime_improvements(
    star_df: pd.DataFrame,
    t_df: pd.DataFrame,
) -> None:
    """Print runtime summary in four sections (STAR d∈{7,9}, T d∈{9,13} at LER=1e-8)."""
    eps = 1e-15

    star_full = _build_layer_frames(star_df).get("full_trotter", pd.DataFrame()).copy()
    t_df_d13 = _filter_t_cultivation_d13_for_plots(t_df)
    t_for_summary = _filter_t_cultivation_main_compile_setting(t_df_d13)
    t_full = (
        _build_layer_frames(t_for_summary).get("full_trotter", pd.DataFrame()).copy()
    )
    t_full_ablation = (
        _build_layer_frames(t_df_d13).get("full_trotter", pd.DataFrame()).copy()
    )

    star_setting = STAR_VS_T_RUNTIME_SETTING
    star_ref = star_full[_setting_filter(star_full, star_setting)].copy()
    star_ref_col = _filter_col_based(star_ref)
    t_full_col = _filter_col_based(t_full)

    def _geomean_ratio(num: np.ndarray, den: np.ndarray) -> float:
        r = np.clip(num, eps, None) / np.clip(den, eps, None)
        return float(np.exp(np.mean(np.log(np.clip(r, eps, None)))))

    def _geomean_actual_over_expected_star_ft(
        qubits: pd.Series, runtime: pd.Series
    ) -> float | None:
        m = pd.DataFrame(
            {
                "n": pd.to_numeric(qubits, errors="coerce"),
                "a": pd.to_numeric(runtime, errors="coerce"),
            }
        ).dropna()
        if m.empty:
            return None
        m = m.sort_values("n")
        n_list = [int(v) for v in m["n"].tolist()]
        act = np.clip(m["a"].to_numpy(dtype=float), eps, None)
        exp_arr = np.clip(
            np.asarray(
                _get_theoretical_lower_bound(n_list)["full_trotter"],
                dtype=float,
            ),
            eps,
            None,
        )
        if exp_arr.size != act.size:
            return None
        return _geomean_ratio(act, exp_arr)

    def _geomean_actual_over_expected_t_ft(
        qubits: pd.Series,
        runtime: pd.Series,
        code_distance: int,
        fidelity_target: float,
    ) -> float | None:
        m = pd.DataFrame(
            {
                "n": pd.to_numeric(qubits, errors="coerce"),
                "a": pd.to_numeric(runtime, errors="coerce"),
            }
        ).dropna()
        if m.empty:
            return None
        m = m.sort_values("n")
        n_list = [int(v) for v in m["n"].tolist()]
        act = np.clip(m["a"].to_numpy(dtype=float), eps, None)
        exp_arr = np.clip(
            np.asarray(
                _get_theoretical_lower_bound_t_cultivation(
                    n_list,
                    code_distance=int(code_distance),
                    fidelity_target=float(fidelity_target),
                )["full_trotter"],
                dtype=float,
            ),
            eps,
            None,
        )
        if exp_arr.size != act.size:
            return None
        return _geomean_ratio(act, exp_arr)

    def _banner(title: str, *subtitle_lines: str) -> None:
        bar = "=" * 72
        print("\n" + bar)
        print(title)
        for line in subtitle_lines:
            print(line)
        print(bar)

    def _ln(msg: str) -> None:
        print(f"  {msg}")

    star_setting_desc = "STAR main compile (col_based, sync. execution)"
    summary_triples = _runtime_summary_t_triples()
    t_triple_note = (
        "T-cultivation lines: "
        + ", ".join(
            f"d={int(cd)}, nf={int(fps)}, LER={float(ft):.0e}"
            for cd, ft, fps in summary_triples
        )
        if summary_triples
        else "(no T summary triples in this run)"
    )

    star_col_agg = pd.DataFrame()
    if not star_ref_col.empty and "code_distance" in star_ref_col.columns:
        star_col_agg = star_ref_col.groupby(
            ["code_distance", "n_qubits", "n_aods"], as_index=False
        ).agg(runtime=("total_time", "mean"))

    def _triple_key_match(
        d: int | float, ler: float, fps: int | float
    ) -> tuple[int, float, int] | None:
        for scd, sft, sfps in summary_triples:
            if int(scd) != int(d) or int(sfps) != int(fps):
                continue
            if np.isclose(float(ler), float(sft), rtol=0.0, atol=0.0):
                return (int(scd), float(sft), int(sfps))
        return None

    # ----- (1) Overall: AOD values in RUNTIME_SUMMARY_OVERALL_AODS, col_based -----
    aods_str = ", ".join(str(int(a)) for a in RUNTIME_SUMMARY_OVERALL_AODS)
    _banner(
        "(1) OVERALL TIME COMPARISON",
        "full_trotter · col_based · " f"AOD ∈ {{{aods_str}}} · {star_setting_desc}",
        "STAR code distances: 7, 9 · T-cultivation: d = 9, 13 at LER = 1e-8 "
        "(subset shown if d=13 omitted in this run)",
    )

    tg = pd.DataFrame()
    if not t_full_col.empty and "code_distance" in t_full_col.columns:
        tg = t_full_col.groupby(
            [
                "code_distance",
                "n_qubits",
                "n_aods",
                "fidelity_target",
                "factory_physical_size",
            ],
            as_index=False,
        ).agg(runtime=("total_time", "mean"))

    def _t_runtime_rows(cd: int, ft: float, fps: int) -> pd.DataFrame:
        if tg.empty:
            return pd.DataFrame()
        cdi = pd.to_numeric(tg["code_distance"], errors="coerce")
        fti = pd.to_numeric(tg["fidelity_target"], errors="coerce")
        fpsi = pd.to_numeric(tg["factory_physical_size"], errors="coerce")
        return tg[
            (cdi == int(cd))
            & np.isclose(fti, float(ft), rtol=0.0, atol=1e-20)
            & (fpsi == int(fps))
        ][["n_qubits", "n_aods", "runtime"]].copy()

    def _pair_ratio_one_aod(
        num: pd.DataFrame, den: pd.DataFrame, label: str, aod: int
    ) -> None:
        m = num.merge(
            den,
            on=["n_qubits", "n_aods"],
            how="inner",
            suffixes=("_num", "_den"),
        )
        m = m[pd.to_numeric(m["n_aods"], errors="coerce") == int(aod)]
        if m.empty:
            _ln(f"{label}: unavailable at AOD={int(aod)}")
            return
        g = _geomean_ratio(m["runtime_num"].to_numpy(), m["runtime_den"].to_numpy())
        _ln(f"{label}: geomean ratio = {g:.4f}x (AOD={int(aod)})")

    t9 = _t_runtime_rows(9, RUNTIME_SUMMARY_T_LER, 2)
    t13 = _t_runtime_rows(13, RUNTIME_SUMMARY_T_LER, 4)
    has_t_d9 = any(int(cd) == 9 for cd, _ft, _fps in summary_triples)
    has_t_d13 = any(int(cd) == 13 for cd, _ft, _fps in summary_triples)
    if not (has_t_d9 and has_t_d13 and not t9.empty and not t13.empty):
        if not has_t_d13:
            _ln(
                "T-cultivation d=9 vs d=13: skipped (d=13 omitted in this run; "
                "rerun with include_t_cultivation_d13=True for both distances)"
            )
        else:
            _ln("T-cultivation d=9 vs d=13: unavailable (missing rows in CSV)")

    if star_col_agg.empty:
        _ln("STAR d=9 vs d=7: unavailable (no col_based STAR rows)")

    for aod_eff in RUNTIME_SUMMARY_OVERALL_AODS:
        aod_i = int(aod_eff)
        _ln(f"── AOD={aod_i} ──")
        if not star_col_agg.empty:
            d9 = star_col_agg[star_col_agg["code_distance"] == 9][
                ["n_qubits", "n_aods", "runtime"]
            ].rename(columns={"runtime": "rt9"})
            d7 = star_col_agg[star_col_agg["code_distance"] == 7][
                ["n_qubits", "n_aods", "runtime"]
            ].rename(columns={"runtime": "rt7"})
            m_sd = d9.merge(d7, on=["n_qubits", "n_aods"], how="inner")
            m2 = m_sd[pd.to_numeric(m_sd["n_aods"], errors="coerce") == aod_i]
            if m2.empty:
                _ln(
                    f"STAR d=9 vs d=7 (geomean runtime ratio): unavailable at AOD={aod_i}"
                )
            else:
                g = _geomean_ratio(m2["rt9"].to_numpy(), m2["rt7"].to_numpy())
                _ln(f"STAR d=9 vs d=7: geomean ratio = {g:.4f}x (AOD={aod_i})")

        if has_t_d9 and has_t_d13 and not t9.empty and not t13.empty:
            _pair_ratio_one_aod(t9, t13, "T-cultivation d=9 vs d=13 (LER=1e-8)", aod_i)

        if not star_col_agg.empty:
            s9 = star_col_agg[star_col_agg["code_distance"] == 9][
                ["n_qubits", "n_aods", "runtime"]
            ].copy()
            _pair_ratio_one_aod(t9, s9, "Cross-architecture T d=9 / STAR d=9", aod_i)
            if has_t_d13 and not t13.empty:
                _pair_ratio_one_aod(
                    t13, s9, "Cross-architecture T d=13 / STAR d=9", aod_i
                )
                _pair_ratio_one_aod(
                    s9, t13, "Cross-architecture STAR d=9 / T d=13", aod_i
                )

    # ----- (2) AOD -----
    _banner(
        "(2) AOD COMPARISON",
        "Geometric mean runtime ratio AOD_lo / AOD_hi at matched n_qubits "
        "(>1 means slower at higher AOD count).",
        "Also: geomean(actual / expected) for full_trotter vs in-code lower bounds.",
        "STAR d ∈ {7, 9} · T-cultivation: " + t_triple_note,
    )

    if not star_col_agg.empty:
        for d in RUNTIME_SUMMARY_STAR_CODE_DISTANCES:
            sd = star_col_agg[star_col_agg["code_distance"] == int(d)]
            if sd.empty:
                continue
            aods = sorted(
                pd.to_numeric(sd["n_aods"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            for a1, a2 in zip(aods[:-1], aods[1:]):
                rt_lo = sd[sd["n_aods"] == a1][["n_qubits", "runtime"]].rename(
                    columns={"runtime": "r1"}
                )
                rt_hi = sd[sd["n_aods"] == a2][["n_qubits", "runtime"]].rename(
                    columns={"runtime": "r2"}
                )
                m = rt_lo.merge(rt_hi, on="n_qubits", how="inner")
                if m.empty:
                    continue
                g = _geomean_ratio(m["r1"].to_numpy(), m["r2"].to_numpy())
                _ln(
                    f"STAR d={int(d)}: AOD {int(a1)}/{int(a2)} runtime ratio = {g:.4f}x"
                )
                s1 = sd[sd["n_aods"] == a1]
                s2 = sd[sd["n_aods"] == a2]
                g1 = _geomean_actual_over_expected_star_ft(
                    s1["n_qubits"], s1["runtime"]
                )
                g2 = _geomean_actual_over_expected_star_ft(
                    s2["n_qubits"], s2["runtime"]
                )
                if g1 is not None:
                    _ln(
                        f"        AOD={int(a1)}: geomean(actual / expected full_trotter) "
                        f"= {g1:.4f}x"
                    )
                else:
                    _ln(
                        f"        AOD={int(a1)}: actual / expected full_trotter: unavailable"
                    )
                if g2 is not None:
                    _ln(
                        f"        AOD={int(a2)}: geomean(actual / expected full_trotter) "
                        f"= {g2:.4f}x"
                    )
                else:
                    _ln(
                        f"        AOD={int(a2)}: actual / expected full_trotter: unavailable"
                    )

    if not tg.empty and all(
        c in t_full.columns
        for c in ("code_distance", "fidelity_target", "factory_physical_size")
    ):
        tga = t_full_col.groupby(
            [
                "code_distance",
                "fidelity_target",
                "factory_physical_size",
                "n_qubits",
                "n_aods",
            ],
            as_index=False,
        ).agg(runtime=("total_time", "mean"))
        for (d, ler, fps), grp in tga.groupby(
            ["code_distance", "fidelity_target", "factory_physical_size"]
        ):
            if _triple_key_match(d, float(ler), int(fps)) is None:
                continue
            aods = sorted(
                pd.to_numeric(grp["n_aods"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            for a1, a2 in zip(aods[:-1], aods[1:]):
                rt_lo = grp[grp["n_aods"] == a1][["n_qubits", "runtime"]].rename(
                    columns={"runtime": "r1"}
                )
                rt_hi = grp[grp["n_aods"] == a2][["n_qubits", "runtime"]].rename(
                    columns={"runtime": "r2"}
                )
                m = rt_lo.merge(rt_hi, on="n_qubits", how="inner")
                if m.empty:
                    continue
                g = _geomean_ratio(m["r1"].to_numpy(), m["r2"].to_numpy())
                _ln(
                    f"T d={int(d)}, LER={float(ler):.0e}, nf={int(fps)}: "
                    f"AOD {int(a1)}/{int(a2)} runtime ratio = {g:.4f}x"
                )
                t1 = grp[grp["n_aods"] == a1]
                t2 = grp[grp["n_aods"] == a2]
                gt1 = _geomean_actual_over_expected_t_ft(
                    t1["n_qubits"], t1["runtime"], int(d), float(ler)
                )
                gt2 = _geomean_actual_over_expected_t_ft(
                    t2["n_qubits"], t2["runtime"], int(d), float(ler)
                )
                if gt1 is not None:
                    _ln(
                        f"        AOD={int(a1)}: geomean(actual / expected full_trotter) "
                        f"= {gt1:.4f}x"
                    )
                else:
                    _ln(
                        f"        AOD={int(a1)}: actual / expected full_trotter: unavailable"
                    )
                if gt2 is not None:
                    _ln(
                        f"        AOD={int(a2)}: geomean(actual / expected full_trotter) "
                        f"= {gt2:.4f}x"
                    )
                else:
                    _ln(
                        f"        AOD={int(a2)}: actual / expected full_trotter: unavailable"
                    )

    # ----- (3) Micro-architecture -----
    _banner(
        "(3) MICRO-ARCHITECTURE COMPARISON",
        "Geometric mean total_time: alternate placement / col_based "
        "(matched n_qubits).",
        "STAR d ∈ {7, 9} · T-cultivation: " + t_triple_note,
    )

    star_dist_set = set(int(x) for x in RUNTIME_SUMMARY_STAR_CODE_DISTANCES)
    if not star_ref.empty:
        sm = star_ref.groupby(
            ["code_distance", "placement", "n_aods", "n_qubits"], as_index=False
        ).agg(runtime=("total_time", "mean"))
        for placement_name in ["seperate_region_row", "checkerboard"]:
            for d in sorted(
                pd.to_numeric(sm["code_distance"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            ):
                if int(d) not in star_dist_set:
                    continue
                for aod in sorted(
                    pd.to_numeric(sm["n_aods"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                ):
                    p = sm[
                        (sm["placement"] == placement_name)
                        & (
                            pd.to_numeric(sm["code_distance"], errors="coerce")
                            == int(d)
                        )
                        & (pd.to_numeric(sm["n_aods"], errors="coerce") == int(aod))
                    ][["n_qubits", "runtime"]].rename(columns={"runtime": "rp"})
                    c = sm[
                        (sm["placement"] == "col_based")
                        & (
                            pd.to_numeric(sm["code_distance"], errors="coerce")
                            == int(d)
                        )
                        & (pd.to_numeric(sm["n_aods"], errors="coerce") == int(aod))
                    ][["n_qubits", "runtime"]].rename(columns={"runtime": "rc"})
                    m = p.merge(c, on="n_qubits", how="inner")
                    if m.empty:
                        continue
                    g = _geomean_ratio(m["rp"].to_numpy(), m["rc"].to_numpy())
                    _ln(
                        f"STAR {placement_name}/col_based · d={int(d)} · "
                        f"AOD={int(aod)}: ratio = {g:.4f}x"
                    )

    t_mic_need = (
        "placement",
        "total_time",
        "n_qubits",
        "n_aods",
        "code_distance",
        "fidelity_target",
        "factory_physical_size",
    )
    if not t_full.empty and all(c in t_full.columns for c in t_mic_need):
        t_mic = _filter_t_cultivation_d13_for_plots(t_full.copy())
        if not t_mic.empty:
            for cd, ft, fps in summary_triples:
                sub = t_mic.loc[_mask_t_cultivation_triple(t_mic, cd, ft, fps)].copy()
                if sub.empty:
                    continue
                tm = sub.groupby(
                    ["code_distance", "placement", "n_aods", "n_qubits"],
                    as_index=False,
                ).agg(runtime=("total_time", "mean"))
                setting_ctx = f"nf={int(fps)}, LER={ft:.0e}"
                for placement_name in ["seperate_region_row", "checkerboard"]:
                    for d in sorted(
                        pd.to_numeric(tm["code_distance"], errors="coerce")
                        .dropna()
                        .astype(int)
                        .unique()
                    ):
                        for aod in sorted(
                            pd.to_numeric(tm["n_aods"], errors="coerce")
                            .dropna()
                            .astype(int)
                            .unique()
                        ):
                            p = tm[
                                (tm["placement"] == placement_name)
                                & (
                                    pd.to_numeric(tm["code_distance"], errors="coerce")
                                    == int(d)
                                )
                                & (
                                    pd.to_numeric(tm["n_aods"], errors="coerce")
                                    == int(aod)
                                )
                            ][["n_qubits", "runtime"]].rename(columns={"runtime": "rp"})
                            c = tm[
                                (tm["placement"] == "col_based")
                                & (
                                    pd.to_numeric(tm["code_distance"], errors="coerce")
                                    == int(d)
                                )
                                & (
                                    pd.to_numeric(tm["n_aods"], errors="coerce")
                                    == int(aod)
                                )
                            ][["n_qubits", "runtime"]].rename(columns={"runtime": "rc"})
                            m = p.merge(c, on="n_qubits", how="inner")
                            if m.empty:
                                continue
                            g = _geomean_ratio(m["rp"].to_numpy(), m["rc"].to_numpy())
                            _ln(
                                f"T {placement_name}/col_based · d={int(d)} · "
                                f"AOD={int(aod)} ({setting_ctx}): ratio = {g:.4f}x"
                            )

    # ----- (4) Ablation -----
    _banner(
        "(4) ABLATION STUDY",
        "STAR: mean % runtime reduction vs vanilla (matched configs).",
        "STAR d ∈ {7, 9} · T-cultivation compile ablation: " + t_triple_note,
    )

    ablation_labels = STAR_SETTING_LABELS
    if not star_full.empty:
        star_ablation = star_full.copy()
        base = star_ablation[
            _setting_filter(star_ablation, STAR_COL_BASED_VANILLA_SETTING)
        ].copy()
        bg = base.groupby(
            ["placement", "code_distance", "n_aods", "n_qubits"],
            as_index=False,
        ).agg(rt_base=("total_time", "mean"))
        for idx in range(len(SETTINGS)):
            setting = SETTINGS[idx]
            if setting[0] != "col_based" or setting == STAR_COL_BASED_VANILLA_SETTING:
                continue
            s = star_ablation[_setting_filter(star_ablation, setting)].copy()
            if s.empty:
                continue
            sg_ab = s.groupby(
                ["placement", "code_distance", "n_aods", "n_qubits"],
                as_index=False,
            ).agg(rt_set=("total_time", "mean"))
            m = bg.merge(
                sg_ab,
                on=["placement", "code_distance", "n_aods", "n_qubits"],
                how="inner",
            )
            if m.empty:
                continue
            _ln(f"STAR · {ablation_labels[idx]}:")
            for placement in sorted(pd.unique(m["placement"])):
                m_place = m[m["placement"] == placement]
                if m_place.empty:
                    continue
                for distance in sorted(
                    pd.to_numeric(m_place["code_distance"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                ):
                    if int(distance) not in star_dist_set:
                        continue
                    m_distance = m_place[
                        pd.to_numeric(m_place["code_distance"], errors="coerce")
                        == int(distance)
                    ]
                    aod_values = sorted(
                        set(
                            int(v)
                            for v in pd.to_numeric(
                                m_distance["n_aods"], errors="coerce"
                            )
                            .dropna()
                            .astype(int)
                            .unique()
                            .tolist()
                        ).intersection({1, 5})
                    )
                    for aod in aod_values:
                        m_da = m_distance[
                            pd.to_numeric(m_distance["n_aods"], errors="coerce")
                            == int(aod)
                        ]
                        if m_da.empty:
                            continue
                        base_vals = np.clip(m_da["rt_base"].to_numpy(), eps, None)
                        set_vals = np.clip(m_da["rt_set"].to_numpy(), eps, None)
                        pct = float(np.mean((base_vals - set_vals) / base_vals) * 100.0)
                        _ln(
                            f"  {placement} · d={int(distance)} · AOD={int(aod)}: "
                            f"avg improvement vs vanilla = {pct:+.2f}%"
                        )

    _ln("T-cultivation compile ablation (col_based, vs vanilla compile row):")
    t_cab_cols = (
        "placement",
        "total_time",
        "n_qubits",
        "n_aods",
        "code_distance",
        "fidelity_target",
        "factory_physical_size",
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
    )
    if t_full_ablation.empty or not all(
        c in t_full_ablation.columns for c in t_cab_cols
    ):
        _ln("  (skipped: missing placement/compile columns or empty full_trotter)")
    else:
        t_work = t_full_ablation.copy()
        if "placement" in t_work.columns:
            t_work = t_work[
                t_work["placement"].astype(str).str.strip() == "col_based"
            ].copy()
        if not summary_triples:
            _ln("  (skipped: no T summary triples in this run)")
        else:
            tr0, dm0, rs0 = _T_COMPILE_ABLATION_GRID[0]
            any_t_cab = False
            for idx in range(
                1, min(len(_T_COMPILE_ABLATION_GRID), len(_T_COMPILE_ABLATION_LABELS))
            ):
                trs, dms, rss = _T_COMPILE_ABLATION_GRID[idx]
                printed_label = False
                for cd, ft, fps in summary_triples:
                    base_mask = _mask_t_cultivation_triple(
                        t_work, cd, ft, fps
                    ) & _mask_t_cultivation_compile(t_work, tr0, dm0, rs0)
                    base_sub = t_work.loc[base_mask].copy()
                    if base_sub.empty:
                        continue
                    bg = base_sub.groupby(["n_aods", "n_qubits"], as_index=False).agg(
                        rt_base=("total_time", "mean")
                    )
                    s_mask = _mask_t_cultivation_triple(
                        t_work, cd, ft, fps
                    ) & _mask_t_cultivation_compile(t_work, trs, dms, rss)
                    s_sub = t_work.loc[s_mask].copy()
                    if s_sub.empty:
                        continue
                    sg_cab = s_sub.groupby(["n_aods", "n_qubits"], as_index=False).agg(
                        rt_set=("total_time", "mean")
                    )
                    m = bg.merge(sg_cab, on=["n_aods", "n_qubits"], how="inner")
                    if m.empty:
                        continue
                    if not printed_label:
                        _ln(f"  {_T_COMPILE_ABLATION_LABELS[idx]}:")
                        printed_label = True
                    aod_values = sorted(
                        set(
                            int(v)
                            for v in pd.to_numeric(m["n_aods"], errors="coerce")
                            .dropna()
                            .astype(int)
                            .unique()
                            .tolist()
                        ).intersection({1, 5})
                    )
                    for aod in aod_values:
                        m_da = m[
                            pd.to_numeric(m["n_aods"], errors="coerce") == int(aod)
                        ]
                        if m_da.empty:
                            continue
                        base_vals = np.clip(m_da["rt_base"].to_numpy(), eps, None)
                        set_vals = np.clip(m_da["rt_set"].to_numpy(), eps, None)
                        pct = float(np.mean((base_vals - set_vals) / base_vals) * 100.0)
                        _ln(
                            f"    d={int(cd)}, AOD={int(aod)}, "
                            f"nf={int(fps)}, LER={ft:.0e}: "
                            f"avg improvement vs vanilla = {pct:+.2f}%"
                        )
                        any_t_cab = True
            if not any_t_cab:
                _ln("  (no overlapping vanilla vs compile rows for col_based)")


def _print_t_cultivation_aod_vs_theoretical_improvement(
    t_layers: dict[str, pd.DataFrame],
) -> None:
    """Print improvement (overhead) of each T-cultivation AOD versus theoretical lower bound."""
    print("\n  [T-cultivation AOD vs Theoretical Lower Bound]")
    eps = 1e-15

    for round_name, layer_df in t_layers.items():
        if layer_df.empty:
            continue

        x_vals = sorted(
            pd.to_numeric(layer_df["n_qubits"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if not x_vals:
            continue

        bounds = _get_theoretical_lower_bound_t_cultivation(
            x_vals,
            code_distance=None,
            fidelity_target=None,
        )
        if round_name == "full_trotter":
            theo_vals = bounds["full_trotter"]
        elif round_name == "zz_layers":
            theo_vals = bounds["zz_layer"]
        elif round_name == "x_layer":
            theo_vals = bounds["x_layer"]
        else:
            continue

        placement_values = ["all"]
        if "placement" in layer_df.columns:
            placement_values = sorted(
                str(v) for v in pd.unique(layer_df["placement"]) if pd.notna(v)
            )

        for placement in placement_values:
            if placement == "all":
                layer_subset = layer_df
            else:
                layer_subset = layer_df[layer_df["placement"] == placement]

            if layer_subset.empty:
                continue

            aod_values = sorted(
                pd.to_numeric(layer_subset["n_aods"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            print(f"    placement={placement}:")
            for aod in aod_values:
                sub_aod = layer_subset[layer_subset["n_aods"] == aod]
                if sub_aod.empty:
                    continue

                mean_times = (
                    pd.to_numeric(sub_aod["total_time"], errors="coerce")
                    .dropna()
                    .tolist()
                )
                if not mean_times:
                    continue

                geomean_actual = float(
                    np.exp(np.mean(np.log(np.maximum(mean_times, eps))))
                )
                geomean_theo = float(
                    np.exp(np.mean(np.log(np.maximum(theo_vals, eps))))
                )
                ratio = geomean_actual / (geomean_theo + eps)
                print(
                    f"      {_format_round_title(round_name)} AOD={aod} / theoretical: {ratio:.4f}x"
                )


def process_t_cultivation_runtime_comparison(
    star_csv_file: str,
    t_cultivation_csv_file: str,
    output_dir: str,
    *,
    include_t_cultivation_d13: bool = False,
    emit_runtime_console: bool = True,
    verbose_runtime_plots: bool = True,
):
    """Generate runtime comparison between STAR and T-cultivation profiling results.

    STAR vs T, T multi-AOD, architecture placement panels, and the four-section console
    summary use T-cultivation rows matching ``T_CULTIVATION_MAIN_COMPILE_SETTING`` only
    (see ``evaluation_fidelity_t_cultivation.MAIN_COMPILE_SETTING``). Compile ablation
    figures still use the full compile grid from the CSV. Column-based slices are used
    where noted in each plot helper.

    When ``include_t_cultivation_d13`` is True, T-cultivation ``d=13`` settings
    are included in every figure and in the four-section console summary. The d=13
    toggle is
    threaded by temporarily overriding the module-level
    ``SHOW_T_CULTIVATION_D13`` flag so all helpers consistently see the same
    setting list and data subset.

    Set ``emit_runtime_console=False`` when running a preliminary pass (e.g. without
    ``d=13``) so only one call prints sections (1)–(4). Set ``verbose_runtime_plots=False``
    to silence per-figure ``Saved:`` / skip messages from this pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    star_df = pd.read_csv(star_csv_file, engine="python", on_bad_lines="skip")
    t_df = pd.read_csv(t_cultivation_csv_file, engine="python", on_bad_lines="skip")
    t_df = _dedupe_duplicate_columns(t_df)
    star_df = _coerce_result_cols_numeric(star_df)
    star_df = _normalize_config_types(star_df)
    t_df = _coerce_result_cols_numeric(t_df)
    t_df = _normalize_config_types(t_df)

    global SHOW_T_CULTIVATION_D13
    previous_show_d13 = SHOW_T_CULTIVATION_D13
    SHOW_T_CULTIVATION_D13 = bool(include_t_cultivation_d13)
    try:
        star_layers = _build_layer_frames(star_df)
        t_df_d13 = _filter_t_cultivation_d13_for_plots(t_df)
        t_layers_ablation = {
            name: layer_df.copy()
            for name, layer_df in _build_layer_frames(t_df_d13).items()
        }
        t_df_main_d13 = _filter_t_cultivation_d13_for_plots(
            _filter_t_cultivation_main_compile_setting(t_df)
        )
        t_layers = {
            name: layer_df.copy()
            for name, layer_df in _build_layer_frames(t_df_main_d13).items()
        }

        _plot_star_vs_t_cultivation_best(
            star_layers,
            t_layers,
            output_dir,
            target_aods=[1, 5],
            t_placement="col_based",
            log_console=verbose_runtime_plots,
        )
        _plot_t_cultivation_multi_aod(
            t_layers, output_dir, verbose=verbose_runtime_plots
        )
        _plot_t_cultivation_compile_ablation(
            t_layers_ablation,
            os.path.join(output_dir, "ablation"),
            placement="col_based",
            verbose=verbose_runtime_plots,
        )
        _plot_t_cultivation_architecture_placement_execution(
            t_layers,
            os.path.join(output_dir, "ablation"),
            verbose=verbose_runtime_plots,
        )
        if emit_runtime_console:
            _print_requested_runtime_improvements(star_df, t_df)
    finally:
        SHOW_T_CULTIVATION_D13 = previous_show_d13

    if verbose_runtime_plots:
        print("Saved T-cultivation runtime comparison plots to", output_dir)


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------


if __name__ == "__main__":
    full_trotter_csv = (
        "output/evaluation/fidelity/star_full_trotter_profiling_results.csv"
    )
    full_trotter_output_dir = "output/analysis_plots/full_trotter"
    process_full_trotter_csv(full_trotter_csv, full_trotter_output_dir)

    t_cultivation_profiling_csv = (
        "output/evaluation/fidelity/t_cultivation_fidelity_profiling_results.csv"
    )
    runtime_compare_output_dir = "output/analysis_plots/t_cultivation_runtime"
    runtime_compare_output_dir_with_d13 = (
        "output/analysis_plots/t_cultivation_runtime_with_d13"
    )
    if os.path.exists(t_cultivation_profiling_csv):
        process_t_cultivation_runtime_comparison(
            full_trotter_csv,
            t_cultivation_profiling_csv,
            runtime_compare_output_dir,
            emit_runtime_console=False,
            verbose_runtime_plots=False,
        )
        process_t_cultivation_runtime_comparison(
            full_trotter_csv,
            t_cultivation_profiling_csv,
            runtime_compare_output_dir_with_d13,
            include_t_cultivation_d13=True,
            verbose_runtime_plots=False,
        )
