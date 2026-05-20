# analyze_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import math
import os
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz


_FIG_FONT_SIZE = 28


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


SETTINGS = [
    (True, "matching", 0, False, False),  # vanilla
    (False, "matching", 0, False, False),  # optimized return
    (
        False,
        "matching",
        1,
        False,
        False,
    ),  # optimized return + skip partial RUS
    (
        False,
        "matching",
        2,
        False,
        False,
    ),  # optimized return + skip whole RUS
    (
        False,
        "matching",
        2,
        True,
        False,
    ),  # optimized decompose return + skip whole RUS
    (
        False,
        "matching",
        0,
        False,
        True,
    ),  # optimized return  + decompose move + asynchronous RUS
    # (
    #     False,
    #     "matching",
    #     1,
    #     False,
    #     True,
    # ),  # optimized return+ skip partial RUS + decompose move + asynchronous RUS
    (
        False,
        "matching",
        2,
        False,
        True,
    ),  # optimized return+ skip whole RUS + decompose move + asynchronous RUS
]

# Paper STAR ablation: omit this SETTINGS index from curves (Opt. return + part. RUS).
STAR_ABLATION_EXCLUDE_SETTING_IDX = 2

# Configuration columns (experimental settings)
CONFIG_COLS = [
    "n_qubits",
    "qubit_cols",
    "qubit_rows",
    "placement",
    "n_aods",
    "consider_skip_rus",
    "tmr_assignment_method",
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
# Paper ``process_full_trotter_csv``: only these STAR code distances get plot folders.
STAR_PAPER_PLOT_DISTANCES: tuple[int, ...] = (7, 9)
# STAR curves on STAR vs T-cultivation combined runtime PDFs (omit d=13 for readability).
STAR_VS_T_RUNTIME_STAR_DISTANCES: tuple[int, ...] = (7, 9)
# Console runtime summary: one readable block (full_trotter, col_based for cross-arch).
RUNTIME_SUMMARY_OVERALL_AODS: tuple[int, ...] = (2, 5)
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

# Combined STAR + T-cultivation 2×2 grid (setting study + AOD comparison).
# Change ``STAR_T_GRID_COMPARISON_CODE_DISTANCE`` when comparing a different logical distance.
STAR_T_GRID_COMPARISON_CODE_DISTANCE: int = 9
STAR_T_GRID_STAR_SETTING_INDEX: int = 4
# AOD band panel: shaded region between these execution-time curves (top / bottom).
STAR_VS_T_AOD_BAND_TOP: int = 1
STAR_VS_T_AOD_BAND_BOTTOM: int = 5
# Overall single-AOD panel (no band): measured curves at this AOD only.
STAR_VS_T_OVERALL_SINGLE_AOD: int = STAR_VS_T_AOD_BAND_TOP
# Combined 2×2 STAR/T grid: setting study (left) uses this AOD.
STAR_T_SETTING_STUDY_AOD: int = STAR_VS_T_OVERALL_SINGLE_AOD
# STAR col_based curves in the setting-study panel (four compiler optimizations).
STAR_SETTING_STUDY_COL_BASED_SETTING_IDXS: tuple[int, ...] = (1, 3, 4, 6)
T_SETTING_STUDY_VANILLA_PLACEMENT: str = "seperate_region_row"


# --- Legend styling (aligned with ``compare_fidelity.py`` overall infidelity figure) ---
_RUNTIME_LEGEND_STAR_DISTANCE_COLORS = [
    "#F1CE63",
    "#F28E2B",
    "#D55E00",
    "#A63603",
]
_RUNTIME_LEGEND_T_DISTANCE_COLORS = [
    "#B2DF8A",
    "#59A14F",
    "#1B7837",
    "#00441B",
]
_RUNTIME_LEGEND_METHOD_STYLES = {
    "STAR": {"marker": "o", "linestyle": "-"},
    "T-cultivation": {"marker": "^", "linestyle": "--"},
}
_RUNTIME_LEGEND_METHOD_COLORS = {
    "STAR": "#F28E2B",
    "T-cultivation": "#59A14F",
}


def _build_runtime_star_vs_t_distance_colors(
    star_distances: list[int],
    t_line_settings: list[tuple[int, float, int]],
) -> dict[str, dict[int, str]]:
    """STAR / T distance hues in the same spirit as ``compare_fidelity._build_architecture_distance_colors``."""
    t_ds = sorted({int(cd) for cd, _ft, _fps in t_line_settings})
    merged = sorted(set(int(d) for d in star_distances) | set(t_ds))
    return {
        "STAR": {
            d: _RUNTIME_LEGEND_STAR_DISTANCE_COLORS[
                i % len(_RUNTIME_LEGEND_STAR_DISTANCE_COLORS)
            ]
            for i, d in enumerate(merged)
        },
        "T-cultivation": {
            d: _RUNTIME_LEGEND_T_DISTANCE_COLORS[
                i % len(_RUNTIME_LEGEND_T_DISTANCE_COLORS)
            ]
            for i, d in enumerate(merged)
        },
    }


def _add_runtime_star_vs_t_legends(
    fig_or_ax,
    architecture_distance_colors: dict[str, dict[int, str]],
    *,
    star_distances: list[int],
    t_line_settings: list[tuple[int, float, int]],
    legend_position: str = "right",
) -> None:
    """Legend for STAR vs T runtime figures.

    ``legend_position="inset"``: on-axes boxes like ``compare_fidelity._add_overall_legends``
    (upper left: expected time + methods; lower right: code distance ``d``).
    ``"right"`` / ``"bottom"``: figure-level legend outside the axes.
    """
    star_colors = architecture_distance_colors.get("STAR", {})
    t_colors = architecture_distance_colors.get("T-cultivation", {})
    t_ds = sorted({int(cd) for cd, _ft, _fps in t_line_settings})
    distances = sorted(set(int(d) for d in star_distances) | set(t_ds))
    star_ms = _RUNTIME_LEGEND_METHOD_STYLES["STAR"]
    t_ms = _RUNTIME_LEGEND_METHOD_STYLES["T-cultivation"]

    arch_handles: list = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=2.2,
            markersize=0,
        ),
    ]
    arch_labels: list[str] = ["Expected time"]
    arch_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=_RUNTIME_LEGEND_METHOD_COLORS["STAR"],
                linewidth=2.5,
                markersize=9,
                marker=star_ms["marker"],
                linestyle=star_ms["linestyle"],
            ),
            Line2D(
                [0],
                [0],
                color=_RUNTIME_LEGEND_METHOD_COLORS["T-cultivation"],
                linewidth=2.5,
                markersize=9,
                marker=t_ms["marker"],
                linestyle=t_ms["linestyle"],
            ),
        ]
    )
    arch_labels.extend(["STAR", "T-cultivation"])

    distance_handles = [
        (
            Line2D(
                [0],
                [0],
                color=star_colors.get(int(d), "#333333"),
                linewidth=2.5,
                markersize=9,
                marker=star_ms["marker"],
                linestyle=star_ms["linestyle"],
            ),
            Line2D(
                [0],
                [0],
                color=t_colors.get(int(d), "#333333"),
                linewidth=2.5,
                markersize=9,
                marker=t_ms["marker"],
                linestyle=t_ms["linestyle"],
            ),
        )
        for d in distances
    ]
    distance_labels = [str(d) for d in distances]

    legend_fs = _FIG_FONT_SIZE
    if legend_position == "inset":
        method_legend = fig_or_ax.legend(
            handles=arch_handles,
            labels=arch_labels,
            loc="upper left",
            frameon=True,
            fontsize=legend_fs,
        )
        fig_or_ax.add_artist(method_legend)
        if distances:
            fig_or_ax.legend(
                handles=distance_handles,
                labels=distance_labels,
                title="d",
                loc="lower right",
                frameon=True,
                fontsize=legend_fs,
                title_fontsize=legend_fs,
                handler_map={tuple: HandlerTuple(ndivide=None)},
            )
        return

    fig = fig_or_ax
    if legend_position == "bottom":
        arch_legend = fig.legend(
            handles=arch_handles,
            labels=arch_labels,
            title="Architecture",
            loc="upper center",
            bbox_to_anchor=(0.24, -0.0),
            ncol=1,
            frameon=True,
            fontsize=legend_fs,
            title_fontsize=legend_fs,
        )
        fig.add_artist(arch_legend)
        fig.legend(
            handles=distance_handles,
            labels=distance_labels,
            title="d",
            loc="upper center",
            bbox_to_anchor=(0.74, -0.0),
            ncol=1,
            frameon=True,
            fontsize=legend_fs,
            title_fontsize=legend_fs,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )
        return

    arch_legend = fig.legend(
        handles=arch_handles,
        labels=arch_labels,
        title="Architecture",
        loc="center left",
        bbox_to_anchor=(0.93, 0.72),
        frameon=True,
        fontsize=legend_fs,
        title_fontsize=legend_fs,
    )
    fig.add_artist(arch_legend)
    fig.legend(
        handles=distance_handles,
        labels=distance_labels,
        title="d",
        loc="center left",
        bbox_to_anchor=(0.93, 0.28),
        frameon=True,
        fontsize=legend_fs,
        title_fontsize=legend_fs,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )


def _setting_filter(df: pd.DataFrame, setting: tuple) -> pd.Series:
    return (
        (df["trivial_return"] == setting[0])
        & (df["tmr_assignment_method"] == setting[1])
        & (df["consider_skip_rus"] == setting[2])
        & (df["decompose_move"] == setting[3])
        & (df["parallel_execution"] == setting[4])
    )


def _get_available_settings(df: pd.DataFrame, max_settings: int = 2) -> list[tuple]:
    """Return up to ``max_settings`` ``SETTINGS`` tuples that have at least one row in ``df``."""
    if df.empty:
        return []
    out: list[tuple] = []
    for setting in SETTINGS:
        if not df[_setting_filter(df, setting)].empty:
            out.append(setting)
            if len(out) >= int(max_settings):
                break
    return out


def _filter_col_based(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "placement" not in df.columns:
        return df
    return df[df["placement"] == "col_based"].copy()


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
    full_trotter_bounds = [x + zz * 8 for x, zz in zip(x_layer_bounds, zz_layer_bounds)]

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
    if code_distance is not None and int(code_distance) < 13:
        stage1_factor = 12.5 * 1 / (1 - 0.7 * 0.7)
    else:
        stage1_factor = 12.5

    # Stage-2: depends on LER target used by T-cultivation setting.
    if code_distance is not None and int(code_distance) <= 13:
        stage2_factor = (1 / 0.4) * (6 + stage1_factor)
    else:
        stage2_factor = (1 / 0.66) * (0.5 + stage1_factor)

    rus_factor = 2.0 + 1.0
    s_gate_overhead = 1.0
    t_cultivation_scale = stage2_factor + rus_factor + s_gate_overhead

    # for x layer bound, we can assume some parallelization and only count half the T gates per qubit
    x_layer_bounds = [t_cultivation_scale * n_t_per_qubit for _ in n_values]

    # for zz layer bound, we can assume more parallelization and only count a quarter of the T gates per qubit
    zz_layer_bounds = [t_cultivation_scale * (n_t_per_qubit / 2) for _ in n_values]

    full_trotter_bounds = [x + 8 * zz for x, zz in zip(x_layer_bounds, zz_layer_bounds)]
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

    plt.xlabel("Number of Qubits")
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
        if label in {"setting_4", "sync", "sync_exec", "sync._exec."}:
            return "Sync. Execution"
        if label in {"setting_6", "async", "async_exec", "async._exec."}:
            return "Async. Execution"
        return str(setting_label)

    if setting_idx == 4:
        return "Sync. Execution"
    if setting_idx == 6:
        return "Async. Execution"
    return "Execution"


def _format_microarch_placement_label(placement: str) -> str:
    normalized = str(placement).strip().lower()
    if normalized == "col_based":
        return "Alternating Column"
    if normalized in {"seperate_region_row", "separate_region_row"}:
        return "Seperate Region"
    return str(placement).replace("_", " ").title()


def _format_nqubit_ticklabels(
    x_vals: list, round_name: str, round_angle_lookup: dict[str, dict[int, int]] | None
) -> list[str]:
    """Tick labels: physical qubit count only (no angle annotations)."""
    _ = (round_name, round_angle_lookup)  # API compatibility with older call sites
    return [str(int(x)) if pd.notna(x) else "" for x in x_vals]


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
    plt.xlabel("Number of Qubits")
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
    # Filter the setting
    setting = SETTINGS[setting_idx]
    df_col = df[
        (df["trivial_return"] == setting[0])
        & (df["tmr_assignment_method"] == setting[1])
        & (df["consider_skip_rus"] == setting[2])
        & (df["decompose_move"] == setting[3])
        & (df["parallel_execution"] == setting[4])
    ].copy()

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

    plt.xlabel("Number of Qubits")
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


# ------------------------------------------------------------
# Fig 2 : controlled placement comparison
# ------------------------------------------------------------
def plot_microarch_comp_setting(df, output_dir, setting_idx=1, tag=""):
    os.makedirs(output_dir, exist_ok=True)
    setting = SETTINGS[setting_idx]
    df_ctrl = df[
        (df["trivial_return"] == setting[0])
        & (df["tmr_assignment_method"] == setting[1])
        & (df["consider_skip_rus"] == setting[2])
        & (df["decompose_move"] == setting[3])
        & (df["parallel_execution"] == setting[4])
    ]

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


# ------------------------------------------------------------
# Ablation study (col-based placement)


def plot_ablation_combined(dfs_dict, output_dir, placement, round_angle_lookup=None):
    """Ablation figures for the paper: full Trotter step only, execution time only.

    Layout is one row with one column per AOD (typically 1 and 5). Separate PDF per
    code distance when ``code_distance`` is present. Omits the "Opt. return + part. RUS"
    curve (see ``STAR_ABLATION_EXCLUDE_SETTING_IDX``). Legend uses two columns.
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

    ablation_labels = [
        "Vanilla",
        "Opt. return",
        "Opt. return + part. RUS",
        "Opt. return + skip whole RUS",
        "Opt. decomp. return + skip whole RUS",
        "Opt. return + async. RUS",
        "Opt. return + skip whole RUS + async. RUS",
    ]

    # Larger typography for ablation figures (two-column paper).
    ablation_font_size = _FIG_FONT_SIZE + 4

    # Paper ablation: full Trotter step only (no ZZ-merge or X-layer columns).
    round_order = ["full_trotter"]
    round_names = [r for r in round_order if r in dfs_dict]
    if not round_names:
        print(f"No full_trotter data in dfs_dict for ablation ({placement}), skipping.")
        return

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
            if label_idx == STAR_ABLATION_EXCLUDE_SETTING_IDX:
                continue
            subdf = df_distance[
                (df_distance["trivial_return"] == setting[0])
                & (df_distance["tmr_assignment_method"] == setting[1])
                & (df_distance["consider_skip_rus"] == setting[2])
                & (df_distance["decompose_move"] == setting[3])
                & (df_distance["parallel_execution"] == setting[4])
            ]
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
                    sub_round = sub_round[
                        (sub_round["trivial_return"] == setting[0])
                        & (sub_round["tmr_assignment_method"] == setting[1])
                        & (sub_round["consider_skip_rus"] == setting[2])
                        & (sub_round["decompose_move"] == setting[3])
                        & (sub_round["parallel_execution"] == setting[4])
                    ]
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
                    capsize=5,
                    linewidth=2.0,
                    color=setting_color_map[setting_idx],
                    label=label,
                )
            ax.set_xticks(x_vals)
            ax.set_xticklabels(
                _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
                rotation=0,
            )
            ax.tick_params(labelsize=ablation_font_size)
            # Don't set ylim for ablation execution time - let it auto-scale
            if show_title:
                ax.set_title(
                    f"AOD = {int(aod_value)}",
                    fontsize=ablation_font_size,
                    pad=5,
                )

        # One row: columns are AOD = 1 and AOD = 5; execution time only (no movement).
        n_aod = len(aods_to_plot)
        # Wide panels so axes span roughly the same width as the 2-col legend below.
        fig_w = max(13.0, 6.9 * n_aod)
        fig_h = 5.9
        fig, axes = plt.subplots(1, n_aod, figsize=(fig_w, fig_h), squeeze=False)

        round_name = round_names[0]
        for col_idx, aod_value in enumerate(aods_to_plot):
            ax = axes[0, col_idx]
            _draw_execution_panel(ax, round_name, aod_value, show_title=True)
            if col_idx == 0:
                ax.set_ylabel(
                    "Execution time",
                    fontsize=ablation_font_size,
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Number of Qubits", fontsize=ablation_font_size)

        fig.suptitle(
            f"Ablation Study — STAR architecture, d = {distance}",
            fontsize=ablation_font_size + 1,
            y=0.985,
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
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=ablation_font_size,
            frameon=True,
            columnspacing=-6,
            handletextpad=0.4,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.82))
        fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.22)

        distance_suffix = f"_distance_{distance}" if distance is not None else ""
        fig.savefig(
            os.path.join(
                output_dir,
                f"ablation_{placement}{distance_suffix}_execution.pdf",
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
    """Create microarchitecture execution-time figures for the paper (no movement charts).

    Uses the full Trotter step only. Layout is one row with one column per AOD
    (typically 1 and 5).
    """
    os.makedirs(output_dir, exist_ok=True)
    if setting_override is None:
        setting = SETTINGS[setting_idx]
    else:
        setting = setting_override

    agg_data = {}
    for round_name, df in dfs_dict.items():
        df_ctrl = df[_setting_filter(df, setting)]
        if df_ctrl.empty:
            continue
        agg_data[round_name] = _aggregate_microarch_trials(df_ctrl)

    if not agg_data:
        print(
            f"No data for microarch setting_idx={setting_idx}, skipping combined plot."
        )
        return

    preferred_round_order = ["full_trotter"]
    round_names = [
        round_name for round_name in preferred_round_order if round_name in agg_data
    ]
    if not round_names:
        print(
            f"No full_trotter data for microarch setting_idx={setting_idx}, "
            "skipping combined plot."
        )
        return

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
    for round_name in round_names:
        y_mins_exec = []
        y_maxs_exec = []

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

        execution_ylim_by_round[round_name] = _build_shared_ylim(
            y_mins_exec,
            y_maxs_exec,
            force_zero_bottom=False,
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
            rotation=0,
        )
        if show_title:
            ax.set_title(
                _format_round_title(round_name), fontsize=_FIG_FONT_SIZE, pad=10
            )

    metric_name = "Execution Time"
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=placement_color_map[p],
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
            label="Expected time",
        )
    ]
    ylim_by_round = execution_ylim_by_round

    n_aod = len(aods_to_plot)
    fig, axes = plt.subplots(
        1,
        n_aod,
        figsize=(5.8 * n_aod, 5.8),
        squeeze=False,
    )

    round_name = round_names[0]
    for col_idx, aod_value in enumerate(aods_to_plot):
        g = agg_data[round_name]
        g_aod = g[g["n_aods"] == aod_value]
        ax = axes[0, col_idx]
        if g_aod.empty:
            ax.axis("off")
            continue
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        _draw_execution_panel(ax, round_name, g_aod, show_title=False)
        ax.set_title(
            f"AOD = {int(aod_value)}",
            fontsize=_FIG_FONT_SIZE,
            pad=10,
        )
        ax.set_ylim(*ylim_by_round[round_name])
        ax.tick_params(axis="both", which="major", pad=1)
        if col_idx == 0:
            ax.set_ylabel(metric_name, labelpad=0)
        else:
            ax.set_ylabel("", labelpad=0)
        ax.set_xlabel("Number of Qubits", labelpad=0)

    fig.suptitle(
        "Microarchitecture Comparison",
        fontsize=_FIG_FONT_SIZE,
        y=0.96,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(4, len(legend_handles)),
        fontsize=_FIG_FONT_SIZE,
        frameon=True,
        columnspacing=0.5,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0.04, 0.20, 0.98, 0.88), pad=0.10, w_pad=0.03, h_pad=0.03)
    fig.subplots_adjust(
        top=0.86,
        bottom=0.36,
        hspace=0.40,
        wspace=0.0,
        left=0.04,
        right=0.98,
    )
    fig.savefig(
        os.path.join(output_dir, f"fig2_setting_{file_setting_tag}_execution.pdf"),
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
    """STAR AOD sweep (paper): column-based placement; one curve per AOD on a single axes."""
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

    if "full_trotter" not in agg_data:
        print(
            f"No full_trotter data for AOD study setting_idx={setting_idx}, "
            "skipping combined plot."
        )
        return
    agg_data = {"full_trotter": agg_data["full_trotter"]}

    first_grouped = list(agg_data.values())[0]
    if "code_distance" in first_grouped.columns:
        distance_values = sorted(
            first_grouped["code_distance"].dropna().astype(int).unique()
        )
    else:
        distance_values = [None]

    file_setting_tag = (
        str(setting_idx)
        if setting_override is None
        else (setting_label or "auto").replace(" ", "_").replace(",", "")
    )

    figure_title = "AOD Comparison: STAR Architecture"
    # if setting_label:
    #     figure_title = f"{figure_title} ({setting_label.replace('_', ' ')})"

    round_items = list(agg_data.items())

    def _save_aod_figure(
        distance_title: str, distance_suffix: str, tag: str, grouped: pd.DataFrame
    ) -> None:
        work = grouped.copy()
        work["placement"] = work["placement"].astype(str).str.strip()
        g_cb = work[work["placement"] == "col_based"].copy()
        if g_cb.empty:
            return

        aod_vals = sorted(g_cb["n_aods"].dropna().astype(int).unique().tolist())
        if not aod_vals:
            return

        aod_color_map = _build_aod_color_map(aod_vals)
        round_name = "full_trotter"

        fig, ax = plt.subplots(figsize=(8.0, 6.9))
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        x_vals: list[int] = []
        for aod in aod_vals:
            sub = g_cb[g_cb["n_aods"] == int(aod)].sort_values("n_qubits")
            if sub.empty:
                continue
            x_vals = sorted(
                set(x_vals).union(sub["n_qubits"].dropna().astype(int).tolist())
            )
            mean_vals = sub["total_time_mean"]
            min_vals = sub["total_time_min"]
            max_vals = sub["total_time_max"]
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
                alpha=1.0,
                label=f"AOD = {int(aod)}",
            )

        if not x_vals:
            plt.close(fig)
            return

        bounds = _get_theoretical_lower_bound(x_vals)
        ax.plot(
            x_vals,
            bounds["full_trotter"],
            "k--",
            linewidth=1.5,
            label="Expected time",
        )

        ax.set_xticks(x_vals)
        ax.set_xticklabels(
            _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
            rotation=0,
        )
        ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
        ax.set_ylabel("Execution time", fontsize=_FIG_FONT_SIZE)
        ax.tick_params(labelsize=_FIG_FONT_SIZE)
        fig.suptitle(distance_title, y=0.94, fontsize=_FIG_FONT_SIZE)
        h, lab = ax.get_legend_handles_labels()
        if h:
            fig.legend(
                h,
                lab,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.10),
                ncol=3,
                fontsize=_FIG_FONT_SIZE,
                frameon=True,
            )
        fig.tight_layout(rect=(0, 0.06, 1, 0.90))
        fig.subplots_adjust(bottom=0.10, top=0.88)
        placement_tag = "_col_based"
        fig.savefig(
            os.path.join(
                output_dir,
                f"total_time_placement_nAOD_setting_{file_setting_tag}_skip_placements-True_combined_{tag}{placement_tag}{distance_suffix}.pdf",
            ),
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)

    for distance in distance_values:
        if distance is None:
            distance_suffix = ""
            distance_title = figure_title
            _, grouped_plot = round_items[0]
        else:
            distance_round_items = []
            for round_name, grouped in round_items:
                grouped_distance = grouped[grouped["code_distance"] == distance].copy()
                if not grouped_distance.empty:
                    distance_round_items.append((round_name, grouped_distance))

            if not distance_round_items:
                continue

            distance_suffix = f"_distance_{distance}"
            distance_title = f"{figure_title}, d={int(distance)}"
            _, grouped_plot = distance_round_items[0]

        # _save_aod_figure(distance_title, distance_suffix, "vertical", grouped_plot)
        _save_aod_figure(distance_title, distance_suffix, "horizontal", grouped_plot)


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
        preferred = [d for d in STAR_PAPER_PLOT_DISTANCES if d in available_distances]
        if not preferred:
            distances_to_plot = sorted(d for d in available_distances if d != 13)
        else:
            distances_to_plot = preferred + sorted(
                (available_distances - set(preferred)) - {13}
            )
        if only_distances is not None:
            requested = {int(d) for d in only_distances}
            distances_to_plot = [d for d in distances_to_plot if int(d) in requested]
            if not distances_to_plot:
                print(f"No data for requested distances {sorted(requested)}; skipping.")
                return
        distances_to_plot = [d for d in distances_to_plot if int(d) != 13]
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

        preferred_indices = [4]
        settings_to_plot: list[tuple[str, tuple]] = []
        for idx in preferred_indices:
            setting = SETTINGS[idx]
            if not df_distance[_setting_filter(df_distance, setting)].empty:
                settings_to_plot.append((f"setting_{idx}", setting))

        if len(settings_to_plot) == 0:
            auto_settings = _get_available_settings(df_distance, max_settings=2)
            settings_to_plot = [
                (f"auto_{i+1}", setting) for i, setting in enumerate(auto_settings)
            ]

        micro_dir = os.path.join(distance_output_dir, "microarch_setting")
        for setting_label, setting in settings_to_plot:
            plot_microarch_comp_setting_combined(
                dfs_dict_micro,
                micro_dir,
                round_angle_lookup=round_angle_lookup,
                setting_override=setting,
                setting_label=setting_label,
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
        f"T placement={effective_t_placement}, STAR=setting_4 · "
        "avg_improvement_vs_STAR = mean((STAR-T)/STAR)*100 (positive => T faster)"
    )

    # STAR: same compile/setting, compare code distances (same curve families as the figure).
    sds_sorted = sorted(int(d) for d in star_distances)
    if len(sds_sorted) >= 2:
        print("    [STAR — distance ratio at matched n_qubits, setting_4]")
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


def _star_vs_t_layer_expected_bounds(
    layer_name: str, x_vals: list[int]
) -> dict[str, list]:
    bounds = _get_theoretical_lower_bound(x_vals)
    if layer_name == "full_trotter":
        return {"star": bounds["full_trotter"]}
    if layer_name == "zz_layers":
        return {"star": bounds["zz_layer"]}
    return {"star": bounds["x_layer"]}


def _t_layer_expected_y(layer_name: str, t_bounds: dict[str, list]) -> list:
    if layer_name == "full_trotter":
        return t_bounds["full_trotter"]
    if layer_name == "zz_layers":
        return t_bounds["zz_layer"]
    return t_bounds["x_layer"]


def _draw_star_vs_t_runtime_panel(
    ax,
    *,
    layer_name: str,
    aod: int,
    star_filtered: pd.DataFrame,
    t_work: pd.DataFrame,
    star_distances: list[int],
    t_line_settings: list[tuple[int, float, int]],
    architecture_distance_colors: dict[str, dict[int, str]],
    single_black_star_expected: bool = False,
) -> tuple[list[float], bool]:
    """Draw one STAR vs T runtime panel; return STAR reference y-values and whether data exist."""
    star_ms = _RUNTIME_LEGEND_METHOD_STYLES["STAR"]
    t_ms = _RUNTIME_LEGEND_METHOD_STYLES["T-cultivation"]
    star_ref_values: list[float] = []
    star_x_all: list[int] = []
    plotted_star = 0
    plotted_t = 0

    for sd in star_distances:
        star_d = star_filtered[
            pd.to_numeric(star_filtered["code_distance"], errors="coerce").astype(int)
            == int(sd)
        ].copy()
        star_summary = _summarize_execution_by_qubits(star_d, method="STAR")
        star_a = star_summary[star_summary["n_aods"] == aod].sort_values("n_qubits")
        if star_a.empty:
            continue
        plotted_star += 1
        star_color = architecture_distance_colors["STAR"].get(
            int(sd), _RUNTIME_LEGEND_METHOD_COLORS["STAR"]
        )
        mean_vals = star_a["execution_time_mean"]
        min_vals = star_a["execution_time_min"]
        max_vals = star_a["execution_time_max"]
        ax.errorbar(
            star_a["n_qubits"],
            mean_vals,
            yerr=[mean_vals - min_vals, max_vals - mean_vals],
            marker=star_ms["marker"],
            linestyle=star_ms["linestyle"],
            capsize=3,
            linewidth=2.0,
            color=star_color,
        )
        star_ref_values.extend(
            star_a["execution_time_mean"].to_numpy(dtype=float).tolist()
        )
        x_sd = sorted(star_a["n_qubits"].dropna().astype(int).unique().tolist())
        if x_sd and not single_black_star_expected:
            y_star_exp = _star_vs_t_layer_expected_bounds(layer_name, x_sd)["star"]
            ax.plot(
                x_sd,
                y_star_exp,
                color=star_color,
                linestyle=":",
                linewidth=1.6,
                alpha=0.9,
            )
        elif x_sd:
            star_x_all.extend(x_sd)

    if single_black_star_expected and star_x_all:
        x_star = sorted(set(int(v) for v in star_x_all))
        y_star_exp = _star_vs_t_layer_expected_bounds(layer_name, x_star)["star"]
        ax.plot(
            x_star,
            y_star_exp,
            color="black",
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            zorder=3,
        )

    if all(
        c in t_work.columns
        for c in ("code_distance", "fidelity_target", "factory_physical_size")
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
            mask = cd_match & ft_match & (t_candidates["factory_physical_size"] == fps)
            t_setting = t_candidates[mask]
            t_summary = _summarize_execution_by_qubits(
                t_setting, method="T cultivation"
            ).sort_values("n_qubits")
            if t_summary.empty:
                continue
            plotted_t += 1
            t_color = architecture_distance_colors["T-cultivation"].get(
                int(cd), _RUNTIME_LEGEND_METHOD_COLORS["T-cultivation"]
            )
            mean_vals = t_summary["execution_time_mean"]
            min_vals = t_summary["execution_time_min"]
            max_vals = t_summary["execution_time_max"]
            ax.errorbar(
                t_summary["n_qubits"],
                mean_vals,
                yerr=[mean_vals - min_vals, max_vals - mean_vals],
                marker=t_ms["marker"],
                linestyle=t_ms["linestyle"],
                capsize=3,
                linewidth=1.8,
                color=t_color,
            )
            x_t = sorted(set(t_summary["n_qubits"].dropna().astype(int).tolist()))
            if x_t:
                t_bounds = _get_theoretical_lower_bound_t_cultivation(
                    x_t, code_distance=cd, fidelity_target=ft
                )
                y_t_exp = _t_layer_expected_y(layer_name, t_bounds)
                ax.plot(
                    x_t,
                    y_t_exp,
                    color=t_color,
                    linestyle="-.",
                    linewidth=1.4,
                    alpha=0.9,
                )

    ax.set_ylim(bottom=0)
    if layer_name in {"zz_layers", "x_layer"}:
        if star_ref_values:
            _add_standard_y_ticks_with_star_reference(
                ax, np.asarray(star_ref_values, dtype=float)
            )
    elif star_ref_values:
        _add_star_reference_tick(ax, np.asarray(star_ref_values, dtype=float))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", pad=1)

    has_data = plotted_star > 0 or plotted_t > 0
    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=_FIG_FONT_SIZE,
        )
    return star_ref_values, has_data


def _merge_execution_means_at_aod_pair(
    summary: pd.DataFrame,
    *,
    aod_top: int,
    aod_bottom: int,
) -> pd.DataFrame | None:
    """Align mean execution time at two AOD values on shared ``n_qubits``."""
    top = summary.loc[
        summary["n_aods"] == int(aod_top), ["n_qubits", "execution_time_mean"]
    ]
    bot = summary.loc[
        summary["n_aods"] == int(aod_bottom), ["n_qubits", "execution_time_mean"]
    ]
    if top.empty or bot.empty:
        return None
    merged = top.merge(
        bot,
        on="n_qubits",
        how="inner",
        suffixes=("_top", "_bottom"),
    ).sort_values("n_qubits")
    if merged.empty:
        return None
    return merged.rename(
        columns={
            "execution_time_mean_top": "y_top",
            "execution_time_mean_bottom": "y_bottom",
        }
    )


def _errorbar_at_aod(
    ax,
    summary: pd.DataFrame,
    *,
    aod: int,
    color: str,
    marker: str,
    linestyle: str,
    linewidth: float,
) -> pd.DataFrame:
    """Plot measured execution time at one AOD; return the subset (may be empty)."""
    sub = summary.loc[summary["n_aods"] == int(aod)].sort_values("n_qubits")
    if sub.empty:
        return sub
    mean_vals = sub["execution_time_mean"]
    min_vals = sub["execution_time_min"]
    max_vals = sub["execution_time_max"]
    ax.errorbar(
        sub["n_qubits"],
        mean_vals,
        yerr=[mean_vals - min_vals, max_vals - mean_vals],
        marker=marker,
        linestyle=linestyle,
        capsize=3,
        linewidth=linewidth,
        color=color,
        zorder=4,
    )
    return sub


def _star_runtime_col_based(star_df: pd.DataFrame, setting: tuple) -> pd.DataFrame:
    """STAR rows for runtime comparison: chosen setting and column-based placement."""
    work = star_df[_setting_filter(star_df, setting)].copy()
    if "placement" in work.columns:
        work = _filter_col_based(work)
    return work


def _draw_star_vs_t_runtime_aod_band_panel(
    ax,
    *,
    layer_name: str,
    star_filtered: pd.DataFrame,
    t_work: pd.DataFrame,
    star_distances: list[int],
    t_line_settings: list[tuple[int, float, int]],
    architecture_distance_colors: dict[str, dict[int, str]],
    aod_top: int = STAR_VS_T_AOD_BAND_TOP,
    aod_bottom: int = STAR_VS_T_AOD_BAND_BOTTOM,
) -> tuple[list[float], bool]:
    """Single panel: shaded band between AOD ``aod_top`` (upper) and ``aod_bottom`` (lower)."""
    star_ms = _RUNTIME_LEGEND_METHOD_STYLES["STAR"]
    t_ms = _RUNTIME_LEGEND_METHOD_STYLES["T-cultivation"]
    star_ref_values: list[float] = []
    plotted = False
    star_x_all: list[int] = []

    for sd in star_distances:
        star_d = star_filtered[
            pd.to_numeric(star_filtered["code_distance"], errors="coerce").astype(int)
            == int(sd)
        ].copy()
        star_summary = _summarize_execution_by_qubits(star_d, method="STAR")
        band = _merge_execution_means_at_aod_pair(
            star_summary, aod_top=aod_top, aod_bottom=aod_bottom
        )
        if band is None:
            continue
        plotted = True
        star_color = architecture_distance_colors["STAR"].get(
            int(sd), _RUNTIME_LEGEND_METHOD_COLORS["STAR"]
        )
        x = band["n_qubits"].astype(int).to_numpy()
        y_top = band["y_top"].to_numpy(dtype=float)
        y_bot = band["y_bottom"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            y_bot,
            y_top,
            color=star_color,
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y_bot,
            color=star_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
        )
        star_at_top = _errorbar_at_aod(
            ax,
            star_summary,
            aod=aod_top,
            color=star_color,
            marker=star_ms["marker"],
            linestyle=star_ms["linestyle"],
            linewidth=2.0,
        )
        if not star_at_top.empty:
            star_ref_values.extend(
                star_at_top["execution_time_mean"].to_numpy(dtype=float).tolist()
            )
            star_x_all.extend(star_at_top["n_qubits"].astype(int).tolist())
        star_ref_values.extend(y_bot.tolist())

    if star_x_all:
        x_star = sorted(set(int(v) for v in star_x_all))
        y_star_exp = _star_vs_t_layer_expected_bounds(layer_name, x_star)["star"]
        ax.plot(
            x_star,
            y_star_exp,
            color="black",
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            zorder=3,
        )

    if all(
        c in t_work.columns
        for c in ("code_distance", "fidelity_target", "factory_physical_size")
    ):
        for cd, ft, fps in t_line_settings:
            cd_match = pd.to_numeric(t_work["code_distance"], errors="coerce").astype(
                int
            ) == int(cd)
            ft_match = np.isclose(
                pd.to_numeric(t_work["fidelity_target"], errors="coerce"),
                float(ft),
                rtol=0.0,
                atol=0.0,
            )
            mask = cd_match & ft_match & (t_work["factory_physical_size"] == fps)
            t_setting = t_work.loc[mask].copy()
            t_summary = _summarize_execution_by_qubits(
                t_setting, method="T cultivation"
            )
            band = _merge_execution_means_at_aod_pair(
                t_summary, aod_top=aod_top, aod_bottom=aod_bottom
            )
            if band is None:
                continue
            plotted = True
            t_color = architecture_distance_colors["T-cultivation"].get(
                int(cd), _RUNTIME_LEGEND_METHOD_COLORS["T-cultivation"]
            )
            x = band["n_qubits"].astype(int).to_numpy()
            y_top = band["y_top"].to_numpy(dtype=float)
            y_bot = band["y_bottom"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                y_bot,
                y_top,
                color=t_color,
                alpha=0.28,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x,
                y_bot,
                color=t_color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                zorder=2,
            )
            t_at_top = _errorbar_at_aod(
                ax,
                t_summary,
                aod=aod_top,
                color=t_color,
                marker=t_ms["marker"],
                linestyle=t_ms["linestyle"],
                linewidth=1.8,
            )
            x_t = (
                t_at_top["n_qubits"].astype(int).tolist()
                if not t_at_top.empty
                else x.tolist()
            )
            if x_t:
                t_bounds = _get_theoretical_lower_bound_t_cultivation(
                    x_t, code_distance=cd, fidelity_target=ft
                )
                y_t_exp = _t_layer_expected_y(layer_name, t_bounds)
                ax.plot(
                    x_t,
                    y_t_exp,
                    color=t_color,
                    linestyle="-.",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=3,
                )

    ax.set_ylim(bottom=0)
    if star_ref_values:
        _add_star_reference_tick(ax, np.asarray(star_ref_values, dtype=float))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", pad=1)
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=_FIG_FONT_SIZE,
        )
    return star_ref_values, plotted


def _apply_star_vs_t_shared_axis_labels(
    axes,
    *,
    n_rows: int,
    n_cols: int,
    row_titles: list[str] | None = None,
    col_titles: list[str] | None = None,
    y_label: str = "Execution time",
    x_label: str = "Number of Qubits",
) -> None:
    """One y-label per row (left column) and one x-label per column (bottom row)."""
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx == 0:
                if row_titles is not None and row_idx < len(row_titles):
                    ax.set_ylabel(
                        f"{row_titles[row_idx]}\n{y_label}",
                        fontsize=_FIG_FONT_SIZE,
                    )
                else:
                    ax.set_ylabel(y_label, fontsize=_FIG_FONT_SIZE)
            else:
                ax.set_ylabel("")
            if row_idx == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=_FIG_FONT_SIZE, labelpad=0)
            else:
                ax.set_xlabel("")
    if col_titles is not None:
        for col_idx, title in enumerate(col_titles):
            if col_idx < n_cols:
                axes[0][col_idx].set_title(title, fontsize=_FIG_FONT_SIZE, pad=10)


def _plot_star_vs_t_cultivation_best(
    star_layers: dict[str, pd.DataFrame],
    t_layers: dict[str, pd.DataFrame],
    output_dir: str,
    target_aods: list[int] | None = None,
    *,
    t_placement: str | None = None,
    log_console: bool = True,
):
    """Compare STAR (d=7, 9 when present) and T-cultivation runtime.

    Writes two PDFs:

    - ``runtime_star_vs_t_cultivation_all_in_one_aod_{aods}.pdf``: rows = AOD,
      columns = circuit layer; legend outside on the right.
    - ``runtime_star_vs_t_cultivation_full_trotter_aod_{aods}.pdf``: one row,
      columns = AOD 2 and 5 (full Trotter only); legend below.
    - ``runtime_star_vs_t_cultivation_overall_aod_1-5_band.pdf``: single full-Trotter
      panel; shaded band between AOD 1 (top) and AOD 5 (bottom) per distance curve.
    - ``runtime_star_vs_t_cultivation_overall_aod_{n}.pdf``: same filters, single AOD
      measured curves only (no band; default AOD 1).

    STAR d=13 is omitted from STAR curves; T-cultivation may include d=13 when
    ``SHOW_T_CULTIVATION_D13`` is enabled. T expected-time curves are colored by
    code distance.
    """
    if target_aods is None:
        target_aods = [2, 5]

    effective_t_placement = t_placement if t_placement is not None else "col_based"
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

    setting_4 = SETTINGS[4]
    star_dist_all: set[int] = set()
    for layer_name in layer_order:
        star_filtered = star_layers[layer_name][
            _setting_filter(star_layers[layer_name], setting_4)
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

    t_line_settings = _t_cultivation_runtime_line_settings()
    architecture_distance_colors = _build_runtime_star_vs_t_distance_colors(
        list(star_distances), list(t_line_settings)
    )
    aod_tag = "-".join(str(a) for a in target_aods)

    def _t_work_for_layer(layer_name: str) -> pd.DataFrame:
        t_work = t_layers[layer_name]
        if "placement" not in t_work.columns:
            return t_work
        return t_work[
            t_work["placement"].astype(str).str.strip() == effective_t_placement
        ].copy()

    # --- Full grid: rows = AOD, columns = layer ---
    fig, axes = plt.subplots(len(target_aods), len(layer_order), figsize=(16, 8.5))
    axes = np.atleast_2d(axes)

    any_runtime_data = False
    for row_idx, aod in enumerate(target_aods):
        for col_idx, layer_name in enumerate(layer_order):
            ax = axes[row_idx, col_idx]
            star_filtered = star_layers[layer_name][
                _setting_filter(star_layers[layer_name], setting_4)
            ].copy()
            _, has_data = _draw_star_vs_t_runtime_panel(
                ax,
                layer_name=layer_name,
                aod=int(aod),
                star_filtered=star_filtered,
                t_work=_t_work_for_layer(layer_name),
                star_distances=star_distances,
                t_line_settings=t_line_settings,
                architecture_distance_colors=architecture_distance_colors,
            )
            any_runtime_data = any_runtime_data or has_data

    _apply_star_vs_t_shared_axis_labels(
        axes,
        n_rows=len(target_aods),
        n_cols=len(layer_order),
        row_titles=[f"AOD = {int(a)}" for a in target_aods],
        col_titles=[pretty_name[k] for k in layer_order],
    )

    if log_console:
        _print_runtime_star_vs_t_all_in_one_improvements(
            star_layers,
            t_layers,
            layer_order,
            star_distances,
            target_aods,
            effective_t_placement,
            setting_4,
            filename_suffix,
        )

    fig.suptitle(
        "Overall execution time comparison: STAR vs T",
        fontsize=_FIG_FONT_SIZE,
        y=0.99,
    )
    if any_runtime_data:
        _add_runtime_star_vs_t_legends(
            fig,
            architecture_distance_colors,
            star_distances=star_distances,
            t_line_settings=t_line_settings,
            legend_position="right",
        )
    fig.tight_layout(rect=(0.04, 0.06, 0.95, 0.90), pad=0.10, w_pad=0.03, h_pad=0.03)
    fig.subplots_adjust(
        top=0.90,
        hspace=0.42,
        wspace=0.28,
        bottom=0.08,
        left=0.06,
        right=0.9,
    )
    all_in_one_path = os.path.join(
        output_dir,
        f"runtime_star_vs_t_cultivation_all_in_one_aod_{aod_tag}{filename_suffix}.pdf",
    )
    fig.savefig(all_in_one_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # --- Compact: full Trotter only, one row per AOD column ---
    if "full_trotter" not in layer_order:
        return

    layer_name = "full_trotter"
    fig_c, axes_c = plt.subplots(1, len(target_aods), figsize=(10, 4.2), squeeze=False)
    axes_c = axes_c.reshape(1, -1)
    any_compact = False
    star_filtered_ft = star_layers[layer_name][
        _setting_filter(star_layers[layer_name], setting_4)
    ].copy()
    t_work_ft = _t_work_for_layer(layer_name)
    for col_idx, aod in enumerate(target_aods):
        _, has_data = _draw_star_vs_t_runtime_panel(
            axes_c[0, col_idx],
            layer_name=layer_name,
            aod=int(aod),
            star_filtered=star_filtered_ft,
            t_work=t_work_ft,
            star_distances=star_distances,
            t_line_settings=t_line_settings,
            architecture_distance_colors=architecture_distance_colors,
        )
        any_compact = any_compact or has_data

    _apply_star_vs_t_shared_axis_labels(
        axes_c,
        n_rows=1,
        n_cols=len(target_aods),
        col_titles=[f"AOD = {int(a)}" for a in target_aods],
    )
    fig_c.suptitle(
        "Overall execution time comparison: STAR vs T",
        fontsize=_FIG_FONT_SIZE,
        y=0.98,
    )
    if any_compact:
        _add_runtime_star_vs_t_legends(
            fig_c,
            architecture_distance_colors,
            star_distances=star_distances,
            t_line_settings=t_line_settings,
            legend_position="bottom",
        )
    fig_c.tight_layout(rect=(0.04, 0.14, 0.98, 0.8), pad=0.10, w_pad=0.12)
    fig_c.subplots_adjust(top=0.8, bottom=0.22, left=0.08, right=0.98)
    compact_path = os.path.join(
        output_dir,
        f"runtime_star_vs_t_cultivation_full_trotter_aod_{aod_tag}{filename_suffix}.pdf",
    )
    fig_c.savefig(compact_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_c)

    # --- Single panel: full Trotter, AOD band between 1 (top) and 5 (bottom) ---
    # STAR: setting 4 + col_based; T: main compile (``t_layers``) + col_based.
    star_filtered_band = _star_runtime_col_based(star_layers[layer_name], setting_4)
    t_work_band = _t_work_for_layer(layer_name)
    fig_b, ax_b = plt.subplots(1, 1, figsize=(10, 7))
    _, has_band = _draw_star_vs_t_runtime_aod_band_panel(
        ax_b,
        layer_name=layer_name,
        star_filtered=star_filtered_band,
        t_work=t_work_band,
        star_distances=star_distances,
        t_line_settings=t_line_settings,
        architecture_distance_colors=architecture_distance_colors,
    )
    ax_b.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE, labelpad=0)
    ax_b.set_ylabel("Execution time", fontsize=_FIG_FONT_SIZE)
    # ax_b.set_title("Full Trotter", fontsize=_FIG_FONT_SIZE, pad=10)
    fig_b.suptitle(
        "Overall execution time comparison: STAR vs T",
        fontsize=_FIG_FONT_SIZE,
        y=0.98,
    )
    if has_band:
        _add_runtime_star_vs_t_legends(
            ax_b,
            architecture_distance_colors,
            star_distances=star_distances,
            t_line_settings=t_line_settings,
            legend_position="inset",
        )
    fig_b.tight_layout(rect=(0.04, 0.06, 0.98, 0.88), pad=0.10)
    fig_b.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.98)
    band_path = os.path.join(
        output_dir,
        f"runtime_star_vs_t_cultivation_overall_aod_"
        f"{STAR_VS_T_AOD_BAND_TOP}-{STAR_VS_T_AOD_BAND_BOTTOM}_band"
        f"{filename_suffix}.pdf",
    )
    fig_b.savefig(band_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_b)

    # --- Overall panel: single AOD, no band ---
    overall_aod = int(STAR_VS_T_OVERALL_SINGLE_AOD)
    fig_n, ax_n = plt.subplots(1, 1, figsize=(10, 7))
    _, has_noband = _draw_star_vs_t_runtime_panel(
        ax_n,
        layer_name=layer_name,
        aod=overall_aod,
        star_filtered=star_filtered_band,
        t_work=t_work_band,
        star_distances=star_distances,
        t_line_settings=t_line_settings,
        architecture_distance_colors=architecture_distance_colors,
        single_black_star_expected=True,
    )
    ax_n.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE, labelpad=0)
    ax_n.set_ylabel("Execution time", fontsize=_FIG_FONT_SIZE)
    fig_n.suptitle(
        "Overall execution time comparison: STAR vs T",
        fontsize=_FIG_FONT_SIZE,
        y=0.98,
    )
    if has_noband:
        _add_runtime_star_vs_t_legends(
            ax_n,
            architecture_distance_colors,
            star_distances=star_distances,
            t_line_settings=t_line_settings,
            legend_position="inset",
        )
    fig_n.tight_layout(rect=(0.04, 0.06, 0.98, 0.88), pad=0.10)
    fig_n.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.98)
    noband_path = os.path.join(
        output_dir,
        f"runtime_star_vs_t_cultivation_overall_aod_{overall_aod}{filename_suffix}.pdf",
    )
    fig_n.savefig(noband_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_n)


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
    """T-cultivation AOD comparison (paper): full Trotter step only.

    One **row** of panels: one column per triple in ``_t_cultivation_runtime_line_settings``.
    Uses **column-based** placement only (same microarchitecture as the main STAR slice).
    When two triples are plotted, each panel title is the code distance (``d = …``).
    """
    os.makedirs(output_dir, exist_ok=True)
    layer_order = [k for k in ["full_trotter"] if k in t_layers]
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

    layer_name = layer_order[0]
    fig_w = max(11.0, 5.6 * n_settings)
    fig_h = 6
    fig, axes = plt.subplots(1, n_settings, figsize=(fig_w, fig_h), squeeze=False)

    plotted_any = False
    for col_idx, (cd, ft, fps) in enumerate(t_line_settings):
        ax = axes[0, col_idx]
        layer_df = t_layers[layer_name]
        if "placement" in layer_df.columns:
            layer_df = _filter_col_based(layer_df)
        mask = _mask_t_cultivation_triple(layer_df, cd, ft, fps)
        sub_df = layer_df.loc[mask].copy()

        if sub_df.empty:
            if n_settings == 2:
                ax.set_title(f"d = {int(cd)}", fontsize=_FIG_FONT_SIZE, pad=10)
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
                ax.set_ylabel("Execution Time", fontsize=_FIG_FONT_SIZE)
            else:
                ax.set_ylabel("")
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

        if n_settings == 2:
            ax.set_title(f"d = {int(cd)}", fontsize=_FIG_FONT_SIZE, pad=0)
        ax.tick_params(axis="both", which="major", pad=1)
        if col_idx == 0:
            ax.set_ylabel("Execution Time", fontsize=_FIG_FONT_SIZE)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Number of Qubits")
        ax.grid(True, alpha=0.3)
        if len(aod_values) > 0:
            ax.legend(fontsize=_FIG_FONT_SIZE)

    if not plotted_any:
        plt.close(fig)
        return

    fig.suptitle("AOD Comparison: T Cultivation", fontsize=_FIG_FONT_SIZE, y=0.995)
    _apply_shared_bottom_legend(fig, ncol=3, anchor_y=0.03)
    fig.tight_layout(rect=(0.04, 0.13, 0.98, 0.93), pad=0.10, w_pad=0.08, h_pad=0.03)
    fig.subplots_adjust(
        top=0.9,
        hspace=0.28,
        wspace=0.22,
        left=0.04,
        right=0.98,
        bottom=0.33,
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
    """T-cultivation compile ablation (paper): full Trotter step only, execution time only.

    Each line is one tuple in ``_T_COMPILE_ABLATION_GRID`` (aligned with
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
    layer_order = [k for k in ["full_trotter"] if k in t_layers]
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

    ablation_font_size = _FIG_FONT_SIZE + 4
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
            ax.set_xticklabels([str(int(v)) for v in x_vals], rotation=0)
            ax.tick_params(labelsize=ablation_font_size)
            if show_title:
                ax.set_title(
                    f"AOD = {int(aod)}",
                    fontsize=ablation_font_size,
                    pad=8,
                )

        metric_name = "Execution Time"
        file_suffix = "execution"
        n_aod = len(aods_to_plot)
        fig, axes = plt.subplots(
            1,
            n_aod,
            figsize=(6.9 * n_aod, 6.6),
            squeeze=False,
        )
        for col_idx, aod in enumerate(aods_to_plot):
            ax = axes[0, col_idx]
            layer_name = layer_order[0]
            _draw_execution_panel(ax, layer_name, aod, show_title=True)
            if col_idx == 0:
                ax.set_ylabel(
                    metric_name,
                    fontsize=ablation_font_size,
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Number of Qubits", fontsize=ablation_font_size)

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
            f"Ablation Study - T-cultivation, {triple_note}",
            fontsize=ablation_font_size,
            y=0.985,
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=ablation_font_size,
            frameon=True,
        )
        # fig.tight_layout(rect=(0, 0.14, 1, 0.97))
        # fig.subplots_adjust(bottom=0.3, top=0.8, wspace=0.28)
        fig.tight_layout(rect=(0, 0.1, 1, 0.85))
        fig.subplots_adjust(top=0.85, bottom=0.38, wspace=0.28)
        output_path = os.path.join(
            output_dir,
            f"t_cultivation_compile_ablation_{placement}_d{distance}_{file_suffix}.pdf",
        )
        # fig.tight_layout(rect=(0, 0.14, 1, 0.90))
        # fig.subplots_adjust(bottom=0.22, wspace=0.28)
        # output_path = os.path.join(
        #     output_dir,
        #     f"t_cultivation_compile_ablation_{placement}_d{distance}_{file_suffix}.pdf",
        # )
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
    """Compare T-cultivation placements at ``T_CULTIVATION_MAIN_COMPILE_SETTING`` (paper).

    One PDF per code distance from ``_t_cultivation_runtime_line_settings()`` (same
    primary triple as compile ablation): **one row** of panels with AOD = 1 and AOD = 5
    when present; each panel shows **full Trotter** execution vs qubit count with one
    curve per placement in ``T_CULTIVATION_ARCHITECTURE_PLACEMENTS``, plus a dashed
    theoretical lower bound for the selected fidelity triple.
    """
    os.makedirs(output_dir, exist_ok=True)
    pretty_round = {
        "full_trotter": "Full Trotter",
        "zz_layers": "ZZ layers",
        "x_layer": "X layer",
    }
    layer_order = [k for k in ["full_trotter"] if k in t_layers]
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
            1,
            len(aods_to_plot),
            figsize=(5.8 * len(aods_to_plot), 5.8),
            squeeze=False,
        )

        for col_idx, aod_value in enumerate(aods_to_plot):
            ax = axes[0, col_idx]
            round_name = round_names[0]
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
                x_vals = sorted(g_aod[SWEEP_COL].dropna().astype(int).unique().tolist())

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
                ax.set_xticklabels([str(int(v)) for v in x_vals], rotation=0)
            else:
                ax.set_xticks([])
            ax.tick_params(labelsize=fs)
            ax.set_title(f"AOD = {aod_value}", fontsize=fs, pad=8)
            if col_idx == 0:
                ax.set_ylabel("Execution time", fontsize=fs)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Number of Qubits", fontsize=fs)

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
            f"Microarchitecture Comparison: T-cultivation, {triple_note}",
            fontsize=fs,
            y=0.99,
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(4, len(legend_handles)),
            fontsize=fs,
            frameon=True,
            columnspacing=0.5,
            handletextpad=0.4,
        )
        fig.tight_layout(
            rect=(0.04, 0.20, 0.98, 0.88), pad=0.10, w_pad=0.03, h_pad=0.03
        )
        fig.subplots_adjust(
            top=0.86,
            bottom=0.36,
            hspace=0.40,
            wspace=0.25,
            left=0.04,
            right=0.98,
        )
        out_path = os.path.join(
            output_dir,
            f"t_cultivation_architecture_placement_d{distance}_execution.pdf",
        )
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)
        if verbose:
            print(f"Saved: {out_path}")


def _primary_t_triple_for_star_t_grid(
    code_distance: int,
) -> tuple[int, float, int] | None:
    for triple in _t_cultivation_runtime_line_settings():
        if int(triple[0]) == int(code_distance):
            return triple
    return None


def _fig_star_t_grid_row_arch_and_aod_titles(
    fig,
    axes,
    row_arch_labels: list[str],
    aod_values: tuple[int, ...] | None = None,
    *,
    column_titles: tuple[str, ...] | None = None,
    y_aod_above_axes: float = 0.014,
    y_arch_above_aod: float = 0.030,
    y_col_above_arch: float = 0.030,
    fontsize_arch: int | None = None,
    fontsize_aod: int | None = None,
    fontsize_col: int | None = None,
) -> None:
    """Row title (architecture) centered over each row; optional per-column subtitles."""
    fs_arch = fontsize_arch if fontsize_arch is not None else _FIG_FONT_SIZE
    fs_aod = fontsize_aod if fontsize_aod is not None else (_FIG_FONT_SIZE - 1)
    fs_col = fontsize_col if fontsize_col is not None else (_FIG_FONT_SIZE - 1)
    nrows, ncols = int(axes.shape[0]), int(axes.shape[1])
    for row, arch in enumerate(row_arch_labels):
        if row >= nrows:
            break
        positions = [axes[row, c].get_position() for c in range(ncols)]
        row_top = max(p.y1 for p in positions)
        y_sub = min(0.94, row_top + y_aod_above_axes)
        if aod_values is not None:
            for c, aod in enumerate(aod_values):
                if c >= ncols:
                    break
                pc = positions[c]
                xc_col = 0.5 * (pc.x0 + pc.x1)
                fig.text(
                    xc_col,
                    y_sub,
                    f"AOD = {int(aod)}",
                    ha="center",
                    va="bottom",
                    fontsize=fs_aod,
                )
        xc_row = 0.5 * (positions[0].x0 + positions[-1].x1)
        y_arch = min(0.97, y_sub + y_arch_above_aod)
        fig.text(xc_row, y_arch, arch, ha="center", va="bottom", fontsize=fs_arch)
    if column_titles is not None:
        for c, title in enumerate(column_titles):
            if c >= ncols:
                break
            col_positions = [axes[r, c].get_position() for r in range(nrows)]
            col_top = max(p.y1 for p in col_positions)
            xc = 0.5 * (col_positions[0].x0 + col_positions[0].x1)
            y_col = min(0.995, col_top + y_col_above_arch + y_arch_above_aod + 0.02)
            fig.text(xc, y_col, title, ha="center", va="bottom", fontsize=fs_col)


def _normalize_placement_name(placement: str) -> str:
    return str(placement).strip()


def _aggregate_star_setting_study(
    dfs_dict_micro: dict[str, pd.DataFrame],
    setting: tuple,
    placement: str,
    code_distance: int,
    aod: int,
) -> pd.DataFrame | None:
    """Mean/min/max execution time vs qubits for one STAR compiler setting."""
    round_name = "full_trotter"
    if round_name not in dfs_dict_micro:
        return None
    df = dfs_dict_micro[round_name].copy()
    df = df.loc[_setting_filter(df, setting)].copy()
    if df.empty:
        return None
    if "code_distance" in df.columns:
        df = df[
            pd.to_numeric(df["code_distance"], errors="coerce").astype(int)
            == int(code_distance)
        ].copy()
    if df.empty:
        return None
    df["placement"] = df["placement"].astype(str).map(_normalize_placement_name)
    df = df[df["placement"] == _normalize_placement_name(placement)].copy()
    if df.empty:
        return None
    df["n_aods"] = pd.to_numeric(df["n_aods"], errors="coerce").astype(int)
    df = df[df["n_aods"] == int(aod)].copy()
    if df.empty:
        return None
    grouped = (
        df.groupby("n_qubits")[["total_time"]]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    return grouped if not grouped.empty else None


def _aggregate_t_setting_study(
    layer_df: pd.DataFrame,
    triple: tuple[int, float, int],
    compile_tuple: tuple[bool, bool, bool],
    placement: str,
    code_distance: int,
    aod: int,
) -> pd.DataFrame | None:
    """Mean/min/max execution time vs qubits for one T-cultivation compile setting."""
    work = layer_df.copy()
    work["placement"] = work["placement"].astype(str).map(_normalize_placement_name)
    work = work[work["placement"] == _normalize_placement_name(placement)].copy()
    cd_t, ft, fps = triple
    tr, dm, rs = compile_tuple
    mask = _mask_t_cultivation_triple(work, cd_t, ft, fps) & _mask_t_cultivation_compile(
        work, tr, dm, rs
    )
    work = work.loc[mask].copy()
    if work.empty:
        return None
    work = work[
        pd.to_numeric(work["code_distance"], errors="coerce").astype(int)
        == int(code_distance)
    ].copy()
    if work.empty:
        return None
    work["n_aods"] = pd.to_numeric(work["n_aods"], errors="coerce").astype(int)
    work = work[work["n_aods"] == int(aod)].copy()
    if work.empty:
        return None
    grouped = (
        work.groupby("n_qubits")[["total_time"]]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    return grouped if not grouped.empty else None


def _draw_errorbar_from_agg(
    ax,
    grouped: pd.DataFrame,
    *,
    color,
    label: str | None = None,
) -> list[int]:
    sub = grouped.sort_values("n_qubits")
    mean_vals = sub["total_time_mean"]
    min_vals = sub["total_time_min"]
    max_vals = sub["total_time_max"]
    ax.errorbar(
        sub["n_qubits"],
        mean_vals,
        yerr=[mean_vals - min_vals, max_vals - mean_vals],
        marker="o",
        capsize=3,
        linewidth=1.8,
        color=color,
        label=label,
    )
    return sub["n_qubits"].dropna().astype(int).tolist()


def _star_grouped_aod_col_based_at_distance(
    dfs_dict_aod: dict[str, pd.DataFrame],
    setting: tuple,
    code_distance: int,
) -> pd.DataFrame | None:
    """Replicate STAR AOD combined aggregation, ``full_trotter`` only, one code distance."""
    agg_data: dict[str, pd.DataFrame] = {}
    for round_name, df in dfs_dict_aod.items():
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
    if "full_trotter" not in agg_data:
        return None
    grouped = agg_data["full_trotter"]
    cd_mask = pd.to_numeric(grouped["code_distance"], errors="coerce").astype(
        int
    ) == int(code_distance)
    gd = grouped.loc[cd_mask].copy()
    if gd.empty:
        return None
    gd["placement"] = gd["placement"].astype(str).str.strip()
    g_cb = gd[gd["placement"] == "col_based"].copy()
    return g_cb if not g_cb.empty else None


def _plot_star_t_setting_and_aod_combined_grid(
    dfs_dict_micro: dict[str, pd.DataFrame],
    dfs_dict_aod: dict[str, pd.DataFrame],
    t_layers: dict[str, pd.DataFrame],
    t_layers_ablation: dict[str, pd.DataFrame],
    output_dir: str,
    *,
    code_distance: int | None = None,
    star_setting_index: int = STAR_T_GRID_STAR_SETTING_INDEX,
    setting_study_aod: int = STAR_T_SETTING_STUDY_AOD,
    round_angle_lookup: dict[str, dict[int, int]] | None = None,
    verbose: bool = True,
) -> None:
    """Single 2×2 figure: rows STAR / T-cultivation; setting study (left) and AOD sweep (right).

    Left column at ``setting_study_aod``: vanilla on ``seperate_region_row`` with
    ``COMPILE_SETTINGS[0]`` / ``SETTINGS[0]``, plus four column-based compiler curves
    (STAR: ``STAR_SETTING_STUDY_COL_BASED_SETTING_IDXS``; T: full
    ``_T_COMPILE_ABLATION_GRID`` on ``col_based``). Right column: AOD = 1–5 at
    ``SETTINGS[star_setting_index]`` / ``T_CULTIVATION_MAIN_COMPILE_SETTING``.
    """
    os.makedirs(output_dir, exist_ok=True)
    cd = int(
        code_distance
        if code_distance is not None
        else STAR_T_GRID_COMPARISON_CODE_DISTANCE
    )
    round_name = "full_trotter"
    aod_setting = int(setting_study_aod)
    star_aod_setting = SETTINGS[int(star_setting_index)]

    star_ablation_labels = [
        "Vanilla",
        "Opt. return",
        "Opt. return + part. RUS",
        "Opt. return + skip whole RUS",
        "Opt. decomp. return + skip whole RUS",
        "Opt. return + async. RUS",
        "Opt. return + skip whole RUS + async. RUS",
    ]

    g_cb = _star_grouped_aod_col_based_at_distance(
        dfs_dict_aod, star_aod_setting, cd
    )
    if g_cb is None or g_cb.empty:
        if verbose:
            print(f"Skipping STAR/T combined grid: no STAR col_based rows at d={cd}.")
        return

    triple = _primary_t_triple_for_star_t_grid(cd)
    if triple is None:
        if verbose:
            print(
                f"Skipping STAR/T combined grid: no T-cultivation triple for d={cd}."
            )
        return
    cd_t, ft, fps = triple

    compile_cols = {
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
    }
    if round_name not in t_layers or round_name not in t_layers_ablation:
        if verbose:
            print("Skipping STAR/T combined grid: missing full_trotter layer frames.")
        return
    t_aod_layer = t_layers[round_name].copy()
    t_setting_layer = t_layers_ablation[round_name].copy()
    if "placement" in t_aod_layer.columns:
        t_aod_layer = _filter_col_based(t_aod_layer)
    if not compile_cols.issubset(t_aod_layer.columns) or not compile_cols.issubset(
        t_setting_layer.columns
    ):
        if verbose:
            print("Skipping STAR/T combined grid: T CSV missing compile columns.")
        return
    tr_m, dm_m, rs_m = T_CULTIVATION_MAIN_COMPILE_SETTING
    t_base = t_aod_layer.loc[
        _mask_t_cultivation_compile(t_aod_layer, tr_m, dm_m, rs_m)
        & _mask_t_cultivation_triple(t_aod_layer, cd_t, ft, fps)
    ].copy()
    t_base = t_base[
        pd.to_numeric(t_base["code_distance"], errors="coerce").astype(int) == cd
    ].copy()
    if t_base.empty:
        if verbose:
            print(f"Skipping STAR/T combined grid: no T-cultivation AOD rows at d={cd}.")
        return

    aod_candidates = [1, 2, 3, 4, 5]
    star_aods = set(
        pd.to_numeric(g_cb["n_aods"], errors="coerce").dropna().astype(int).unique()
    )
    t_aods = set(
        pd.to_numeric(t_base["n_aods"], errors="coerce").dropna().astype(int).unique()
    )
    plot_aods = [a for a in aod_candidates if int(a) in star_aods and int(a) in t_aods]
    if not plot_aods:
        if verbose:
            print(
                f"Skipping STAR/T combined grid: no overlapping AOD in {{1,…,5}} "
                f"at d={cd}."
            )
        return

    setting_cmap = plt.get_cmap("tab10")
    star_setting_colors = {
        "vanilla": setting_cmap(0),
        **{
            idx: setting_cmap((1 + i) % 10)
            for i, idx in enumerate(STAR_SETTING_STUDY_COL_BASED_SETTING_IDXS)
        },
    }
    t_setting_colors = {
        "vanilla": setting_cmap(0),
        **{i: setting_cmap((1 + i) % 10) for i in range(len(_T_COMPILE_ABLATION_GRID))},
    }
    aod_color_map = _build_aod_color_map(plot_aods)

    def _finalize_setting_panel(ax, x_points: list[int], *, theory: str) -> None:
        x_u = sorted(set(int(x) for x in x_points))
        if x_u:
            if theory == "star":
                b = _get_theoretical_lower_bound(x_u)
            else:
                b = _get_theoretical_lower_bound_t_cultivation(
                    x_u, code_distance=cd_t, fidelity_target=ft
                )
            ax.plot(x_u, b["full_trotter"], color="black", linestyle="--", linewidth=1.5)
            ax.set_xticks(x_u)
            ax.set_xticklabels(
                _format_nqubit_ticklabels(x_u, round_name, round_angle_lookup),
                rotation=0,
            )
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.relim(visible_only=True)
        ax.autoscale_view()
        y_lo, y_hi = ax.get_ylim()
        if y_hi > 0.0:
            ax.set_ylim(max(0.0, y_lo), y_hi)

    def _draw_star_setting_panel(ax) -> list[int]:
        x_pts: list[int] = []
        vanilla = _aggregate_star_setting_study(
            dfs_dict_micro,
            SETTINGS[0],
            T_SETTING_STUDY_VANILLA_PLACEMENT,
            cd,
            aod_setting,
        )
        if vanilla is not None:
            x_pts.extend(
                _draw_errorbar_from_agg(
                    ax,
                    vanilla,
                    color=star_setting_colors["vanilla"],
                    label="Vanilla",
                )
            )
        for idx in STAR_SETTING_STUDY_COL_BASED_SETTING_IDXS:
            if idx >= len(SETTINGS) or idx >= len(star_ablation_labels):
                continue
            grouped = _aggregate_star_setting_study(
                dfs_dict_micro,
                SETTINGS[idx],
                "col_based",
                cd,
                aod_setting,
            )
            if grouped is None:
                continue
            x_pts.extend(
                _draw_errorbar_from_agg(
                    ax,
                    grouped,
                    color=star_setting_colors[idx],
                    label=star_ablation_labels[idx],
                )
            )
        _finalize_setting_panel(ax, x_pts, theory="star")
        return x_pts

    def _draw_t_setting_panel(ax) -> list[int]:
        x_pts: list[int] = []
        vanilla_compile = _T_COMPILE_ABLATION_GRID[0]
        vanilla = _aggregate_t_setting_study(
            t_setting_layer,
            triple,
            vanilla_compile,
            T_SETTING_STUDY_VANILLA_PLACEMENT,
            cd,
            aod_setting,
        )
        if vanilla is not None:
            x_pts.extend(
                _draw_errorbar_from_agg(
                    ax,
                    vanilla,
                    color=t_setting_colors["vanilla"],
                    label="Vanilla",
                )
            )
        for compile_idx, compile_tuple in enumerate(_T_COMPILE_ABLATION_GRID):
            grouped = _aggregate_t_setting_study(
                t_setting_layer,
                triple,
                compile_tuple,
                "col_based",
                cd,
                aod_setting,
            )
            if grouped is None:
                continue
            x_pts.extend(
                _draw_errorbar_from_agg(
                    ax,
                    grouped,
                    color=t_setting_colors[compile_idx],
                    label=_T_COMPILE_ABLATION_LABELS[compile_idx],
                )
            )
        _finalize_setting_panel(ax, x_pts, theory="t")
        return x_pts

    def _draw_star_aod_panel(ax) -> list[int]:
        x_pts: list[int] = []
        for aod in plot_aods:
            sub = g_cb[g_cb["n_aods"] == int(aod)].sort_values("n_qubits")
            if sub.empty:
                continue
            mean_vals = sub["total_time_mean"]
            min_vals = sub["total_time_min"]
            max_vals = sub["total_time_max"]
            ax.errorbar(
                sub["n_qubits"],
                mean_vals,
                yerr=[mean_vals - min_vals, max_vals - mean_vals],
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=aod_color_map.get(int(aod)),
            )
            x_pts.extend(sub["n_qubits"].dropna().astype(int).tolist())
        _finalize_setting_panel(ax, x_pts, theory="star")
        return x_pts

    def _draw_t_aod_panel(ax) -> list[int]:
        x_pts: list[int] = []
        for aod in plot_aods:
            t_a = t_base[t_base["n_aods"] == int(aod)].copy()
            summ = _summarize_execution_by_qubits(t_a, method="T cultivation").sort_values(
                "n_qubits"
            )
            if summ.empty:
                continue
            mean_vals = summ["execution_time_mean"]
            min_vals = summ["execution_time_min"]
            max_vals = summ["execution_time_max"]
            ax.errorbar(
                summ["n_qubits"],
                mean_vals,
                yerr=[mean_vals - min_vals, max_vals - mean_vals],
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=aod_color_map.get(int(aod)),
            )
            x_pts.extend(summ["n_qubits"].dropna().astype(int).tolist())
        _finalize_setting_panel(ax, x_pts, theory="t")
        return x_pts

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), squeeze=False)
    row_labels = ["STAR architecture", "T-cultivation"]
    _draw_star_setting_panel(axes[0, 0])
    _draw_star_aod_panel(axes[0, 1])
    _draw_t_setting_panel(axes[1, 0])
    _draw_t_aod_panel(axes[1, 1])

    for row in range(2):
        y_lo_m, y_hi_m = None, None
        for col in range(2):
            ax = axes[row, col]
            lo, hi = ax.get_ylim()
            y_lo_m = lo if y_lo_m is None else min(y_lo_m, lo)
            y_hi_m = hi if y_hi_m is None else max(y_hi_m, hi)
        y_lo_m = max(0.0, y_lo_m)
        for col in range(2):
            axes[row, col].set_ylim(y_lo_m, y_hi_m)

    for row in range(2):
        axes[row, 0].set_ylabel("Execution time", fontsize=_FIG_FONT_SIZE, labelpad=2)
        axes[row, 1].set_ylabel("")
        axes[row, 0].set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE, labelpad=0)
        axes[row, 1].set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE, labelpad=0)
        for col in range(2):
            axes[row, col].tick_params(axis="both", which="major", pad=1)

    setting_legend_handles: list = [
        Line2D(
            [0],
            [0],
            color=star_setting_colors["vanilla"],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            label="Vanilla",
        ),
    ]
    for idx in STAR_SETTING_STUDY_COL_BASED_SETTING_IDXS:
        if idx >= len(star_ablation_labels):
            continue
        setting_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=star_setting_colors[idx],
                marker="o",
                linestyle="-",
                linewidth=1.8,
                label=star_ablation_labels[idx],
            )
        )
    aod_legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Expected time",
        ),
    ]
    for aod in aod_candidates:
        if int(aod) not in plot_aods:
            continue
        aod_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=aod_color_map.get(int(aod)),
                marker="o",
                linestyle="-",
                linewidth=1.8,
                label=f"AOD = {int(aod)}",
            )
        )

    fig.legend(
        handles=setting_legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.02),
        ncol=2,
        fontsize=_FIG_FONT_SIZE - 2,
        frameon=True,
        title=f"Setting study (AOD = {aod_setting})",
        title_fontsize=_FIG_FONT_SIZE - 2,
    )
    fig.legend(
        handles=aod_legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        ncol=min(3, len(aod_legend_handles)),
        fontsize=_FIG_FONT_SIZE - 2,
        frameon=True,
        title="AOD comparison",
        title_fontsize=_FIG_FONT_SIZE - 2,
    )

    fig.suptitle(
        f"STAR vs T-cultivation (d = {cd})",
        fontsize=_FIG_FONT_SIZE + 1,
        y=0.995,
    )
    fig.tight_layout(rect=(0.06, 0.18, 0.98, 0.82), pad=0.10, w_pad=0.18, h_pad=0.40)
    fig.subplots_adjust(
        top=0.84,
        bottom=0.30,
        left=0.07,
        right=0.98,
        wspace=0.28,
        hspace=0.48,
    )
    _fig_star_t_grid_row_arch_and_aod_titles(
        fig,
        axes,
        row_labels,
        column_titles=("Setting study", "AOD comparison"),
    )
    fig.text(
        0.5,
        0.155,
        f"Microarchitecture opt.: column-based placement · Vanilla: "
        f"{_format_microarch_placement_label(T_SETTING_STUDY_VANILLA_PLACEMENT)}",
        ha="center",
        va="center",
        fontsize=_FIG_FONT_SIZE - 4,
    )

    out_path = os.path.join(output_dir, f"star_t_setting_and_aod_comparison_d{cd}.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
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
        target_aods = [2, 5]

    # Align with plotting comparison setting.
    setting_4 = SETTINGS[4]
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
            _setting_filter(star_layers[layer_name], setting_4)
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
            d for d in STAR_PAPER_PLOT_DISTANCES if d in star_distances
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

    star_setting = SETTINGS[4]
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

    star_setting_desc = "STAR setting_4 (sync. execution)"
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

    ablation_labels = [
        "Vanilla",
        "Opt. return",
        "Opt. return + part. RUS",
        "Opt. return + skip whole RUS",
        "Opt. decomp. return + skip whole RUS",
        "Opt. return + async. RUS",
        "Opt. return + skip whole RUS + async. RUS",
    ]
    if not star_full.empty:
        star_ablation = star_full.copy()
        base = star_ablation[_setting_filter(star_ablation, SETTINGS[0])].copy()
        bg = base.groupby(
            ["placement", "code_distance", "n_aods", "n_qubits"],
            as_index=False,
        ).agg(rt_base=("total_time", "mean"))
        for idx in range(1, min(len(SETTINGS), len(ablation_labels))):
            s = star_ablation[_setting_filter(star_ablation, SETTINGS[idx])].copy()
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

        dfs_dict_micro, _dfs_dict_ablation_rt, dfs_dict_aod = _build_round_plot_dicts(
            star_df
        )
        round_angle_lookup_star_t = _build_round_angle_lookup(dfs_dict_micro)
        _plot_star_t_setting_and_aod_combined_grid(
            dfs_dict_micro,
            dfs_dict_aod,
            t_layers,
            t_layers_ablation,
            output_dir,
            code_distance=STAR_T_GRID_COMPARISON_CODE_DISTANCE,
            round_angle_lookup=round_angle_lookup_star_t,
            verbose=verbose_runtime_plots,
        )

        _plot_star_vs_t_cultivation_best(
            star_layers,
            t_layers,
            output_dir,
            target_aods=[2, 5],
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
    full_trotter_output_dir = "output/prx_quantum/full_trotter"
    process_full_trotter_csv(full_trotter_csv, full_trotter_output_dir)

    t_cultivation_profiling_csv = (
        "output/evaluation/fidelity/t_cultivation_fidelity_profiling_results.csv"
    )
    runtime_compare_output_dir = "output/prx_quantum/t_cultivation_runtime"
    runtime_compare_output_dir_with_d13 = (
        "output/prx_quantum/t_cultivation_runtime_with_d13"
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
