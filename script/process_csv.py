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


def _apply_shared_bottom_legend(fig, ncol: int = 4, anchor_y: float = 0.01) -> None:
    """Collapse subplot legends into a single figure legend at the bottom."""
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


def _add_star_reference_tick(ax, star_values: np.ndarray) -> None:
    """Add a readable y-tick near STAR duration in mixed STAR vs T plots.

    Generates nice round ticks (5000, 10000, etc.) and adds star median as one extra tick.
    """
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

    # Generate nice round ticks (multiples of 5000)
    tick_step = 5000
    base_ticks = []
    tick = max(tick_step, int(np.floor(y_min / tick_step)) * tick_step)
    if tick == 0:
        tick = tick_step
    while tick <= y_max:
        if tick >= y_min:
            base_ticks.append(tick)
        tick += tick_step

    # Add star reference tick and sort
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


def _setting_filter(df: pd.DataFrame, setting: tuple) -> pd.Series:
    return (
        (df["trivial_return"] == setting[0])
        & (df["tmr_assignment_method"] == setting[1])
        & (df["consider_skip_rus"] == setting[2])
        & (df["decompose_move"] == setting[3])
        & (df["parallel_execution"] == setting[4])
    )


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


def _get_available_settings(df: pd.DataFrame, max_settings: int = 2) -> list[tuple]:
    setting_cols = [
        "trivial_return",
        "tmr_assignment_method",
        "consider_skip_rus",
        "decompose_move",
        "parallel_execution",
    ]
    available = (
        df.groupby(setting_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return [
        tuple(row[col] for col in setting_cols)
        for _, row in available.head(max_settings).iterrows()
    ]


def _get_circuit_lower_bound_components() -> dict[str, tuple[int, int]]:
    """Return the number of CNOT and H gate layers surrounding each Rz round.

    Parses the logical TFIM circuit (CNOT/Rz/CNOT ZZ layers + H/Rz/H transverse field)
    and groups non-Rz instructions between consecutive Rz instructions.
    Trailing instructions after the last Rz are attributed to the last round.

    The gate-layer structure is identical for all even-by-even periodic lattices,
    so a fixed 4x4 representative layout is used.

    Returns:
        dict mapping "round_i" and "full_trotter" -> (n_cnot_layers, n_h_layers)
    """
    global _LOWER_BOUND_COMPONENTS
    if _LOWER_BOUND_COMPONENTS is not None:
        return _LOWER_BOUND_COMPONENTS

    qc = generate_one_layer_2d_tfim_circuit_cz(
        n_qubits=16, qubit_layout=(4, 4), J=1.0, h=1.0, dt=1.0, logical=True, order=2
    )

    rz_positions = [i for i, inst in enumerate(qc) if inst.get("gate") == "Rz"]
    n_rz = len(rz_positions)

    components: dict[str, tuple[int, int]] = {}
    prev_pos = 0
    for rz_idx, rz_pos in enumerate(rz_positions):
        segment = qc[prev_pos:rz_pos]
        n_cnot = sum(1 for inst in segment if inst.get("gate") == "CNOT")
        n_h = sum(1 for inst in segment if inst.get("gate") == "H")
        if rz_idx == n_rz - 1:
            trailing = qc[rz_pos + 1 :]
            n_cnot += sum(1 for inst in trailing if inst.get("gate") == "CNOT")
            n_h += sum(1 for inst in trailing if inst.get("gate") == "H")
        components[f"round_{rz_idx}"] = (n_cnot, n_h)
        prev_pos = rz_pos + 1

    total_cnot = sum(c for c, _ in components.values())
    total_h = sum(h for _, h in components.values())
    components["full_trotter"] = (total_cnot, total_h)

    _LOWER_BOUND_COMPONENTS = components
    return components


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


def _format_microarch_placement_label(placement: str) -> str:
    normalized = str(placement).strip().lower()
    if normalized == "col_based":
        return "Column-Based"
    if normalized in {"seperate_region_row", "separate_region_row"}:
        return "Seperate Row Region"
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

    ablation_labels = [
        "Vanilla",
        "Opt. return",
        "Opt. return + part. RUS",
        "Opt. return + skip whole RUS",
        "Opt. decomp. return + skip whole RUS",
        "Opt. return + async. RUS",
        "Opt. return + skip whole RUS + async. RUS",
    ]

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
            # Don't set ylim for ablation execution time - let it auto-scale
            if show_title:
                ax.set_title(
                    _format_round_title(round_name), fontsize=_FIG_FONT_SIZE, pad=10
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
            ax.set_ylim(*movement_ylim_by_round[round_name])
            if show_title:
                ax.set_title(
                    _format_round_title(round_name), fontsize=_FIG_FONT_SIZE, pad=10
                )

        for metric_name, draw_panel, file_suffix in [
            ("Execution Time", _draw_execution_panel, "execution"),
            ("Movement Time", _draw_movement_panel, "movement"),
        ]:
            # Use taller subplots for execution time to prevent line crowding
            height_multiplier = 5.5 if metric_name == "Execution Time" else 4.4
            fig, axes = plt.subplots(
                len(aods_to_plot),
                len(round_names),
                figsize=(5.8 * len(round_names), height_multiplier * len(aods_to_plot)),
                squeeze=False,
            )

            for row_idx, aod_value in enumerate(aods_to_plot):
                for col_idx, round_name in enumerate(round_names):
                    ax = axes[row_idx, col_idx]
                    draw_panel(ax, round_name, aod_value, show_title=(row_idx == 0))
                    if col_idx == 0:
                        ax.set_ylabel(f"AOD = {aod_value}\n{metric_name}")
                    else:
                        ax.set_ylabel("")
                    if row_idx == len(aods_to_plot) - 1:
                        ax.set_xlabel("Number of Qubits")
                    else:
                        ax.set_xlabel("")

            fig.suptitle(
                f"Ablation Study, {metric_name}, distance={distance}",
                fontsize=_FIG_FONT_SIZE,
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
                bbox_to_anchor=(0.5, -0.005),
                ncol=4,
                fontsize=_FIG_FONT_SIZE,
                frameon=True,
            )
            # fig.tight_layout(rect=(0.03, 0.15, 0.98, 0.99))
            # fig.subplots_adjust(top=0.92, bottom=0.15, hspace=0.15, wspace=0.25)

            fig.tight_layout(rect=(0, 0.28, 1, 1))
            fig.subplots_adjust(bottom=0.15, wspace=0.32)

            distance_suffix = f"_distance_{distance}" if distance is not None else ""
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"ablation_{placement}{distance_suffix}_{file_suffix}.pdf",
                )
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
    for optional in ("trial", "fidelity_target", "factory_physical_size"):
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
        df_ctrl = df[_setting_filter(df, setting)]
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
                    label="theoretical lower bound",
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
    min_aods = first_grouped["n_aods"].min()
    max_aods = first_grouped["n_aods"].max()
    execution_mode = _format_setting_title(setting_label, setting_idx)

    figure_title = f"AOD number comparison, {execution_mode}"
    file_setting_tag = (
        str(setting_idx)
        if setting_override is None
        else (setting_label or "auto").replace(" ", "_").replace(",", "")
    )

    cmap_placement = plt.get_cmap("tab10")
    base_colors = [cmap_placement(i) for i in range(10)]

    round_items = list(agg_data.items())

    def _draw_aod_round(ax, round_name, grouped):
        placements_to_plot = [placement] if placement else ["col_based", "checkerboard"]
        for p_idx, p in enumerate(placements_to_plot):
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
                    color=base_colors[p_idx % len(base_colors)],
                    alpha=0.3 + 0.7 * (aod - min_aods) / (max_aods - min_aods + 1e-5),
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
                label="theoretical lower bound",
            )
        elif round_name == MERGED_ROUND_NAME:
            zz_bounds = bounds["zz_layer"]
            ax.plot(
                x_vals,
                zz_bounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
            )
        elif round_name == f"round_{INDIVIDUAL_ROUND}":
            x_bounds = bounds["x_layer"]
            ax.plot(
                x_vals,
                x_bounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
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


def process_full_trotter_csv(csv_file: str, output_dir: str):
    """Generate full-trotter and per-round analyses for microarchitecture, ablation, and AOD studies."""
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
        distances_to_plot = [d for d in [7, 9] if d in available_distances]
        if len(distances_to_plot) == 0:
            distances_to_plot = sorted(available_distances)
    else:
        distances_to_plot = [None]

    for distance in distances_to_plot:
        if distance is None:
            df_distance = df
            distance_output_dir = output_dir
        else:
            df_distance = df[df["code_distance"] == distance]
            if df_distance.empty:
                continue
            distance_output_dir = os.path.join(output_dir, f"distance_{distance}")

        dfs_dict_micro, dfs_dict_ablation, dfs_dict_aod = _build_round_plot_dicts(
            df_distance
        )
        round_angle_lookup = _build_round_angle_lookup(dfs_dict_micro)

        preferred_indices = [4, 6]
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
        for placement in ["checkerboard", "col_based"]:
            plot_ablation_combined(
                dfs_dict_ablation,
                os.path.join(ablation_dir, placement),
                placement,
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
    bool_cols = ["trivial_return", "decompose_move", "parallel_execution"]
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
    return (
        f"d={int(code_distance)}, size={int(factory_physical_size)}, "
        f"LER={fidelity_target:g}"
    )


def _format_t_cultivation_setting_label(
    code_distance: float | int,
    fidelity_target: float,
    factory_physical_size: float | int,
) -> str:
    """Multi-line setting label for row titles in the T-cultivation AOD plot."""
    return (
        f"d={int(code_distance)}\n"
        f"size={int(factory_physical_size)}\n"
        f"LER={fidelity_target:g}"
    )


def _dedupe_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep first occurrence when CSV has duplicated column names (e.g. two ``code_distance``)."""
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


# T-cultivation runtime lines vs STAR (matches fidelity evaluation triples).
# Each tuple is (code_distance, fidelity_target, factory_physical_size).
T_CULTIVATION_RUNTIME_LINE_SETTINGS: list[tuple[int, float, int]] = [
    (7, 1e-8, 2),
    (13, 1e-8, 4),
    (13, 1e-10, 4),
]


def _plot_star_vs_t_cultivation_best(
    star_layers: dict[str, pd.DataFrame],
    t_layers: dict[str, pd.DataFrame],
    output_dir: str,
    target_aods: list[int] | None = None,
):
    """Compare STAR (d=7,d=9) and T-cultivation runtime in one 2x3 figure."""
    if target_aods is None:
        target_aods = [2, 5]

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

    # Use setting 4 (index 4)
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

    star_distances = [d for d in [7, 9] if d in star_dist_all]
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
    t_style_colors = ["tab:orange", "tab:green", "tab:purple"]

    for row_idx, aod in enumerate(target_aods):
        for col_idx, layer_name in enumerate(layer_order):
            ax = axes[row_idx][col_idx]
            star_filtered = star_layers[layer_name][
                _setting_filter(star_layers[layer_name], setting_4)
            ].copy()
            t_work = t_layers[layer_name]

            star_ref_values: list[float] = []
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
                star_ref_values.extend(
                    star_a["execution_time_mean"].to_numpy(dtype=float).tolist()
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
                for cd, ft, fps in T_CULTIVATION_RUNTIME_LINE_SETTINGS:
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
                        T_CULTIVATION_RUNTIME_LINE_SETTINGS.index((cd, ft, fps))
                        % len(t_style_colors)
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
                        label=f"T Cultivation, d={int(cd)}, size={int(fps)}, LER={float(ft):g}",
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

    fig.suptitle(
        "STAR vs T cultivation runtime · "
        "Columns: Full Trotter | ZZ layers | X layer · "
        "Rows: AOD 2 (top), AOD 5 (bottom)",
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
        f"{'-'.join(str(a) for a in target_aods)}.pdf",
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


def _plot_t_cultivation_multi_aod(
    t_layers: dict[str, pd.DataFrame],
    output_dir: str,
):
    """T-cultivation AOD comparison: one row per setting (same triples as STAR plot), three columns for layers."""
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
        print("Skipping T-cultivation multi-AOD plot: missing setting columns.")
        return

    n_settings = len(T_CULTIVATION_RUNTIME_LINE_SETTINGS)
    fig, axes = plt.subplots(
        n_settings,
        len(layer_order),
        figsize=(5.6 * len(layer_order), 4.8 * n_settings),
        squeeze=False,
    )

    plotted_any = False
    for row_idx, (cd, ft, fps) in enumerate(T_CULTIVATION_RUNTIME_LINE_SETTINGS):
        row_label = _format_t_cultivation_setting_label(cd, ft, fps)
        for col_idx, layer_name in enumerate(layer_order):
            ax = axes[row_idx][col_idx]
            layer_df = t_layers[layer_name]
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
                    label=f"AOD={aod}",
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
    _apply_shared_bottom_legend(fig, ncol=5, anchor_y=0.03)
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


def process_t_cultivation_runtime_comparison(
    star_csv_file: str,
    t_cultivation_csv_file: str,
    output_dir: str,
):
    """Generate runtime comparison between STAR and T-cultivation profiling results."""
    os.makedirs(output_dir, exist_ok=True)

    star_df = pd.read_csv(star_csv_file, engine="python", on_bad_lines="skip")
    t_df = pd.read_csv(t_cultivation_csv_file, engine="python", on_bad_lines="skip")
    t_df = _dedupe_duplicate_columns(t_df)
    star_df = _coerce_result_cols_numeric(star_df)
    star_df = _normalize_config_types(star_df)
    t_df = _coerce_result_cols_numeric(t_df)
    t_df = _normalize_config_types(t_df)

    star_layers = _build_layer_frames(star_df)
    t_layers = _build_layer_frames(t_df)

    _plot_star_vs_t_cultivation_best(
        star_layers, t_layers, output_dir, target_aods=[2, 5]
    )
    _plot_t_cultivation_multi_aod(t_layers, output_dir)
    print("Saved T-cultivation runtime comparison plots to", output_dir)


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def process_csv(csv_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    df = _coerce_result_cols_numeric(df)
    df = _normalize_config_types(df)
    # plot_microarch_comp_average_all(df, output_dir + "/microarch_all")
    # plot_microarch_comp_setting(df, output_dir + "/microarch_setting", setting_idx=4)
    plot_nAOD_placement_lines(df, output_dir, setting_idx=6, skip_placements=True)
    plot_nAOD_placement_lines(df, output_dir, setting_idx=4, skip_placements=True)

    print("Saved plots to", output_dir)


if __name__ == "__main__":
    # Microarchitecture / timing sweep (e.g. analog rotation or STAR CSV export).
    # T-cultivation *fidelity* and distance-9 extrapolation use
    # ``compare_fidelity.load_and_process_data`` (``t_cultivation_fidelity_results.csv``,
    # or ``evaluation_results.csv`` if it contains the same fidelity columns).
    csv_file = "output/evaluation/evaluation_results.csv"
    output_dir = "output/analysis_plots"
    try:
        process_csv(csv_file, output_dir)
    except Exception as e:
        print(f"Skipping one-round processing due to error: {e}")

    full_trotter_csv = (
        "output/evaluation/fidelity/star_full_trotter_profiling_results.csv"
    )
    full_trotter_output_dir = "output/analysis_plots/full_trotter"
    process_full_trotter_csv(full_trotter_csv, full_trotter_output_dir)

    t_cultivation_profiling_csv = (
        "output/evaluation/fidelity/t_cultivation_fidelity_profiling_results.csv"
    )
    runtime_compare_output_dir = "output/analysis_plots/t_cultivation_runtime"
    if os.path.exists(t_cultivation_profiling_csv):
        process_t_cultivation_runtime_comparison(
            full_trotter_csv,
            t_cultivation_profiling_csv,
            runtime_compare_output_dir,
        )
