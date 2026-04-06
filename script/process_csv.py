# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import math
import os
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz


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


def _setting_filter(df: pd.DataFrame, setting: tuple) -> pd.Series:
    return (
        (df["trivial_return"] == setting[0])
        & (df["tmr_assignment_method"] == setting[1])
        & (df["consider_skip_rus"] == setting[2])
        & (df["decompose_move"] == setting[3])
        & (df["parallel_execution"] == setting[4])
    )


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


def _plot_ablation_total_time(gs, labels, prefix, output_dir):

    plt.figure(figsize=(7, 5))

    for g, label in zip(gs, labels):
        sub = g.sort_values(SWEEP_COL)
        # print(sub)
        plt.errorbar(
            sub[SWEEP_COL],
            sub["total_time_mean"],
            yerr=sub["total_time_std"],
            marker="o",
            capsize=4,
            label=label,
        )

    # print the average improvement percentage for each setting compared to the previous setting
    # print compared with the vanilla setting
    vanilla = gs[0].sort_values(SWEEP_COL)
    for i in range(1, len(gs)):
        prev = gs[i - 1].sort_values(SWEEP_COL)
        curr = gs[i].sort_values(SWEEP_COL)
        # Align by n_qubits
        merged = pd.merge(
            prev[[SWEEP_COL, "total_time_mean"]],
            curr[[SWEEP_COL, "total_time_mean"]],
            on=SWEEP_COL,
            suffixes=("_prev", "_curr"),
        )
        merged["improvement"] = (
            (merged["total_time_mean_prev"] - merged["total_time_mean_curr"])
            / merged["total_time_mean_prev"]
            * 100
        )
        avg_improvement = merged["improvement"].mean()
        print(
            f"Average improvement from '{labels[i-1]}' to '{labels[i]}': {avg_improvement:.2f}%"
        )
        merged = pd.merge(
            vanilla[[SWEEP_COL, "total_time_mean"]],
            curr[[SWEEP_COL, "total_time_mean"]],
            on=SWEEP_COL,
            suffixes=("_vanilla", "_curr"),
        )
        merged["improvement"] = (
            (merged["total_time_mean_vanilla"] - merged["total_time_mean_curr"])
            / merged["total_time_mean_vanilla"]
            * 100
        )
        avg_improvement = merged["improvement"].mean()
        print(
            f"Average improvement from vanilla to '{labels[i]}': {avg_improvement:.2f}%"
        )

    plt.xlabel(SWEEP_COL)
    plt.ylabel("total_time")
    plt.legend(title="settings")
    plt.title("Ablation Study: Total Execution Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_total_time.pdf"))
    plt.close()


def _plot_ablation_movement_bar(gs, labels, prefix, output_dir):
    """
    gs: list of DataFrames, each grouped by ['n_aods', 'n_qubits'] with mean/std columns
    labels: list of strings for each DataFrame (ablation setting)
    """
    if len(gs) == 0:
        print("No data provided to plot_ablation_movement_bar")
        return

    # get all n_qubits (x-axis)
    x_vals = sorted(gs[0]["n_qubits"].unique())
    n_lines = len(gs)
    width = 0.8 / n_lines  # side-by-side bars

    plt.figure(figsize=(7, 5))

    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i) for i in range(10)]
    handles = []
    for i, (g, label) in enumerate(zip(gs, labels)):
        sub = g.sort_values("n_qubits")

        # x positions for this line
        x = [v + i * width for v in range(len(x_vals))]

        move = sub["movement_time_mean"].values
        ret = sub["return_movement_time_mean"].values

        base_color = base_colors[i % len(base_colors)]

        # lighter shade for return
        lighter = mcolors.to_rgba(base_color, alpha=0.35)

        # stacked bars: bottom=move (black), top=return (grey)
        plt.bar(
            x,
            move,
            width=width,
            color=base_color,
        )
        plt.bar(
            x,
            ret,
            width=width,
            bottom=move,
            color=lighter,
        )
        patch = mpatches.Patch(color=base_color, label=label)
        handles.append(patch)

    # x-ticks at center of grouped bars
    plt.xticks([r + width * (n_lines / 2) for r in range(len(x_vals))], x_vals)
    plt.xlabel("n_qubits")
    plt.ylabel("movement time")

    # Legend: only Move / Return
    move_patch = mpatches.Patch(color="black", label="Move")
    return_patch = mpatches.Patch(color="grey", label="Return")
    handles.append(move_patch)
    handles.append(return_patch)
    plt.legend(title="settings/move type", handles=handles)
    plt.title("Ablation Study: Movement Time Breakdown")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_movement.pdf"))
    plt.close()


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
    plt.ylabel("total_time")
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
            total_time_count=("total_time", "count"),
            movement_time_mean=("movement_time", "mean"),
            movement_time_std=("movement_time", "std"),
            movement_time_count=("movement_time", "count"),
            return_movement_time_mean=("return_movement_time", "mean"),
            return_movement_time_std=("return_movement_time", "std"),
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
                    merged_angle_count = sum(
                        count
                        for round_idx, count in enumerate(counts)
                        if round_idx != INDIVIDUAL_ROUND
                    )
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
    if round_name == MERGED_ROUND_NAME:
        return "ZZ layers"
    if round_name == f"round_{INDIVIDUAL_ROUND}":
        return "X layer"
    return round_name


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
    plt.ylabel("total_time")
    plt.ylim(bottom=20, top=200)
    plt.title("Total time for placement × #aod")
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(fontsize=7, ncol=2)
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
# ------------------------------------------------------------
def plot_ablation(df, output_dir, placement, tag=""):
    os.makedirs(output_dir, exist_ok=True)
    df_col = df[df["placement"] == placement]
    labels = [
        "vanilla",
        "opt return",
        "opt return + part. RUS",
        "opt return + skip whole RUS",
        "opt decomp return + skip whole RUS",
        "opt return + async. RUS",
        # "opt return + skip part. RUS + async. RUS",
        "opt return + skip whole RUS + async. RUS",
    ]

    gs = []
    labels_used = []
    for label, setting in zip(labels, SETTINGS):
        subdf = df_col[
            (df_col["trivial_return"] == setting[0])
            & (df_col["tmr_assignment_method"] == setting[1])
            & (df_col["consider_skip_rus"] == setting[2])
            & (df_col["decompose_move"] == setting[3])
            & (df_col["parallel_execution"] == setting[4])
        ]
        if subdf.empty:
            continue

        g = (
            subdf.groupby(["n_aods", "n_qubits"])[RESULT_COLS]
            .agg(["mean", "std"])
            .reset_index()
        )
        g.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c for c in g.columns
        ]
        if not g.empty:
            gs.append(g)
            labels_used.append(label)

    if len(gs) == 0:
        print(f"No data found for ablation ({placement}, tag={tag}), skipping.")
        return

    all_aods = sorted(
        set().union(*[set(g["n_aods"].unique()) for g in gs if "n_aods" in g.columns])
    )
    if len(all_aods) == 0:
        print(f"No AOD data found for ablation ({placement}, tag={tag}), skipping.")
        return

    suffix = f"_{tag}" if tag else ""
    target_aod = all_aods[1] if len(all_aods) > 1 else all_aods[0]

    tmp_gs = []
    tmp_labels = []
    for g, lbl in zip(gs, labels_used):
        sub = g[g["n_aods"] == target_aod]
        if not sub.empty:
            tmp_gs.append(sub)
            tmp_labels.append(lbl)

    if len(tmp_gs) == 0:
        print(
            f"No data for target AOD={target_aod} in ablation ({placement}, tag={tag}), skipping."
        )
        return

    _plot_ablation_total_time(
        tmp_gs, tmp_labels, f"ablation_nAOD{target_aod}{suffix}", output_dir
    )
    _plot_ablation_movement_bar(
        tmp_gs, tmp_labels, f"ablation_nAOD{target_aod}{suffix}", output_dir
    )


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
    """Create combined microarch plots with rows for each round.

    dfs_dict: dict of {round_name -> dataframe}
    """
    os.makedirs(output_dir, exist_ok=True)
    if setting_override is None:
        setting = SETTINGS[setting_idx]
    else:
        setting = setting_override

    # Filter and aggregate data for each round
    agg_data = {}
    for round_name, df in dfs_dict.items():
        df_ctrl = df[_setting_filter(df, setting)]

        if df_ctrl.empty:
            continue

        g = _aggregate_microarch_trials(df_ctrl)
        agg_data[round_name] = g

    if not agg_data:
        print(
            f"No data for microarch setting_idx={setting_idx}, skipping combined plot."
        )
        return

    # Get AODs from first round
    aods = sorted(agg_data[list(agg_data.keys())[0]]["n_aods"].unique())
    if len(aods) == 0:
        print(f"No AOD values for microarch setting_idx={setting_idx}, skipping.")
        return

    round_names = list(agg_data.keys())

    aods_to_plot = [aods[0], aods[-1]]
    if 5 in aods and 5 not in aods_to_plot:
        aods_to_plot.append(5)

    if setting_label is not None:
        execution_mode_title = f" - {setting_label}"
    else:
        execution_mode_title = ""
        if setting_idx == 4:
            execution_mode_title = " - sync. execution"
        elif setting_idx == 6:
            execution_mode_title = " - async. execution"

    file_setting_tag = (
        str(setting_idx)
        if setting_override is None
        else (setting_label or "auto").replace(" ", "_").replace(",", "")
    )

    def _draw_round_plots(total_ax, movement_ax, round_name, g_aod):
        lines = sorted(g_aod["placement"].unique())

        for line_name in lines:
            sub = g_aod[g_aod["placement"] == line_name].sort_values(SWEEP_COL)
            total_ax.errorbar(
                sub[SWEEP_COL],
                sub["total_time_mean"],
                yerr=sub["total_time_std"],
                marker="o",
                capsize=4,
                label=str(line_name),
            )

        total_x_vals = sorted(g_aod[SWEEP_COL].unique())
        bounds = _get_theoretical_lower_bound(total_x_vals)
        if round_name == "full_trotter":
            ideal_rounds = bounds["full_trotter"]
            total_ax.plot(
                total_x_vals,
                ideal_rounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
            )
        elif round_name == MERGED_ROUND_NAME:
            zz_bounds = bounds["zz_layer"]
            total_ax.plot(
                total_x_vals,
                zz_bounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
            )
        elif round_name == f"round_{INDIVIDUAL_ROUND}":
            x_bounds = bounds["x_layer"]
            total_ax.plot(
                total_x_vals,
                x_bounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
            )
        total_ax.set_xticks(total_x_vals)
        total_ax.set_xticklabels(
            _format_nqubit_ticklabels(total_x_vals, round_name, round_angle_lookup),
            rotation=45,
            ha="right",
        )
        total_ax.set_xlabel("n_qubits (angles)")
        total_ax.set_ylabel("total_time")
        total_ax.legend(title="microarchitecture")
        total_ax.set_title(f"Total Execution Time - {_format_round_title(round_name)}")

        x_vals = sorted(g_aod[SWEEP_COL].unique())
        width = 0.8 / len(lines)
        cmap = plt.get_cmap("tab10")
        base_colors = [cmap(i) for i in range(10)]
        handles = []

        for i, line_name in enumerate(lines):
            sub = g_aod[g_aod["placement"] == line_name].sort_values(SWEEP_COL)
            x = [v + i * width for v in range(len(x_vals))]
            move = sub["movement_time_mean"].values
            ret = sub["return_movement_time_mean"].values
            base_color = base_colors[i % len(base_colors)]
            lighter = mcolors.to_rgba(base_color, alpha=0.35)

            movement_ax.bar(
                x,
                move,
                width=width,
                color=base_color,
            )
            movement_ax.bar(
                x,
                ret,
                width=width,
                bottom=move,
                color=lighter,
            )
            patch = mpatches.Patch(color=base_color, label=line_name)
            handles.append(patch)

        movement_ax.set_xticks(
            [r + width * (len(lines) / 2) for r in range(len(x_vals))]
        )
        movement_ax.set_xticklabels(
            _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
            rotation=45,
            ha="right",
        )
        movement_ax.set_xlabel("n_qubits (angles)")
        movement_ax.set_ylabel("movement time")
        movement_ax.set_ylim(bottom=0, top=1100)
        movement_ax.set_title(
            f"Movement Time Breakdown - {_format_round_title(round_name)}"
        )

        move_patch = mpatches.Patch(color="black", label="Move")
        return_patch = mpatches.Patch(color="grey", label="Return")
        handles.extend([move_patch, return_patch])
        movement_ax.legend(title="microarch/move type", handles=handles, fontsize=8)

    # Create combined figures for each AOD in both orientations
    for aod_val in aods_to_plot:
        # vertical: execution time in a column
        fig, axes = plt.subplots(
            len(round_names), 2, figsize=(14, 4.5 * len(round_names))
        )
        if len(round_names) == 1:
            axes = axes.reshape(1, 2)

        for row_idx, round_name in enumerate(round_names):
            g = agg_data[round_name]
            g_aod = g[g["n_aods"] == aod_val]
            if g_aod.empty:
                continue
            _draw_round_plots(axes[row_idx, 0], axes[row_idx, 1], round_name, g_aod)

        fig.suptitle(
            f"microarchitecture evaluation{execution_mode_title} (n_aods={aod_val})",
            fontsize=20,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.985))
        fig.savefig(
            os.path.join(
                output_dir,
                f"fig2_nAOD{aod_val}_setting_{file_setting_tag}_combined_vertical.pdf",
            )
        )
        # Keep backward-compatible filename mapped to the vertical layout.
        plt.close(fig)

        # horizontal: execution time in a row
        fig, axes = plt.subplots(
            2, len(round_names), figsize=(4.5 * len(round_names), 9)
        )
        if len(round_names) == 1:
            axes = axes.reshape(2, 1)

        for col_idx, round_name in enumerate(round_names):
            g = agg_data[round_name]
            g_aod = g[g["n_aods"] == aod_val]
            if g_aod.empty:
                continue
            _draw_round_plots(axes[0, col_idx], axes[1, col_idx], round_name, g_aod)

        fig.suptitle(
            f"microarchitecture evaluation{execution_mode_title} (n_aods={aod_val})",
            fontsize=20,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.985))
        fig.savefig(
            os.path.join(
                output_dir,
                f"fig2_nAOD{aod_val}_setting_{file_setting_tag}_combined_horizontal.pdf",
            )
        )
        plt.close(fig)


def plot_ablation_combined(dfs_dict, output_dir, placement, round_angle_lookup=None):
    """Create combined ablation plots with rows for each round."""
    os.makedirs(output_dir, exist_ok=True)

    labels = [
        "vanilla",
        "opt return",
        "opt return + skip whole RUS",
        "opt decomp return + skip whole RUS",
        "opt return + skip whole RUS + async. RUS",
    ]
    setting_indices = [0, 1, 3, 4, 6]

    placement_all_df = pd.concat(
        [
            round_df[round_df["placement"] == placement]
            for round_df in dfs_dict.values()
        ],
        ignore_index=True,
    )

    setting_candidates = []
    for label, setting_idx in zip(labels, setting_indices):
        setting = SETTINGS[setting_idx]
        if not placement_all_df[_setting_filter(placement_all_df, setting)].empty:
            setting_candidates.append((label, setting))

    if len(setting_candidates) == 0:
        auto_settings = _get_available_settings(placement_all_df, max_settings=5)
        setting_candidates = [
            (f"cfg_{idx+1}", setting) for idx, setting in enumerate(auto_settings)
        ]

    # Process each round
    agg_data = {}
    for round_name, df in dfs_dict.items():
        df_col = df[df["placement"] == placement]

        gs = []
        labels_used = []
        for label, setting in setting_candidates:
            subdf = df_col[_setting_filter(df_col, setting)]
            if subdf.empty:
                continue

            g = (
                subdf.groupby(["n_aods", "n_qubits"])[RESULT_COLS]
                .agg(["mean", "std"])
                .reset_index()
            )
            g.columns = [
                "_".join(c).strip("_") if isinstance(c, tuple) else c for c in g.columns
            ]
            if not g.empty:
                gs.append(g)
                labels_used.append(label)

        if len(gs) > 0:
            agg_data[round_name] = (gs, labels_used)

    if not agg_data:
        print(f"No data found for ablation ({placement}), skipping combined plot.")
        return

    # Get all AODs across all settings/rounds
    all_aods = sorted(
        {aod for gs, _ in agg_data.values() for g in gs for aod in g["n_aods"].unique()}
    )
    if len(all_aods) == 0:
        print(f"No AOD data found for ablation ({placement}), skipping.")
        return

    # Pick AOD with most setting coverage on full_trotter (fallback: all rounds)
    def _coverage_for_aod(round_name, aod):
        gs_labels = agg_data.get(round_name)
        if gs_labels is None:
            return 0
        gs, _ = gs_labels
        return sum(1 for g in gs if not g[g["n_aods"] == aod].empty)

    if "full_trotter" in agg_data:
        primary_target_aod = max(
            all_aods, key=lambda a: (_coverage_for_aod("full_trotter", a), -a)
        )
    else:
        primary_target_aod = max(
            all_aods,
            key=lambda a: (
                sum(_coverage_for_aod(round_name, a) for round_name in agg_data.keys()),
                -a,
            ),
        )

    target_aods = [primary_target_aod]
    if 5 in all_aods and 5 not in target_aods:
        target_aods.append(5)

    round_items = list(agg_data.items())

    def _draw_ablation_round(
        total_ax, movement_ax, round_name, gs, labels_used, target_aod
    ):
        tmp_gs = []
        tmp_labels = []
        for g, lbl in zip(gs, labels_used):
            sub = g[g["n_aods"] == target_aod]
            if not sub.empty:
                tmp_gs.append(sub)
                tmp_labels.append(lbl)

        if len(tmp_gs) == 0:
            return

        for g, label in zip(tmp_gs, tmp_labels):
            sub = g.sort_values(SWEEP_COL)
            total_ax.errorbar(
                sub[SWEEP_COL],
                sub["total_time_mean"],
                yerr=sub["total_time_std"],
                marker="o",
                capsize=4,
                label=label,
            )

        total_x_vals = sorted(tmp_gs[0][SWEEP_COL].unique())
        bounds = _get_theoretical_lower_bound(total_x_vals)
        if round_name == "full_trotter":
            ideal_rounds = bounds["full_trotter"]
            total_ax.plot(
                total_x_vals,
                ideal_rounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
            )
        elif round_name == MERGED_ROUND_NAME:
            zz_bounds = bounds["zz_layer"]
            total_ax.plot(
                total_x_vals,
                zz_bounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
            )
        elif round_name == f"round_{INDIVIDUAL_ROUND}":
            x_bounds = bounds["x_layer"]
            total_ax.plot(
                total_x_vals,
                x_bounds,
                "k--",
                linewidth=1.5,
                label="theoretical lower bound",
            )
        total_ax.set_xticks(total_x_vals)
        total_ax.set_xticklabels(
            _format_nqubit_ticklabels(total_x_vals, round_name, round_angle_lookup),
            rotation=45,
            ha="right",
        )
        total_ax.set_xlabel("n_qubits (angles)")
        total_ax.set_ylabel("total_time")
        total_ax.legend(title="settings", fontsize=8)
        total_ax.set_title(f"Total Execution Time - {_format_round_title(round_name)}")

        x_vals = sorted(tmp_gs[0][SWEEP_COL].unique())
        n_lines = len(tmp_gs)
        width = 0.8 / n_lines

        cmap = plt.get_cmap("tab10")
        base_colors = [cmap(i) for i in range(10)]
        handles = []

        for i, (g, label) in enumerate(zip(tmp_gs, tmp_labels)):
            sub = g.sort_values("n_qubits")
            x = [v + i * width for v in range(len(x_vals))]
            move = sub["movement_time_mean"].values
            ret = sub["return_movement_time_mean"].values
            base_color = base_colors[i % len(base_colors)]
            lighter = mcolors.to_rgba(base_color, alpha=0.35)

            movement_ax.bar(
                x,
                move,
                width=width,
                color=base_color,
            )
            movement_ax.bar(
                x,
                ret,
                width=width,
                bottom=move,
                color=lighter,
            )
            patch = mpatches.Patch(color=base_color, label=label)
            handles.append(patch)

        movement_ax.set_xticks([r + width * (n_lines / 2) for r in range(len(x_vals))])
        movement_ax.set_xticklabels(
            _format_nqubit_ticklabels(x_vals, round_name, round_angle_lookup),
            rotation=45,
            ha="right",
        )
        movement_ax.set_xlabel("n_qubits (angles)")
        movement_ax.set_ylabel("movement time")
        movement_ax.set_title(
            f"Movement Time Breakdown - {_format_round_title(round_name)}"
        )

        move_patch = mpatches.Patch(color="black", label="Move")
        return_patch = mpatches.Patch(color="grey", label="Return")
        handles.extend([move_patch, return_patch])
        movement_ax.legend(title="settings/move type", handles=handles, fontsize=8)

    for target_aod in target_aods:
        # vertical: each round in a row, metrics in columns
        fig, axes = plt.subplots(
            len(round_items), 2, figsize=(14, 4.5 * len(round_items))
        )
        if len(round_items) == 1:
            axes = axes.reshape(1, -1)

        for row_idx, (round_name, (gs, labels_used)) in enumerate(round_items):
            _draw_ablation_round(
                axes[row_idx, 0],
                axes[row_idx, 1],
                round_name,
                gs,
                labels_used,
                target_aod,
            )

        fig.suptitle(f"ablation study (n_aods={target_aod})")
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        fig.savefig(
            os.path.join(
                output_dir,
                f"ablation_nAOD{target_aod}_{placement}_combined_vertical.pdf",
            )
        )
        # Keep backward-compatible filename mapped to the vertical layout.
        plt.close(fig)

        # horizontal: each round in a column, metrics in rows
        fig, axes = plt.subplots(
            2, len(round_items), figsize=(4.5 * len(round_items), 9)
        )
        if len(round_items) == 1:
            axes = axes.reshape(2, 1)

        for col_idx, (round_name, (gs, labels_used)) in enumerate(round_items):
            _draw_ablation_round(
                axes[0, col_idx],
                axes[1, col_idx],
                round_name,
                gs,
                labels_used,
                target_aod,
            )

        fig.suptitle(f"ablation study (n_aods={target_aod})")
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        fig.savefig(
            os.path.join(
                output_dir,
                f"ablation_nAOD{target_aod}_{placement}_combined_horizontal.pdf",
            )
        )
        plt.close(fig)


def plot_nAOD_placement_lines_combined(
    dfs_dict,
    output_dir,
    setting_idx=4,
    round_angle_lookup=None,
    setting_override=None,
    setting_label=None,
):
    """Create combined AOD study plots with rows for each round and one figure per code distance."""
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
            .agg(["mean", "std"])
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

    # Get placements and AODs
    first_grouped = list(agg_data.values())[0]
    placements = sorted(first_grouped["placement"].unique())
    if "code_distance" in first_grouped.columns:
        distance_values = sorted(
            first_grouped["code_distance"].dropna().astype(int).unique()
        )
    else:
        distance_values = [None]
    min_aods = first_grouped["n_aods"].min()
    max_aods = first_grouped["n_aods"].max()
    if setting_idx == 4:
        execution_mode = "sync. execution"
    elif setting_idx == 6:
        execution_mode = "async. execution"
    else:
        execution_mode = "execution"

    if setting_label is not None:
        execution_mode = setting_label

    figure_title = f"AOD number comparison ({execution_mode})"
    file_setting_tag = (
        str(setting_idx)
        if setting_override is None
        else (setting_label or "auto").replace(" ", "_").replace(",", "")
    )

    cmap_placement = plt.get_cmap("tab10")
    base_colors = [cmap_placement(i) for i in range(10)]

    round_items = list(agg_data.items())

    def _draw_aod_round(ax, round_name, grouped):
        for p_idx, placement in enumerate(placements):
            if placement not in ["col_based", "checkerboard"]:
                continue
            sub = grouped[grouped["placement"] == placement]

            aods_values = sorted(sub["n_aods"].unique())
            for aod in aods_values:
                sub_aod = sub[sub["n_aods"] == aod].sort_values("n_qubits")
                ax.errorbar(
                    sub_aod["n_qubits"],
                    sub_aod["total_time_mean"],
                    yerr=sub_aod["total_time_std"],
                    marker="o",
                    capsize=3,
                    color=base_colors[p_idx % len(base_colors)],
                    alpha=0.3 + 0.7 * (aod - min_aods) / (max_aods - min_aods + 1e-5),
                    label=f"{placement}, #aod={aod}",
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
        ax.set_xlabel("n_qubits (angles)")
        ax.set_ylabel("total_time")
        # Apply y-limit only for single trotter rounds, not for full_trotter
        if round_name != "full_trotter":
            ax.set_ylim(0, 300)
        ax.set_title(
            f"Total time for placement × #aod - {_format_round_title(round_name)}"
        )
        ax.legend(fontsize=7, ncol=2)

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
            len(distance_round_items), 1, figsize=(7, 4.5 * len(distance_round_items))
        )
        if len(distance_round_items) == 1:
            axes = [axes]

        for row_idx, (round_name, grouped) in enumerate(distance_round_items):
            _draw_aod_round(axes[row_idx], round_name, grouped)

        fig.suptitle(distance_title)
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        fig.savefig(
            os.path.join(
                output_dir,
                f"total_time_placement_nAOD_setting_{file_setting_tag}_skip_placements-True_combined_vertical{distance_suffix}.pdf",
            )
        )
        plt.close(fig)

        fig, axes = plt.subplots(
            1, len(distance_round_items), figsize=(4.5 * len(distance_round_items), 4.5)
        )
        if len(distance_round_items) == 1:
            axes = [axes]

        for col_idx, (round_name, grouped) in enumerate(distance_round_items):
            _draw_aod_round(axes[col_idx], round_name, grouped)

        fig.suptitle(distance_title)
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        fig.savefig(
            os.path.join(
                output_dir,
                f"total_time_placement_nAOD_setting_{file_setting_tag}_skip_placements-True_combined_horizontal{distance_suffix}.pdf",
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
    """Summarize T-cultivation execution time per trotter step."""
    t_df = t_df.copy()
    numeric_cols = [
        "n_qubits",
        "code_distance",
        "factory_physical_size",
        "fidelity_target",
        "step_execution_time",
    ]
    for col in numeric_cols:
        if col in t_df.columns:
            t_df[col] = pd.to_numeric(t_df[col], errors="coerce")

    summary = (
        t_df.groupby(
            ["n_qubits", "code_distance", "factory_physical_size", "fidelity_target"],
            as_index=False,
        )
        .agg(
            execution_time_mean=("step_execution_time", "mean"),
            execution_time_std=("step_execution_time", "std"),
            n_samples=("step_execution_time", "count"),
        )
        .assign(method="t_cultivation")
    )
    summary["execution_time_std"] = summary["execution_time_std"].fillna(0.0)
    return summary


def process_execution_time_comparison(
    star_csv: str,
    t_cultivation_csv: str,
    output_dir: str,
):
    """
    Generate execution-time comparison CSVs:
    1) STAR vs T-cultivation
    2) T-cultivation standalone (three settings)
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(star_csv):
        raise FileNotFoundError(f"Missing STAR csv: {star_csv}")
    if not os.path.exists(t_cultivation_csv):
        raise FileNotFoundError(f"Missing T-cultivation csv: {t_cultivation_csv}")

    star_df = pd.read_csv(star_csv, engine="python", on_bad_lines="skip")
    t_df = pd.read_csv(t_cultivation_csv, engine="python", on_bad_lines="skip")

    star_summary = _summarize_star_full_trotter_execution(star_df)
    t_summary = _summarize_t_cultivation_execution(t_df)

    # T-cultivation-only summary (keep all settings explicitly)
    t_only_path = os.path.join(output_dir, "t_cultivation_execution_time_summary.csv")
    t_summary.sort_values(
        ["code_distance", "factory_physical_size", "fidelity_target", "n_qubits"]
    ).to_csv(t_only_path, index=False)

    # STAR vs T-cultivation comparison
    star_comp = star_summary.copy()
    star_comp["factory_physical_size"] = pd.NA
    star_comp["fidelity_target"] = pd.NA

    comparison_cols = [
        "method",
        "n_qubits",
        "code_distance",
        "factory_physical_size",
        "fidelity_target",
        "execution_time_mean",
        "execution_time_std",
        "n_samples",
    ]
    comparison = pd.concat(
        [star_comp[comparison_cols], t_summary[comparison_cols]], ignore_index=True
    ).sort_values(
        ["n_qubits", "code_distance", "method", "factory_physical_size", "fidelity_target"]
    )
    comparison_path = os.path.join(
        output_dir, "star_vs_t_cultivation_execution_time_comparison.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    print(f"Saved T-cultivation summary CSV to: {t_only_path}")
    print(f"Saved STAR vs T-cultivation comparison CSV to: {comparison_path}")


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
    plot_ablation(df, output_dir + "/ablation_checkerboard", "checkerboard")
    plot_nAOD_placement_lines(df, output_dir, setting_idx=6, skip_placements=True)
    plot_nAOD_placement_lines(df, output_dir, setting_idx=4, skip_placements=True)

    print("Saved plots to", output_dir)


if __name__ == "__main__":
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

    execution_time_output_dir = "output/analysis_plots/execution_time_comparison"
    t_cultivation_execution_csv = (
        "output/evaluation/fidelity/t_cultivation_circuit_execution_results.csv"
    )
    try:
        process_execution_time_comparison(
            star_csv=full_trotter_csv,
            t_cultivation_csv=t_cultivation_execution_csv,
            output_dir=execution_time_output_dir,
        )
    except Exception as e:
        print(f"Skipping execution-time comparison due to error: {e}")
