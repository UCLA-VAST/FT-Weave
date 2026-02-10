# analyze_results.py
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os

mapping = {"True": 1, "False": 0, "2": 2}

SETTINGS = [
    (True, "naive", False),  # vanilla
    (False, "naive", False),  # optimized return
    (False, "matching", False),  # optimized return + matching TMR
    (False, "matching", True),  # optimized return + matching TMR + skip partial RUS
    (False, "matching", 2),
]  # optimized return + matching TMR + skip

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


# ------------------------------
# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
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

    for l in lines:
        sub = df[df[line_col] == l].sort_values(SWEEP_COL)

        plt.errorbar(
            sub[SWEEP_COL],
            sub["total_time_mean"],
            yerr=sub["total_time_std"],
            marker="o",
            capsize=4,
            label=str(l),
        )

    plt.xlabel(SWEEP_COL)
    plt.ylabel("total_time")
    plt.legend(title="microarchitecture")
    plt.title("Total Execution Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_total_time.pdf"))
    plt.close()


def _plot_movement_bar(df, line_col, prefix, output_dir):
    lines = sorted(df[line_col].unique())
    x_vals = sorted(df[SWEEP_COL].unique())
    width = 0.8 / len(lines)

    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i) for i in range(10)]

    plt.figure(figsize=(7, 5))
    handles = []
    for i, l in enumerate(lines):
        sub = df[df[line_col] == l].sort_values(SWEEP_COL)

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
        patch = mpatches.Patch(color=base_color, label=l)
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
    df, output_dir, sweep_col="n_qubits", setting_idx=3, skip_placements=True
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
    plt.title("Total time for placement × #aod")
    plt.legend(fontsize=7, ncol=2)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"total_time_placement_nAOD_setting_{setting_idx}_skip_placements-{skip_placements}.pdf",
        )
    )
    plt.close()


# ------------------------------------------------------------
# Fig 1 : placement comparison (average over settings)
# ------------------------------------------------------------
def plot_microarch_comp_average_all(df, output_dir):
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
def plot_microarch_comp_setting(df, output_dir, setting_idx=1):
    os.makedirs(output_dir, exist_ok=True)
    setting = SETTINGS[setting_idx]
    df_ctrl = df[
        (df["trivial_return"] == setting[0])
        & (df["tmr_assignment_method"] == setting[1])
        & (df["consider_skip_rus"] == setting[2])
    ]

    g = (
        df_ctrl.groupby(["n_aods", "placement", "n_qubits"])[RESULT_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )

    g.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in g.columns
    ]
    aods = sorted(g["n_aods"].unique())
    for a in [aods[0], aods[-1]]:  # only plot for lowest and highest AODs
        _plot_total_time(
            g[g["n_aods"] == a],
            "placement",
            f"fig2_nAOD{a}_setting_{setting_idx}",
            output_dir,
        )
        _plot_movement_bar(
            g[g["n_aods"] == a],
            "placement",
            f"fig2_nAOD{a}_setting_{setting_idx}",
            output_dir,
        )


# ------------------------------------------------------------
# Ablation study (col-based placement)
# ------------------------------------------------------------
def plot_ablation(df, output_dir, placement):
    os.makedirs(output_dir, exist_ok=True)
    df_col = df[df["placement"] == placement]
    labels = [
        "vanilla",
        "optimized return",
        "optimized return + matching TMR",
        "optimized return + matching TMR + skip parital RUS",
        "optimized return + matching TMR + skip whole RUS",
    ]
    df_list = []
    for setting in SETTINGS:
        df_tmp = df_col[
            (df_col["trivial_return"] == setting[0])
            & (df_col["tmr_assignment_method"] == setting[1])
            & (df_col["consider_skip_rus"] == setting[2])
        ]
        df_list.append(df_tmp)
    gs = []
    for subdf in df_list:
        g = (
            subdf.groupby(["n_aods", "n_qubits"])[RESULT_COLS]
            .agg(["mean", "std"])
            .reset_index()
        )

        g.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c for c in g.columns
        ]

        gs.append(g)

    aods = sorted(gs[0]["n_aods"].unique())
    for a in [aods[0], aods[-1]]:  # only plot for lowest and highest AODs
        tmp_gs = [g[g["n_aods"] == a] for g in gs]
        _plot_ablation_total_time(tmp_gs, labels, f"ablation_nAOD{a}", output_dir)
        _plot_ablation_movement_bar(tmp_gs, labels, f"ablation_nAOD{a}", output_dir)


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def process_csv(csv_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    df["consider_skip_rus"] = df["consider_skip_rus"].map(mapping)
    plot_microarch_comp_average_all(df, output_dir + "/microarch_all")
    plot_microarch_comp_setting(df, output_dir + "/microarch_setting", setting_idx=4)
    plot_ablation(df, output_dir + "/ablation_checkerboard", "checkerboard")
    plot_nAOD_placement_lines(df, output_dir, setting_idx=4, skip_placements=False)
    plot_nAOD_placement_lines(df, output_dir, setting_idx=4)

    print("Saved plots to", output_dir)


if __name__ == "__main__":
    csv_file = "output/evaluation/evaluation_results.csv"
    output_dir = "output/analysis_plots"
    process_csv(csv_file, output_dir)
