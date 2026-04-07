import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def get_n_qubit(layout_str):
    if isinstance(layout_str, str):
        rows, cols = eval(layout_str)
        return rows * cols
    return layout_str[0] * layout_str[1]


def get_code_distances(star_df):
    if "code_distance" not in star_df.columns:
        return [None]
    return sorted(star_df["code_distance"].dropna().astype(int).unique())


def normalize_star_df(star_df):
    star_df = star_df.copy()
    if "skip_rus" not in star_df.columns and "consider_skip_rus" in star_df.columns:
        star_df["skip_rus"] = star_df["consider_skip_rus"]
    return star_df


def format_t_cultivation_setting_label(
    code_distance, fidelity_target, factory_physical_size
):
    return (
        f"d={int(code_distance)}\n"
        f"size={int(factory_physical_size)}\n"
        f"LER={fidelity_target:g}"
    )


def load_and_process_data():
    """Load raw, star, and optional t-cultivation fidelity results."""
    raw_df = pd.read_csv("output/evaluation/fidelity/raw_fidelity_results.csv")
    star_df = pd.read_csv("output/evaluation/fidelity/star_fidelity_results.csv")
    t_cultivation_path = "output/evaluation/fidelity/t_cultivation_fidelity_results.csv"
    t_cultivation_df = (
        pd.read_csv(t_cultivation_path) if os.path.exists(t_cultivation_path) else None
    )

    return raw_df, star_df, t_cultivation_df


def generate_comparison_csv(raw_df, star_df, output_path, t_cultivation_df=None):
    """Generate a comparison CSV with raw, STAR, and optional T-cultivation rows."""

    star_df = normalize_star_df(star_df)

    # Prepare comparison data
    comparison_data = []

    # Add raw results (aggregate by n_qubit)
    for n_qubit in sorted(raw_df["n_qubit"].unique()):
        raw_subset = raw_df[raw_df["n_qubit"] == n_qubit]
        avg_fidelity = raw_subset["fidelity"].mean()

        comparison_data.append(
            {
                "method": "raw",
                "n_qubit": n_qubit,
                "qubit_layout": f"({int(np.sqrt(n_qubit))}, {int(np.sqrt(n_qubit))})",
                "placement": "N/A",
                "n_aods": "N/A",
                "skip_rus": "N/A",
                "trivial_return": "N/A",
                "decompose_move": "N/A",
                "parallel_execution": "N/A",
                "mean_fidelity": avg_fidelity,
                "std_fidelity": raw_subset["fidelity"].std(),
                "n_samples": len(raw_subset),
                "infidelity": 1 - avg_fidelity,
            }
        )

    # Add STAR results (aggregate by configuration)
    # Group by all configuration parameters
    star_grouped = star_df.groupby(
        [
            "qubit_layout",
            "placement",
            "n_aods",
            "skip_rus",
            "trivial_return",
            "decompose_move",
            "parallel_execution",
        ]
    )

    for config, group_df in star_grouped:
        (
            qubit_layout,
            placement,
            n_aods,
            skip_rus,
            trivial_return,
            decompose_move,
            parallel_execution,
        ) = config

        # Parse qubit layout to get n_qubit
        if isinstance(qubit_layout, str):
            rows, cols = eval(qubit_layout)
            n_qubit = rows * cols
        else:
            n_qubit = qubit_layout[0] * qubit_layout[1]

        comparison_data.append(
            {
                "method": "star",
                "n_qubit": n_qubit,
                "qubit_layout": str(qubit_layout),
                "placement": placement,
                "n_aods": n_aods,
                "skip_rus": skip_rus,
                "trivial_return": trivial_return,
                "decompose_move": decompose_move,
                "parallel_execution": parallel_execution,
                "mean_fidelity": group_df["fidelity"].mean(),
                "std_fidelity": group_df["fidelity"].std(),
                "n_samples": len(group_df),
                "infidelity": 1 - group_df["fidelity"].mean(),
            }
        )

    # Add T-cultivation results (aggregate by configuration + three settings)
    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_df = t_cultivation_df.copy()
        t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
        t_df["fidelity_total"] = pd.to_numeric(
            t_df.get("fidelity_total"), errors="coerce"
        )
        t_df = t_df.dropna(subset=["fidelity_total"])
        if not t_df.empty:
            t_grouped = t_df.groupby(
                [
                    "n_qubit",
                    "qubit_layout",
                    "placement",
                    "n_aods",
                    "code_distance",
                    "fidelity_target",
                    "factory_physical_size",
                ]
            )
            for config, group_df in t_grouped:
                (
                    n_qubit,
                    qubit_layout,
                    placement,
                    n_aods,
                    code_distance,
                    fidelity_target,
                    factory_physical_size,
                ) = config
                mean_fidelity = group_df["fidelity_total"].mean()
                comparison_data.append(
                    {
                        "method": "t_cultivation",
                        "n_qubit": n_qubit,
                        "qubit_layout": str(qubit_layout),
                        "placement": placement,
                        "n_aods": n_aods,
                        "skip_rus": "N/A",
                        "trivial_return": "N/A",
                        "decompose_move": "N/A",
                        "parallel_execution": "N/A",
                        "code_distance": code_distance,
                        "fidelity_target": fidelity_target,
                        "factory_physical_size": factory_physical_size,
                        "mean_fidelity": mean_fidelity,
                        "std_fidelity": group_df["fidelity_total"].std(),
                        "n_samples": len(group_df),
                        "infidelity": 1 - mean_fidelity,
                    }
                )

    # Create DataFrame and save
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(["n_qubit", "method", "mean_fidelity"])
    comparison_df.to_csv(output_path, index=False)
    print(f"Comparison CSV saved to: {output_path}")

    return comparison_df


def plot_raw_infidelity_breakdown(raw_df, output_dir):
    """Plot stacked infidelity breakdown for raw results by error term."""

    # Calculate infidelity for each error source
    error_terms = [
        "fidelity_cz",
        "fidelity_1q",
        "fidelity_move",
        "fidelity_idle",
        "fidelity_init",
        "fidelity_measurement",
    ]

    # Create stacked bar plot showing contribution of each error term
    fig, ax = plt.subplots(figsize=(10, 6))

    n_qubits = sorted(raw_df["n_qubit"].unique())
    infidelity_data = {term: [] for term in error_terms}

    for n in n_qubits:
        subset = raw_df[raw_df["n_qubit"] == n]
        for term in error_terms:
            # Calculate contribution to total infidelity
            infidelity_data[term].append(1 - subset[term].mean())

    # Create stacked bar chart
    x = np.arange(len(n_qubits))
    width = 0.6

    bottom = np.zeros(len(n_qubits))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(error_terms))]

    for idx, (term, color) in enumerate(zip(error_terms, colors)):
        values = infidelity_data[term]
        ax.bar(
            x,
            values,
            width,
            label=term.replace("fidelity_", "").upper(),
            bottom=bottom,
            color=color,
            alpha=0.8,
        )
        bottom += values

    ax.set_xlabel("Number of Qubits", fontsize=14)
    ax.set_ylabel("Total Infidelity", fontsize=14)
    ax.set_title("Raw Infidelity Breakdown by Error Source", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(n_qubits, fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    output_path = os.path.join(output_dir, "raw_infidelity_stacked.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Raw infidelity stacked plot saved to: {output_path}")
    plt.close()


def plot_star_infidelity_breakdown(star_df, output_dir):
    """Plot infidelity breakdown for STAR results: 2 rows (placements) × 4 cols (error terms)."""

    # Error terms in STAR data (exclude CNOT and 1Q as they are constant)
    error_terms = [
        "fidelity_of_rz_injection",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
    ]

    star_df = normalize_star_df(star_df)
    star_df["n_qubit"] = star_df["qubit_layout"].apply(get_n_qubit)

    # Create merged figure for all placements
    placements = sorted(star_df["placement"].unique())
    fig, axes = plt.subplots(
        len(placements), len(error_terms), figsize=(18, 6 * len(placements))
    )

    for placement_idx, placement in enumerate(placements):
        placement_df = star_df[star_df["placement"] == placement]

        for term_idx, term in enumerate(error_terms):
            ax = axes[placement_idx, term_idx]

            # For each n_aods configuration, plot the trend
            for n_aods in sorted(placement_df["n_aods"].unique()):
                aod_subset = placement_df[placement_df["n_aods"] == n_aods]
                grouped = aod_subset.groupby("n_qubit")

                n_qubits = sorted(aod_subset["n_qubit"].unique())
                infidelities = [1 - grouped.get_group(n)[term].mean() for n in n_qubits]

                ax.plot(
                    n_qubits,
                    infidelities,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=f"n_aods={n_aods}",
                    alpha=0.7,
                )

            ax.set_xlabel("Number of Qubits", fontsize=14)
            ax.set_ylabel("Infidelity (1 - Fidelity)", fontsize=14)
            term_name = term.replace("fidelity_", "").replace("of_", "").upper()
            ax.set_title(term_name, fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            ax.tick_params(axis="both", labelsize=12)
            if placement_idx == 0:
                ax.legend(fontsize=10)

    # Set row labels
    for placement_idx, placement in enumerate(placements):
        axes[placement_idx, 0].set_ylabel(
            f"{placement.upper()}\nInfidelity", fontsize=14, fontweight="bold"
        )

    fig.suptitle("STAR Infidelity Breakdown by Error Type and Placement", fontsize=18)
    fig.tight_layout()
    output_path = os.path.join(output_dir, "star_infidelity_breakdown.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"STAR infidelity breakdown plot saved to: {output_path}")
    plt.close(fig)

    # Create one clustered stacked chart by distance, averaged over STAR configurations
    stacked_terms = [
        "fidelity_of_rz_injection",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
        "fidelity_cnot",
        "fidelity_1q",
    ]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(stacked_terms))]

    if "code_distance" in star_df.columns:
        available_distances = sorted(
            star_df["code_distance"].dropna().astype(int).unique()
        )
        distances = [d for d in [7, 9] if d in available_distances]
        if len(distances) == 0:
            distances = available_distances
    else:
        distances = []

    if len(distances) == 0:
        print("Skipping STAR stacked distance cluster plot: no code_distance available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    grouped = (
        star_df[star_df["code_distance"].isin(distances)]
        .groupby(["n_qubit", "code_distance"], as_index=False)[stacked_terms]
        .mean()
    )

    n_qubits = sorted(grouped["n_qubit"].unique())
    x_group = np.arange(len(n_qubits))
    group_width = 0.8
    bar_width = group_width / len(distances)

    xtick_positions = []
    xtick_labels = []

    for dist_idx, code_distance in enumerate(distances):
        offset = (dist_idx - (len(distances) - 1) / 2) * bar_width
        x = x_group + offset

        subset = (
            grouped[grouped["code_distance"] == code_distance]
            .set_index("n_qubit")
            .reindex(n_qubits)
        )

        bottom = np.zeros(len(n_qubits))
        for term, color in zip(stacked_terms, colors):
            values = (1 - subset[term]).fillna(0).values
            label = term.replace("fidelity_", "").replace("of_", "").upper()
            if term == "fidelity_1q":
                label = "H"
            ax.bar(
                x,
                values,
                bar_width * 0.9,
                label=label if dist_idx == 0 else "_nolegend_",
                bottom=bottom,
                color=color,
                alpha=0.85,
            )
            bottom += values

        xtick_positions.extend(x.tolist())
        xtick_labels.extend([str(code_distance)] * len(x))

    ax.set_ylabel("Total Infidelity", fontsize=14)
    ax.set_title("STAR Infidelity Breakdown (Distance 7 and 9)", fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=10)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_xlabel("Code Distance", fontsize=12, labelpad=2)

    secax = ax.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    secax.set_xticks(x_group)
    secax.set_xticklabels([str(n) for n in n_qubits], fontsize=11)
    secax.set_xlabel("Number of Qubits", fontsize=12, labelpad=10)
    secax.spines["bottom"].set_position(("outward", 42))

    fig.tight_layout()
    output_path = os.path.join(output_dir, "star_infidelity_stacked_best.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"STAR infidelity stacked plot saved to: {output_path}")
    plt.close(fig)


def plot_t_cultivation_infidelity_breakdown(t_cultivation_df, output_dir):
    """Plot stacked infidelity breakdown for T-cultivation results by setting."""

    stacked_terms = [
        "fidelity_t_injection",
        "fidelity_t_teleportation",
        "fidelity_clifford",
    ]

    t_df = t_cultivation_df.copy()
    t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
    for column in ["fidelity_total", *stacked_terms]:
        if column in t_df.columns:
            t_df[column] = pd.to_numeric(t_df[column], errors="coerce")

    t_df = t_df.dropna(subset=["n_qubit", *stacked_terms])
    if t_df.empty:
        print("Skipping T-cultivation stacked infidelity plot: no usable data")
        return

    grouped = (
        t_df.groupby(
            ["n_qubit", "code_distance", "fidelity_target", "factory_physical_size"],
            as_index=False,
        )[stacked_terms]
        .mean()
        .sort_values(
            ["n_qubit", "code_distance", "factory_physical_size", "fidelity_target"]
        )
    )

    settings = [
        (code_distance, fidelity_target, factory_physical_size)
        for code_distance, fidelity_target, factory_physical_size in sorted(
            grouped[["code_distance", "fidelity_target", "factory_physical_size"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    ]

    if len(settings) == 0:
        print("Skipping T-cultivation stacked infidelity plot: no settings available")
        return

    n_qubits = sorted(grouped["n_qubit"].unique())

    total_infidelity = (1 - grouped[stacked_terms]).sum(axis=1)
    grouped = grouped.assign(total_infidelity=total_infidelity)
    setting_totals = grouped.groupby(
        ["code_distance", "fidelity_target", "factory_physical_size"]
    )["total_infidelity"].max()
    dominant_setting = setting_totals.idxmax()
    dominant_group = grouped[
        (grouped["code_distance"] == dominant_setting[0])
        & (grouped["fidelity_target"] == dominant_setting[1])
        & (grouped["factory_physical_size"] == dominant_setting[2])
    ]
    other_max = setting_totals.drop(dominant_setting).max()
    dominant_min = dominant_group["total_infidelity"].min()

    use_broken_axis = (
        pd.notna(other_max) and pd.notna(dominant_min) and other_max < dominant_min
    )

    if use_broken_axis:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
        )
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labelbottom=False, bottom=False)
        ax_bottom.tick_params(top=False)
        bottom_ylim_max = max(other_max * 1.4, other_max + 1e-4, 1e-4)
        top_ylim_min = max(dominant_min * 0.8, bottom_ylim_max * 1.5)
        ax_bottom.set_ylim(0, bottom_ylim_max)
        ax_top.set_ylim(top_ylim_min, grouped["total_infidelity"].max() * 1.08)
    else:
        fig, ax_bottom = plt.subplots(figsize=(12, 6))
        ax_top = None
    x_group = np.arange(len(n_qubits))
    group_width = 0.8
    bar_width = group_width / len(settings)

    xtick_positions = []
    xtick_labels = []

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(stacked_terms))]

    def draw_bars(ax):
        for setting_idx, (
            code_distance,
            fidelity_target,
            factory_physical_size,
        ) in enumerate(settings):
            offset = (setting_idx - (len(settings) - 1) / 2) * bar_width
            x = x_group + offset

            subset = (
                grouped[
                    (grouped["code_distance"] == code_distance)
                    & (grouped["fidelity_target"] == fidelity_target)
                    & (grouped["factory_physical_size"] == factory_physical_size)
                ]
                .set_index("n_qubit")
                .reindex(n_qubits)
            )

            bottom = np.zeros(len(n_qubits))
            for term, color in zip(stacked_terms, colors):
                values = (1 - subset[term]).fillna(0).values
                label = (
                    term.replace("fidelity_", "")
                    .replace("teleportation", "teleport")
                    .upper()
                )
                ax.bar(
                    x,
                    values,
                    bar_width * 0.9,
                    label=(
                        label if setting_idx == 0 and ax is ax_bottom else "_nolegend_"
                    ),
                    bottom=bottom,
                    color=color,
                    alpha=0.85,
                )
                bottom += values

            xtick_positions.extend(x.tolist())
            xtick_labels.extend(
                [
                    format_t_cultivation_setting_label(
                        code_distance, fidelity_target, factory_physical_size
                    )
                ]
                * len(x)
            )

    draw_bars(ax_bottom)
    if ax_top is not None:
        draw_bars(ax_top)

    if ax_top is not None:
        ax_top.grid(True, alpha=0.3, axis="y")
        ax_top.tick_params(axis="y", labelsize=12)
    ax_bottom.set_ylabel("Total Infidelity", fontsize=14, labelpad=22)
    ax_bottom.tick_params(axis="y", labelsize=12)
    ax_bottom.grid(True, alpha=0.3, axis="y")
    ax_bottom.set_xticks(xtick_positions)
    ax_bottom.set_xticklabels(xtick_labels, fontsize=8)
    ax_bottom.set_xlabel("T-cultivation Setting", fontsize=12, labelpad=2)
    ax_bottom.tick_params(axis="x", pad=2)

    if ax_top is not None:
        ax_top.set_title("T-cultivation Infidelity Breakdown by Setting", fontsize=16)
        handles, labels = ax_bottom.get_legend_handles_labels()
        filtered = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if not label.startswith("_")
        ]
        if filtered:
            handles, labels = zip(*filtered)
            ax_top.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
                fontsize=10,
                frameon=True,
                framealpha=0.95,
            )

        ax_top.text(
            0,
            0,
            "//",
            transform=ax_top.transAxes,
            fontsize=16,
            va="center",
            ha="left",
        )
        ax_bottom.text(
            0,
            1,
            "//",
            transform=ax_bottom.transAxes,
            fontsize=16,
            va="center",
            ha="left",
        )
    else:
        ax_bottom.set_title(
            "T-cultivation Infidelity Breakdown by Setting", fontsize=16
        )
        handles, labels = ax_bottom.get_legend_handles_labels()
        filtered = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if not label.startswith("_")
        ]
        if filtered:
            handles, labels = zip(*filtered)
            ax_bottom.legend(handles, labels, loc="upper left", fontsize=10)

    secax = ax_bottom.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    secax.set_xticks(x_group)
    secax.set_xticklabels([str(n) for n in n_qubits], fontsize=11)
    secax.set_xlabel("Number of Qubits", fontsize=12, labelpad=10)
    secax.spines["bottom"].set_position(("outward", 42))

    if ax_top is not None:
        fig.subplots_adjust(hspace=0.05, top=0.92, bottom=0.20, left=0.12, right=0.98)
    else:
        fig.tight_layout()
    output_path = os.path.join(output_dir, "t_cultivation_infidelity_stacked_best.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"T-cultivation infidelity stacked plot saved to: {output_path}")
    plt.close(fig)


def plot_star_settings_comparison(star_df, output_dir):
    """Plot merged STAR ablation comparison: rows=placements × cols=AOD (1,2,5)."""

    star_df = normalize_star_df(star_df)
    star_df["n_qubit"] = star_df["qubit_layout"].apply(get_n_qubit)

    for col in ["trivial_return", "decompose_move", "parallel_execution"]:
        if star_df[col].dtype == object:
            star_df[col] = (
                star_df[col].astype(str).str.lower().map({"true": True, "false": False})
            )

    ablation_settings_all = [
        ("Vanilla", True, 0, False, False),
        ("Optimized Return", False, 0, False, False),
        ("Optimized Return + Skip Partial RUS", False, 1, False, False),
        ("Optimized Return + Skip Whole RUS", False, 2, False, False),
        ("Optimized Decompose Return + Skip Whole RUS", False, 2, True, False),
        (
            "Optimized Return + Skip Whole RUS + Async RUS",
            False,
            2,
            False,
            True,
        ),
    ]

    aod_values = [1, 2, 5]
    placements = sorted(star_df["placement"].unique())
    n_placements = len(placements)

    for code_distance in get_code_distances(star_df):
        if code_distance is None:
            distance_df = star_df
            distance_suffix = ""
            title_suffix = ""
        else:
            distance_df = star_df[star_df["code_distance"] == code_distance]
            distance_suffix = f"_distance_{code_distance}"
            title_suffix = f" (Distance = {code_distance})"

        if distance_df.empty:
            continue

        fig, axes = plt.subplots(
            n_placements, len(aod_values), figsize=(18, 6 * n_placements)
        )
        if n_placements == 1:
            axes = axes.reshape(1, len(aod_values))

        col_ylims: list[tuple[float, float] | None] = [None] * len(aod_values)

        for col_idx, n_aods in enumerate(aod_values):
            aod_df = distance_df[distance_df["n_aods"] == n_aods]
            if aod_df.empty:
                print(f"No data found for n_aods={n_aods}, distance={code_distance}")
                continue

            if n_aods == 1:
                ablation_settings = ablation_settings_all[:-1]
            else:
                ablation_settings = ablation_settings_all

            for row_idx, placement in enumerate(placements):
                ax = axes[row_idx, col_idx]
                placement_df = aod_df[aod_df["placement"] == placement]

                for (
                    label,
                    trivial_return,
                    skip_rus,
                    decompose_move,
                    parallel_execution,
                ) in ablation_settings:
                    subset = placement_df[
                        (placement_df["trivial_return"] == trivial_return)
                        & (placement_df["skip_rus"] == skip_rus)
                        & (placement_df["decompose_move"] == decompose_move)
                        & (placement_df["parallel_execution"] == parallel_execution)
                    ]

                    if subset.empty:
                        continue

                    grouped = subset.groupby("n_qubit")["fidelity"]
                    means = grouped.mean().sort_index()
                    mins = grouped.min().reindex(means.index)
                    maxs = grouped.max().reindex(means.index)
                    lower_err = (means - mins).clip(lower=0)
                    upper_err = (maxs - means).clip(lower=0)
                    yerr = np.vstack([lower_err.values, upper_err.values])

                    ax.errorbar(
                        means.index,
                        means.values,
                        yerr=yerr,
                        marker="o",
                        linewidth=2,
                        markersize=7,
                        capsize=4,
                        label=label,
                    )

                ax.set_xlabel("Number of Qubits", fontsize=14)
                ax.set_ylabel("Fidelity", fontsize=14)
                ax.set_title(f"{placement}, AOD={n_aods}", fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=12)
                ax.legend(fontsize=10)

                current = ax.get_ylim()
                prev_ylim = col_ylims[col_idx]
                if prev_ylim is None:
                    col_ylims[col_idx] = current
                else:
                    col_ylims[col_idx] = (
                        min(prev_ylim[0], current[0]),
                        max(prev_ylim[1], current[1]),
                    )

        for col_idx in range(len(aod_values)):
            if col_ylims[col_idx] is not None:
                for row_idx in range(n_placements):
                    axes[row_idx, col_idx].set_ylim(col_ylims[col_idx])

        fig.suptitle(f"STAR Ablation Study{title_suffix}", fontsize=18, y=0.995)
        fig.tight_layout()
        output_path = os.path.join(
            output_dir, f"star_ablation_fidelity{distance_suffix}.pdf"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"STAR ablation plot saved to: {output_path}")


def plot_overall_fidelity_comparison(
    raw_df, star_df, output_dir, t_cultivation_df=None
):
    """Create overall fidelity comparison with STAR and optional T-cultivation lines."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw fidelity
    raw_data = (
        raw_df.groupby("n_qubit", as_index=False)["fidelity"]
        .mean()
        .sort_values("n_qubit")
    )
    ax.plot(
        raw_data["n_qubit"],
        raw_data["fidelity"],
        marker="s",
        linewidth=3,
        markersize=10,
        label="Raw (Physical)",
        color="red",
        alpha=0.7,
    )

    star_data = normalize_star_df(star_df)
    star_data["n_qubit"] = star_data["qubit_layout"].apply(get_n_qubit)
    colors = {7: "tab:blue", 9: "tab:orange"}

    for code_distance in get_code_distances(star_data):
        if code_distance is None:
            distance_df = star_data
            label = "STAR"
            color = None
        else:
            distance_df = star_data[star_data["code_distance"] == code_distance]
            label = f"STAR (distance={code_distance})"
            color = colors.get(code_distance)

        if distance_df.empty:
            continue

        grouped = distance_df.groupby("n_qubit")["fidelity"]
        means = grouped.mean().sort_index()
        mins = grouped.min().reindex(means.index)
        maxs = grouped.max().reindex(means.index)

        line = ax.plot(
            means.index,
            means.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=label,
            alpha=0.8,
            color=color,
        )[0]
        ax.fill_between(
            means.index,
            mins.values,
            maxs.values,
            color=line.get_color(),
            alpha=0.15,
        )

    # Plot optional T-cultivation lines by setting:
    # (code_distance, factory_physical_size, fidelity_target)
    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_df = t_cultivation_df.copy()
        t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
        t_df["fidelity_total"] = pd.to_numeric(
            t_df.get("fidelity_total"), errors="coerce"
        )
        t_df = t_df.dropna(subset=["fidelity_total"])
        if not t_df.empty:
            setting_cols = ["code_distance", "factory_physical_size", "fidelity_target"]
            for setting, setting_df in t_df.groupby(setting_cols):
                code_distance, factory_physical_size, fidelity_target = setting
                grouped = setting_df.groupby("n_qubit")["fidelity_total"]
                means = grouped.mean().sort_index()
                mins = grouped.min().reindex(means.index)
                maxs = grouped.max().reindex(means.index)
                label = (
                    "T-cultivation "
                    f"(d={int(code_distance)}, size={int(factory_physical_size)}, "
                    f"target={fidelity_target:g})"
                )
                line = ax.plot(
                    means.index,
                    means.values,
                    marker="^",
                    linewidth=2,
                    markersize=8,
                    linestyle="--",
                    label=label,
                    alpha=0.9,
                )[0]
                ax.fill_between(
                    means.index,
                    mins.values,
                    maxs.values,
                    color=line.get_color(),
                    alpha=0.12,
                )

    ax.set_xlabel("Number of Qubits", fontsize=14)
    ax.set_ylabel("Mean Fidelity", fontsize=14)
    ax.set_title(
        "Overall Fidelity Comparison: Raw vs STAR vs T-cultivation", fontsize=16
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "overall_fidelity_comparison.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Overall fidelity comparison plot saved to: {output_path}")
    plt.close()


def main():
    """Main function to generate comparison CSV and plots."""
    print("=" * 80)
    print("FIDELITY COMPARISON AND VISUALIZATION")
    print("=" * 80 + "\n")

    # Create output directory
    output_dir = "output/evaluation/fidelity/comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    raw_df, star_df, t_cultivation_df = load_and_process_data()
    t_count = 0 if t_cultivation_df is None else len(t_cultivation_df)
    print(
        f"Loaded {len(raw_df)} raw results, {len(star_df)} STAR results, "
        f"and {t_count} T-cultivation results\n"
    )

    # Generate comparison CSV
    print("Generating comparison CSV...")
    comparison_csv_path = os.path.join(output_dir, "fidelity_comparison.csv")
    comparison_df = generate_comparison_csv(
        raw_df, star_df, comparison_csv_path, t_cultivation_df=t_cultivation_df
    )
    print(f"Generated {len(comparison_df)} comparison rows\n")

    # Generate plots
    print("Generating plots...")
    print("\n1. Raw infidelity stacked breakdown...")
    plot_raw_infidelity_breakdown(raw_df, output_dir)

    print("\n2. STAR infidelity breakdown...")
    plot_star_infidelity_breakdown(star_df, output_dir)

    print("\n3. T-cultivation infidelity breakdown...")
    if t_cultivation_df is not None:
        plot_t_cultivation_infidelity_breakdown(t_cultivation_df, output_dir)
    else:
        print("Skipping T-cultivation stacked infidelity plot: no data loaded")

    print("\n4. STAR settings comparison (1 AOD vs 5 AOD)...")
    plot_star_settings_comparison(star_df, output_dir)

    print("\n5. Overall fidelity comparison...")
    plot_overall_fidelity_comparison(
        raw_df, star_df, output_dir, t_cultivation_df=t_cultivation_df
    )

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
