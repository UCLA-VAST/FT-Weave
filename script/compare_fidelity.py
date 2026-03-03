import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


def load_and_process_data():
    """Load raw and star fidelity results."""
    raw_df = pd.read_csv("output/evaluation/fidelity/raw_fidelity_results.csv")
    star_df = pd.read_csv("output/evaluation/fidelity/star_fidelity_results.csv")

    return raw_df, star_df


def generate_comparison_csv(raw_df, star_df, output_path):
    """Generate a comparison CSV with one line for raw and lines for each STAR setting."""

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
    fig, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xticklabels(n_qubits)
    ax.legend(loc="upper left", fontsize=10)
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

    # Parse qubit layout to get n_qubit
    def get_n_qubit(layout_str):
        if isinstance(layout_str, str):
            rows, cols = eval(layout_str)
            return rows * cols
        return layout_str[0] * layout_str[1]

    star_df["n_qubit"] = star_df["qubit_layout"].apply(get_n_qubit)

    # Create merged figure for all placements
    placements = sorted(star_df["placement"].unique())
    fig, axes = plt.subplots(len(placements), len(error_terms), figsize=(16, 10))

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

            ax.set_xlabel("Number of Qubits", fontsize=11)
            ax.set_ylabel("Infidelity (1 - Fidelity)", fontsize=11)
            term_name = (
                term.replace("fidelity_", "").replace("fidelity_of_", "").upper()
            )
            ax.set_title(term_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            if placement_idx == 0:
                ax.legend(fontsize=8)

    # Set row labels
    for placement_idx, placement in enumerate(placements):
        axes[placement_idx, 0].set_ylabel(
            f"{placement.upper()}\nInfidelity", fontsize=12, fontweight="bold"
        )

    fig.suptitle("STAR Infidelity Breakdown by Error Type and Placement", fontsize=16)
    fig.tight_layout()
    output_path = os.path.join(output_dir, "star_infidelity_breakdown.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"STAR infidelity breakdown plot saved to: {output_path}")
    plt.close(fig)

    # Create a single stacked bar chart by averaging STAR over all configurations
    # Include all STAR fidelity sources for stacked view
    stacked_terms = [
        "fidelity_of_rz_injection",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
        "fidelity_cnot",
        "fidelity_1q",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(stacked_terms))]

    # Average over all STAR configurations and placements per qubit count
    grouped = star_df.groupby("n_qubit", as_index=False)[stacked_terms].mean()
    n_qubits = grouped["n_qubit"].tolist()

    infidelity_data = {term: (1 - grouped[term]).tolist() for term in stacked_terms}

    x = np.arange(len(n_qubits))
    width = 0.6
    bottom = np.zeros(len(n_qubits))

    for term, color in zip(stacked_terms, colors):
        values = infidelity_data[term]
        label = term.replace("fidelity_", "").replace("fidelity_of_", "").upper()
        if term == "fidelity_1q":
            label = "H"
        ax.bar(
            x,
            values,
            width,
            label=label,
            bottom=bottom,
            color=color,
            alpha=0.8,
        )
        bottom += values

    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Total Infidelity", fontsize=12)
    ax.set_title(
        "STAR Infidelity Breakdown (Averaged Over All Configurations)", fontsize=13
    )
    ax.set_xticks(x)
    ax.set_xticklabels(n_qubits)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "STAR Infidelity Stacked (Averaged Over All Configurations)", fontsize=15
    )
    fig.tight_layout()
    output_path = os.path.join(output_dir, "star_infidelity_stacked_best.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"STAR infidelity stacked plot saved to: {output_path}")
    plt.close(fig)


def plot_star_settings_comparison(star_df, output_dir):
    """Plot merged STAR ablation comparison: rows=placements × cols=AOD (1,2,5)."""

    def get_n_qubit(layout_str):
        if isinstance(layout_str, str):
            rows, cols = eval(layout_str)
            return rows * cols
        return layout_str[0] * layout_str[1]

    star_df = star_df.copy()
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

    fig, axes = plt.subplots(
        n_placements, len(aod_values), figsize=(21, 5 * n_placements)
    )
    if n_placements == 1:
        axes = axes.reshape(1, len(aod_values))

    # Store y-limits for each column to sync placement rows
    col_ylims = [None] * len(aod_values)

    for col_idx, n_aods in enumerate(aod_values):
        aod_df = star_df[star_df["n_aods"] == n_aods]
        if aod_df.empty:
            print(f"No data found for n_aods={n_aods}")
            continue

        # Use different settings for AOD=1 (exclude async)
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

            ax.set_xlabel("Number of Qubits", fontsize=11)
            ax.set_ylabel("Fidelity", fontsize=11)
            ax.set_title(f"{placement}, AOD={n_aods}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Track y-limits per column for syncing
            if col_ylims[col_idx] is None:
                col_ylims[col_idx] = ax.get_ylim()
            else:
                current = ax.get_ylim()
                col_ylims[col_idx] = (
                    min(col_ylims[col_idx][0], current[0]),
                    max(col_ylims[col_idx][1], current[1]),
                )

    # Apply synchronized y-limits per column
    for col_idx in range(len(aod_values)):
        if col_ylims[col_idx] is not None:
            for row_idx in range(n_placements):
                axes[row_idx, col_idx].set_ylim(col_ylims[col_idx])

    fig.suptitle("STAR Ablation Study", fontsize=16, y=0.995)
    fig.tight_layout()
    output_path = os.path.join(output_dir, "star_ablation_fidelity.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"STAR ablation plot saved to: {output_path}")


def plot_overall_fidelity_comparison(raw_df, star_df, output_dir):
    """Create overall fidelity comparison with STAR averaged over all configurations."""

    fig, ax = plt.subplots(figsize=(12, 8))

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

    # Plot STAR fidelity averaged over all configurations
    def get_n_qubit(layout_str):
        if isinstance(layout_str, str):
            rows, cols = eval(layout_str)
            return rows * cols
        return layout_str[0] * layout_str[1]

    star_data = star_df.copy()
    star_data["n_qubit"] = star_data["qubit_layout"].apply(get_n_qubit)
    grouped = star_data.groupby("n_qubit")["fidelity"]
    means = grouped.mean().sort_index()
    mins = grouped.min().reindex(means.index)
    maxs = grouped.max().reindex(means.index)

    line = ax.plot(
        means.index,
        means.values,
        marker="o",
        linewidth=2,
        markersize=8,
        label="STAR (avg over all configs)",
        alpha=0.8,
    )[0]
    ax.fill_between(
        means.index,
        mins.values,
        maxs.values,
        color=line.get_color(),
        alpha=0.2,
        label="STAR (min-max)",
    )

    ax.set_xlabel("Number of Qubits", fontsize=14)
    ax.set_ylabel("Mean Fidelity", fontsize=14)
    ax.set_title("Overall Fidelity Comparison: Raw vs STAR", fontsize=16)
    ax.legend(fontsize=12)
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
    raw_df, star_df = load_and_process_data()
    print(f"Loaded {len(raw_df)} raw results and {len(star_df)} STAR results\n")

    # Generate comparison CSV
    print("Generating comparison CSV...")
    comparison_csv_path = os.path.join(output_dir, "fidelity_comparison.csv")
    comparison_df = generate_comparison_csv(raw_df, star_df, comparison_csv_path)
    print(f"Generated {len(comparison_df)} comparison rows\n")

    # Generate plots
    print("Generating plots...")
    print("\n1. Raw infidelity stacked breakdown...")
    plot_raw_infidelity_breakdown(raw_df, output_dir)

    print("\n2. STAR infidelity breakdown...")
    plot_star_infidelity_breakdown(star_df, output_dir)

    print("\n3. STAR settings comparison (1 AOD vs 5 AOD)...")
    plot_star_settings_comparison(star_df, output_dir)

    print("\n4. Overall fidelity comparison...")
    plot_overall_fidelity_comparison(raw_df, star_df, output_dir)

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
