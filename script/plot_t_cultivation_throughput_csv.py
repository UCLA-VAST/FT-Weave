import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


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


def _setting_subtitle(distance: int, fidelity_target: float) -> str:
    return f"distance-{distance}, LER: {fidelity_target:g}"


def _mean_min_max(grouped) -> pd.DataFrame:
    out = grouped.agg(["mean", "min", "max"]).reset_index()
    out["lower_err"] = (out["mean"] - out["min"]).clip(lower=0.0)
    out["upper_err"] = (out["max"] - out["mean"]).clip(lower=0.0)
    return out


def _set_integer_x_ticks(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "setting_index",
        "distance",
        "fidelity_target",
        "factory_physical_size",
        "trial",
        "seed",
        "n_qubits",
        "k_t_per_qubit",
        "ratio",
        "n_factories",
        "makespan",
        "metric_value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "setting_index" not in df.columns:
        raise ValueError("CSV is missing required column: setting_index")
    if "figure" not in df.columns:
        raise ValueError("CSV is missing required column: figure")
    if "metric_mode" not in df.columns:
        raise ValueError("CSV is missing required column: metric_mode")

    return df


def _save_fig1(df: pd.DataFrame, output_dir: str, metric_mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    ax_time, ax_metric = axes

    df1 = df[df["figure"] == "fig1"].copy()
    if df1.empty:
        plt.close(fig)
        return

    for setting_idx in sorted(df1["setting_index"].dropna().astype(int).unique()):
        s = df1[df1["setting_index"] == setting_idx].copy()
        if s.empty:
            continue

        distance = int(s["distance"].dropna().iloc[0])
        fidelity_target = float(s["fidelity_target"].dropna().iloc[0])
        label = _setting_subtitle(distance, fidelity_target)

        time_stats = _mean_min_max(s.groupby("n_factories")["makespan"])
        metric_stats = _mean_min_max(s.groupby("n_factories")["metric_value"])

        ax_time.errorbar(
            time_stats["n_factories"],
            time_stats["mean"],
            yerr=[time_stats["lower_err"], time_stats["upper_err"]],
            capsize=3,
            marker="o",
            label=label,
        )
        ax_metric.errorbar(
            metric_stats["n_factories"],
            metric_stats["mean"],
            yerr=[metric_stats["lower_err"], metric_stats["upper_err"]],
            capsize=3,
            marker="o",
            label=label,
        )

    ax_time.set_ylabel("Execution Time")
    ax_time.set_xlabel("Number of factories")
    ax_time.grid(True, alpha=0.3)
    _set_integer_x_ticks(ax_time)

    ax_metric.set_xlabel("Number of factories")
    if metric_mode == "throughput":
        ax_metric.set_ylabel("Single-factory throughput (T / time / factory)")
    else:
        ax_metric.set_ylabel("Cycles per T per Factory")
    ax_metric.grid(True, alpha=0.3)
    _set_integer_x_ticks(ax_metric)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=max(10, _FIG_FONT_SIZE - 4),
        frameon=True,
    )

    fig.suptitle(
        "Comparison: Number of Qubits < Number of Factories\n (1 qubits, each with 10 T gates)",
        y=0.94,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(
        os.path.join(output_dir, "fig_1_t_cultivation_one_qubit_vs_factories.pdf"),
        dpi=200,
    )
    plt.close(fig)


def _save_fig2(df: pd.DataFrame, output_dir: str, metric_mode: str) -> None:
    df2 = df[df["figure"] == "fig2"].copy()
    if df2.empty:
        return

    settings = sorted(df2["setting_index"].dropna().astype(int).unique())
    fig, axes = plt.subplots(
        2, len(settings), figsize=(5.8 * len(settings), 8.2), sharex="col"
    )
    if len(settings) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, setting_idx in enumerate(settings):
        s = df2[df2["setting_index"] == setting_idx].copy()
        if s.empty:
            continue

        distance = int(s["distance"].dropna().iloc[0])
        fidelity_target = float(s["fidelity_target"].dropna().iloc[0])

        for ratio in sorted(s["ratio"].dropna().astype(int).unique()):
            sr = s[s["ratio"] == ratio]
            time_stats = _mean_min_max(sr.groupby("n_qubits")["makespan"])
            metric_stats = _mean_min_max(sr.groupby("n_qubits")["metric_value"])

            axes[0, col].errorbar(
                time_stats["n_qubits"],
                time_stats["mean"],
                yerr=[time_stats["lower_err"], time_stats["upper_err"]],
                capsize=3,
                marker="o",
                label=f"#Factory/#Qubit={ratio}",
            )
            axes[1, col].errorbar(
                metric_stats["n_qubits"],
                metric_stats["mean"],
                yerr=[metric_stats["lower_err"], metric_stats["upper_err"]],
                capsize=3,
                marker="o",
                label=f"#Factory/#Qubit={ratio}",
            )

        axes[0, col].set_title(_setting_subtitle(distance, fidelity_target))
        axes[0, col].grid(True, alpha=0.3)
        _set_integer_x_ticks(axes[0, col])

        axes[1, col].set_xlabel("Number of qubits")
        axes[1, col].grid(True, alpha=0.3)
        _set_integer_x_ticks(axes[1, col])

    axes[0, 0].set_ylabel("Execution Time")
    if metric_mode == "throughput":
        axes[1, 0].set_ylabel("Single-factory throughput (T / time / factory)")
    else:
        axes[1, 0].set_ylabel("Cycles per T per Factory")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        bbox_to_anchor=(0.5, 0.0),
        fontsize=max(10, _FIG_FONT_SIZE - 4),
        frameon=True,
    )

    fig.suptitle(
        "Comparison for Different Qubit to Factory Ratio\n (10 qubits, each with 10 T gates)"
    )
    fig.suptitle(
        "Comparison for Different Qubit to Factory Ratio\n (10 qubits, each with 10 T gates)",
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(
        os.path.join(output_dir, "fig_2_t_cultivation_ratio_sweep.pdf"),
        dpi=200,
    )
    plt.close(fig)


def _save_fig3(df: pd.DataFrame, output_dir: str, metric_mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    ax_time, ax_metric = axes

    df3 = df[df["figure"] == "fig3"].copy()
    if df3.empty:
        plt.close(fig)
        return

    for setting_idx in sorted(df3["setting_index"].dropna().astype(int).unique()):
        s = df3[df3["setting_index"] == setting_idx].copy()
        if s.empty:
            continue

        distance = int(s["distance"].dropna().iloc[0])
        fidelity_target = float(s["fidelity_target"].dropna().iloc[0])
        label = _setting_subtitle(distance, fidelity_target)

        time_stats = _mean_min_max(s.groupby("n_factories")["makespan"])
        metric_stats = _mean_min_max(s.groupby("n_factories")["metric_value"])

        ax_time.errorbar(
            time_stats["n_factories"],
            time_stats["mean"],
            yerr=[time_stats["lower_err"], time_stats["upper_err"]],
            capsize=3,
            marker="o",
            label=label,
        )
        ax_metric.errorbar(
            metric_stats["n_factories"],
            metric_stats["mean"],
            yerr=[metric_stats["lower_err"], metric_stats["upper_err"]],
            capsize=3,
            marker="o",
            label=label,
        )

    ax_time.set_ylabel("Circuit duration")
    ax_time.set_xlabel("Number of factories")
    ax_time.grid(True, alpha=0.3)
    _set_integer_x_ticks(ax_time)

    ax_metric.set_xlabel("Number of factories")
    if metric_mode == "throughput":
        ax_metric.set_ylabel("Single-Factory Throughput (T / Time / Factory)")
    else:
        ax_metric.set_ylabel("Cycles per T per Factory")
    ax_metric.grid(True, alpha=0.3)
    _set_integer_x_ticks(ax_metric)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        bbox_to_anchor=(0.5, 0.005),
        fontsize=max(10, _FIG_FONT_SIZE - 4),
        frameon=True,
    )

    fig.suptitle(
        "Comparison: Number of Factories < Number of Qubits\n (10 qubits, each with 10 T gates)"
    )
    fig.suptitle(
        "Comparison: Number of Factories < Number of Qubits\n (10 qubits, each with 10 T gates)",
        y=0.94,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(
        os.path.join(output_dir, "fig_3_t_cultivation_10q_factory_sweep.pdf"),
        dpi=200,
    )
    plt.close(fig)


def process_t_cultivation_throughput_csv(csv_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = _load_csv(csv_path)

    metric_modes = [m for m in df["metric_mode"].dropna().astype(str).unique() if m]
    metric_mode = metric_modes[0] if metric_modes else "cycles_per_t"

    _save_fig1(df, output_dir, metric_mode)
    _save_fig2(df, output_dir, metric_mode)
    _save_fig3(df, output_dir, metric_mode)

    print(f"Saved throughput figures to {output_dir}")


if __name__ == "__main__":
    csv_file = (
        "output/t_cultivation/factory_throughput/t_cultivation_throughput_trials.csv"
    )
    output_dir = "output/t_cultivation/factory_throughput"

    process_t_cultivation_throughput_csv(csv_file, output_dir)
