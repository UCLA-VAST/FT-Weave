import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    }
)

# process_csv.py SETTINGS[5]
# (trivial_return, tmr_assignment_method, consider_skip_rus, decompose_move, parallel_execution)
_STAR_BEST_SETTING = (False, "matching", 0, False, True)


def get_n_qubit(layout_str):
    if isinstance(layout_str, str):
        rows, cols = eval(layout_str)
        return rows * cols
    return layout_str[0] * layout_str[1]


def _resolve_t_cultivation_fidelity_csv_path() -> str | None:
    required = {
        "fidelity_total",
        "code_distance",
        "fidelity_target",
        "factory_physical_size",
    }
    candidates = [
        os.path.join("output", "evaluation", "evaluation_results.csv"),
        os.path.join("output", "evaluation", "fidelity", "t_cultivation_fidelity_results.csv"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        if required.issubset(set(header.columns)):
            return path
    return None


def _normalize_star_df(star_df):
    out = star_df.copy()
    if "skip_rus" not in out.columns and "consider_skip_rus" in out.columns:
        out["skip_rus"] = out["consider_skip_rus"]
    return out


def _select_star_best_setting_col_based(star_df):
    out = _normalize_star_df(star_df)
    out = out[out["placement"] == "col_based"].copy()
    if out.empty:
        return out
    trivial_return, tmr_method, skip_rus, decompose_move, parallel_execution = _STAR_BEST_SETTING
    mask = (
        (out["trivial_return"] == trivial_return)
        & (out["tmr_assignment_method"] == tmr_method)
        & (out["skip_rus"] == skip_rus)
        & (out["decompose_move"] == decompose_move)
        & (out["parallel_execution"] == parallel_execution)
    )
    return out[mask].copy()


def _exclude_star_distances(df, excluded_distances):
    if not excluded_distances or "code_distance" not in df.columns:
        return df
    out = df.copy()
    cd = pd.to_numeric(out["code_distance"], errors="coerce")
    return out[~cd.isin([int(d) for d in excluded_distances])].copy()


def load_data():
    raw_df = pd.read_csv("output/evaluation/fidelity/raw_fidelity_results.csv")
    star_df = pd.read_csv("output/evaluation/fidelity/star_fidelity_results.csv")
    t_path = _resolve_t_cultivation_fidelity_csv_path()
    t_df = pd.read_csv(t_path) if t_path else None
    return raw_df, star_df, t_df


def _plot_stacked_bars_infidelity(ax, df, x_col, components, title):
    if df.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return
    x_values = sorted(df[x_col].dropna().unique())
    positions = np.arange(len(x_values))
    width = 0.72
    bottom = np.zeros(len(x_values), dtype=float)

    for col, label, color in components:
        if col not in df.columns:
            continue
        values = []
        for x in x_values:
            v = pd.to_numeric(df[df[x_col] == x][col], errors="coerce").mean()
            if pd.isna(v):
                values.append(0.0)
            else:
                fv = float(v)
                values.append(max(0.0, min(1.0, 1.0 - fv)))
        values = np.array(values, dtype=float)
        ax.bar(
            positions,
            values,
            width=width,
            bottom=bottom,
            label=label,
            color=color,
            alpha=0.9,
        )
        bottom += values

    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(v)) for v in x_values], rotation=0)
    ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
    ax.set_ylabel("Total Infidelity", fontsize=_FIG_FONT_SIZE)
    ax.set_title(title, fontsize=_FIG_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_grouped_stacked_infidelity(
    ax,
    df,
    x_col,
    setting_cols,
    components,
    title,
    *,
    setting_label_fn,
    qubit_axis_outward=54,
):
    if df.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return

    x_values = sorted(pd.to_numeric(df[x_col], errors="coerce").dropna().unique())
    if not x_values:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return

    missing_cols = [c for c in setting_cols if c not in df.columns]
    if missing_cols:
        ax.set_title(f"{title}\n(missing setting columns)")
        ax.axis("off")
        return

    settings = sorted(df[setting_cols].drop_duplicates().itertuples(index=False, name=None))
    if not settings:
        ax.set_title(f"{title}\n(no settings)")
        ax.axis("off")
        return

    n_settings = len(settings)
    bar_width = 0.7
    group_gap = 0.9
    group_span = n_settings * bar_width
    x_centers = []
    setting_positions = []
    setting_labels = []

    for i, x in enumerate(x_values):
        group_start = i * (group_span + group_gap)
        x_centers.append(group_start + (group_span - bar_width) / 2.0)
        for j, setting in enumerate(settings):
            x_pos = group_start + j * bar_width
            setting_positions.append(x_pos)
            setting_labels.append(setting_label_fn(setting))

            sdf = df[pd.to_numeric(df[x_col], errors="coerce") == x].copy()
            for col_name, col_val in zip(setting_cols, setting):
                sdf = sdf[sdf[col_name] == col_val]

            bottom = 0.0
            for col, label, color in components:
                if col not in sdf.columns:
                    continue
                v = pd.to_numeric(sdf[col], errors="coerce").mean()
                val = 0.0 if pd.isna(v) else max(0.0, min(1.0, 1.0 - float(v)))
                ax.bar(
                    x_pos,
                    val,
                    width=bar_width * 0.92,
                    bottom=bottom,
                    color=color,
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.2,
                    label=label if (i == 0 and j == 0) else None,
                )
                bottom += val

    ax.set_xticks(setting_positions)
    ax.set_xticklabels(
        setting_labels,
        rotation=35,
        ha="center",
        fontsize=max(10, _FIG_FONT_SIZE - 3),
    )
    ax.set_xlabel("Setting", fontsize=_FIG_FONT_SIZE, labelpad=2)
    ax.set_ylabel("Total Infidelity", fontsize=_FIG_FONT_SIZE)
    ax.set_title(title, fontsize=_FIG_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)

    secax = ax.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    secax.set_xticks(x_centers)
    secax.set_xticklabels([str(int(v)) for v in x_values], fontsize=_FIG_FONT_SIZE)
    secax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE, labelpad=8)
    secax.tick_params(axis="x", pad=2)
    secax.spines["bottom"].set_position(("outward", qubit_axis_outward))

    # Keep setting labels close to bars.
    ax.tick_params(axis="x", pad=1, length=3)

    component_handles, component_labels = ax.get_legend_handles_labels()
    return component_handles, component_labels


def _plot_raw_fidelity_breakdown(raw_df, output_dir, *, include_idle_component=True, filename=None):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    components = [
        ("fidelity_cz", "CZ", "#4C78A8"),
        ("fidelity_1q", "1Q", "#F58518"),
        ("fidelity_move", "Move", "#54A24B"),
        ("fidelity_init", "Init", "#E45756"),
        ("fidelity_measurement", "Measurement", "#72B7B2"),
    ]
    if include_idle_component:
        components.append(("fidelity_idle", "Idle", "#9467BD"))
    _plot_stacked_bars_infidelity(
        ax,
        raw_df.copy(),
        "n_qubit",
        components,
        "Raw Infidelity Breakdown"
        + ("" if include_idle_component else " (Without Idle)"),
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.subplots_adjust(right=0.75)
    if filename is None:
        filename = (
            "raw_fidelity_breakdown_stacked.pdf"
            if include_idle_component
            else "raw_fidelity_breakdown_stacked_no_idle.pdf"
        )
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_star_fidelity_breakdown(
    star_df,
    output_dir,
    *,
    include_idle_component=True,
    excluded_star_distances=None,
    filename=None,
):
    data = _select_star_best_setting_col_based(star_df)
    if data.empty:
        data = _normalize_star_df(star_df)
    data = data.copy()
    data["n_qubit"] = data["qubit_layout"].apply(get_n_qubit)

    data = _exclude_star_distances(data, excluded_star_distances)
    if data.empty or "code_distance" not in data.columns:
        return
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    components = [
        ("fidelity_cnot", "CNOT", "#4C78A8"),
        ("fidelity_1q", "H", "#8C564B"),
        ("fidelity_of_rz_injection", "RZ Injection", "#C44E52"),
        ("fidelity_of_rz_teleportaion", "Teleportation-CNOT", "#F28E2B"),
        ("fidelity_of_rz_s", "Rz Correction", "#59A14F"),
    ]
    if include_idle_component:
        components.append(("fidelity_idle_model", "Idle", "#4DBBD5"))
    handles, labels = _plot_grouped_stacked_infidelity(
        ax,
        data,
        "n_qubit",
        ["code_distance"],
        components,
        "STAR Infidelity Breakdown"
        + ("" if include_idle_component else " (Without Idle)"),
        setting_label_fn=lambda s: f"d={int(s[0])}",
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            frameon=True,
        )
    fig.subplots_adjust(bottom=0.34, right=0.83)
    if filename is None:
        filename = (
            "star_fidelity_breakdown_stacked.pdf"
            if include_idle_component
            else "star_fidelity_breakdown_stacked_no_idle.pdf"
        )
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_t_cultivation_fidelity_breakdown(
    t_df, output_dir, *, include_idle_component=True, filename=None
):
    if t_df is None or t_df.empty:
        return
    data = t_df.copy()
    data["n_qubit"] = data["qubit_layout"].apply(get_n_qubit)
    if data.empty or "code_distance" not in data.columns:
        return
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    components = [
        ("fidelity_cnot", "CNOT", "#4C78A8"),
        ("fidelity_h", "H", "#8C564B"),
        ("fidelity_of_rz_teleportaion", "Teleportation-CNOT", "#F58518"),
        ("fidelity_of_rz_s", "Rz decomposition-S", "#54A24B"),
        ("fidelity_of_rz_h", "Rz decomposition-H", "#BCBD22"),
        ("fidelity_of_t_gate", "T", "#9467BD"),
    ]
    if include_idle_component:
        components.append(("fidelity_idle_model", "Idle", "#4DBBD5"))
    handles, labels = _plot_grouped_stacked_infidelity(
        ax,
        data,
        "n_qubit",
        ["code_distance", "factory_physical_size", "fidelity_target"],
        components,
        "T-cultivation Infidelity Breakdown"
        + ("" if include_idle_component else " (Without Idle)"),
        setting_label_fn=lambda s: f"{int(s[0])}/{int(s[1])}/{float(s[2]):g}",
        qubit_axis_outward=74,
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.86, 0.5),
            frameon=True,
        )
    fig.subplots_adjust(bottom=0.40, right=0.85)
    if filename is None:
        filename = (
            "t_cultivation_fidelity_breakdown_stacked.pdf"
            if include_idle_component
            else "t_cultivation_fidelity_breakdown_stacked_no_idle.pdf"
        )
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_overall(
    raw_df,
    star_df,
    output_dir,
    t_cultivation_df=None,
    *,
    include_idle: bool,
    include_raw: bool = True,
    excluded_star_distances=None,
    filename=None,
):
    fig, ax = plt.subplots(figsize=(10, 5.0))

    if include_raw:
        raw_mean = (
            raw_df.groupby("n_qubit", as_index=False)["fidelity"].mean().sort_values("n_qubit")
        )
        ax.plot(
            raw_mean["n_qubit"],
            raw_mean["fidelity"],
            marker="s",
            linewidth=3,
            markersize=10,
            label="Raw (Physical)",
            color="red",
            alpha=0.7,
        )

    star_data = _select_star_best_setting_col_based(star_df)
    if star_data.empty:
        star_data = _normalize_star_df(star_df)
    star_data = _exclude_star_distances(star_data, excluded_star_distances)
    star_data["n_qubit"] = star_data["qubit_layout"].apply(get_n_qubit)
    y_star = "fidelity_with_idle" if include_idle else "fidelity"
    if y_star not in star_data.columns:
        y_star = "fidelity"
    star_data[y_star] = pd.to_numeric(star_data[y_star], errors="coerce")
    star_data = star_data.dropna(subset=["n_qubit", y_star])

    for d in sorted(star_data["code_distance"].dropna().astype(int).unique()):
        sd = star_data[star_data["code_distance"] == d].copy()
        if sd.empty:
            continue
        grouped = sd.groupby("n_qubit")[y_star]
        means = grouped.mean().sort_index()
        mins = grouped.min().reindex(means.index)
        maxs = grouped.max().reindex(means.index)
        label = f"STAR (distance={d})"
        line = ax.plot(
            means.index,
            means.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=label,
            alpha=0.85,
        )[0]
        ax.fill_between(
            means.index,
            mins.values,
            maxs.values,
            color=line.get_color(),
            alpha=0.15,
        )

    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_df = t_cultivation_df.copy()
        t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
        y_t = "fidelity_total_with_idle" if include_idle else "fidelity_total"
        if y_t not in t_df.columns:
            y_t = "fidelity_total"
        t_df[y_t] = pd.to_numeric(t_df[y_t], errors="coerce")
        t_df = t_df.dropna(subset=["n_qubit", y_t])

        for (code_distance, factory_size, fidelity_target), sdf in t_df.groupby(
            ["code_distance", "factory_physical_size", "fidelity_target"]
        ):
            grouped = sdf.groupby("n_qubit")[y_t]
            means = grouped.mean().sort_index()
            mins = grouped.min().reindex(means.index)
            maxs = grouped.max().reindex(means.index)
            label = (
                f"T-cultivation (d={int(code_distance)}, size={int(factory_size)}, "
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

    ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
    ax.set_ylabel("Mean Fidelity", fontsize=_FIG_FONT_SIZE)
    if include_idle:
        ax.set_title("Overall Fidelity Comparison (Including Idle Error)", fontsize=_FIG_FONT_SIZE)
    else:
        ax.set_title("Overall Fidelity Comparison", fontsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=_FIG_FONT_SIZE)
    ax.legend(
        fontsize=_FIG_FONT_SIZE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
    )
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(right=0.74)

    if filename is None:
        filename = (
            "overall_fidelity_comparison_with_idle.pdf"
            if include_idle
            else "overall_fidelity_comparison.pdf"
        )
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("=" * 80)
    print("FIDELITY COMPARISON PLOTTING")
    print("=" * 80)

    output_dir = "output/evaluation/fidelity/comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    raw_df, star_df, t_df = load_data()
    print(
        f"Loaded raw={len(raw_df)}, star={len(star_df)}, "
        f"t_cultivation={0 if t_df is None else len(t_df)}"
    )

    print("\n1. Plot raw stacked fidelity breakdown...")
    _plot_raw_fidelity_breakdown(raw_df, output_dir)

    print("\n2. Plot STAR stacked fidelity breakdown...")
    _plot_star_fidelity_breakdown(star_df, output_dir)

    print("\n3. Plot T-cultivation stacked fidelity breakdown...")
    _plot_t_cultivation_fidelity_breakdown(t_df, output_dir)

    print("\n4. Plot overall fidelity (without idle, with raw line)...")
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=False,
    )

    print("\n5. Plot overall fidelity (with idle, with raw line)...")
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=True,
    )

    print("\n6. Plot overall fidelity (without idle, no raw line)...")
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=False,
        include_raw=False,
        filename="overall_fidelity_comparison_no_raw.pdf",
    )
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=True,
        include_raw=False,
        filename="overall_fidelity_comparison_with_idle_no_raw.pdf",
    )

    print("\n7. Plot stacked fidelity breakdowns (without idle)...")
    _plot_raw_fidelity_breakdown(
        raw_df, output_dir, include_idle_component=False
    )
    _plot_star_fidelity_breakdown(
        star_df, output_dir, include_idle_component=False
    )
    _plot_t_cultivation_fidelity_breakdown(
        t_df, output_dir, include_idle_component=False
    )

    print("\n8. Plot overall fidelity variants without STAR d=13 (with raw line)...")
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=False,
        excluded_star_distances=[13],
        filename="overall_fidelity_comparison_no_star_d13.pdf",
    )
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=True,
        excluded_star_distances=[13],
        filename="overall_fidelity_comparison_with_idle_no_star_d13.pdf",
    )

    print("\n9. Plot overall fidelity variants without STAR d=13 (no raw line)...")
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=False,
        include_raw=False,
        excluded_star_distances=[13],
        filename="overall_fidelity_comparison_no_star_d13_no_raw.pdf",
    )
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_idle=True,
        include_raw=False,
        excluded_star_distances=[13],
        filename="overall_fidelity_comparison_with_idle_no_star_d13_no_raw.pdf",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
