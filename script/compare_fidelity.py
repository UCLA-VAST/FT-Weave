import os

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
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

# Fidelity comparisons should compare runs under the same runtime architecture.
# STAR setting 4 matches evaluation_fidelity_star.py / process_csv.py SETTINGS[4].
# (trivial_return, tmr_assignment_method, consider_skip_rus, decompose_move, parallel_execution)
_COMPARISON_N_AODS = 1
_COMPARISON_PLACEMENT = "col_based"
_STAR_COMPARISON_SETTING_INDEX = 4
_STAR_COMPARISON_SETTING = (False, "matching", 2, True, False)
# Must match ``MAIN_COMPILE_SETTING`` in ``evaluation_fidelity_t_cultivation.py``:
# (trivial_return, decompose_move, redistribute_stage1_success).
_T_MAIN_COMPILE_SETTING = (False, True, True)
_T_COMPARISON_SETTINGS = [
    (7, 1e-8, 2),
    (9, 1e-8, 2),
    (13, 1e-8, 4),
]
_EXCLUDED_STAR_DISTANCES = {13}
_EXCLUDE_T_CULTIVATION_D13 = False
_EXCLUDED_T_CULTIVATION_TARGETS = {1e-9}
_STAR_DISTANCE_COLORS = [
    "#F1CE63",
    "#F28E2B",
    "#D55E00",
    "#A63603",
]
_T_DISTANCE_COLORS = [
    "#B2DF8A",
    "#59A14F",
    "#1B7837",
    "#00441B",
]
_METHOD_STYLES = {
    "Physical": {"marker": "s", "linestyle": "-", "color": "black"},
    "STAR": {"marker": "o", "linestyle": "-"},
    "T-cultivation": {"marker": "^", "linestyle": "--"},
}
_METHOD_LEGEND_COLORS = {
    "STAR": "#F28E2B",
    "T-cultivation": "#59A14F",
}


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
    }
    candidates = [
        os.path.join("output", "evaluation", "evaluation_results.csv"),
        os.path.join(
            "output", "evaluation", "fidelity", "t_cultivation_fidelity_results.csv"
        ),
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


def _filter_comparison_architecture(df):
    out = df.copy()
    if "placement" in out.columns:
        out = out[out["placement"] == _COMPARISON_PLACEMENT].copy()
    if "n_aods" in out.columns:
        n_aods = pd.to_numeric(out["n_aods"], errors="coerce")
        out = out[n_aods == _COMPARISON_N_AODS].copy()
    return out


def _select_star_comparison_setting(star_df):
    out = _normalize_star_df(star_df)
    out = _filter_comparison_architecture(out)
    if out.empty:
        return out
    trivial_return, tmr_method, skip_rus, decompose_move, parallel_execution = (
        _STAR_COMPARISON_SETTING
    )
    mask = (
        (out["trivial_return"] == trivial_return)
        & (out["tmr_assignment_method"] == tmr_method)
        & (out["skip_rus"] == skip_rus)
        & (out["decompose_move"] == decompose_move)
        & (out["parallel_execution"] == parallel_execution)
    )
    return out[mask].copy()


def _select_t_cultivation_comparison_data(t_df):
    if t_df is None:
        return None
    out = _filter_comparison_architecture(t_df)
    compile_cols = {
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
    }
    if compile_cols.issubset(out.columns):
        tr, dm, rs = _T_MAIN_COMPILE_SETTING
        out = out[
            (out["trivial_return"] == tr)
            & (out["decompose_move"] == dm)
            & (out["redistribute_stage1_success"] == rs)
        ].copy()
    required = {"code_distance", "fidelity_target", "factory_physical_size"}
    if required.issubset(out.columns):
        cd = pd.to_numeric(out["code_distance"], errors="coerce")
        target = pd.to_numeric(out["fidelity_target"], errors="coerce")
        size = pd.to_numeric(out["factory_physical_size"], errors="coerce")
        selected = pd.Series(False, index=out.index)
        for setting_cd, setting_target, setting_size in _T_COMPARISON_SETTINGS:
            target_match = pd.Series(
                np.isclose(
                    target.to_numpy(dtype=float),
                    float(setting_target),
                    rtol=0.0,
                    atol=1e-20,
                ),
                index=out.index,
            )
            selected |= (
                (cd == int(setting_cd))
                & target_match
                & (size == int(setting_size))
            )
        out = out[selected].copy()
    return out


def _exclude_star_d13(df):
    if "code_distance" not in df.columns:
        return df
    out = df.copy()
    cd = pd.to_numeric(out["code_distance"], errors="coerce")
    return out[~cd.isin(_EXCLUDED_STAR_DISTANCES)].copy()


def _exclude_t_cultivation_targets(df):
    if df is None or "fidelity_target" not in df.columns:
        return df
    out = df.copy()
    target = pd.to_numeric(out["fidelity_target"], errors="coerce")
    return out[~target.isin(_EXCLUDED_T_CULTIVATION_TARGETS)].copy()


def _exclude_t_cultivation_d13(df):
    if (
        not _EXCLUDE_T_CULTIVATION_D13
        or df is None
        or "code_distance" not in df.columns
    ):
        return df
    out = df.copy()
    cd = pd.to_numeric(out["code_distance"], errors="coerce")
    return out[cd != 13].copy()


def _build_architecture_distance_colors(star_df, t_df):
    distances: set[int] = set()
    if "code_distance" in star_df.columns:
        distances.update(
            pd.to_numeric(star_df["code_distance"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
    if t_df is not None and "code_distance" in t_df.columns:
        distances.update(
            pd.to_numeric(t_df["code_distance"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
    ordered_distances = sorted(distances)
    return {
        "STAR": {
            d: _STAR_DISTANCE_COLORS[i % len(_STAR_DISTANCE_COLORS)]
            for i, d in enumerate(ordered_distances)
        },
        "T-cultivation": {
            d: _T_DISTANCE_COLORS[i % len(_T_DISTANCE_COLORS)]
            for i, d in enumerate(ordered_distances)
        },
    }


def _add_overall_legends(
    ax,
    *,
    include_raw: bool,
    architecture_distance_colors: dict[str, dict[int, str]],
):
    method_handles = []
    if include_raw:
        method_handles.append(
            Line2D(
                [0],
                [0],
                label="Physical",
                linewidth=3,
                markersize=9,
                **_METHOD_STYLES["Physical"],
            )
        )
    method_handles.extend(
        [
            Line2D(
                [0],
                [0],
                label="STAR",
                color=_METHOD_LEGEND_COLORS["STAR"],
                linewidth=2.5,
                markersize=9,
                **_METHOD_STYLES["STAR"],
            ),
            Line2D(
                [0],
                [0],
                label="T-cultivation",
                color=_METHOD_LEGEND_COLORS["T-cultivation"],
                linewidth=2.5,
                markersize=9,
                **_METHOD_STYLES["T-cultivation"],
            ),
        ]
    )
    method_legend = ax.legend(
        handles=method_handles,
        loc="upper left",
        frameon=True,
        fontsize=_FIG_FONT_SIZE,
    )
    ax.add_artist(method_legend)

    star_colors = architecture_distance_colors.get("STAR", {})
    t_colors = architecture_distance_colors.get("T-cultivation", {})
    distances = sorted(set(star_colors))
    if distances:
        distance_handles = [
            (
                Line2D(
                    [0],
                    [0],
                    color=star_colors[d],
                    linewidth=2.5,
                    markersize=9,
                    **_METHOD_STYLES["STAR"],
                ),
                Line2D(
                    [0],
                    [0],
                    color=t_colors[d],
                    linewidth=2.5,
                    markersize=9,
                    **_METHOD_STYLES["T-cultivation"],
                ),
            )
            for d in distances
        ]
        ax.legend(
            handles=distance_handles,
            labels=[str(d) for d in distances],
            title="d",
            loc="lower right",
            frameon=True,
            fontsize=_FIG_FONT_SIZE,
            title_fontsize=_FIG_FONT_SIZE,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )


def load_data():
    raw_df = pd.read_csv("output/evaluation/fidelity/raw_fidelity_results.csv")
    star_df = pd.read_csv("output/evaluation/fidelity/star_fidelity_results.csv")
    star_df = _exclude_star_d13(star_df)
    star_df = _select_star_comparison_setting(star_df)
    t_path = _resolve_t_cultivation_fidelity_csv_path()
    t_df = pd.read_csv(t_path) if t_path else None
    t_df = _exclude_t_cultivation_targets(t_df)
    t_df = _exclude_t_cultivation_d13(t_df)
    t_df = _select_t_cultivation_comparison_data(t_df)
    return raw_df, star_df, t_df


def _collect_overall_infidelity_series(raw_df, star_df, t_cultivation_df):
    """Return plotted overall curves as infidelity series indexed by n_qubit."""
    series: dict[str, pd.Series] = {}

    raw_mean = (
        raw_df.groupby("n_qubit", as_index=False)["fidelity"]
        .mean()
        .sort_values("n_qubit")
    )
    if not raw_mean.empty:
        values = pd.to_numeric(raw_mean["fidelity"], errors="coerce")
        series["Physical"] = pd.Series(
            1.0 - values.to_numpy(dtype=float),
            index=raw_mean["n_qubit"].astype(int),
        ).dropna()

    star_data = _select_star_comparison_setting(star_df)
    if not star_data.empty and "code_distance" in star_data.columns:
        star_data = star_data.copy()
        star_data["n_qubit"] = star_data["qubit_layout"].apply(get_n_qubit)
        star_data["fidelity"] = pd.to_numeric(star_data["fidelity"], errors="coerce")
        star_data = star_data.dropna(subset=["n_qubit", "fidelity"])
        for d in sorted(star_data["code_distance"].dropna().astype(int).unique()):
            sd = star_data[star_data["code_distance"] == d]
            means = sd.groupby("n_qubit")["fidelity"].mean().sort_index()
            series[f"STAR d={d}"] = (1.0 - means).dropna()

    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_df = _select_t_cultivation_comparison_data(t_cultivation_df)
        if t_df is None:
            return series
        t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
        t_df["fidelity_total"] = pd.to_numeric(t_df["fidelity_total"], errors="coerce")
        t_df = t_df.dropna(subset=["n_qubit", "fidelity_total"])
        for d in sorted(t_df["code_distance"].dropna().astype(int).unique()):
            sd = t_df[t_df["code_distance"] == d]
            means = sd.groupby("n_qubit")["fidelity_total"].mean().sort_index()
            series[f"T-cultivation d={d}"] = (1.0 - means).dropna()

    return series


def _print_pairwise_infidelity_improvements(raw_df, star_df, t_cultivation_df):
    """Print pairwise lower-infidelity factors between every overall curve."""
    series = _collect_overall_infidelity_series(raw_df, star_df, t_cultivation_df)
    labels = sorted(series)
    if len(labels) < 2:
        return

    print("\nPairwise infidelity improvement between overall curves:")
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            joined = pd.concat([series[left], series[right]], axis=1, join="inner")
            joined.columns = [left, right]
            joined = joined.dropna()
            joined = joined[(joined[left] > 0) & (joined[right] > 0)]
            if joined.empty:
                continue
            # Geometric mean is less dominated by the largest system size on a
            # log-scale infidelity plot.
            left_mean = float(np.exp(np.log(joined[left]).mean()))
            right_mean = float(np.exp(np.log(joined[right]).mean()))
            if left_mean <= right_mean:
                better, worse = left, right
                factor = right_mean / left_mean
            else:
                better, worse = right, left
                factor = left_mean / right_mean
            qubits = ", ".join(str(int(q)) for q in joined.index)
            print(
                f"  {better} is {factor:.2f}x lower infidelity than {worse} "
                f"(common n={qubits})"
            )


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

    settings = sorted(
        df[setting_cols].drop_duplicates().itertuples(index=False, name=None)
    )
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


def _plot_raw_fidelity_breakdown(raw_df, output_dir, *, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6.2))
    components = [
        ("fidelity_cz", "CZ", "#4C78A8"),
        ("fidelity_1q", "1Q", "#F58518"),
        ("fidelity_move", "Move", "#54A24B"),
        ("fidelity_init", "Init", "#E45756"),
        ("fidelity_measurement", "Measurement", "#72B7B2"),
        ("fidelity_idle", "Idle", "#9467BD"),
    ]
    _plot_stacked_bars_infidelity(
        ax,
        raw_df.copy(),
        "n_qubit",
        components,
        "Raw Infidelity Breakdown",
    )
    ax.legend(loc="upper left", frameon=True, fontsize=_FIG_FONT_SIZE - 2)
    fig.subplots_adjust(right=0.96)
    if filename is None:
        filename = "raw_fidelity_breakdown_stacked.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_star_fidelity_breakdown(
    star_df,
    output_dir,
    *,
    filename=None,
):
    data = _select_star_comparison_setting(star_df)
    data = data.copy()
    data["n_qubit"] = data["qubit_layout"].apply(get_n_qubit)

    if data.empty or "code_distance" not in data.columns:
        return
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    components = [
        ("fidelity_cnot", "CNOT", "#4C78A8"),
        ("fidelity_1q", "H", "#8C564B"),
        ("fidelity_of_rz_injection", "Rz(theta)", "#C44E52"),
        ("fidelity_of_rz_teleportaion", "Teleportation-CNOT", "#F28E2B"),
        ("fidelity_of_rz_s", "Rz-S", "#59A14F"),
        # Simulator-native idle: ``fidelity_idle`` = (1 - p_I) ** n_se_q,
        # produced by the SE_q-based logical-SE scheduler.
        ("fidelity_idle", "Idle", "#4DBBD5"),
    ]
    handles, labels = _plot_grouped_stacked_infidelity(
        ax,
        data,
        "n_qubit",
        ["code_distance"],
        components,
        "STAR Infidelity Breakdown",
        setting_label_fn=lambda s: f"d={int(s[0])}",
    )
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            frameon=True,
            fontsize=_FIG_FONT_SIZE - 2,
            ncol=2,
        )
    fig.subplots_adjust(bottom=0.34, right=0.96)
    if filename is None:
        filename = "star_fidelity_breakdown_stacked.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_t_cultivation_fidelity_breakdown(t_df, output_dir, *, filename=None):
    if t_df is None or t_df.empty:
        return
    data = t_df.copy()
    data["n_qubit"] = data["qubit_layout"].apply(get_n_qubit)
    if data.empty or "code_distance" not in data.columns:
        return
    fig, ax = plt.subplots(figsize=(12.4, 6.8))
    components = [
        ("fidelity_cnot", "CNOT", "#4C78A8"),
        ("fidelity_h", "H", "#8C564B"),
        ("fidelity_of_rz_teleportaion", "Teleportation-CNOT", "#F58518"),
        ("fidelity_of_rz_s", "Rz-S", "#54A24B"),
        ("fidelity_of_rz_h", "Rz-H", "#BCBD22"),
        # Cultivated T-state imperfection: one F_T factor per teleportation
        # CNOT (= per consumed magic state).
        ("fidelity_of_t_gate", "T state", "#E45756"),
        # Gridsynth approximation: per-Rz state infidelity ~ epsilon ** 2.
        ("fidelity_synthesis", "Rz approx.", "#9467BD"),
        # ``fidelity_idle`` from the simulator (SE_q-based).
        ("fidelity_idle", "Idle", "#4DBBD5"),
    ]
    handles, labels = _plot_grouped_stacked_infidelity(
        ax,
        data,
        "n_qubit",
        ["code_distance"],
        components,
        "T-cultivation Infidelity Breakdown",
        setting_label_fn=lambda s: f"d={int(s[0])}",
        qubit_axis_outward=54,
    )
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            frameon=True,
            fontsize=_FIG_FONT_SIZE - 2,
            ncol=2,
        )
    fig.subplots_adjust(bottom=0.34, right=0.96)
    if filename is None:
        filename = "t_cultivation_fidelity_breakdown_stacked.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _populate_overall_ax(
    ax,
    raw_df,
    star_df,
    t_cultivation_df,
    *,
    include_raw: bool,
    as_infidelity: bool = False,
    architecture_distance_colors: dict[str, dict[int, str]] | None = None,
):
    """Render STAR / T-cultivation / (optionally) raw curves on ``ax``.

    All fidelities here already include the SE_q-based idle term: the STAR
    simulator returns ``fidelity`` and the T-cultivation simulator returns
    ``fidelity_total`` with idle (and gridsynth synthesis for T) folded in.

    When ``as_infidelity`` is True, ``1 - fidelity`` is plotted instead, with
    the y-axis flipped to log scale by the caller.
    """

    def _to_y(series):
        return (1.0 - series) if as_infidelity else series

    if architecture_distance_colors is None:
        architecture_distance_colors = _build_architecture_distance_colors(
            star_df, t_cultivation_df
        )

    if include_raw:
        raw_mean = (
            raw_df.groupby("n_qubit", as_index=False)["fidelity"]
            .mean()
            .sort_values("n_qubit")
        )
        ax.plot(
            raw_mean["n_qubit"],
            _to_y(raw_mean["fidelity"]),
            linewidth=3,
            markersize=10,
            label="Physical",
            alpha=0.7,
            **_METHOD_STYLES["Physical"],
        )

    star_data = _select_star_comparison_setting(star_df)
    if not star_data.empty:
        star_data["n_qubit"] = star_data["qubit_layout"].apply(get_n_qubit)
        star_data["fidelity"] = pd.to_numeric(star_data["fidelity"], errors="coerce")
        star_data = star_data.dropna(subset=["n_qubit", "fidelity"])

        for d in sorted(star_data["code_distance"].dropna().astype(int).unique()):
            sd = star_data[star_data["code_distance"] == d].copy()
            if sd.empty:
                continue
            grouped = sd.groupby("n_qubit")["fidelity"]
            means = grouped.mean().sort_index()
            mins = grouped.min().reindex(means.index)
            maxs = grouped.max().reindex(means.index)
            # When plotting infidelity, ``1 - fidelity`` flips min/max ordering.
            y_means = _to_y(means)
            y_lo = _to_y(maxs) if as_infidelity else mins
            y_hi = _to_y(mins) if as_infidelity else maxs
            color = architecture_distance_colors["STAR"].get(d)
            line = ax.plot(
                means.index,
                y_means.values,
                color=color,
                linewidth=2,
                markersize=8,
                label=f"STAR d={d}",
                alpha=0.85,
                **_METHOD_STYLES["STAR"],
            )[0]
            ax.fill_between(
                means.index,
                y_lo.values,
                y_hi.values,
                color=line.get_color(),
                alpha=0.15,
            )

    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_df = _select_t_cultivation_comparison_data(t_cultivation_df)
        if t_df is None or t_df.empty:
            return
        t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
        t_df["fidelity_total"] = pd.to_numeric(t_df["fidelity_total"], errors="coerce")
        t_df = t_df.dropna(subset=["n_qubit", "fidelity_total"])

        for code_distance, sdf in t_df.groupby("code_distance"):
            grouped = sdf.groupby("n_qubit")["fidelity_total"]
            means = grouped.mean().sort_index()
            mins = grouped.min().reindex(means.index)
            maxs = grouped.max().reindex(means.index)
            y_means = _to_y(means)
            y_lo = _to_y(maxs) if as_infidelity else mins
            y_hi = _to_y(mins) if as_infidelity else maxs
            d = int(code_distance)
            color = architecture_distance_colors["T-cultivation"].get(d)
            line = ax.plot(
                means.index,
                y_means.values,
                color=color,
                linewidth=2,
                markersize=8,
                label=f"T-cultivation d={d}",
                alpha=0.9,
                **_METHOD_STYLES["T-cultivation"],
            )[0]
            ax.fill_between(
                means.index,
                y_lo.values,
                y_hi.values,
                color=line.get_color(),
                alpha=0.12,
            )

    ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=_FIG_FONT_SIZE)
    ax.grid(True, alpha=0.3, which="both" if as_infidelity else "major")


def _plot_overall(
    raw_df,
    star_df,
    output_dir,
    t_cultivation_df=None,
    *,
    include_raw: bool = True,
    filename=None,
):
    fig, ax = plt.subplots(figsize=(10, 6.4))
    architecture_distance_colors = _build_architecture_distance_colors(
        star_df, t_cultivation_df
    )
    _populate_overall_ax(
        ax,
        raw_df,
        star_df,
        t_cultivation_df,
        include_raw=include_raw,
        architecture_distance_colors=architecture_distance_colors,
    )
    ax.set_ylabel("Mean Fidelity", fontsize=_FIG_FONT_SIZE)
    ax.set_title("Overall Fidelity Comparison", fontsize=_FIG_FONT_SIZE)
    _add_overall_legends(
        ax,
        include_raw=include_raw,
        architecture_distance_colors=architecture_distance_colors,
    )
    fig.subplots_adjust(right=0.96)

    if filename is None:
        filename = "overall_fidelity_comparison.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_overall_infidelity(
    raw_df,
    star_df,
    output_dir,
    t_cultivation_df=None,
    *,
    include_raw: bool = True,
    filename=None,
):
    """Overall infidelity (``1 - fidelity``) on a log y-axis."""
    fig, ax = plt.subplots(figsize=(10, 6.4))
    architecture_distance_colors = _build_architecture_distance_colors(
        star_df, t_cultivation_df
    )
    _populate_overall_ax(
        ax,
        raw_df,
        star_df,
        t_cultivation_df,
        include_raw=include_raw,
        as_infidelity=True,
        architecture_distance_colors=architecture_distance_colors,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Mean Infidelity", fontsize=_FIG_FONT_SIZE)
    ax.set_title("Overall Infidelity Comparison", fontsize=_FIG_FONT_SIZE)
    _add_overall_legends(
        ax,
        include_raw=include_raw,
        architecture_distance_colors=architecture_distance_colors,
    )
    fig.subplots_adjust(right=0.96)

    if filename is None:
        filename = (
            "overall_infidelity_comparison.pdf"
            if include_raw
            else "overall_infidelity_comparison_no_raw.pdf"
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

    print(
        "Using comparison filters: "
        f"placement={_COMPARISON_PLACEMENT}, "
        f"n_aods={_COMPARISON_N_AODS}, "
        f"STAR setting={_STAR_COMPARISON_SETTING_INDEX}, "
        f"T settings={_T_COMPARISON_SETTINGS}"
    )

    print("Loading data...")
    raw_df, star_df, t_df = load_data()
    print(
        f"Loaded raw={len(raw_df)}, star={len(star_df)}, "
        f"t_cultivation={0 if t_df is None else len(t_df)}"
    )
    _print_pairwise_infidelity_improvements(raw_df, star_df, t_df)

    print("\n1. Plot raw stacked fidelity breakdown...")
    _plot_raw_fidelity_breakdown(raw_df, output_dir)

    print("\n2. Plot STAR stacked fidelity breakdown...")
    _plot_star_fidelity_breakdown(star_df, output_dir)

    print("\n3. Plot T-cultivation stacked fidelity breakdown...")
    _plot_t_cultivation_fidelity_breakdown(t_df, output_dir)

    print("\n4. Plot overall fidelity (with raw line)...")
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
    )

    print("\n5. Plot overall fidelity (no raw line)...")
    _plot_overall(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
        include_raw=False,
        filename="overall_fidelity_comparison_no_raw.pdf",
    )

    print("\n5a. Plot overall infidelity (log y, with raw line)...")
    _plot_overall_infidelity(
        raw_df,
        star_df,
        output_dir,
        t_cultivation_df=t_df,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
