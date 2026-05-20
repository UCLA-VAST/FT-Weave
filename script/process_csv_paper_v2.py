"""Plot STAR vs T-cultivation runtime: single 2×2 setting-study + AOD comparison figure."""

import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

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

# (placement, prepare_lookahead_angles, trivial_return, consider_skip_rus,
#  decompose_move, parallel_execution) — keep in sync with evaluation_fidelity_star.py
SETTINGS = [
    ("seperate_region_row", False, True, 0, False, False),
    ("col_based", False, True, 0, False, False),
    ("col_based", True, True, 0, False, False),
    ("col_based", True, False, 0, True, False),
    ("col_based", True, False, 2, True, False),
    ("col_based", True, False, 2, False, True),
]

RESULT_COLS = ["total_time", "movement_time", "return_movement_time"]

STAR_T_GRID_COMPARISON_CODE_DISTANCE: int = 9
STAR_T_GRID_STAR_SETTING_INDEX: int = 4
STAR_T_SETTING_STUDY_AOD: int = 1
# Setting-study left panel: crop y-axis when Vanilla (index 0) dominates the scale.
STAR_SETTING_STUDY_VANILLA_IDX: int = 0
# Broken-axis split: bottom panel [0, break], top panel [break, Vanilla max].
STAR_SETTING_STUDY_Y_BREAK: float = 1200.0
STAR_SETTING_STUDY_Y_CROP_GAP_RATIO: float = 1.25
STAR_SETTING_STUDY_Y_CROP_PAD_FRAC: float = 0.1
T_SETTING_STUDY_VANILLA_PLACEMENT: str = "seperate_region_row"
# T setting-study: vanilla on separate region (grid[0]); other compile tuples on col_based.
T_SETTING_STUDY_VANILLA_COMPILE: tuple[bool, bool, bool] = (True, False, False)
T_SETTING_STUDY_COL_BASED_COMPILE_IDXS: tuple[int, ...] = (1, 2, 3)

T_CULTIVATION_MAIN_COMPILE_SETTING: tuple[bool, bool, bool] = (False, True, True)

SHOW_T_CULTIVATION_D13 = False
T_CULTIVATION_RUNTIME_LINE_SETTINGS: list[tuple[int, float, int]] = [
    (9, 1e-8, 2),
    (13, 1e-8, 4),
]

_T_COMPILE_ABLATION_GRID: list[tuple[bool, bool, bool]] = [
    (True, False, False),
    (True, False, False),
    (False, True, False),
    (False, True, True),
]
_T_COMPILE_ABLATION_LABELS: list[str] = [
    "Vanilla",
    "(a) Opt. microarch.",
    "(b) + Opt. move",
    "(c) + Opt. patch redist.",
]

_STAR_ABLATION_LABELS = [
    "Vanilla",
    "(a) Opt. microarch.",
    "(b) + Opt. angle assign.",
    "(c) + Opt. move",
    "(d) + Opt. skip RUS",
    "(e) + Async. RUS",
]


def _star_setting_study_entries() -> list[tuple[int, str]]:
    """(``SETTINGS`` index, legend label) for the STAR setting-study panel."""
    n = min(len(SETTINGS), len(_STAR_ABLATION_LABELS))
    return [(i, _STAR_ABLATION_LABELS[i]) for i in range(n)]


def _t_setting_study_entries() -> list[tuple[str, int, str]]:
    """(placement, compile-grid index, legend label) for the T setting-study panel."""
    entries: list[tuple[str, int, str]] = [
        (
            T_SETTING_STUDY_VANILLA_PLACEMENT,
            0,
            _T_COMPILE_ABLATION_LABELS[0],
        ),
    ]
    for compile_idx in T_SETTING_STUDY_COL_BASED_COMPILE_IDXS:
        if compile_idx >= len(_T_COMPILE_ABLATION_GRID) or compile_idx >= len(
            _T_COMPILE_ABLATION_LABELS
        ):
            continue
        entries.append(
            ("col_based", compile_idx, _T_COMPILE_ABLATION_LABELS[compile_idx])
        )
    return entries


def _setting_filter(df: pd.DataFrame, setting: tuple) -> pd.Series:
    placement, prepare_lookahead, trivial_return, skip_rus, decompose_move, parallel = (
        setting
    )
    mask = (df["placement"].astype(str).str.strip() == str(placement).strip()) & (
        df["trivial_return"] == trivial_return
    )
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


_AOD_COLORBAR_MIN = 1
_AOD_COLORBAR_MAX = 5
# Dedicated red ramp for the AOD panel (distinct from left-panel tab10 strategies).
_AOD_LIGHT_RGB = (0.98, 0.72, 0.72)
_AOD_DARK_RGB = (0.62, 0.0, 0.0)


def _build_aod_color_map(plot_aods: list[int] | None = None) -> dict[int, tuple]:
    """Light red at AOD = 1, progressively darker through AOD = 5 (T-cultivation panel)."""
    light_rgb = np.array(_AOD_LIGHT_RGB)
    dark_rgb = np.array(_AOD_DARK_RGB)
    aods = sorted(
        {
            int(a)
            for a in (
                plot_aods or list(range(_AOD_COLORBAR_MIN, _AOD_COLORBAR_MAX + 1))
            )
        }
    )
    out: dict[int, tuple] = {}
    denom = max(_AOD_COLORBAR_MAX - _AOD_COLORBAR_MIN, 1)
    for aod in aods:
        t = (int(aod) - _AOD_COLORBAR_MIN) / denom
        rgb = (1.0 - t) * light_rgb + t * dark_rgb
        out[int(aod)] = tuple(np.clip(rgb, 0.0, 1.0))
    return out


def _build_aod_color_map_from_base(
    base_rgb: tuple[float, float, float],
    plot_aods: list[int] | None = None,
) -> dict[int, tuple]:
    """AOD = 1 matches *base_rgb*; higher AOD values are progressively darker."""
    base = np.clip(np.asarray(base_rgb, dtype=float), 0.0, 1.0)
    dark_rgb = base * 0.38
    aods = sorted(
        {
            int(a)
            for a in (
                plot_aods or list(range(_AOD_COLORBAR_MIN, _AOD_COLORBAR_MAX + 1))
            )
        }
    )
    out: dict[int, tuple] = {}
    denom = max(_AOD_COLORBAR_MAX - _AOD_COLORBAR_MIN, 1)
    for aod in aods:
        if int(aod) == _AOD_COLORBAR_MIN:
            out[int(aod)] = tuple(base)
            continue
        t = (int(aod) - _AOD_COLORBAR_MIN) / denom
        rgb = (1.0 - t) * base + t * dark_rgb
        out[int(aod)] = tuple(np.clip(rgb, 0.0, 1.0))
    return out


def _aod_listed_colormap(aod_color_map: dict[int, tuple]) -> mcolors.ListedColormap:
    full_map = _build_aod_color_map()
    colors = [
        aod_color_map.get(a, full_map.get(a, _AOD_DARK_RGB))
        for a in range(_AOD_COLORBAR_MIN, _AOD_COLORBAR_MAX + 1)
    ]
    return mcolors.ListedColormap(colors, name="aod_sweep", N=_AOD_COLORBAR_MAX)


def _expected_time_legend_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Expected time",
    )


def _get_theoretical_lower_bound(n_values: list) -> dict[str, list]:
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
    _ = fidelity_target
    n_t_per_qubit = 40
    if code_distance is not None and int(code_distance) < 13:
        stage1_factor = 12.5 * 1 / (1 - 0.7 * 0.7)
    else:
        stage1_factor = 12.5
    if code_distance is not None and int(code_distance) <= 13:
        stage2_factor = (1 / 0.4) * (6 + stage1_factor)
    else:
        stage2_factor = (1 / 0.66) * (0.5 + stage1_factor)
    t_cultivation_scale = stage2_factor + 2.0 + 1.0 + 1.0
    x_layer_bounds = [t_cultivation_scale * n_t_per_qubit for _ in n_values]
    zz_layer_bounds = [t_cultivation_scale * (n_t_per_qubit / 2) for _ in n_values]
    full_trotter_bounds = [x + 8 * zz for x, zz in zip(x_layer_bounds, zz_layer_bounds)]
    return {
        "x_layer": x_layer_bounds,
        "zz_layer": zz_layer_bounds,
        "full_trotter": full_trotter_bounds,
    }


def aggregate_full_trotter(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consecutive Rz rounds into one full-trotter-step row."""
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
        "trivial_return",
        "decompose_move",
        "parallel_execution",
        "prepare_lookahead_angles",
    ]
    for optional in (
        "trial",
        "fidelity_target",
        "factory_physical_size",
        "redistribute_stage1_success",
    ):
        if optional in df.columns and optional not in config_cols:
            config_cols.append(optional)

    work = df.copy().reset_index(drop=False).rename(columns={"index": "_row_order"})
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


def _star_full_trotter_dict(star_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"full_trotter": aggregate_full_trotter(star_df)}


def _format_nqubit_ticklabels(
    x_vals: list,
    round_name: str,
    round_angle_lookup: dict[str, dict[int, int]] | None,
) -> list[str]:
    _ = (round_name, round_angle_lookup)
    return [str(int(x)) if pd.notna(x) else "" for x in x_vals]


def _format_y_tick_thousands(value: float, _pos: float) -> str:
    scaled = value / 1000.0
    if abs(scaled) < 1e-9:
        return "0"
    if abs(scaled - round(scaled)) < 1e-6:
        return str(int(round(scaled)))
    return f"{scaled:g}"


def _apply_y_axis_thousands(axes: list) -> None:
    formatter = FuncFormatter(_format_y_tick_thousands)
    for ax in axes:
        ax.yaxis.set_major_formatter(formatter)


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
        "prepare_lookahead_angles",
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


def _build_layer_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.copy()
    default_cols = {
        "consider_skip_rus": 0,
        "trivial_return": False,
        "decompose_move": False,
        "parallel_execution": False,
        "prepare_lookahead_angles": True,
    }
    for col, default_val in default_cols.items():
        if col not in work.columns:
            work[col] = default_val
    return {"full_trotter": aggregate_full_trotter(work)}


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


def _dedupe_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _t_cultivation_runtime_line_settings() -> list[tuple[int, float, int]]:
    if SHOW_T_CULTIVATION_D13:
        return T_CULTIVATION_RUNTIME_LINE_SETTINGS
    return [
        setting
        for setting in T_CULTIVATION_RUNTIME_LINE_SETTINGS
        if int(setting[0]) != 13
    ]


def _filter_t_cultivation_d13_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    if SHOW_T_CULTIVATION_D13 or "code_distance" not in df.columns:
        return df
    out = df.copy()
    cd = pd.to_numeric(out["code_distance"], errors="coerce")
    return out[cd != 13].copy()


def _mask_t_cultivation_triple(
    df: pd.DataFrame, cd: int, ft: float, fps: int
) -> pd.Series:
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
    return (
        (df["trivial_return"] == trivial_return)
        & (df["decompose_move"] == decompose_move)
        & (df["redistribute_stage1_success"] == redistribute_stage1_success)
    )


def _filter_t_cultivation_main_compile_setting(df: pd.DataFrame) -> pd.DataFrame:
    cols = ("trivial_return", "decompose_move", "redistribute_stage1_success")
    if df.empty or not all(c in df.columns for c in cols):
        return df
    tr, dm, rs = T_CULTIVATION_MAIN_COMPILE_SETTING
    return df.loc[_mask_t_cultivation_compile(df, tr, dm, rs)].copy()


def _primary_t_triple_for_star_t_grid(
    code_distance: int,
) -> tuple[int, float, int] | None:
    for triple in _t_cultivation_runtime_line_settings():
        if int(triple[0]) == int(code_distance):
            return triple
    return None


def _save_setting_aod_architecture_figure(
    *,
    draw_setting,
    draw_aod,
    setting_legend_handles: list,
    aod_color_map: dict[int, tuple],
    figure_title: str,
    out_path: str,
    xlabel: str = "Number of Qubits",
    column_titles: tuple[str, str] = (
        "Compilation Strategies, AOD = 1",
        "AOD comparison",
    ),
    y_axis_thousands: bool = False,
) -> None:
    """One row, two columns: setting study (left) and AOD sweep (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), squeeze=False)
    draw_setting(axes[0, 0])
    draw_aod(axes[0, 1])

    panel_title_fs = _FIG_FONT_SIZE - 1
    legend_fs = _FIG_FONT_SIZE - 4
    axes[0, 0].set_title(column_titles[0], fontsize=panel_title_fs, pad=8)
    axes[0, 1].set_title(column_titles[1], fontsize=panel_title_fs, pad=8)
    y_label = "Execution time (×10³)" if y_axis_thousands else "Execution time"
    axes[0, 0].set_ylabel(y_label, fontsize=_FIG_FONT_SIZE, labelpad=2)
    if y_axis_thousands:
        _apply_y_axis_thousands([axes[0, 0], axes[0, 1]])
    axes[0, 1].set_ylabel("")
    for col in range(2):
        axes[0, col].set_xlabel(xlabel, fontsize=_FIG_FONT_SIZE, labelpad=0)
        axes[0, col].tick_params(axis="both", which="major", pad=1)

    fig.suptitle(figure_title, fontsize=_FIG_FONT_SIZE + 1, y=0.96)
    fig.tight_layout(rect=(0.06, 0.14, 0.86, 0.90), pad=0.10, w_pad=0.1)
    fig.subplots_adjust(top=0.83, bottom=0.34, left=0.07, right=0.90, wspace=0.15)

    combined_legend = [*setting_legend_handles, _expected_time_legend_handle()]
    fig.legend(
        handles=combined_legend,
        loc="lower center",
        bbox_to_anchor=(0.46, -0.05),
        ncol=3,
        fontsize=legend_fs,
        frameon=True,
        columnspacing=0.6,
        handletextpad=0.35,
        labelspacing=0.35,
    )

    aod_cmap = _aod_listed_colormap(aod_color_map)
    aod_norm = mcolors.BoundaryNorm(
        boundaries=np.arange(
            _AOD_COLORBAR_MIN - 0.5, _AOD_COLORBAR_MAX + 1.5, dtype=float
        ),
        ncolors=_AOD_COLORBAR_MAX,
    )
    sm = ScalarMappable(norm=aod_norm, cmap=aod_cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes[0, 1],
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
        ticks=list(range(_AOD_COLORBAR_MIN, _AOD_COLORBAR_MAX + 1)),
    )
    cbar.ax.tick_params(labelsize=legend_fs)
    cbar.outline.set_linewidth(0.8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _normalize_placement_name(placement: str) -> str:
    return str(placement).strip()


def _aggregate_star_setting_study(
    dfs_dict_micro: dict[str, pd.DataFrame],
    setting: tuple,
    code_distance: int,
    aod: int,
) -> pd.DataFrame | None:
    round_name = "full_trotter"
    if round_name not in dfs_dict_micro:
        return None
    df = (
        dfs_dict_micro[round_name]
        .loc[_setting_filter(dfs_dict_micro[round_name], setting)]
        .copy()
    )
    if df.empty:
        return None
    if "code_distance" in df.columns:
        df = df[
            pd.to_numeric(df["code_distance"], errors="coerce").astype(int)
            == int(code_distance)
        ].copy()
    if df.empty:
        return None
    df["n_aods"] = pd.to_numeric(df["n_aods"], errors="coerce").astype(int)
    df = df[df["n_aods"] == int(aod)].copy()
    if df.empty:
        return None
    grouped = (
        df.groupby("n_qubits")[["total_time"]].agg(["mean", "min", "max"]).reset_index()
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
    work = layer_df.copy()
    work["placement"] = work["placement"].astype(str).map(_normalize_placement_name)
    work = work[work["placement"] == _normalize_placement_name(placement)].copy()
    cd_t, ft, fps = triple
    tr, dm, rs = compile_tuple
    mask = _mask_t_cultivation_triple(
        work, cd_t, ft, fps
    ) & _mask_t_cultivation_compile(work, tr, dm, rs)
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


def _agg_y_extent(grouped: pd.DataFrame) -> tuple[float, float]:
    """Min/max execution time including error bars."""
    sub = grouped.sort_values("n_qubits")
    mean_vals = sub["total_time_mean"]
    min_vals = sub["total_time_min"]
    max_vals = sub["total_time_max"]
    y_lo = float((mean_vals - (mean_vals - min_vals)).min())
    y_hi = float((mean_vals + (max_vals - mean_vals)).max())
    return y_lo, y_hi


def _compute_star_setting_break_y(
    y_extents: dict[int, tuple[float, float]],
    *,
    vanilla_idx: int = STAR_SETTING_STUDY_VANILLA_IDX,
    pad_frac: float = STAR_SETTING_STUDY_Y_CROP_PAD_FRAC,
) -> float:
    """Pick a break so optimized curves fit below and every Vanilla point appears on bot or top."""
    v_lo, v_hi = y_extents[vanilla_idx]
    other_hi = [
        hi for idx, (_lo, hi) in y_extents.items() if int(idx) != int(vanilla_idx)
    ]
    zoom_hi = max(other_hi) if other_hi else v_lo
    # Above all non-Vanilla data and above Vanilla minimum (low-qubit points on bottom panel).
    break_y = max(zoom_hi * (1.0 + pad_frac), v_lo * (1.0 + 0.5 * pad_frac))
    # Leave headroom for the upper Vanilla segment.
    break_y = min(break_y, v_hi * (1.0 - 0.04))
    if break_y <= v_lo:
        break_y = (v_lo + v_hi) / 2.0
    return float(break_y)


def _star_setting_broken_axis_needed(
    y_extents: dict[int, tuple[float, float]],
    *,
    vanilla_idx: int = STAR_SETTING_STUDY_VANILLA_IDX,
    gap_ratio: float = STAR_SETTING_STUDY_Y_CROP_GAP_RATIO,
) -> bool:
    if vanilla_idx not in y_extents:
        return False
    vanilla_hi = float(y_extents[vanilla_idx][1])
    vanilla_lo = float(y_extents[vanilla_idx][0])
    other_hi = [
        hi for idx, (_lo, hi) in y_extents.items() if int(idx) != int(vanilla_idx)
    ]
    if not other_hi:
        return False
    zoom_hi = max(other_hi)
    if vanilla_hi <= zoom_hi * float(gap_ratio):
        return False
    # Need a gap large enough to split Vanilla across panels.
    return (vanilla_hi - vanilla_lo) > (zoom_hi - vanilla_lo) * 0.15


def _draw_axis_break_between(
    ax_upper,
    ax_lower,
    *,
    d: float = 0.018,
) -> None:
    """Diagonal break marks between a top (Vanilla) and bottom (zoomed) panel."""
    kwargs = {
        "color": "black",
        "clip_on": False,
        "linewidth": 1.2,
        "solid_capstyle": "round",
    }
    ax_upper.plot(
        (-d, +d),
        (-d, +d),
        transform=ax_upper.transAxes,
        **kwargs,
    )
    ax_upper.plot(
        (1 - d, 1 + d),
        (-d, +d),
        transform=ax_upper.transAxes,
        **kwargs,
    )
    ax_lower.plot(
        (-d, +d),
        (1 - d, 1 + d),
        transform=ax_lower.transAxes,
        **kwargs,
    )
    ax_lower.plot(
        (1 - d, 1 + d),
        (1 - d, 1 + d),
        transform=ax_lower.transAxes,
        **kwargs,
    )


def _create_star_setting_broken_axes(ax_parent):
    """Split the setting-study cell into top (Vanilla) and bottom (other settings) axes."""
    fig = ax_parent.figure
    ax_parent.set_visible(False)
    gs = ax_parent.get_subplotspec().subgridspec(
        2,
        1,
        height_ratios=[1, 4],
        hspace=0.06,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.spines.bottom.set_visible(False)
    ax_bot.spines.top.set_visible(False)
    ax_top.set_axisbelow(True)
    ax_bot.set_axisbelow(True)
    ax_top.grid(True, alpha=0.3)
    ax_bot.grid(True, alpha=0.3)
    ax_parent._broken_setting_axis = True  # type: ignore[attr-defined]
    return ax_top, ax_bot


def _set_ylim_from_extent(
    ax,
    y_lo: float,
    y_hi: float,
    *,
    pad_frac: float = STAR_SETTING_STUDY_Y_CROP_PAD_FRAC,
    floor_zero: bool = True,
) -> None:
    span = max(y_hi - y_lo, 1e-9)
    pad = span * float(pad_frac)
    bottom = 0.0 if floor_zero else y_lo - pad
    ax.set_ylim(bottom, y_hi + pad)


def _star_setting_zoom_inset_needed(
    y_extents: dict[int, tuple[float, float]],
    *,
    vanilla_idx: int = STAR_SETTING_STUDY_VANILLA_IDX,
    gap_ratio: float = STAR_SETTING_STUDY_Y_CROP_GAP_RATIO,
) -> bool:
    return _star_setting_broken_axis_needed(
        y_extents, vanilla_idx=vanilla_idx, gap_ratio=gap_ratio
    )


def _add_star_setting_zoom_inset(
    ax,
    prepared: list[tuple[int, str, pd.DataFrame]],
    *,
    star_setting_colors: dict[int, tuple],
    vanilla_idx: int,
    y_extents: dict[int, tuple[float, float]],
    x_points: list[int],
) -> None:
    """Inset on the setting-study axes: optimized curves only, expanded y-scale."""
    other_hi = [
        hi for idx, (_lo, hi) in y_extents.items() if int(idx) != int(vanilla_idx)
    ]
    if not other_hi:
        return
    zoom_hi = max(other_hi)
    inset_fs = max(_FIG_FONT_SIZE - 10, 14)
    axins = ax.inset_axes([0.50, 0.12, 0.47, 0.50])
    for setting_idx, _label, grouped in prepared:
        if int(setting_idx) == int(vanilla_idx):
            continue
        _draw_errorbar_from_agg(
            axins,
            grouped,
            color=star_setting_colors[setting_idx],
            label=None,
        )
    x_u = sorted(set(int(x) for x in x_points))
    if x_u:
        b = _get_theoretical_lower_bound(x_u)
        axins.plot(
            x_u,
            b["full_trotter"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
    _set_ylim_from_extent(axins, 0.0, zoom_hi)
    axins.set_xlim(ax.get_xlim())
    axins.grid(True, alpha=0.3)
    axins.tick_params(axis="both", labelsize=inset_fs)
    # axins.set_title(
    #     "Optimized settings (zoom)",
    #     fontsize=inset_fs,
    #     pad=4,
    # )
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)


def _t_aod_panel_base_rgb(
    t_setting_colors: dict[tuple[str, int], tuple],
) -> tuple[float, float, float]:
    """Match AOD = 1 to the compile setting used in the T AOD sweep panel."""
    tr_m, dm_m, rs_m = T_CULTIVATION_MAIN_COMPILE_SETTING
    for placement, compile_idx, _label in _t_setting_study_entries():
        if compile_idx >= len(_T_COMPILE_ABLATION_GRID):
            continue
        if placement == "col_based" and _T_COMPILE_ABLATION_GRID[compile_idx] == (
            tr_m,
            dm_m,
            rs_m,
        ):
            return mcolors.to_rgb(t_setting_colors[(placement, compile_idx)])
    return mcolors.to_rgb(t_setting_colors[("col_based", 3)])


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
    agg_data: dict[str, pd.DataFrame] = {}
    for round_name, df in dfs_dict_aod.items():
        df_col = df.loc[_setting_filter(df, setting)].copy()
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
    gd = grouped.loc[
        pd.to_numeric(grouped["code_distance"], errors="coerce").astype(int)
        == int(code_distance)
    ].copy()
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
    verbose: bool = True,
) -> None:
    """Two 1×2 figures: STAR and T-cultivation, each with setting study and AOD 1–5."""
    os.makedirs(output_dir, exist_ok=True)
    cd = int(
        code_distance
        if code_distance is not None
        else STAR_T_GRID_COMPARISON_CODE_DISTANCE
    )
    round_name = "full_trotter"
    aod_setting = int(setting_study_aod)
    star_aod_setting = SETTINGS[int(star_setting_index)]

    g_cb = _star_grouped_aod_col_based_at_distance(dfs_dict_aod, star_aod_setting, cd)
    if g_cb is None:
        g_cb = pd.DataFrame()

    triple = _primary_t_triple_for_star_t_grid(cd)
    cd_t, ft, fps = (0, 0.0, 0)
    t_setting_layer = pd.DataFrame()
    t_base = pd.DataFrame()
    compile_cols = {
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
    }
    if (
        triple is not None
        and round_name in t_layers
        and round_name in t_layers_ablation
    ):
        cd_t, ft, fps = triple
        t_aod_layer = t_layers[round_name].copy()
        t_setting_layer = t_layers_ablation[round_name].copy()
        if "placement" in t_aod_layer.columns:
            t_aod_layer = _filter_col_based(t_aod_layer)
        if compile_cols.issubset(t_aod_layer.columns) and compile_cols.issubset(
            t_setting_layer.columns
        ):
            tr_m, dm_m, rs_m = T_CULTIVATION_MAIN_COMPILE_SETTING
            t_base = t_aod_layer.loc[
                _mask_t_cultivation_compile(t_aod_layer, tr_m, dm_m, rs_m)
                & _mask_t_cultivation_triple(t_aod_layer, cd_t, ft, fps)
            ].copy()
            t_base = t_base[
                pd.to_numeric(t_base["code_distance"], errors="coerce").astype(int)
                == cd
            ].copy()

    aod_candidates = [1, 2, 3, 4, 5]
    star_aods: set[int] = set()
    if not g_cb.empty:
        star_aods = set(
            pd.to_numeric(g_cb["n_aods"], errors="coerce").dropna().astype(int).unique()
        )
    t_aods: set[int] = set()
    if not t_base.empty:
        t_aods = set(
            pd.to_numeric(t_base["n_aods"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
    star_plot_aods = [a for a in aod_candidates if int(a) in star_aods]
    t_plot_aods = [a for a in aod_candidates if int(a) in t_aods]
    if not star_plot_aods and not t_plot_aods:
        if verbose:
            print(f"Skipping figures: no AOD in {{1,…,5}} at d={cd}.")
        return

    setting_cmap = plt.get_cmap("tab10")
    star_study_entries = _star_setting_study_entries()
    t_study_entries = _t_setting_study_entries()
    star_setting_colors = {
        setting_idx: setting_cmap(i % 10)
        for i, (setting_idx, _label) in enumerate(star_study_entries)
    }
    t_setting_colors = {
        (placement, compile_idx): setting_cmap(i % 10)
        for i, (placement, compile_idx, _label) in enumerate(t_study_entries)
    }
    star_best_rgb = mcolors.to_rgb(star_setting_colors[STAR_T_GRID_STAR_SETTING_INDEX])
    star_aod_colors = _build_aod_color_map_from_base(star_best_rgb, star_plot_aods)
    t_best_rgb = _t_aod_panel_base_rgb(t_setting_colors)
    t_aod_colors = _build_aod_color_map_from_base(t_best_rgb, t_plot_aods)
    star_aod_default = star_aod_colors.get(_AOD_COLORBAR_MIN, star_best_rgb)
    t_aod_default = t_aod_colors.get(_AOD_COLORBAR_MIN, t_best_rgb)

    def _finalize_panel(ax, x_points: list[int], *, theory: str) -> None:
        x_u = sorted(set(int(x) for x in x_points))
        if x_u:
            if theory == "star":
                b = _get_theoretical_lower_bound(x_u)
            else:
                b = _get_theoretical_lower_bound_t_cultivation(
                    x_u, code_distance=cd_t, fidelity_target=ft
                )
            ax.plot(
                x_u, b["full_trotter"], color="black", linestyle="--", linewidth=1.5
            )
            ax.set_xticks(x_u)
            ax.set_xticklabels(
                _format_nqubit_ticklabels(x_u, round_name, None), rotation=0
            )
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.relim(visible_only=True)
        ax.autoscale_view()
        _lo, y_hi = ax.get_ylim()
        if y_hi > 0.0:
            ax.set_ylim(0.0, y_hi)

    def _draw_star_setting_panel(ax) -> None:
        prepared: list[tuple[int, str, pd.DataFrame]] = []
        y_extents: dict[int, tuple[float, float]] = {}
        for setting_idx, label in star_study_entries:
            if setting_idx >= len(SETTINGS):
                continue
            grouped = _aggregate_star_setting_study(
                dfs_dict_micro,
                SETTINGS[setting_idx],
                cd,
                aod_setting,
            )
            if grouped is None:
                continue
            prepared.append((int(setting_idx), label, grouped))
            y_extents[int(setting_idx)] = _agg_y_extent(grouped)

        vanilla_idx = STAR_SETTING_STUDY_VANILLA_IDX
        x_pts: list[int] = []
        for setting_idx, label, grouped in prepared:
            x_pts.extend(
                _draw_errorbar_from_agg(
                    ax,
                    grouped,
                    color=star_setting_colors[setting_idx],
                    label=label,
                )
            )
        _finalize_panel(ax, x_pts, theory="star")
        if _star_setting_zoom_inset_needed(y_extents):
            _add_star_setting_zoom_inset(
                ax,
                prepared,
                star_setting_colors=star_setting_colors,
                vanilla_idx=vanilla_idx,
                y_extents=y_extents,
                x_points=x_pts,
            )

    def _draw_t_setting_panel(ax) -> None:
        x_pts: list[int] = []
        for placement, compile_idx, label in t_study_entries:
            if compile_idx >= len(_T_COMPILE_ABLATION_GRID):
                continue
            grouped = _aggregate_t_setting_study(
                t_setting_layer,
                triple,
                _T_COMPILE_ABLATION_GRID[compile_idx],
                placement,
                cd,
                aod_setting,
            )
            if grouped is None:
                continue
            x_pts.extend(
                _draw_errorbar_from_agg(
                    ax,
                    grouped,
                    color=t_setting_colors[(placement, compile_idx)],
                    label=label,
                )
            )
        _finalize_panel(ax, x_pts, theory="t")

    def _draw_star_aod_panel(ax) -> None:
        x_pts: list[int] = []
        for aod in star_plot_aods:
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
                color=star_aod_colors.get(int(aod), star_aod_default),
            )
            x_pts.extend(sub["n_qubits"].dropna().astype(int).tolist())
        _finalize_panel(ax, x_pts, theory="star")

    def _draw_t_aod_panel(ax) -> None:
        x_pts: list[int] = []
        for aod in t_plot_aods:
            t_a = t_base[t_base["n_aods"] == int(aod)].copy()
            summ = _summarize_execution_by_qubits(
                t_a, method="T cultivation"
            ).sort_values("n_qubits")
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
                color=t_aod_colors.get(int(aod), t_aod_default),
            )
            x_pts.extend(summ["n_qubits"].dropna().astype(int).tolist())
        _finalize_panel(ax, x_pts, theory="t")

    star_setting_legend_handles = [
        Line2D(
            [0],
            [0],
            color=star_setting_colors[setting_idx],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            label=label,
        )
        for setting_idx, label in star_study_entries
        if setting_idx in star_setting_colors
    ]

    t_setting_legend_handles = [
        Line2D(
            [0],
            [0],
            color=t_setting_colors[(placement, compile_idx)],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            label=label,
        )
        for placement, compile_idx, label in t_study_entries
        if (placement, compile_idx) in t_setting_colors
    ]

    plot_star = "full_trotter" in dfs_dict_micro
    plot_t = triple is not None and not t_setting_layer.empty

    if plot_star:
        star_path = os.path.join(
            output_dir, f"star_setting_and_aod_comparison_d{cd}.pdf"
        )
        _save_setting_aod_architecture_figure(
            draw_setting=_draw_star_setting_panel,
            draw_aod=_draw_star_aod_panel,
            setting_legend_handles=star_setting_legend_handles,
            aod_color_map=star_aod_colors,
            figure_title="STAR architecture",
            out_path=star_path,
            xlabel="Number of Qubits/Factories",
        )
        if verbose:
            print(f"Saved: {star_path}")

    if plot_t:
        t_path = os.path.join(
            output_dir, f"t_cultivation_setting_and_aod_comparison_d{cd}.pdf"
        )
        _save_setting_aod_architecture_figure(
            draw_setting=_draw_t_setting_panel,
            draw_aod=_draw_t_aod_panel,
            setting_legend_handles=t_setting_legend_handles,
            aod_color_map=t_aod_colors,
            figure_title="T-cultivation",
            out_path=t_path,
            xlabel="Number of Qubits/Factories",
            y_axis_thousands=True,
        )
        if verbose:
            print(f"Saved: {t_path}")


def process_star_t_setting_and_aod_figure(
    star_csv_file: str,
    t_cultivation_csv_file: str,
    output_dir: str,
    *,
    include_t_cultivation_d13: bool = False,
    code_distance: int = STAR_T_GRID_COMPARISON_CODE_DISTANCE,
    verbose: bool = True,
) -> None:
    """Load profiling CSVs and write STAR and T-cultivation 1×2 comparison PDFs."""
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
        dfs_star = _star_full_trotter_dict(star_df)
        t_df_ablation = _filter_t_cultivation_d13_for_plots(t_df)
        t_layers_ablation = _build_layer_frames(t_df_ablation)
        t_df_main = _filter_t_cultivation_d13_for_plots(
            _filter_t_cultivation_main_compile_setting(t_df)
        )
        t_layers = _build_layer_frames(t_df_main)
        _plot_star_t_setting_and_aod_combined_grid(
            dfs_star,
            dfs_star,
            t_layers,
            t_layers_ablation,
            output_dir,
            code_distance=code_distance,
            verbose=verbose,
        )
    finally:
        SHOW_T_CULTIVATION_D13 = previous_show_d13


if __name__ == "__main__":
    star_csv = "output/evaluation/fidelity/star_full_trotter_profiling_results.csv"
    t_csv = "output/evaluation/fidelity/t_cultivation_fidelity_profiling_results.csv"
    out_dir = "output/prx_quantum/"

    if os.path.exists(star_csv) and os.path.exists(t_csv):
        process_star_t_setting_and_aod_figure(star_csv, t_csv, out_dir, verbose=True)
    else:
        print("Missing profiling CSV(s); expected:")
        print(f"  {star_csv}")
        print(f"  {t_csv}")
