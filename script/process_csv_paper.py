"""Plot STAR vs T-cultivation runtime figures (setting study, AOD, runtime profile)."""

import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from src.t_cultivation.config import STAGE_1_SUCCESS_RATE

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
    ("seperate_region_row", False, True, 0, False, False),  # Basline
    ("seperate_region_row", False, False, 0, True, False),  # Baseline + opt mov
    # ("seperate_region_row", False, False, 2, True, False),
    # ("seperate_region_row", True, True, 0, False, False),
    # ("seperate_region_row", True, False, 0, True, False),
    # ("seperate_region_row", True, False, 2, True, False),
    # ("col_based", False, True, 0, False, False),
    # ("col_based", True, True, 0, False, False),
    # ("col_based", True, False, 0, True, False),
    # ("col_based", True, False, 2, True, False),
    # ("col_based", True, False, 2, False, True),
    ("col_based", False, False, 0, True, False),  # Baseline + opt mov + opt micro arch
    ("col_based", True, False, 0, True, True),  # Greedy with opt micro arch
    (
        "col_based",
        False,
        False,
        2,
        True,
        False,
    ),  # Efficient Parallel Execution with opt micro arch
    # ("col_based", True, False, 2, True, False),
    # ("col_based", False, False, 2, False, True),
]

RESULT_COLS = ["total_time", "movement_time", "return_movement_time"]

STAR_T_GRID_COMPARISON_CODE_DISTANCE: int = 9
STAR_T_GRID_STAR_SETTING_INDEX: int = 4
STAR_T_SETTING_STUDY_AOD: int = 5
# Setting-study dual-AOD bar chart: back = first AOD (full color), front = second (lighter).
SETTING_STUDY_DUAL_AODS: tuple[int, ...] = (1, 5)
SETTING_STUDY_DUAL_AOD_FRONT_BLEND: float = 0.52
# Denser y-axis ticks on AOD / execution-time panels.
STAR_AOD_COMPARISON_Y_NBINS: int = 8
# AOD line panel: sync vs best strategy, AOD = 1, 2, 3, 5.
AOD_COMPARISON_AODS: tuple[int, ...] = (1, 2, 3, 5)
T_AOD_SYNC_SETTING_INDEX: int = 0
_LINE_MARKERSIZE: float = 13.0
_LINE_MARKER_EDGEWIDTH: float = 2.0
_LINE_MARKER_FACE_BLEND: float = 0.52
# Setting-study panel: one clearly distinct color per strategy (no repeated hue family).
_SETTING_STUDY_CATEGORICAL_HEX: tuple[str, ...] = (
    "#31A354",  # green
    "#3182BD",  # blue
    "#DE2D26",  # red
    "#F16913",  # orange
    "#756BB1",  # purple
    "#636363",  # gray (fallback only)
)
T_CULTIVATION_Y_NBINS: int = 10
# Setting-study left panel: crop y-axis when Vanilla (index 0) dominates the scale.
STAR_SETTING_STUDY_VANILLA_IDX: int = 0
STAR_AOD_SYNC_SETTING_INDEX: int = STAR_SETTING_STUDY_VANILLA_IDX
# Broken-axis split: bottom panel [0, break], top panel [break, Vanilla max].
STAR_SETTING_STUDY_Y_BREAK: float = 1200.0
STAR_SETTING_STUDY_Y_CROP_GAP_RATIO: float = 1.25
STAR_SETTING_STUDY_Y_CROP_PAD_FRAC: float = 0.1
# Broken-axis layout for STAR AOD panel (modest gap + sparse upper ticks).
STAR_BROKEN_AXIS_HEIGHT_RATIOS: tuple[int, int] = (2, 5)
STAR_BROKEN_INNER_HSPACE: float = 0.11
STAR_BROKEN_TOP_Y_NBINS: int = 3
STAR_BROKEN_CLIP_MARGIN_FRAC: float = 0.012
AOD_BOTTOM_LEGEND_ROW_STEP: float = 0.042
AOD_BOTTOM_LEGEND_BASE_Y: float = 0.015
# T setting-study: configure placement per ablation entry (like STAR `SETTINGS`).

T_CULTIVATION_MAIN_COMPILE_SETTING: tuple[str, bool, bool, bool] = (
    "col_based",
    False,
    True,
    True,
)

# Runtime stacked-bar profile: col_based, compare AOD = 1 vs 5 at best compile setting.
RUNTIME_PROFILE_AODS: tuple[int, ...] = (1, 5)
RUNTIME_PROFILE_AOD_HATCH: dict[int, str | None] = {1: None, 5: "///"}
# STAR runtime profile: col_based, (d) + Opt. skip RUS — index 4 in ``SETTINGS``.
RUNTIME_PROFILE_STAR_SETTING: tuple = (
    "col_based",
    True,
    False,
    2,
    True,
    False,
)
RUNTIME_PROFILE_T_SETTING_IDX: int = 3  # col_based + Opt. patch redist.
_RUNTIME_PROFILE_MOVE_COMPONENTS: list[tuple[str, str, str]] = [
    ("forward_move_mean", "Forward move", "#4C78A8"),
    ("return_move_mean", "Return move", "#9ECAE9"),
]
_RUNTIME_PROFILE_T_COMPONENTS: list[tuple[str, str, str]] = [
    *_RUNTIME_PROFILE_MOVE_COMPONENTS,
    ("stage1_mean", "Check Stage", "#F5B318BD"),
    ("stage2_mean", "Escape Stage", "#B1BC1B"),
    ("rus_mean", "CNOT", "#54A24B"),
]
_RUNTIME_PROFILE_STAR_COMPONENTS: list[tuple[str, str, str]] = [
    *_RUNTIME_PROFILE_MOVE_COMPONENTS,
    ("tmr_mean", "TMR", "#F58518"),
    ("rus_mean", "CNOT", "#54A24B"),
]

_T_PROFILING_REQUIRED_COLS: tuple[str, ...] = (
    "total_time",
    "movement_time",
    "return_movement_time",
    "stage1_time",
    "stage2_time",
    "RUS_round",
)

_RUNTIME_PROFILE_STACK_RTOL: float = 2.0
_RUNTIME_PROFILE_STACK_ATOL: float = 2.0

SHOW_T_CULTIVATION_D13 = False
T_CULTIVATION_RUNTIME_LINE_SETTINGS: list[tuple[int, float, int]] = [
    (9, 1e-8, 2),
    (13, 1e-8, 4),
]

_T_SETTING_ABLATION_GRID: list[tuple[str, bool, bool, bool]] = [
    # (placement, trivial_return, decompose_move, redistribute_stage1_success)
    ("seperate_region_row", True, False, False),  # Vanilla (separate-region)
    ("seperate_region_row", False, True, False),  # + Opt. move (same placement)
    # ("col_based", True, False, False),  # Opt. microarch. (microarchitecture comparison)
    ("col_based", False, True, False),  # + Opt. move
    ("col_based", False, True, True),  # + Opt. patch redist.
]
_T_COMPILE_ABLATION_LABELS: list[str] = [
    "Sync. execution",
    " + Routing opt.",
    " + Microarch. opt.",
    " + Check-stage patch redist.",
]

# _STAR_ABLATION_LABELS = [
#     "Vanilla",
#     "(a) Opt. microarch.",
#     "(b) + Opt. angle assign.",
#     "(c) + Opt. move",
#     "(d) + Opt. skip RUS",
#     "(e) + Async. RUS",
#     # "(c) - angle assign",
#     # "(d) - angle assign",
#     # "(e) - angle assign",
# ]

_STAR_ABLATION_LABELS = [
    "Baseline",
    " + Routing opt.",
    " + Microarch. opt.",
    "Greedy exec.",
    "High parallelism exec.",
    # " + Lookahead angle prep.",  # "(d) + Lookahead angle prep.",
    # "Async. execution",
]


def _star_setting_study_entries() -> list[tuple[int, str]]:
    """(``SETTINGS`` index, legend label) for the STAR setting-study panel."""
    n = min(len(SETTINGS), len(_STAR_ABLATION_LABELS))
    return [(i, _STAR_ABLATION_LABELS[i]) for i in range(n)]


def _t_setting_study_entries() -> list[tuple[int, str]]:
    """(``_T_SETTING_ABLATION_GRID`` index, legend label) for the T setting-study panel."""
    n = min(len(_T_SETTING_ABLATION_GRID), len(_T_COMPILE_ABLATION_LABELS))
    return [(i, _T_COMPILE_ABLATION_LABELS[i]) for i in range(n)]


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
# Grayscale AOD ramp for the AOD panel (no hue/blue; AOD=5 is lighter).
_AOD_DARK_RGB = (0.0, 0.0, 0.0)  # AOD = 1
_AOD_LIGHT_RGB = (0.72, 0.72, 0.72)  # AOD = 5


def _blend_rgb_toward_white(
    rgb: tuple[float, float, float], blend: float
) -> tuple[float, float, float]:
    """Lighten *rgb* by blending toward white (``blend`` in [0, 1])."""
    t = float(np.clip(blend, 0.0, 1.0))
    base = np.array(mcolors.to_rgb(rgb), dtype=float)
    white = np.ones(3, dtype=float)
    return tuple(np.clip((1.0 - t) * base + t * white, 0.0, 1.0))


def _sample_sequential_palette(
    hex_colors: tuple[str, ...] | list[str], n: int
) -> list[tuple[float, float, float]]:
    """Pick *n* evenly spaced colors from a light→dark hex ramp."""
    if n <= 0:
        return []
    ramp = list(hex_colors)
    if n == 1:
        return [mcolors.to_rgb(ramp[len(ramp) // 2])]
    idx = np.linspace(0, len(ramp) - 1, n)
    return [mcolors.to_rgb(ramp[int(round(i))]) for i in idx]


def _setting_study_categorical_colors(
    entries: list[tuple[int, str]],
) -> dict[int, tuple[float, float, float]]:
    """Assign a unique categorical color to each compilation strategy."""
    if len(entries) > len(_SETTING_STUDY_CATEGORICAL_HEX):
        raise ValueError(
            f"Need {len(entries)} distinct setting-study colors but only "
            f"{len(_SETTING_STUDY_CATEGORICAL_HEX)} are defined."
        )
    return {
        setting_idx: mcolors.to_rgb(_SETTING_STUDY_CATEGORICAL_HEX[i])
        for i, (setting_idx, _label) in enumerate(entries)
    }


def _marker_facecolor(
    edge_rgb: tuple[float, float, float],
    *,
    blend: float = _LINE_MARKER_FACE_BLEND,
) -> tuple[float, float, float]:
    return _blend_rgb_toward_white(edge_rgb, blend)


def _plot_styled_errorbar(
    ax,
    x,
    y,
    yerr,
    color,
    *,
    linestyle: str = "-",
    label: str | None = None,
    zorder: int = 2,
):
    """Line + marker with light fill, dark edge, vertical std bars (no caps)."""
    edge = mcolors.to_rgb(color)
    face = _marker_facecolor(edge)
    return ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        marker="o",
        markersize=_LINE_MARKERSIZE,
        markerfacecolor=face,
        markeredgecolor=color,
        markeredgewidth=_LINE_MARKER_EDGEWIDTH,
        capsize=0,
        elinewidth=1.6,
        ecolor=color,
        label=label,
        zorder=zorder,
    )


def _aod_column_legend_handles(
    aod_color_map: dict[int, tuple[float, float, float]],
) -> list[Line2D]:
    """Legend rows: colored marker + AOD number (reference figure style)."""
    handles: list[Line2D] = []
    for aod in sorted(aod_color_map):
        edge = mcolors.to_rgb(aod_color_map[int(aod)])
        face = _marker_facecolor(edge)
        handles.append(
            Line2D(
                [0],
                [0],
                color=edge,
                marker="o",
                markersize=_LINE_MARKERSIZE - 1,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=_LINE_MARKER_EDGEWIDTH,
                linewidth=0,
                label=str(int(aod)),
            )
        )
    return handles


def _build_aod_color_map(plot_aods: list[int] | None = None) -> dict[int, tuple]:
    """Grayscale ramp: AOD=1 is dark, AOD=5 is light."""
    light_rgb = np.array(_AOD_LIGHT_RGB, dtype=float)
    dark_rgb = np.array(_AOD_DARK_RGB, dtype=float)
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
        rgb = (1.0 - t) * dark_rgb + t * light_rgb
        out[int(aod)] = tuple(np.clip(rgb, 0.0, 1.0))
    return out


def _build_aod_color_map_from_base(
    base_rgb: tuple[float, float, float],
    plot_aods: list[int] | None = None,
) -> dict[int, tuple]:
    """Light→dark lightness ramp anchored on the upper-panel setting color.

    AOD = min uses a lighter tint; AOD = max matches *base_rgb* exactly.
    """
    base = np.clip(np.asarray(mcolors.to_rgb(base_rgb), dtype=float), 0.0, 1.0)
    light_rgb = np.array(_blend_rgb_toward_white(tuple(base), 0.58), dtype=float)
    aods = sorted(
        {
            int(a)
            for a in (
                plot_aods or list(range(_AOD_COLORBAR_MIN, _AOD_COLORBAR_MAX + 1))
            )
        }
    )
    if not aods:
        return {}
    out: dict[int, tuple] = {}
    aod_min, aod_max = int(aods[0]), int(aods[-1])
    denom = max(aod_max - aod_min, 1)
    for aod in aods:
        t = (int(aod) - aod_min) / denom
        rgb = (1.0 - t) * light_rgb + t * base
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
    # full_trotter_bounds = [x + zz * 8 for x, zz in zip(x_layer_bounds, zz_layer_bounds)]
    full_trotter_bounds = [
        x * 2 + zz * 4 for x, zz in zip(x_layer_bounds, zz_layer_bounds)
    ]
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
    # Stage-1: success-rate-dependent expected attempts.
    stage_1_parallelization = 4
    if code_distance is not None and int(code_distance) < 13:
        stage_1_parallelization = 2

    effective_stage_1_success_rate = (
        1 - (1 - STAGE_1_SUCCESS_RATE) ** stage_1_parallelization
    )

    # stage1_factor = 12.5 * effective_stage_1_success_rate / (1 - effective_stage_1_success_rate ** 2)

    # Stage-2: depends on LER target used by T-cultivation setting.
    if code_distance is not None and int(code_distance) <= 13:
        expected_attempt = 1 / (effective_stage_1_success_rate * 0.66)
        overhead = 12.5 + 0.5 * effective_stage_1_success_rate
    else:
        expected_attempt = 1 / (effective_stage_1_success_rate * 0.66)
        overhead = 12.5 + 6 * effective_stage_1_success_rate

    rus_factor = 2.0 + 1.0
    s_gate_overhead = 1.0
    expected_time_for_one_resource = (
        (overhead) * expected_attempt + rus_factor + s_gate_overhead
    )
    # print(f"expected_time_for_one_resource: {expected_time_for_one_resource}")
    # input()
    # for x layer bound, we can assume some parallelization and only count half the T gates per qubit
    x_layer_bounds = [expected_time_for_one_resource * n_t_per_qubit for _ in n_values]

    expected_time_for_one_resource = (
        (overhead) * expected_attempt / 2 + rus_factor + s_gate_overhead
    )
    # for zz layer bound, we can assume more parallelization and only count a quarter of the T gates per qubit
    zz_layer_bounds = [expected_time_for_one_resource * n_t_per_qubit for _ in n_values]

    full_trotter_bounds = [
        x * 2 + 4 * zz for x, zz in zip(x_layer_bounds, zz_layer_bounds)
    ]
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
        "stage1_time",
        "stage2_time",
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


def _apply_panel_y_ticks(ax, *, nbins: int = STAR_AOD_COMPARISON_Y_NBINS) -> None:
    """Use more y-axis tick marks on execution-time panels."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=int(nbins), prune="lower"))


def _apply_broken_top_y_ticks(
    ax,
    y_lo: float,
    y_hi: float,
    *,
    nbins: int = STAR_BROKEN_TOP_Y_NBINS,
) -> None:
    """Sparse y ticks on the upper segment of a broken axis."""
    ax.set_ylim(float(y_lo), float(y_hi))
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=int(nbins), prune="both", min_n_ticks=2)
    )


def _apply_broken_bot_y_ticks(
    ax,
    y_hi: float,
    *,
    nbins: int = STAR_AOD_COMPARISON_Y_NBINS,
) -> None:
    """Bottom segment: hide the tick nearest the break to reduce clutter."""
    ax.set_ylim(0.0, float(y_hi))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=int(nbins), prune="upper"))


def _coerce_result_cols_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = RESULT_COLS + [
        "stage1_time",
        "stage2_time",
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
    cols = (
        "placement",
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
    )
    if df.empty or not all(c in df.columns for c in cols):
        return df
    placement, tr, dm, rs = T_CULTIVATION_MAIN_COMPILE_SETTING
    work = df.copy()
    work["placement"] = work["placement"].astype(str).map(_normalize_placement_name)
    mask = (
        work["placement"] == _normalize_placement_name(placement)
    ) & _mask_t_cultivation_compile(work, tr, dm, rs)
    return work.loc[mask].copy()


def _t_setting_ablation_index(setting: tuple[str, bool, bool, bool]) -> int | None:
    """Return index in ``_T_SETTING_ABLATION_GRID`` for *setting*, or None."""
    for idx, entry in enumerate(_T_SETTING_ABLATION_GRID):
        if entry == setting:
            return int(idx)
    return None


def _runtime_profile_t_setting() -> tuple[str, bool, bool, bool]:
    """Setting used for the T-cultivation runtime-profile panel."""
    idx = _t_setting_ablation_index(T_CULTIVATION_MAIN_COMPILE_SETTING)
    if idx is not None:
        return _T_SETTING_ABLATION_GRID[idx]
    placement, tr, dm, rs = T_CULTIVATION_MAIN_COMPILE_SETTING
    return (str(placement).strip(), tr, dm, rs)


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
    aod_legend_handles: list | None = None,
    aod_color_map: dict[int, tuple] | None = None,
    aod_line_legend_handles: list | None = None,
    show_aod_colorbar: bool = True,
    figure_title: str,
    out_path: str,
    xlabel: str = "Number of Qubits",
    column_titles: tuple[str, str] = (
        "Compilation Strategies, AOD = 1",
        "AOD comparison",
    ),
    y_axis_thousands: bool = False,
    legend_handlelength: float = 1.0,
    legend_y_blend: float = 0.5,
    panel_hspace: float = 0.7,
    bottom_axis_y_shift: float = 0.0,
) -> None:
    """Two rows, one column: setting study (top) and AOD sweep (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 11.5), squeeze=False)
    draw_setting(axes[0, 0])
    draw_aod(axes[1, 0])

    # STAR setting-study can optionally split into broken-axis sub-axes.
    setting_axes = getattr(axes[0, 0], "_broken_setting_axes", None)
    if setting_axes:
        ax_setting_top, ax_setting_bot = setting_axes
    else:
        ax_setting_top = ax_setting_bot = axes[0, 0]

    aod_axes = getattr(axes[1, 0], "_broken_aod_axes", None)
    if aod_axes:
        ax_aod_top, ax_aod_bot = aod_axes
    else:
        ax_aod_top = ax_aod_bot = axes[1, 0]

    panel_title_fs = _FIG_FONT_SIZE - 1
    legend_fs = _FIG_FONT_SIZE - 4
    ax_setting_top.set_title(column_titles[0], fontsize=panel_title_fs, pad=8)
    ax_aod_bot.set_title(column_titles[1], fontsize=panel_title_fs, pad=8)
    y_label = "Execution time (×10³)" if y_axis_thousands else "Execution time"
    ax_setting_bot.set_ylabel(y_label, fontsize=_FIG_FONT_SIZE, labelpad=2)
    ax_aod_bot.set_ylabel(y_label, fontsize=_FIG_FONT_SIZE, labelpad=2)
    if y_axis_thousands:
        _apply_y_axis_thousands([ax_setting_bot, ax_aod_bot])
        for _ax in (ax_setting_top, ax_setting_bot, ax_aod_top, ax_aod_bot):
            _apply_panel_y_ticks(_ax, nbins=T_CULTIVATION_Y_NBINS)
    ax_setting_top.tick_params(axis="both", which="major", pad=1)
    ax_setting_bot.tick_params(axis="both", which="major", pad=1)
    ax_aod_top.tick_params(axis="both", which="major", pad=1)
    ax_aod_bot.tick_params(axis="both", which="major", pad=1)
    ax_aod_bot.set_xlabel(xlabel, fontsize=_FIG_FONT_SIZE, labelpad=0)

    fig.suptitle(figure_title, fontsize=_FIG_FONT_SIZE + 1, y=0.98)
    has_aod_bottom_legend = bool(
        getattr(axes[1, 0], "_aod_bottom_legend_rows", None)
    )
    bottom_margin = 0.16 if has_aod_bottom_legend else 0.08
    fig.tight_layout(rect=(0.08, bottom_margin, 0.90, 0.94), pad=0.10, h_pad=0.35)
    fig.subplots_adjust(
        top=0.92, bottom=bottom_margin, left=0.12, right=0.88, hspace=panel_hspace
    )
    if bottom_axis_y_shift:
        bot_pos = ax_aod_bot.get_position()
        ax_aod_bot.set_position(
            [
                bot_pos.x0,
                bot_pos.y0 - float(bottom_axis_y_shift),
                bot_pos.width,
                bot_pos.height,
            ]
        )
        if aod_axes:
            top_pos_aod = ax_aod_top.get_position()
            ax_aod_top.set_position(
                [
                    top_pos_aod.x0,
                    top_pos_aod.y0 - float(bottom_axis_y_shift) * 0.35,
                    top_pos_aod.width,
                    top_pos_aod.height,
                ]
            )

    combined_legend = [*setting_legend_handles, _expected_time_legend_handle()]
    top_pos = axes[0, 0].get_position()
    bot_pos = ax_aod_bot.get_position()
    blend = float(np.clip(legend_y_blend, 0.0, 1.0))
    legend_y = (1.0 - blend) * top_pos.y0 + blend * bot_pos.y1
    if figure_title == "T-cultivation":
        legend_y -= 0.0
    else:
        legend_y += 0.06
    fig.legend(
        handles=combined_legend,
        loc="center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        fontsize=legend_fs,
        frameon=True,
        handlelength=legend_handlelength,
        handletextpad=0.35,
        labelspacing=0.35,
        columnspacing=0.6,
    )

    # Separate AOD legend in the upper-left of the setting-study panel.
    if aod_legend_handles:
        # Anchor AOD legend visually at the upper-left of the top panel, but as
        # a figure-level legend so it is not clipped by the lower axes.
        aod_legend_x = top_pos.x0
        # Slightly below the top of the setting panel to avoid the title text.
        aod_legend_y = top_pos.y1 - 0.02
        fig.legend(
            handles=aod_legend_handles,
            loc="upper left",
            bbox_to_anchor=(aod_legend_x, aod_legend_y),
            fontsize=legend_fs,
            frameon=True,
            handlelength=legend_handlelength,
            handletextpad=0.35,
            labelspacing=0.35,
        )

    _apply_aod_bottom_legend(fig, axes[1, 0], legend_fs=legend_fs)

    if aod_line_legend_handles:
        axes[1, 0].legend(
            handles=aod_line_legend_handles,
            title="AOD",
            loc="upper right",
            fontsize=legend_fs,
            title_fontsize=legend_fs,
            frameon=True,
            handlelength=1.2,
            handletextpad=0.35,
            labelspacing=0.35,
            borderpad=0.35,
        )
    elif show_aod_colorbar and aod_color_map is not None:
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
            ax=axes[1, 0],
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


def _trim_extreme_total_time_samples(times: pd.Series) -> pd.Series:
    """Drop one min and one max sample, then summarize the remainder."""
    vals = pd.to_numeric(times, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return pd.Series(
            {
                "total_time_mean": np.nan,
                "total_time_std": np.nan,
                "total_time_min": np.nan,
                "total_time_max": np.nan,
            }
        )
    if vals.size <= 2:
        return pd.Series(
            {
                "total_time_mean": float(np.mean(vals)),
                "total_time_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "total_time_min": float(np.min(vals)),
                "total_time_max": float(np.max(vals)),
            }
        )
    idx_min = int(np.argmin(vals))
    idx_max = int(np.argmax(vals))
    mask = np.ones(vals.size, dtype=bool)
    mask[idx_min] = False
    if idx_max != idx_min:
        mask[idx_max] = False
    trimmed = vals[mask]
    if trimmed.size == 0:
        trimmed = vals
    return pd.Series(
        {
            "total_time_mean": float(np.mean(trimmed)),
            "total_time_std": (
                float(np.std(trimmed, ddof=1)) if trimmed.size > 1 else 0.0
            ),
            "total_time_min": float(np.min(trimmed)),
            "total_time_max": float(np.max(trimmed)),
        }
    )


def _aggregate_total_time_by_qubits(
    df: pd.DataFrame, *, trim_extremes: bool = False
) -> pd.DataFrame:
    """Per *n_qubits*: mean/min/max of ``total_time`` (optional min/max sample drop)."""
    if trim_extremes:
        grouped = (
            df.groupby("n_qubits", as_index=False)[["total_time"]]
            .apply(lambda g: _trim_extreme_total_time_samples(g["total_time"]))
            .reset_index(drop=True)
        )
    else:
        grouped = (
            df.groupby("n_qubits")[["total_time"]]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        grouped.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in grouped.columns
        ]
        grouped["total_time_std"] = grouped["total_time_std"].fillna(0.0)
    return grouped


def _aggregate_star_setting_study(
    dfs_dict_micro: dict[str, pd.DataFrame],
    setting: tuple,
    code_distance: int,
    aod: int,
    *,
    trim_extremes: bool = False,
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
    grouped = _aggregate_total_time_by_qubits(df, trim_extremes=trim_extremes)
    return grouped if not grouped.empty else None


def _aggregate_t_setting_study(
    layer_df: pd.DataFrame,
    triple: tuple[int, float, int],
    setting: tuple[str, bool, bool, bool],
    code_distance: int,
    aod: int,
    *,
    trim_extremes: bool = False,
) -> pd.DataFrame | None:
    work = layer_df.copy()
    work["placement"] = work["placement"].astype(str).map(_normalize_placement_name)
    placement, tr, dm, rs = setting
    work = work[work["placement"] == _normalize_placement_name(placement)].copy()
    cd_t, ft, fps = triple
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
    grouped = _aggregate_total_time_by_qubits(work, trim_extremes=trim_extremes)
    return grouped if not grouped.empty else None


def _agg_y_extent(grouped: pd.DataFrame) -> tuple[float, float]:
    """Mean ± std extent for y-axis limits."""
    sub = grouped.sort_values("n_qubits")
    mean_vals = pd.to_numeric(sub["total_time_mean"], errors="coerce")
    std_vals = pd.to_numeric(
        sub.get("total_time_std", pd.Series(0.0, index=sub.index)),
        errors="coerce",
    ).fillna(0.0)
    y_lo = float((mean_vals - std_vals).min())
    y_hi = float((mean_vals + std_vals).max())
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
    d: float = 0.009,
    lw: float = 1.0,
) -> None:
    """Diagonal break marks between a top (high-y) and bottom (zoomed) panel."""
    kwargs = {
        "color": "0.25",
        "clip_on": False,
        "linewidth": lw,
        "solid_capstyle": "round",
        "zorder": 12,
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


def _grouped_aod_y_extent(grouped: pd.DataFrame | None) -> tuple[float, float]:
    """Mean ± std y-range for one AOD sweep dataframe."""
    if grouped is None or grouped.empty:
        return 0.0, 0.0
    mean_vals = pd.to_numeric(grouped["total_time_mean"], errors="coerce")
    std_vals = pd.to_numeric(
        grouped.get("total_time_std", pd.Series(0.0, index=grouped.index)),
        errors="coerce",
    ).fillna(0.0)
    return float((mean_vals - std_vals).min()), float((mean_vals + std_vals).max())


def _star_aod_broken_axis_needed(
    sync_extent: tuple[float, float],
    best_extent: tuple[float, float],
    *,
    gap_ratio: float = STAR_SETTING_STUDY_Y_CROP_GAP_RATIO,
) -> bool:
    sync_lo, sync_hi = sync_extent
    best_lo, best_hi = best_extent
    if sync_hi <= 0.0 or best_hi <= 0.0:
        return False
    if sync_hi <= best_hi * float(gap_ratio):
        return False
    return (sync_hi - sync_lo) > (best_hi - best_lo) * 0.15


def _compute_star_aod_break_y(
    sync_extent: tuple[float, float],
    best_extent: tuple[float, float],
    *,
    pad_frac: float = STAR_SETTING_STUDY_Y_CROP_PAD_FRAC,
) -> float:
    sync_lo, sync_hi = sync_extent
    _best_lo, best_hi = best_extent
    break_y = max(best_hi * (1.0 + pad_frac), sync_lo * (1.0 + 0.5 * pad_frac))
    break_y = min(break_y, sync_hi * (1.0 - 0.04))
    if break_y <= sync_lo:
        break_y = (sync_lo + sync_hi) / 2.0
    return float(break_y)


def _create_broken_y_axes(
    ax_parent,
    *,
    height_ratios: tuple[int, int] = (1, 4),
    hspace: float = 0.06,
    attr_name: str = "_broken_setting_axes",
):
    """Split a panel into top (high-y) and bottom (zoomed) sub-axes."""
    fig = ax_parent.figure
    ax_parent.set_visible(False)
    gs = ax_parent.get_subplotspec().subgridspec(
        2,
        1,
        height_ratios=list(height_ratios),
        hspace=float(hspace),
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
    setattr(ax_parent, attr_name, (ax_top, ax_bot))
    return ax_top, ax_bot


def _create_star_aod_broken_axes(ax_parent):
    """Broken AOD panel with a wide gap and a compact upper segment."""
    return _create_broken_y_axes(
        ax_parent,
        height_ratios=STAR_BROKEN_AXIS_HEIGHT_RATIOS,
        hspace=STAR_BROKEN_INNER_HSPACE,
        attr_name="_broken_aod_axes",
    )


def _create_star_setting_broken_axes(ax_parent):
    """Split the setting-study cell into top (Vanilla) and bottom (other settings) axes."""
    return _create_broken_y_axes(ax_parent, attr_name="_broken_setting_axes")


def _mean_total_time_from_grouped(grouped: pd.DataFrame | None) -> float | None:
    if grouped is None or grouped.empty:
        return None
    vals = pd.to_numeric(grouped["total_time_mean"], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _pick_best_star_setting_index(
    dfs_dict_micro: dict[str, pd.DataFrame],
    code_distance: int,
    entries: list[tuple[int, str]],
    *,
    aods: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> int:
    """Return ``SETTINGS`` index with lowest mean execution time (col_based AOD sweep)."""
    best_idx: int | None = None
    best_mean = math.inf
    for setting_idx, _label in entries:
        if setting_idx >= len(SETTINGS):
            continue
        setting = SETTINGS[setting_idx]
        if str(setting[0]).strip() != "col_based":
            continue
        sample_means: list[float] = []
        for aod in aods:
            grouped = _aggregate_star_setting_study(
                dfs_dict_micro, setting, code_distance, int(aod)
            )
            m = _mean_total_time_from_grouped(grouped)
            if m is not None:
                sample_means.append(m)
        if not sample_means:
            continue
        avg = float(np.mean(sample_means))
        if avg < best_mean:
            best_mean = avg
            best_idx = int(setting_idx)
    if best_idx is None:
        return int(STAR_T_GRID_STAR_SETTING_INDEX)
    return best_idx


def _pick_best_t_setting_key(
    t_setting_layer: pd.DataFrame,
    triple: tuple[int, float, int] | None,
    code_distance: int,
    entries: list[tuple[int, str]],
    *,
    aods: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> int:
    """Return `_T_SETTING_ABLATION_GRID` index with lowest mean execution time for AOD panel."""
    if triple is None or t_setting_layer.empty:
        return int(RUNTIME_PROFILE_T_SETTING_IDX)

    best_idx: int | None = None
    best_mean = math.inf
    for setting_idx, _label in entries:
        if setting_idx >= len(_T_SETTING_ABLATION_GRID):
            continue
        placement, _tr, _dm, _rs = _T_SETTING_ABLATION_GRID[setting_idx]
        if str(placement).strip() != "col_based":
            continue
        sample_means: list[float] = []
        for aod in aods:
            grouped = _aggregate_t_setting_study(
                t_setting_layer,
                triple,
                _T_SETTING_ABLATION_GRID[setting_idx],
                code_distance,
                int(aod),
            )
            m = _mean_total_time_from_grouped(grouped)
            if m is not None:
                sample_means.append(m)
        if not sample_means:
            continue
        avg = float(np.mean(sample_means))
        if avg < best_mean:
            best_mean = avg
            best_idx = int(setting_idx)
    if best_idx is None:
        return int(RUNTIME_PROFILE_T_SETTING_IDX)
    return best_idx


def _build_t_aod_base_df(
    t_aod_layer: pd.DataFrame,
    triple: tuple[int, float, int],
    code_distance: int,
    compile_tuple: tuple[bool, bool, bool],
) -> pd.DataFrame:
    """Col-based T-cultivation rows for one compile tuple (all AODs, one distance)."""
    cd_t, ft, fps = triple
    tr, dm, rs = compile_tuple
    work = t_aod_layer.copy()
    if "placement" in work.columns:
        work = _filter_col_based(work)
    t_base = work.loc[
        _mask_t_cultivation_compile(work, tr, dm, rs)
        & _mask_t_cultivation_triple(work, cd_t, ft, fps)
    ].copy()
    if t_base.empty:
        return t_base
    return t_base[
        pd.to_numeric(t_base["code_distance"], errors="coerce").astype(int)
        == int(code_distance)
    ].copy()


def _t_aod_panel_base_rgb(
    t_setting_colors: dict[int, tuple],
) -> tuple[float, float, float]:
    """Match AOD = 1 to the compile/placement setting used in the T AOD sweep panel."""
    main_idx = _t_setting_ablation_index(T_CULTIVATION_MAIN_COMPILE_SETTING)
    if main_idx is not None and main_idx in t_setting_colors:
        return mcolors.to_rgb(t_setting_colors[main_idx])
    placement_m, tr_m, dm_m, rs_m = T_CULTIVATION_MAIN_COMPILE_SETTING
    for setting_idx, _label in _t_setting_study_entries():
        if setting_idx >= len(_T_SETTING_ABLATION_GRID):
            continue
        placement, tr, dm, rs = _T_SETTING_ABLATION_GRID[setting_idx]
        if str(placement).strip() != str(placement_m).strip():
            continue
        if (tr, dm, rs) == (tr_m, dm_m, rs_m):
            if setting_idx in t_setting_colors:
                return mcolors.to_rgb(t_setting_colors[setting_idx])
    # Fallback: last entry's color (typically best col_based setting).
    if t_setting_colors:
        return mcolors.to_rgb(t_setting_colors[max(t_setting_colors.keys())])
    return (0.2, 0.2, 0.2)


def _draw_setting_study_grouped_bars(
    ax,
    series: list[tuple],
    *,
    round_name: str = "full_trotter",
) -> list[int]:
    """Grouped bars for the compilation-strategy panel.

    Each *series* entry is ``(color, legend_label, grouped_df)`` with columns
    ``n_qubits``, ``total_time_mean``, ``total_time_min``, ``total_time_max``.
    """
    if not series:
        return []
    x_vals = sorted(
        {
            int(nq)
            for _color, _label, grouped in series
            for nq in grouped["n_qubits"].dropna().astype(int).unique()
        }
    )
    if not x_vals:
        return []
    # Use categorical positions for uniform spacing.
    x_centers = np.arange(len(x_vals), dtype=float)
    n_series = len(series)
    group_width = 0.72
    bar_width = group_width / max(n_series, 1)

    for series_idx, (color, label, grouped) in enumerate(series):
        sub = grouped.sort_values("n_qubits")
        by_q = sub.set_index("n_qubits")
        offset = (series_idx - (n_series - 1) / 2.0) * bar_width
        positions = x_centers + offset
        heights: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []
        for nq in x_vals:
            if int(nq) in by_q.index:
                row = by_q.loc[int(nq)]
                mean = float(row["total_time_mean"])
                std = float(row.get("total_time_std", 0.0) or 0.0)
                heights.append(mean)
                yerr_lo.append(max(0.0, std))
                yerr_hi.append(max(0.0, std))
            else:
                heights.append(0.0)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
        ax.bar(
            positions,
            heights,
            width=bar_width,
            color=color,
            label=label,
            alpha=0.92,
            edgecolor="black",
            linewidth=0.5,
            yerr=[yerr_lo, yerr_hi],
            capsize=0,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        _format_nqubit_ticklabels(x_vals, round_name, None),
        rotation=0,
    )
    ax.set_xlim(-0.65, float(len(x_centers) - 1) + 0.65)
    return x_vals


def _draw_setting_study_dual_aod_bars(
    ax,
    series: list[tuple],
    *,
    compare_aods: tuple[int, ...] = SETTING_STUDY_DUAL_AODS,
    round_name: str = "full_trotter",
    front_blend: float = SETTING_STUDY_DUAL_AOD_FRONT_BLEND,
) -> list[int]:
    """Grouped bars with two overlapping bars per setting (lower AOD in front, lighter).

    Each *series* entry is
    ``(color, legend_label, grouped_back, grouped_front)`` for *compare_aods[0]*
    and *compare_aods[1]*.
    """
    if not series or len(compare_aods) < 2:
        return []
    aod_back, aod_front = int(compare_aods[0]), int(compare_aods[1])

    x_vals = sorted(
        {
            int(nq)
            for _color, _label, g_back, _g_front in series
            for g in (g_back, _g_front)
            if g is not None and not g.empty
            for nq in g["n_qubits"].dropna().astype(int).unique()
        }
    )
    if not x_vals:
        return []
    # Use categorical positions for uniform spacing.
    x_centers = np.arange(len(x_vals), dtype=float)
    n_series = len(series)
    group_width = 0.72
    bar_width = group_width / max(n_series, 1)
    front_width = bar_width * 0.88

    def _bar_heights_and_err(
        grouped: pd.DataFrame | None,
    ) -> tuple[list[float], list[float], list[float]]:
        if grouped is None or grouped.empty:
            return (
                [0.0] * len(x_vals),
                [0.0] * len(x_vals),
                [0.0] * len(x_vals),
            )
        by_q = grouped.sort_values("n_qubits").set_index("n_qubits")
        heights: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []
        for nq in x_vals:
            if int(nq) in by_q.index:
                row = by_q.loc[int(nq)]
                mean = float(row["total_time_mean"])
                std = float(row.get("total_time_std", 0.0) or 0.0)
                heights.append(mean)
                yerr_lo.append(max(0.0, std))
                yerr_hi.append(max(0.0, std))
            else:
                heights.append(0.0)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
        return heights, yerr_lo, yerr_hi

    for series_idx, (color, label, grouped_back, grouped_front) in enumerate(series):
        offset = (series_idx - (n_series - 1) / 2.0) * bar_width
        positions = x_centers + offset
        h_back, err_lo, err_hi = _bar_heights_and_err(grouped_back)
        h_front, _, _ = _bar_heights_and_err(grouped_front)
        base_rgb = mcolors.to_rgb(color)
        light_rgb = _blend_rgb_toward_white(base_rgb, front_blend)

        ax.bar(
            positions,
            h_back,
            width=bar_width,
            color=base_rgb,
            label=None,
            alpha=0.92,
            edgecolor="black",
            linewidth=0.5,
            yerr=[err_lo, err_hi],
            capsize=0,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
            zorder=1,
        )
        ax.bar(
            positions,
            h_front,
            width=front_width,
            color=light_rgb,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.45,
            zorder=2,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        _format_nqubit_ticklabels(x_vals, round_name, None),
        rotation=0,
    )
    ax.set_xlim(-0.65, float(len(x_centers) - 1) + 0.65)
    return x_vals


def _dual_aod_bar_legend_handles(
    sample_color,
    *,
    compare_aods: tuple[int, ...] = SETTING_STUDY_DUAL_AODS,
    front_blend: float = SETTING_STUDY_DUAL_AOD_FRONT_BLEND,
) -> list:
    return [
        Patch(
            facecolor=_AOD_DARK_RGB,
            edgecolor="black",
            label=f"AOD = {int(compare_aods[0])}",
        ),
        Patch(
            facecolor=_AOD_LIGHT_RGB,
            edgecolor="black",
            label=f"AOD = {int(compare_aods[1])}",
        ),
    ]


def _setting_study_chart_suffix(
    setting_chart: str, *, bar_trim_extremes: bool = False
) -> str:
    if setting_chart == "bar_dual_aod":
        return "_bars_aod1_aod5"
    return "_bars_trim_extremes" if bar_trim_extremes else "_bars"


def _setting_study_legend_handle(*, color, label: str, setting_chart: str):
    if setting_chart in {"bar", "bar_dual_aod"}:
        return Patch(facecolor=color, edgecolor="black", label=label)
    raise ValueError(f"Unsupported setting_chart for legend: {setting_chart}")


def _validate_t_profiling_workframe(
    df: pd.DataFrame, *, context: str = ""
) -> pd.DataFrame:
    """Require measured stage/CNOT profiling columns from the evaluation CSV."""
    suffix = f" ({context})" if context else ""
    if df.empty:
        raise ValueError(f"T-cultivation profiling data is empty{suffix}.")

    _validate_t_profiling_csv_columns(df)

    work = df.copy()
    for col in _T_PROFILING_REQUIRED_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    invalid_mask = work[list(_T_PROFILING_REQUIRED_COLS)].isna().any(axis=1)
    if invalid_mask.any():
        raise ValueError(
            "T-cultivation profiling CSV has non-numeric values in required columns"
            f"{suffix} ({int(invalid_mask.sum())} row(s))."
        )

    work = work.fillna(0.0)
    if (work["stage1_time"] + work["stage2_time"] <= 0.0).all():
        raise ValueError(
            "T-cultivation profiling CSV has no stage-1/stage-2 time recorded"
            f"{suffix}. "
            "Re-run script/evaluation_fidelity_t_cultivation.py."
        )

    return work


def _validate_t_profiling_csv_columns(
    df: pd.DataFrame, csv_path: str | None = None
) -> None:
    """Fail fast if the T-cultivation profiling CSV lacks required columns."""
    missing = [c for c in _T_PROFILING_REQUIRED_COLS if c not in df.columns]
    if not missing:
        return
    path_hint = f" ({csv_path})" if csv_path else ""
    raise ValueError(
        "T-cultivation profiling CSV is missing required columns"
        f"{path_hint}: {missing}. "
        "Re-run script/evaluation_fidelity_t_cultivation.py to regenerate "
        "t_cultivation_fidelity_profiling_results.csv."
    )


def _runtime_profile_stack_sum(
    grouped: pd.DataFrame, component_cols: list[str]
) -> np.ndarray:
    out = np.zeros(len(grouped), dtype=float)
    for col in component_cols:
        out += grouped[col].to_numpy(dtype=float)
    return out


def _assert_runtime_profile_stack_matches_total(
    grouped: pd.DataFrame,
    component_cols: list[str],
    *,
    context: str = "",
    total_col: str = "total_mean",
) -> None:
    """Raise when stacked bar components do not sum to mean wall-clock ``total_time``."""
    stack = _runtime_profile_stack_sum(grouped, component_cols)
    total = grouped[total_col].to_numpy(dtype=float)
    if np.allclose(
        stack, total, rtol=_RUNTIME_PROFILE_STACK_RTOL, atol=_RUNTIME_PROFILE_STACK_ATOL
    ):
        return
    diff = stack - total
    bad = ~np.isclose(
        stack, total, rtol=_RUNTIME_PROFILE_STACK_RTOL, atol=_RUNTIME_PROFILE_STACK_ATOL
    )
    if not bad.any():
        return
    details = "; ".join(
        f"n={int(grouped.loc[idx, 'n_qubits'])}: stack={stack[idx]:.1f} "
        f"vs total={grouped.loc[idx, total_col]:.1f} (diff={diff[idx]:+.1f})"
        for idx in grouped.index[bad]
    )
    suffix = f" ({context})" if context else ""
    raise ValueError(
        "Runtime profile stack does not match total execution time"
        f"{suffix}: {details}"
    )


def _finalize_runtime_profile_grouped(
    grouped: pd.DataFrame,
    components: list[tuple[str, str, str]],
    *,
    context: str = "",
) -> pd.DataFrame:
    component_cols = [col for col, _label, _color in components]
    _assert_runtime_profile_stack_matches_total(
        grouped, component_cols, context=context
    )
    return grouped


def _aggregate_star_runtime_profile_by_qubits(
    df: pd.DataFrame, *, context: str = ""
) -> pd.DataFrame | None:
    """Mean forward/return move, TMR, and RUS time per ``n_qubits`` (STAR panel)."""
    if df.empty or "n_qubits" not in df.columns:
        return None
    work = df.copy()
    work["forward_move"] = pd.to_numeric(
        work.get("movement_time"), errors="coerce"
    ).fillna(0.0)
    work["return_move"] = pd.to_numeric(
        work.get("return_movement_time"), errors="coerce"
    ).fillna(0.0)
    work["tmr_time"] = 6 * pd.to_numeric(work.get("TMR_round"), errors="coerce").fillna(
        0.0
    )
    work["rus_time"] = pd.to_numeric(work.get("RUS_round"), errors="coerce").fillna(0.0)
    grouped = (
        work.groupby("n_qubits", as_index=False)
        .agg(
            forward_move_mean=("forward_move", "mean"),
            return_move_mean=("return_move", "mean"),
            tmr_mean=("tmr_time", "mean"),
            rus_mean=("rus_time", "mean"),
            total_mean=("total_time", "mean"),
        )
        .sort_values("n_qubits")
    )
    if grouped.empty:
        return None
    star_ctx = context or "STAR runtime profile"
    return _finalize_runtime_profile_grouped(
        grouped, _RUNTIME_PROFILE_STAR_COMPONENTS, context=star_ctx
    )


def _aggregate_t_runtime_profile_by_qubits(
    df: pd.DataFrame, *, context: str = ""
) -> pd.DataFrame | None:
    """Mean movement, stage-1/2, CNOT time, and ``n_cnot`` per ``n_qubits`` (T panel)."""
    if "n_qubits" not in df.columns:
        raise ValueError(
            "T-cultivation profiling data has no n_qubits column"
            f"{f' ({context})' if context else ''}."
        )
    work = _validate_t_profiling_workframe(df, context=context)
    work["forward_move"] = pd.to_numeric(
        work.get("movement_time"), errors="coerce"
    ).fillna(0.0)
    work["return_move"] = pd.to_numeric(
        work.get("return_movement_time"), errors="coerce"
    ).fillna(0.0)
    grouped = (
        work.groupby("n_qubits", as_index=False)
        .agg(
            forward_move_mean=("forward_move", "mean"),
            return_move_mean=("return_move", "mean"),
            stage1_mean=("stage1_time", "mean"),
            stage2_mean=("stage2_time", "mean"),
            rus_mean=("RUS_round", "mean"),
            total_mean=("total_time", "mean"),
        )
        .sort_values("n_qubits")
    )
    if grouped.empty:
        return None
    t_ctx = context or "T-cultivation runtime profile"
    return _finalize_runtime_profile_grouped(
        grouped, _RUNTIME_PROFILE_T_COMPONENTS, context=t_ctx
    )


def _star_runtime_profile_frame(
    dfs_dict_micro: dict[str, pd.DataFrame],
    code_distance: int,
    aod: int,
    *,
    setting: tuple = RUNTIME_PROFILE_STAR_SETTING,
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
        raise ValueError(
            "No STAR profiling rows for runtime profile setting "
            f"{setting} (code_distance={code_distance}, AOD={aod})."
        )
    if "code_distance" in df.columns:
        df = df[
            pd.to_numeric(df["code_distance"], errors="coerce").astype(int)
            == int(code_distance)
        ].copy()
    if df.empty:
        raise ValueError(
            f"No STAR profiling rows at code_distance={code_distance} for setting "
            f"{setting}, AOD={aod}."
        )
    df["n_aods"] = pd.to_numeric(df["n_aods"], errors="coerce").astype(int)
    df = df[df["n_aods"] == int(aod)].copy()
    if df.empty:
        raise ValueError(
            f"No STAR profiling rows at AOD={aod} for setting {setting}, "
            f"code_distance={code_distance}."
        )
    ctx = f"STAR setting={setting}, code_distance={code_distance}, AOD={aod}"
    return _aggregate_star_runtime_profile_by_qubits(df, context=ctx)


def _t_runtime_profile_frame(
    layer_df: pd.DataFrame,
    triple: tuple[int, float, int],
    setting: tuple[str, bool, bool, bool],
    code_distance: int,
    aod: int,
) -> pd.DataFrame | None:
    work = layer_df.copy()
    if work.empty or "placement" not in work.columns:
        return None
    placement, tr, dm, rs = setting
    work["placement"] = work["placement"].astype(str).map(_normalize_placement_name)
    work = work[work["placement"] == _normalize_placement_name(placement)].copy()
    if work.empty:
        return None
    cd_t, ft, fps = triple
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
    work["n_aods"] = pd.to_numeric(work["n_aods"], errors="coerce").astype(int)
    work = work[work["n_aods"] == int(aod)].copy()
    if work.empty:
        return None
    ctx = f"code_distance={code_distance}, AOD={aod}, " f"setting={setting}"
    return _aggregate_t_runtime_profile_by_qubits(work, context=ctx)


def _profile_values_for_x(
    profile: pd.DataFrame, x_vals: list[int], col: str
) -> np.ndarray:
    by_qubit = profile.set_index("n_qubits")
    out = np.zeros(len(x_vals), dtype=float)
    for i, nq in enumerate(x_vals):
        if int(nq) in by_qubit.index:
            out[i] = float(by_qubit.loc[int(nq), col])
    return out


def _draw_runtime_profile_grouped_aod_bars(
    ax,
    profiles_by_aod: dict[int, pd.DataFrame | None],
    components: list[tuple[str, str, str]],
    *,
    aods: tuple[int, ...] = RUNTIME_PROFILE_AODS,
    y_axis_thousands: bool = False,
) -> None:
    """Grouped stacked bars: AOD = 1 (solid) and AOD = 5 (diagonal hatch) per qubit count."""
    available = {
        int(aod): prof
        for aod in aods
        if (prof := profiles_by_aod.get(int(aod))) is not None and not prof.empty
    }
    if not available:
        ax.axis("off")
        return

    x_vals = sorted(
        {
            int(nq)
            for prof in available.values()
            for nq in prof["n_qubits"].dropna().astype(int).unique()
        }
    )
    x_centers = np.arange(len(x_vals), dtype=float)
    n_aod = len(available)
    bar_width = 0.8 / max(n_aod, 1)
    aod_list = [int(a) for a in aods if int(a) in available]

    for aod_idx, aod in enumerate(aod_list):
        prof = available[aod]
        offset = (aod_idx - (n_aod - 1) / 2.0) * bar_width
        positions = x_centers + offset
        bottom = np.zeros(len(x_vals), dtype=float)
        hatch = RUNTIME_PROFILE_AOD_HATCH.get(int(aod))
        label_components = aod_idx == 0
        for col, label, color in components:
            values = _profile_values_for_x(prof, x_vals, col)
            ax.bar(
                positions,
                values,
                width=bar_width,
                bottom=bottom,
                label=label if label_components else None,
                color=color,
                alpha=0.92,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.4,
            )
            bottom += values

    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(int(v)) for v in x_vals], rotation=0)
    ax.set_ylim(bottom=0.0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    if y_axis_thousands:
        _apply_y_axis_thousands([ax])


def _runtime_profile_aod_legend_handles() -> list:
    handles: list = []
    for aod in RUNTIME_PROFILE_AODS:
        hatch = RUNTIME_PROFILE_AOD_HATCH.get(int(aod))
        handles.append(
            Patch(
                facecolor="0.75",
                edgecolor="0.35",
                hatch=hatch,
                label=f"AOD = {int(aod)}",
            )
        )
    return handles


def _plot_star_t_runtime_profile_figure(
    dfs_dict_micro: dict[str, pd.DataFrame],
    t_layers_ablation: dict[str, pd.DataFrame],
    output_dir: str,
    *,
    code_distance: int = STAR_T_GRID_COMPARISON_CODE_DISTANCE,
    aods: tuple[int, ...] = RUNTIME_PROFILE_AODS,
    t_setting_idx: int | None = None,
    verbose: bool = True,
) -> None:
    """Stacked runtime bars: STAR (top) vs T-cultivation (bottom), AOD 1 vs 5 grouped."""

    def _has_profile(profiles: dict[int, pd.DataFrame | None]) -> bool:
        return any(prof is not None and not prof.empty for prof in profiles.values())

    os.makedirs(output_dir, exist_ok=True)
    cd = int(code_distance)
    round_name = "full_trotter"
    star_profiles = {
        int(aod): _star_runtime_profile_frame(
            dfs_dict_micro, cd, int(aod), setting=RUNTIME_PROFILE_STAR_SETTING
        )
        for aod in aods
    }
    triple = _primary_t_triple_for_star_t_grid(cd)
    t_profiles: dict[int, pd.DataFrame | None] = {int(aod): None for aod in aods}
    t_profile_setting = _runtime_profile_t_setting()
    if t_setting_idx is not None and 0 <= int(t_setting_idx) < len(
        _T_SETTING_ABLATION_GRID
    ):
        t_profile_setting = _T_SETTING_ABLATION_GRID[int(t_setting_idx)]
    if triple is not None and round_name in t_layers_ablation:
        for aod in aods:
            t_profiles[int(aod)] = _t_runtime_profile_frame(
                t_layers_ablation[round_name],
                triple,
                t_profile_setting,
                cd,
                int(aod),
            )
        if verbose and not _has_profile(t_profiles):
            print(
                "Runtime profile: no T-cultivation rows for "
                f"setting={t_profile_setting}, d={cd}, AOD in {aods}."
            )

    if not _has_profile(star_profiles) and not _has_profile(t_profiles):
        if verbose:
            print(f"Skipping runtime profile: no data at d={cd}, AOD in {aods}.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 11.5), squeeze=False)
    legend_fs = _FIG_FONT_SIZE - 4
    panel_title_fs = _FIG_FONT_SIZE - 1

    if _has_profile(star_profiles):
        _draw_runtime_profile_grouped_aod_bars(
            axes[0, 0], star_profiles, _RUNTIME_PROFILE_STAR_COMPONENTS, aods=aods
        )
        axes[0, 0].set_title("STAR", fontsize=panel_title_fs, pad=8)
    else:
        axes[0, 0].set_title("STAR\n(no data)", fontsize=panel_title_fs)
        axes[0, 0].axis("off")

    if _has_profile(t_profiles):
        _draw_runtime_profile_grouped_aod_bars(
            axes[1, 0],
            t_profiles,
            _RUNTIME_PROFILE_T_COMPONENTS,
            aods=aods,
            y_axis_thousands=True,
        )
        axes[1, 0].set_title("T-cultivation", fontsize=panel_title_fs, pad=8)
    else:
        axes[1, 0].set_title("T-cultivation\n(no data)", fontsize=panel_title_fs)
        axes[1, 0].axis("off")

    y_label_star = "Execution time"
    y_label_t = "Execution time (×10³)"
    axes[0, 0].set_ylabel(y_label_star, fontsize=_FIG_FONT_SIZE, labelpad=2)
    axes[1, 0].set_ylabel(y_label_t, fontsize=_FIG_FONT_SIZE, labelpad=2)
    for row in range(2):
        axes[row, 0].tick_params(axis="both", which="major", pad=1)
    axes[1, 0].set_xlabel(
        "Number of Qubits/Factories", fontsize=_FIG_FONT_SIZE, labelpad=0
    )

    fig.suptitle(
        "Runtime profile",
        fontsize=_FIG_FONT_SIZE + 1,
        y=0.98,
    )
    fig.tight_layout(rect=(0.08, 0.06, 0.72, 0.94), pad=0.10, h_pad=0.35)
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.14, right=0.72, hspace=0.28)
    seen_labels: set[str] = set()
    legend_handles: list = []
    for _components in (
        _RUNTIME_PROFILE_STAR_COMPONENTS,
        _RUNTIME_PROFILE_T_COMPONENTS,
    ):
        for _col, label, color in _components:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=4, label=label)
            )
    legend_handles.extend(_runtime_profile_aod_legend_handles())
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.72, 0.5),
        fontsize=legend_fs,
        frameon=True,
        handlelength=1.0,
        handleheight=0.65,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.45,
    )

    out_path = os.path.join(
        output_dir, f"runtime_profile_star_vs_t_cultivation_d{cd}.pdf"
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    if verbose:
        print(f"Saved: {out_path}")


def _star_grouped_aod_at_distance(
    dfs_dict_aod: dict[str, pd.DataFrame],
    setting: tuple,
    code_distance: int,
) -> pd.DataFrame | None:
    """Per (n_aods, n_qubits) mean/std for one compile setting at *code_distance*."""
    round_name = "full_trotter"
    if round_name not in dfs_dict_aod:
        return None
    df = (
        dfs_dict_aod[round_name]
        .loc[_setting_filter(dfs_dict_aod[round_name], setting)]
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
    grouped = (
        df.groupby(["n_aods", "n_qubits"])[["total_time"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    grouped["total_time_std"] = grouped["total_time_std"].fillna(0.0)
    return grouped if not grouped.empty else None


def _t_grouped_aod_at_distance(
    layer_df: pd.DataFrame,
    triple: tuple[int, float, int],
    setting: tuple[str, bool, bool, bool],
    code_distance: int,
) -> pd.DataFrame | None:
    """Per (n_aods, n_qubits) mean/std for one T compile setting at *code_distance*."""
    work = layer_df.copy()
    work["placement"] = work["placement"].astype(str).map(_normalize_placement_name)
    placement, tr, dm, rs = setting
    work = work[work["placement"] == _normalize_placement_name(placement)].copy()
    cd_t, ft, fps = triple
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
    grouped = (
        work.groupby(["n_aods", "n_qubits"])[["total_time"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    grouped["total_time_std"] = grouped["total_time_std"].fillna(0.0)
    return grouped if not grouped.empty else None


def _draw_aod_sweep_on_panel(
    ax,
    grouped: pd.DataFrame | None,
    plot_aods: list[int],
    aod_colors: dict[int, tuple[float, float, float]],
    *,
    linestyle: str = "-",
    y_clip: tuple[float | None, float | None] | None = None,
) -> list[int]:
    x_pts: list[int] = []
    if grouped is None or grouped.empty:
        return x_pts
    clip_lo, clip_hi = (None, None) if y_clip is None else y_clip
    for aod in plot_aods:
        sub = grouped[grouped["n_aods"].astype(int) == int(aod)].sort_values("n_qubits")
        if sub.empty:
            continue
        mean_vals = pd.to_numeric(sub["total_time_mean"], errors="coerce")
        std_vals = pd.to_numeric(sub["total_time_std"], errors="coerce").fillna(0.0)
        keep = pd.Series(True, index=sub.index)
        if clip_lo is not None:
            keep &= mean_vals >= float(clip_lo)
        if clip_hi is not None:
            keep &= mean_vals <= float(clip_hi)
        if not bool(keep.any()):
            continue
        sub = sub.loc[keep]
        mean_vals = mean_vals.loc[keep]
        std_vals = std_vals.loc[keep]
        color = aod_colors.get(int(aod), (0.2, 0.2, 0.2))
        _plot_styled_errorbar(
            ax,
            sub["n_qubits"],
            mean_vals,
            std_vals,
            color,
            linestyle=linestyle,
        )
        x_pts.extend(sub["n_qubits"].dropna().astype(int).tolist())
    return x_pts


def _aod_row_style_handle(
    base_rgb: tuple[float, float, float],
    *,
    linestyle: str | tuple = "-",
) -> Line2D:
    edge = mcolors.to_rgb(base_rgb)
    return Line2D(
        [0],
        [0],
        color=edge,
        linestyle=linestyle,
        linewidth=2.5,
        marker="o",
        markersize=_LINE_MARKERSIZE - 2,
        markerfacecolor=_marker_facecolor(edge),
        markeredgecolor=edge,
        markeredgewidth=_LINE_MARKER_EDGEWIDTH,
    )


def _build_aod_bottom_legend_rows(
    *,
    sync_aod_colors: dict[int, tuple[float, float, float]],
    best_aod_colors: dict[int, tuple[float, float, float]],
    sync_base_rgb: tuple[float, float, float],
    best_base_rgb: tuple[float, float, float],
    sync_label: str = "Sync. execution",
    best_label: str = "Best strategy",
) -> list[tuple[list[Line2D], list[str]]]:
    """One legend row per setting: strategy swatch + AOD 1/2/3/5 markers."""
    rows: list[tuple[list[Line2D], list[str]]] = []
    for setting_label, base_rgb, aod_colors, linestyle in (
        (sync_label, sync_base_rgb, sync_aod_colors, "-"),
        (best_label, best_base_rgb, best_aod_colors, (0, (5, 2))),
    ):
        handles = [_aod_row_style_handle(base_rgb, linestyle=linestyle)]
        handles.extend(_aod_column_legend_handles(aod_colors))
        labels = [setting_label] + [str(int(a)) for a in sorted(aod_colors)]
        rows.append((handles, labels))
    return rows


def _attach_aod_bottom_legend_spec(
    ax,
    *,
    sync_aod_colors: dict[int, tuple[float, float, float]],
    best_aod_colors: dict[int, tuple[float, float, float]],
    sync_base_rgb: tuple[float, float, float],
    best_base_rgb: tuple[float, float, float],
    sync_label: str = "Sync. execution",
    best_label: str = "Best strategy",
) -> None:
    ax._aod_bottom_legend_rows = _build_aod_bottom_legend_rows(  # type: ignore[attr-defined]
        sync_aod_colors=sync_aod_colors,
        best_aod_colors=best_aod_colors,
        sync_base_rgb=sync_base_rgb,
        best_base_rgb=best_base_rgb,
        sync_label=sync_label,
        best_label=best_label,
    )


def _apply_aod_bottom_legend(fig, ax_aod_parent, *, legend_fs: float) -> None:
    rows = getattr(ax_aod_parent, "_aod_bottom_legend_rows", None)
    if not rows:
        return
    n_rows = len(rows)
    for row_idx, (handles, labels) in enumerate(rows):
        y = AOD_BOTTOM_LEGEND_BASE_Y + (n_rows - 1 - row_idx) * AOD_BOTTOM_LEGEND_ROW_STEP
        fig.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=len(handles),
            fontsize=legend_fs,
            frameon=True,
            handlelength=1.4,
            handletextpad=0.35,
            columnspacing=0.65,
            borderpad=0.25,
        )


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
    setting_charts: tuple[str, ...] = ("bar", "bar_dual_aod"),
    bar_trim_extremes: bool = False,
    verbose: bool = True,
) -> None:
    """STAR and T-cultivation figures: setting-study bars and AOD comparison.

    Writes one PDF per entry in *setting_charts* (default: ``_bars`` and
    ``_bars_aod1_aod5`` for overlapping AOD = 1 vs 5 bars).
    When *bar_trim_extremes* is True, single-AOD bar panels drop min/max samples.
    Setting-study bars use distinct green/blue/red hues per strategy. The AOD panel
    plots sync and best at AOD = 1, 2, 3, 5; each strategy uses a light→dark ramp of
    its own setting color (AOD 1 lightest, AOD 5 darkest).
    """
    os.makedirs(output_dir, exist_ok=True)
    cd = int(
        code_distance
        if code_distance is not None
        else STAR_T_GRID_COMPARISON_CODE_DISTANCE
    )
    round_name = "full_trotter"
    aod_setting = int(setting_study_aod)

    star_study_entries = _star_setting_study_entries()
    t_study_entries = _t_setting_study_entries()
    star_setting_colors = _setting_study_categorical_colors(star_study_entries)
    t_setting_colors = _setting_study_categorical_colors(t_study_entries)

    star_aod_setting_idx = _pick_best_star_setting_index(
        dfs_dict_micro, cd, star_study_entries
    )
    star_aod_setting = SETTINGS[star_aod_setting_idx]
    star_aod_setting_label = next(
        (lbl for idx, lbl in star_study_entries if idx == star_aod_setting_idx),
        str(star_aod_setting_idx),
    )

    g_star_sync = _star_grouped_aod_at_distance(
        dfs_dict_aod, SETTINGS[STAR_AOD_SYNC_SETTING_INDEX], cd
    )
    g_star_best = _star_grouped_aod_at_distance(dfs_dict_aod, star_aod_setting, cd)

    triple = _primary_t_triple_for_star_t_grid(cd)
    cd_t, ft, fps = (0, 0.0, 0)
    t_setting_layer = pd.DataFrame()
    t_base = pd.DataFrame()
    t_aod_layer = pd.DataFrame()
    g_t_sync: pd.DataFrame | None = None
    g_t_best: pd.DataFrame | None = None
    t_aod_setting_idx: int = int(RUNTIME_PROFILE_T_SETTING_IDX)
    t_aod_setting_label = ""
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
        if compile_cols.issubset(t_aod_layer.columns) and compile_cols.issubset(
            t_setting_layer.columns
        ):
            t_aod_setting_idx = _pick_best_t_setting_key(
                t_setting_layer, triple, cd, t_study_entries
            )
            placement, tr, dm, rs = _T_SETTING_ABLATION_GRID[t_aod_setting_idx]
            t_aod_compile = (tr, dm, rs)
            t_aod_setting_label = next(
                (
                    lbl
                    for idx, lbl in t_study_entries
                    if int(idx) == int(t_aod_setting_idx)
                ),
                str(t_aod_setting_idx),
            )
            t_base = _build_t_aod_base_df(
                _filter_col_based(t_aod_layer.copy()), triple, cd, t_aod_compile
            )
            g_t_sync = _t_grouped_aod_at_distance(
                t_aod_layer,
                triple,
                _T_SETTING_ABLATION_GRID[T_AOD_SYNC_SETTING_INDEX],
                cd,
            )
            g_t_best = _t_grouped_aod_at_distance(
                t_aod_layer,
                triple,
                _T_SETTING_ABLATION_GRID[t_aod_setting_idx],
                cd,
            )

    aod_candidates = list(AOD_COMPARISON_AODS)
    star_aods: set[int] = set()
    for frame in (g_star_sync, g_star_best):
        if frame is not None and not frame.empty:
            star_aods.update(
                pd.to_numeric(frame["n_aods"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
    t_aods: set[int] = set()
    for frame in (g_t_sync, g_t_best):
        if frame is not None and not frame.empty:
            t_aods.update(
                pd.to_numeric(frame["n_aods"], errors="coerce")
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

    if verbose:
        print(
            f"  AOD panel (d={cd}): STAR setting idx {star_aod_setting_idx} "
            f"({star_aod_setting_label}); "
            f"T-cultivation {t_aod_setting_label or t_aod_setting_idx}."
        )

    star_sync_base = mcolors.to_rgb(star_setting_colors[STAR_AOD_SYNC_SETTING_INDEX])
    star_best_base = mcolors.to_rgb(star_setting_colors[star_aod_setting_idx])
    star_sync_aod_colors = _build_aod_color_map_from_base(
        star_sync_base, star_plot_aods
    )
    star_best_aod_colors = _build_aod_color_map_from_base(
        star_best_base, star_plot_aods
    )

    t_sync_base = mcolors.to_rgb(t_setting_colors[T_AOD_SYNC_SETTING_INDEX])
    t_best_base = mcolors.to_rgb(t_setting_colors[t_aod_setting_idx])
    t_sync_aod_colors = _build_aod_color_map_from_base(t_sync_base, t_plot_aods)
    t_best_aod_colors = _build_aod_color_map_from_base(t_best_base, t_plot_aods)
    star_shared_y_break: float | None = None
    star_shared_panel_hi: float | None = None

    def _apply_theory_line(
        ax,
        x_points: list[int],
        *,
        theory: str,
        x_positions: dict[int, float] | None = None,
        y_clip: tuple[float | None, float | None] | None = None,
    ) -> None:
        x_u = sorted(set(int(x) for x in x_points))
        if not x_u:
            return
        if theory == "star":
            b = _get_theoretical_lower_bound(x_u)
        else:
            b = _get_theoretical_lower_bound_t_cultivation(
                x_u, code_distance=cd_t, fidelity_target=ft
            )
        if x_positions:
            x_plot = [float(x_positions[int(x)]) for x in x_u if int(x) in x_positions]
            y_theory = [
                b["full_trotter"][i]
                for i, x in enumerate(x_u)
                if int(x) in x_positions
            ]
        else:
            x_plot = [float(x) for x in x_u]
            y_theory = [float(y) for y in b["full_trotter"][: len(x_plot)]]
        if y_clip is not None:
            clip_lo, clip_hi = y_clip
            filtered: list[tuple[float, float]] = []
            for x_val, y_val in zip(x_plot, y_theory):
                if clip_lo is not None and float(y_val) < float(clip_lo):
                    continue
                if clip_hi is not None and float(y_val) > float(clip_hi):
                    continue
                filtered.append((float(x_val), float(y_val)))
            if len(filtered) < 2:
                return
            x_plot, y_theory = zip(*filtered)
        ax.plot(
            x_plot,
            y_theory,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

    def _apply_panel_style(ax) -> None:
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.relim(visible_only=True)
        ax.autoscale_view()
        _lo, y_hi = ax.get_ylim()
        if y_hi > 0.0:
            ax.set_ylim(0.0, y_hi)

    def _finalize_panel(
        ax,
        x_points: list[int],
        *,
        theory: str,
        x_positions: dict[int, float] | None = None,
    ) -> None:
        _apply_theory_line(ax, x_points, theory=theory, x_positions=x_positions)
        _apply_panel_style(ax)

    def _draw_star_aod_curves(
        ax,
        *,
        y_clip: tuple[float | None, float | None] | None = None,
    ) -> list[int]:
        x_pts: list[int] = []
        x_pts.extend(
            _draw_aod_sweep_on_panel(
                ax,
                g_star_sync,
                star_plot_aods,
                star_sync_aod_colors,
                linestyle="-",
                y_clip=y_clip,
            )
        )
        x_pts.extend(
            _draw_aod_sweep_on_panel(
                ax,
                g_star_best,
                star_plot_aods,
                star_best_aod_colors,
                linestyle=(0, (5, 2)),
                y_clip=y_clip,
            )
        )
        return x_pts

    def _draw_star_aod_panel(ax) -> None:
        nonlocal star_shared_y_break, star_shared_panel_hi
        sync_extent = _grouped_aod_y_extent(g_star_sync)
        best_extent = _grouped_aod_y_extent(g_star_best)
        sync_hi = sync_extent[1]
        best_hi = best_extent[1]
        panel_hi = max(sync_hi, best_hi)

        use_break = star_shared_y_break is not None or _star_aod_broken_axis_needed(
            sync_extent, best_extent
        )
        if use_break:
            if star_shared_y_break is not None:
                y_break = float(star_shared_y_break)
                panel_hi = float(star_shared_panel_hi or panel_hi)
            else:
                y_break = _compute_star_aod_break_y(sync_extent, best_extent)
            y_pad = 0.06 * max(panel_hi - y_break, 1e-9)
            y_top_hi = panel_hi + y_pad
            clip_margin = max(
                y_break * STAR_BROKEN_CLIP_MARGIN_FRAC,
                0.01 * max(panel_hi - y_break, 1.0),
            )
            bot_clip = (None, y_break - clip_margin)
            top_clip = (y_break + clip_margin, None)

            ax_top, ax_bot = _create_star_aod_broken_axes(ax)
            x_pts = _draw_star_aod_curves(ax_bot, y_clip=bot_clip)
            x_pts.extend(_draw_star_aod_curves(ax_top, y_clip=top_clip))

            _apply_broken_bot_y_ticks(ax_bot, y_break)
            _apply_broken_top_y_ticks(ax_top, y_break, y_top_hi)

            _apply_theory_line(ax_bot, x_pts, theory="star", y_clip=bot_clip)
            _apply_theory_line(ax_top, x_pts, theory="star", y_clip=top_clip)
            _draw_axis_break_between(ax_top, ax_bot)

            for _ax in (ax_top, ax_bot):
                _ax.set_axisbelow(True)
                _ax.grid(True, alpha=0.3)

            _attach_aod_bottom_legend_spec(
                ax,
                sync_aod_colors=star_sync_aod_colors,
                best_aod_colors=star_best_aod_colors,
                sync_base_rgb=star_sync_base,
                best_base_rgb=star_best_base,
                sync_label="Sync. execution",
                best_label=star_aod_setting_label or "Best strategy",
            )
            return

        x_pts = _draw_star_aod_curves(ax)
        _finalize_panel(ax, x_pts, theory="star")
        _apply_panel_y_ticks(ax, nbins=STAR_AOD_COMPARISON_Y_NBINS)
        _attach_aod_bottom_legend_spec(
            ax,
            sync_aod_colors=star_sync_aod_colors,
            best_aod_colors=star_best_aod_colors,
            sync_base_rgb=star_sync_base,
            best_base_rgb=star_best_base,
            sync_label="Sync. execution",
            best_label=star_aod_setting_label or "Best strategy",
        )

    def _draw_star_setting_panel(ax, *, setting_chart: str) -> None:
        nonlocal star_shared_y_break, star_shared_panel_hi
        prepared: list[tuple[int, str, pd.DataFrame]] = []
        y_extents: dict[int, tuple[float, float]] = {}
        trim_bars = bool(bar_trim_extremes and setting_chart == "bar")
        for setting_idx, label in star_study_entries:
            if setting_idx >= len(SETTINGS):
                continue
            grouped = _aggregate_star_setting_study(
                dfs_dict_micro,
                SETTINGS[setting_idx],
                cd,
                aod_setting,
                trim_extremes=trim_bars,
            )
            if grouped is None:
                continue
            prepared.append((int(setting_idx), label, grouped))
            y_extents[int(setting_idx)] = _agg_y_extent(grouped)

        vanilla_idx = STAR_SETTING_STUDY_VANILLA_IDX
        if setting_chart == "bar_dual_aod":
            aod_back, aod_front = SETTING_STUDY_DUAL_AODS[0], SETTING_STUDY_DUAL_AODS[1]
            dual_series: list[tuple] = []
            for setting_idx, label in star_study_entries:
                if setting_idx >= len(SETTINGS):
                    continue
                g_back = _aggregate_star_setting_study(
                    dfs_dict_micro,
                    SETTINGS[setting_idx],
                    cd,
                    int(aod_back),
                    trim_extremes=False,
                )
                g_front = _aggregate_star_setting_study(
                    dfs_dict_micro,
                    SETTINGS[setting_idx],
                    cd,
                    int(aod_front),
                    trim_extremes=False,
                )
                if g_back is None and g_front is None:
                    continue
                dual_series.append(
                    (star_setting_colors[setting_idx], label, g_back, g_front)
                )
            if vanilla_idx in y_extents:

                def _series_hi(g: pd.DataFrame | None) -> float:
                    if g is None or g.empty:
                        return 0.0
                    vals = pd.to_numeric(g["total_time_max"], errors="coerce").dropna()
                    if vals.empty:
                        return 0.0
                    return float(vals.max())

                dual_top_hi = 0.0
                for _color, _label, g_back, g_front in dual_series:
                    dual_top_hi = max(
                        dual_top_hi, _series_hi(g_back), _series_hi(g_front)
                    )
                if dual_top_hi <= 0.0:
                    dual_top_hi = max(float(_hi) for (_lo, _hi) in y_extents.values())

                y_break = _compute_star_setting_break_y(
                    y_extents, vanilla_idx=vanilla_idx
                )
                y_hi_sorted = sorted(
                    [float(_hi) for (_lo, _hi) in y_extents.values()],
                    reverse=True,
                )
                if len(y_hi_sorted) >= 2:
                    y_break = min(y_break, y_hi_sorted[1] * 0.985)
                y_break = min(y_break, dual_top_hi * 0.985)
                ax_top, ax_bot = _create_star_setting_broken_axes(ax)
                x_pts = _draw_setting_study_dual_aod_bars(ax_top, dual_series)
                _draw_setting_study_dual_aod_bars(ax_bot, dual_series)

                y_top_hi = dual_top_hi
                y_pad = 0.08 * max(y_top_hi - y_break, 1e-9)
                ax_bot.set_ylim(0.0, y_break)
                ax_top.set_ylim(y_break, y_top_hi + y_pad)
                star_shared_y_break = float(y_break)
                star_shared_panel_hi = float(y_top_hi + y_pad)

                x_u = sorted(set(int(x) for x in x_pts))
                if x_u:
                    b = _get_theoretical_lower_bound(x_u)
                    x_pos = {int(nq): float(i) for i, nq in enumerate(x_u)}
                    for _ax in (ax_top, ax_bot):
                        _ax.plot(
                            [x_pos[int(nq)] for nq in x_u],
                            b["full_trotter"],
                            color="black",
                            linestyle="--",
                            linewidth=1.5,
                        )
                _draw_axis_break_between(ax_top, ax_bot)
                return

            x_pts = _draw_setting_study_dual_aod_bars(ax, dual_series)
            x_pos = {int(nq): float(i) for i, nq in enumerate(x_pts)}
        elif setting_chart == "bar":
            bar_series = [
                (star_setting_colors[setting_idx], label, grouped)
                for setting_idx, label, grouped in prepared
            ]
            if vanilla_idx in y_extents and _star_setting_broken_axis_needed(y_extents):
                y_break = _compute_star_setting_break_y(
                    y_extents, vanilla_idx=vanilla_idx
                )
                y_hi_sorted = sorted(
                    [float(_hi) for (_lo, _hi) in y_extents.values()],
                    reverse=True,
                )
                if len(y_hi_sorted) >= 2:
                    y_break = min(y_break, y_hi_sorted[1] * 0.985)
                y_top_hi = float(y_extents[vanilla_idx][1])
                ax_top, ax_bot = _create_star_setting_broken_axes(ax)
                x_pts = _draw_setting_study_grouped_bars(ax_top, bar_series)
                _draw_setting_study_grouped_bars(ax_bot, bar_series)
                y_pad = 0.08 * max(y_top_hi - y_break, 1e-9)
                ax_bot.set_ylim(0.0, y_break)
                ax_top.set_ylim(y_break, y_top_hi + y_pad)
                star_shared_y_break = float(y_break)
                star_shared_panel_hi = float(y_top_hi + y_pad)
                x_u = sorted(set(int(x) for x in x_pts))
                if x_u:
                    b = _get_theoretical_lower_bound(x_u)
                    x_pos = {int(nq): float(i) for i, nq in enumerate(x_u)}
                    for _ax in (ax_top, ax_bot):
                        _ax.plot(
                            [x_pos[int(nq)] for nq in x_u],
                            b["full_trotter"],
                            color="black",
                            linestyle="--",
                            linewidth=1.5,
                        )
                _draw_axis_break_between(ax_top, ax_bot)
                return
            x_pts = _draw_setting_study_grouped_bars(ax, bar_series)
            x_pos = {int(nq): float(i) for i, nq in enumerate(x_pts)}
        else:
            raise ValueError(
                f"Unsupported setting_chart for STAR setting-study: {setting_chart}"
            )
        _finalize_panel(ax, x_pts, theory="star", x_positions=x_pos)

    def _draw_t_setting_panel(ax, *, setting_chart: str) -> None:
        prepared: list[tuple[tuple[str, int], str, pd.DataFrame]] = []
        trim_bars = bool(bar_trim_extremes and setting_chart == "bar")
        for setting_idx, label in t_study_entries:
            if setting_idx >= len(_T_SETTING_ABLATION_GRID):
                continue
            placement, tr, dm, rs = _T_SETTING_ABLATION_GRID[setting_idx]
            grouped = _aggregate_t_setting_study(
                t_setting_layer,
                triple,
                (placement, tr, dm, rs),
                cd,
                aod_setting,
                trim_extremes=trim_bars,
            )
            if grouped is None:
                continue
            prepared.append(((placement, setting_idx), label, grouped))

        if setting_chart == "bar_dual_aod":
            aod_back, aod_front = SETTING_STUDY_DUAL_AODS[0], SETTING_STUDY_DUAL_AODS[1]
            dual_series = []
            x_pts = []
            for setting_idx, label in t_study_entries:
                if setting_idx >= len(_T_SETTING_ABLATION_GRID):
                    continue
                placement, tr, dm, rs = _T_SETTING_ABLATION_GRID[setting_idx]
                g_back = _aggregate_t_setting_study(
                    t_setting_layer,
                    triple,
                    (placement, tr, dm, rs),
                    cd,
                    int(aod_back),
                )
                g_front = _aggregate_t_setting_study(
                    t_setting_layer,
                    triple,
                    (placement, tr, dm, rs),
                    cd,
                    int(aod_front),
                )
                if g_back is None and g_front is None:
                    continue
                dual_series.append(
                    (t_setting_colors[setting_idx], label, g_back, g_front)
                )
            x_pts = _draw_setting_study_dual_aod_bars(ax, dual_series)
            x_pos = {int(nq): float(i) for i, nq in enumerate(x_pts)}
        elif setting_chart == "bar":
            bar_series = [
                (t_setting_colors[key[1]], label, grouped)
                for key, label, grouped in prepared
            ]
            x_pts = _draw_setting_study_grouped_bars(ax, bar_series)
            x_pos = {int(nq): float(i) for i, nq in enumerate(x_pts)}
        else:
            raise ValueError(
                f"Unsupported setting_chart for T setting-study: {setting_chart}"
            )
        _finalize_panel(ax, x_pts, theory="t", x_positions=x_pos)

    def _draw_t_aod_panel(ax) -> None:
        x_pts: list[int] = []
        x_pts.extend(
            _draw_aod_sweep_on_panel(
                ax,
                g_t_sync,
                t_plot_aods,
                t_sync_aod_colors,
                linestyle="-",
            )
        )
        x_pts.extend(
            _draw_aod_sweep_on_panel(
                ax,
                g_t_best,
                t_plot_aods,
                t_best_aod_colors,
                linestyle=(0, (5, 2)),
            )
        )
        _finalize_panel(ax, x_pts, theory="t")
        _apply_panel_y_ticks(ax, nbins=T_CULTIVATION_Y_NBINS)
        _attach_aod_bottom_legend_spec(
            ax,
            sync_aod_colors=t_sync_aod_colors,
            best_aod_colors=t_best_aod_colors,
            sync_base_rgb=t_sync_base,
            best_base_rgb=t_best_base,
            sync_label="Sync. execution",
            best_label=t_aod_setting_label or "Best strategy",
        )

    plot_star = "full_trotter" in dfs_dict_micro
    plot_t = triple is not None and not t_setting_layer.empty

    for setting_chart in setting_charts:
        if setting_chart == "bar_dual_aod":
            setting_column_title = (
                "Compilation Strategies"
                # f"Compilation Strategies, AOD = {SETTING_STUDY_DUAL_AODS[0]} "
                # f"vs {SETTING_STUDY_DUAL_AODS[1]}"
            )
        else:
            setting_column_title = "Compilation Strategies"
        star_column_titles = (
            setting_column_title,
            # f"AOD comparison ({star_aod_setting_label})",
            "AOD comparison",
        )
        t_column_titles = (
            setting_column_title,
            "AOD comparison",
            # f"AOD comparison ({t_aod_setting_label or 'best setting'})",
        )

        chart_suffix = _setting_study_chart_suffix(
            setting_chart, bar_trim_extremes=bar_trim_extremes
        )
        star_setting_legend_handles = [
            _setting_study_legend_handle(
                color=star_setting_colors[setting_idx],
                label=label,
                setting_chart=setting_chart,
            )
            for setting_idx, label in star_study_entries
            if setting_idx in star_setting_colors
        ]
        t_setting_legend_handles = [
            _setting_study_legend_handle(
                color=t_setting_colors[setting_idx],
                label=label,
                setting_chart=setting_chart,
            )
            for setting_idx, label in t_study_entries
            if setting_idx in t_setting_colors
        ]
        aod_legend_handles = None
        if setting_chart == "bar_dual_aod":
            # Separate legend for AOD numbers (no hue/blue; only black/gray).
            aod_legend_handles = _dual_aod_bar_legend_handles("black")

        def _draw_star_setting(ax, _chart=setting_chart):
            _draw_star_setting_panel(ax, setting_chart=_chart)

        def _draw_t_setting(ax, _chart=setting_chart):
            _draw_t_setting_panel(ax, setting_chart=_chart)

        if plot_star:
            star_path = os.path.join(
                output_dir,
                f"star_setting_and_aod_comparison_d{cd}{chart_suffix}.pdf",
            )
            _save_setting_aod_architecture_figure(
                draw_setting=_draw_star_setting,
                draw_aod=_draw_star_aod_panel,
                setting_legend_handles=star_setting_legend_handles,
                aod_legend_handles=aod_legend_handles,
                show_aod_colorbar=False,
                figure_title="STAR architecture",
                out_path=star_path,
                xlabel="Number of Qubits/Factories",
                legend_y_blend=0.72,
                panel_hspace=0.48,
                bottom_axis_y_shift=0.04,
                column_titles=star_column_titles,
            )
            if verbose:
                print(f"Saved: {star_path}")

        if plot_t:
            t_path = os.path.join(
                output_dir,
                f"t_cultivation_setting_and_aod_comparison_d{cd}{chart_suffix}.pdf",
            )
            _save_setting_aod_architecture_figure(
                draw_setting=_draw_t_setting,
                draw_aod=_draw_t_aod_panel,
                setting_legend_handles=t_setting_legend_handles,
                aod_legend_handles=aod_legend_handles,
                show_aod_colorbar=False,
                figure_title="T-cultivation",
                out_path=t_path,
                xlabel="Number of Qubits/Factories",
                y_axis_thousands=True,
                column_titles=t_column_titles,
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
    setting_study_aod: int = STAR_T_SETTING_STUDY_AOD,
    setting_charts: tuple[str, ...] = ("bar", "bar_dual_aod"),
    bar_trim_extremes: bool = False,
    verbose: bool = True,
) -> None:
    """Load profiling CSVs and write STAR and T-cultivation stacked comparison PDFs."""
    os.makedirs(output_dir, exist_ok=True)

    star_df = pd.read_csv(star_csv_file, engine="python", on_bad_lines="skip")
    t_df = pd.read_csv(t_cultivation_csv_file, engine="python", on_bad_lines="skip")
    t_df = _dedupe_duplicate_columns(t_df)
    _validate_t_profiling_csv_columns(t_df, t_cultivation_csv_file)
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
            setting_study_aod=setting_study_aod,
            setting_charts=setting_charts,
            bar_trim_extremes=bar_trim_extremes,
            verbose=verbose,
        )
        _plot_star_t_runtime_profile_figure(
            dfs_star,
            t_layers_ablation,
            output_dir,
            code_distance=code_distance,
            verbose=verbose,
        )
    finally:
        SHOW_T_CULTIVATION_D13 = previous_show_d13


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot STAR vs T-cultivation setting/AOD comparison figures."
    )
    parser.add_argument(
        "--bar-trim-extremes",
        action="store_true",
        help=(
            "For bar charts, drop the min and max total_time sample per qubit count "
            "before computing bar height and error bars."
        ),
    )
    args = parser.parse_args()

    star_csv = "output/evaluation/fidelity/star_full_trotter_profiling_results.csv"
    t_csv = "output/evaluation/fidelity/t_cultivation_fidelity_profiling_results.csv"
    out_dir = "output/prx_quantum/"

    if os.path.exists(star_csv) and os.path.exists(t_csv):
        process_star_t_setting_and_aod_figure(
            star_csv,
            t_csv,
            out_dir,
            bar_trim_extremes=args.bar_trim_extremes,
            verbose=True,
        )
    else:
        print("Missing profiling CSV(s); expected:")
        print(f"  {star_csv}")
        print(f"  {t_csv}")
