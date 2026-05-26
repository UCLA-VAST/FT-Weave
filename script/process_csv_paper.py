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
    ("seperate_region_row", False, True, 0, False, False),
    ("seperate_region_row", False, False, 0, True, False),
    # ("seperate_region_row", False, False, 2, True, False),
    # ("seperate_region_row", True, True, 0, False, False),
    # ("seperate_region_row", True, False, 0, True, False),
    # ("seperate_region_row", True, False, 2, True, False),
    # ("col_based", False, True, 0, False, False),
    # ("col_based", True, True, 0, False, False),
    # ("col_based", True, False, 0, True, False),
    # ("col_based", True, False, 2, True, False),
    # ("col_based", True, False, 2, False, True),
    ("col_based", False, False, 0, True, False),
    ("col_based", False, False, 2, True, False),
    ("col_based", True, False, 2, True, False),
    ("col_based", False, False, 2, False, True),
]

RESULT_COLS = ["total_time", "movement_time", "return_movement_time"]

STAR_T_GRID_COMPARISON_CODE_DISTANCE: int = 9
STAR_T_GRID_STAR_SETTING_INDEX: int = 4
STAR_T_SETTING_STUDY_AOD: int = 5
# Setting-study dual-AOD bar chart: back = first AOD (full color), front = second (lighter).
SETTING_STUDY_DUAL_AODS: tuple[int, ...] = (1, 5)
SETTING_STUDY_DUAL_AOD_FRONT_BLEND: float = 0.52
# Right-hand STAR figure panel: denser y-axis ticks on the AOD sweep.
STAR_AOD_COMPARISON_Y_NBINS: int = 8
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
RUNTIME_PROFILE_T_COMPILE_IDX: int = 3  # "(c) + Opt. patch redist."
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

_T_COMPILE_ABLATION_GRID: list[tuple[bool, bool, bool]] = [
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
    "Sync. execution",
    " + Routing opt.",
    " + Microarch. opt.",
    " + Dropout",
    " + Lookahead angle prep.",  # "(d) + Lookahead angle prep.",
    "Async. execution",
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
# Grayscale AOD ramp for the AOD panel (no hue/blue; AOD=5 is lighter).
_AOD_DARK_RGB = (0.0, 0.0, 0.0)  # AOD = 1
_AOD_LIGHT_RGB = (0.72, 0.72, 0.72)  # AOD = 5


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
    aod_legend_handles: list | None = None,
    aod_color_map: dict[int, tuple],
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

    panel_title_fs = _FIG_FONT_SIZE - 1
    legend_fs = _FIG_FONT_SIZE - 4
    ax_setting_top.set_title(column_titles[0], fontsize=panel_title_fs, pad=8)
    axes[1, 0].set_title(column_titles[1], fontsize=panel_title_fs, pad=8)
    y_label = "Execution time (×10³)" if y_axis_thousands else "Execution time"
    ax_setting_bot.set_ylabel(y_label, fontsize=_FIG_FONT_SIZE, labelpad=2)
    axes[1, 0].set_ylabel(y_label, fontsize=_FIG_FONT_SIZE, labelpad=2)
    if y_axis_thousands:
        _apply_y_axis_thousands([ax_setting_bot, axes[1, 0]])
    for row in range(2):
        if row == 0:
            ax_setting_top.tick_params(axis="both", which="major", pad=1)
            ax_setting_bot.tick_params(axis="both", which="major", pad=1)
        else:
            axes[row, 0].tick_params(axis="both", which="major", pad=1)
    axes[1, 0].set_xlabel(xlabel, fontsize=_FIG_FONT_SIZE, labelpad=0)

    fig.suptitle(figure_title, fontsize=_FIG_FONT_SIZE + 1, y=0.98)
    fig.tight_layout(rect=(0.08, 0.06, 0.90, 0.94), pad=0.10, h_pad=0.85)
    fig.subplots_adjust(
        top=0.92, bottom=0.08, left=0.12, right=0.88, hspace=panel_hspace
    )
    if bottom_axis_y_shift:
        bot_pos = axes[1, 0].get_position()
        axes[1, 0].set_position(
            [
                bot_pos.x0,
                bot_pos.y0 - float(bottom_axis_y_shift),
                bot_pos.width,
                bot_pos.height,
            ]
        )

    combined_legend = [*setting_legend_handles, _expected_time_legend_handle()]
    top_pos = axes[0, 0].get_position()
    bot_pos = axes[1, 0].get_position()
    blend = float(np.clip(legend_y_blend, 0.0, 1.0))
    legend_y = (1.0 - blend) * top_pos.y0 + blend * bot_pos.y1
    if figure_title == "T-cultivation":
        legend_y -= 0.0
    else:
        legend_y += 0.15
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
                "total_time_min": np.nan,
                "total_time_max": np.nan,
            }
        )
    if vals.size <= 2:
        return pd.Series(
            {
                "total_time_mean": float(np.mean(vals)),
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
            .agg(["mean", "min", "max"])
            .reset_index()
        )
        grouped.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in grouped.columns
        ]
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
    compile_tuple: tuple[bool, bool, bool],
    placement: str,
    code_distance: int,
    aod: int,
    *,
    trim_extremes: bool = False,
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
    grouped = _aggregate_total_time_by_qubits(work, trim_extremes=trim_extremes)
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
    ax_parent._broken_setting_axes = (ax_top, ax_bot)  # type: ignore[attr-defined]
    return ax_top, ax_bot


def _blend_rgb_toward_white(
    rgb: tuple[float, float, float], blend: float
) -> tuple[float, float, float]:
    """Lighten *rgb* by blending toward white (``blend`` in [0, 1])."""
    t = float(np.clip(blend, 0.0, 1.0))
    base = np.array(mcolors.to_rgb(rgb), dtype=float)
    white = np.ones(3, dtype=float)
    return tuple(np.clip((1.0 - t) * base + t * white, 0.0, 1.0))


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
    entries: list[tuple[str, int, str]],
    *,
    aods: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> tuple[str, int]:
    """Return (placement, compile_idx) with lowest mean execution time for the AOD panel."""
    if triple is None or t_setting_layer.empty:
        return ("col_based", RUNTIME_PROFILE_T_COMPILE_IDX)

    best_key: tuple[str, int] | None = None
    best_mean = math.inf
    for placement, compile_idx, _label in entries:
        if str(placement).strip() != "col_based":
            continue
        if compile_idx >= len(_T_COMPILE_ABLATION_GRID):
            continue
        sample_means: list[float] = []
        for aod in aods:
            grouped = _aggregate_t_setting_study(
                t_setting_layer,
                triple,
                _T_COMPILE_ABLATION_GRID[compile_idx],
                placement,
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
            best_key = (placement, int(compile_idx))
    if best_key is None:
        return ("col_based", RUNTIME_PROFILE_T_COMPILE_IDX)
    return best_key


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
    t_setting_colors: dict[tuple[str, int], tuple],
    *,
    t_setting_key: tuple[str, int] | None = None,
) -> tuple[float, float, float]:
    """Match AOD = 1 to the compile setting used in the T AOD sweep panel."""
    if t_setting_key is not None and t_setting_key in t_setting_colors:
        return mcolors.to_rgb(t_setting_colors[t_setting_key])
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
                lo = float(row["total_time_min"])
                hi = float(row["total_time_max"])
                heights.append(mean)
                yerr_lo.append(max(0.0, mean - lo))
                yerr_hi.append(max(0.0, hi - mean))
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
            capsize=2.5,
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
                lo = float(row["total_time_min"])
                hi = float(row["total_time_max"])
                heights.append(mean)
                yerr_lo.append(max(0.0, mean - lo))
                yerr_hi.append(max(0.0, hi - mean))
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
            capsize=2.5,
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
    compile_tuple: tuple[bool, bool, bool],
    code_distance: int,
    aod: int,
) -> pd.DataFrame | None:
    work = _filter_col_based(layer_df.copy())
    if work.empty:
        return None
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
    work["n_aods"] = pd.to_numeric(work["n_aods"], errors="coerce").astype(int)
    work = work[work["n_aods"] == int(aod)].copy()
    ctx = (
        f"code_distance={code_distance}, AOD={aod}, "
        f"compile={compile_tuple}, placement=col_based"
    )
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
    t_compile_idx: int = RUNTIME_PROFILE_T_COMPILE_IDX,
    verbose: bool = True,
) -> None:
    """Stacked runtime bars: STAR (top) vs T-cultivation (bottom), AOD 1 vs 5 grouped."""
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
    if (
        triple is not None
        and round_name in t_layers_ablation
        and t_compile_idx < len(_T_COMPILE_ABLATION_GRID)
    ):
        compile_tuple = _T_COMPILE_ABLATION_GRID[t_compile_idx]
        for aod in aods:
            t_profiles[int(aod)] = _t_runtime_profile_frame(
                t_layers_ablation[round_name],
                triple,
                compile_tuple,
                cd,
                int(aod),
            )

    def _has_profile(profiles: dict[int, pd.DataFrame | None]) -> bool:
        return any(prof is not None and not prof.empty for prof in profiles.values())

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
    setting_charts: tuple[str, ...] = ("bar", "bar_dual_aod"),
    bar_trim_extremes: bool = False,
    verbose: bool = True,
) -> None:
    """STAR and T-cultivation figures: setting-study bars and AOD 1–5.

    Writes one PDF per entry in *setting_charts* (default: ``_bars`` and
    ``_bars_aod1_aod5`` for overlapping AOD = 1 vs 5 bars).
    When *bar_trim_extremes* is True, single-AOD bar panels drop min/max samples.
    The AOD comparison panel uses the col_based setting with the best mean runtime
    (picked at runtime); AOD=1/5 are rendered in a consistent grayscale ramp.
    """
    os.makedirs(output_dir, exist_ok=True)
    cd = int(
        code_distance
        if code_distance is not None
        else STAR_T_GRID_COMPARISON_CODE_DISTANCE
    )
    round_name = "full_trotter"
    aod_setting = int(setting_study_aod)

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

    star_aod_setting_idx = _pick_best_star_setting_index(
        dfs_dict_micro, cd, star_study_entries
    )
    star_aod_setting = SETTINGS[star_aod_setting_idx]
    star_aod_setting_label = next(
        (lbl for idx, lbl in star_study_entries if idx == star_aod_setting_idx),
        str(star_aod_setting_idx),
    )

    g_cb = _star_grouped_aod_col_based_at_distance(dfs_dict_aod, star_aod_setting, cd)
    if g_cb is None:
        g_cb = pd.DataFrame()

    triple = _primary_t_triple_for_star_t_grid(cd)
    cd_t, ft, fps = (0, 0.0, 0)
    t_setting_layer = pd.DataFrame()
    t_base = pd.DataFrame()
    t_aod_setting_key: tuple[str, int] = ("col_based", RUNTIME_PROFILE_T_COMPILE_IDX)
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
        if "placement" in t_aod_layer.columns:
            t_aod_layer = _filter_col_based(t_aod_layer)
        if compile_cols.issubset(t_aod_layer.columns) and compile_cols.issubset(
            t_setting_layer.columns
        ):
            t_aod_setting_key = _pick_best_t_setting_key(
                t_setting_layer, triple, cd, t_study_entries
            )
            t_aod_compile = _T_COMPILE_ABLATION_GRID[t_aod_setting_key[1]]
            t_aod_setting_label = next(
                (
                    lbl
                    for pl, cidx, lbl in t_study_entries
                    if (pl, cidx) == t_aod_setting_key
                ),
                str(t_aod_setting_key),
            )
            t_base = _build_t_aod_base_df(t_aod_layer, triple, cd, t_aod_compile)

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

    if verbose:
        print(
            f"  AOD panel (d={cd}): STAR setting idx {star_aod_setting_idx} "
            f"({star_aod_setting_label}); "
            f"T-cultivation {t_aod_setting_label or t_aod_setting_key}."
        )

    # AOD lines: keep the best-setting base color and sweep toward darker shades.
    # For STAR this uses the selected best STAR setting (e.g., Dropout case).
    star_best_rgb = mcolors.to_rgb(star_setting_colors[star_aod_setting_idx])
    star_aod_colors = _build_aod_color_map_from_base(star_best_rgb, star_plot_aods)
    t_best_rgb = _t_aod_panel_base_rgb(
        t_setting_colors, t_setting_key=t_aod_setting_key
    )
    t_aod_colors = _build_aod_color_map_from_base(t_best_rgb, t_plot_aods)
    star_aod_default = star_aod_colors.get(_AOD_COLORBAR_MIN, star_best_rgb)
    t_aod_default = t_aod_colors.get(_AOD_COLORBAR_MIN, t_best_rgb)

    def _finalize_panel(
        ax,
        x_points: list[int],
        *,
        theory: str,
        x_positions: dict[int, float] | None = None,
    ) -> None:
        x_u = sorted(set(int(x) for x in x_points))
        if x_u:
            if theory == "star":
                b = _get_theoretical_lower_bound(x_u)
            else:
                b = _get_theoretical_lower_bound_t_cultivation(
                    x_u, code_distance=cd_t, fidelity_target=ft
                )
            if x_positions:
                x_plot = [
                    float(x_positions[int(x)]) for x in x_u if int(x) in x_positions
                ]
            else:
                x_plot = x_u
            ax.plot(
                x_plot,
                b["full_trotter"][: len(x_plot)],
                color="black",
                linestyle="--",
                linewidth=1.5,
            )
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.relim(visible_only=True)
        ax.autoscale_view()
        _lo, y_hi = ax.get_ylim()
        if y_hi > 0.0:
            ax.set_ylim(0.0, y_hi)

    def _draw_star_setting_panel(ax, *, setting_chart: str) -> None:
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

                # STAR: the dual-AOD bars can be dominated by the largest qubit counts.
                # Add a y-axis “skip point” via a broken-axis split so smaller bars
                # remain readable.
                y_break = _compute_star_setting_break_y(
                    y_extents, vanilla_idx=vanilla_idx
                )
                # Ensure the top panel still shows bar tops for the next tallest
                # strategies (e.g., blue/orange at large qubit count), not only
                # the single tallest series.
                y_hi_sorted = sorted(
                    [float(_hi) for (_lo, _hi) in y_extents.values()],
                    reverse=True,
                )
                if len(y_hi_sorted) >= 2:
                    y_break = min(y_break, y_hi_sorted[1] * 0.985)
                # Keep split below the highest bar top from the dual-AOD data.
                y_break = min(y_break, dual_top_hi * 0.985)
                ax_top, ax_bot = _create_star_setting_broken_axes(ax)
                x_pts = _draw_setting_study_dual_aod_bars(ax_top, dual_series)
                _draw_setting_study_dual_aod_bars(ax_bot, dual_series)

                y_top_hi = dual_top_hi
                y_pad = 0.08 * max(y_top_hi - y_break, 1e-9)
                ax_bot.set_ylim(0.0, y_break)
                ax_top.set_ylim(y_break, y_top_hi + y_pad)

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
                trim_extremes=trim_bars,
            )
            if grouped is None:
                continue
            prepared.append(((placement, compile_idx), label, grouped))

        if setting_chart == "bar_dual_aod":
            aod_back, aod_front = SETTING_STUDY_DUAL_AODS[0], SETTING_STUDY_DUAL_AODS[1]
            dual_series = []
            x_pts = []
            for placement, compile_idx, label in t_study_entries:
                if compile_idx >= len(_T_COMPILE_ABLATION_GRID):
                    continue
                g_back = _aggregate_t_setting_study(
                    t_setting_layer,
                    triple,
                    _T_COMPILE_ABLATION_GRID[compile_idx],
                    placement,
                    cd,
                    int(aod_back),
                )
                g_front = _aggregate_t_setting_study(
                    t_setting_layer,
                    triple,
                    _T_COMPILE_ABLATION_GRID[compile_idx],
                    placement,
                    cd,
                    int(aod_front),
                )
                if g_back is None and g_front is None:
                    continue
                key = (placement, compile_idx)
                dual_series.append((t_setting_colors[key], label, g_back, g_front))
            x_pts = _draw_setting_study_dual_aod_bars(ax, dual_series)
            x_pos = {int(nq): float(i) for i, nq in enumerate(x_pts)}
        elif setting_chart == "bar":
            bar_series = [
                (t_setting_colors[key], label, grouped)
                for key, label, grouped in prepared
            ]
            x_pts = _draw_setting_study_grouped_bars(ax, bar_series)
            x_pos = {int(nq): float(i) for i, nq in enumerate(x_pts)}
        else:
            raise ValueError(
                f"Unsupported setting_chart for T setting-study: {setting_chart}"
            )
        _finalize_panel(ax, x_pts, theory="t", x_positions=x_pos)

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
        _apply_panel_y_ticks(ax, nbins=STAR_AOD_COMPARISON_Y_NBINS)

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
                color=t_setting_colors[(placement, compile_idx)],
                label=label,
                setting_chart=setting_chart,
            )
            for placement, compile_idx, label in t_study_entries
            if (placement, compile_idx) in t_setting_colors
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
                aod_color_map=star_aod_colors,
                figure_title="STAR architecture",
                out_path=star_path,
                xlabel="Number of Qubits/Factories",
                legend_y_blend=0.72,
                panel_hspace=0.8,
                bottom_axis_y_shift=0.14,
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
                aod_color_map=t_aod_colors,
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
