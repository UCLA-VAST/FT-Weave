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
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MaxNLocator,
)
from matplotlib.transforms import Bbox
from src.t_cultivation.config import STAGE_1_SUCCESS_RATE

_FIG_FONT_SIZE = 28
_FIG_AXIS_DARK = "0.12"

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
    # ("col_based", False, False, 0, True, False),  # Baseline + opt mov + opt micro arch
    ("col_based", True, False, 2, False, True),  # Greedy with opt micro arch
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
AOD_COMPARISON_STEP_PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (2, 3), (3, 5))
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
# Fixed hidden band on STAR compilation-strategy broken y-axis (all bar tops stay visible).
STAR_SETTING_STUDY_Y_GAP_LO: float = 2000.0
STAR_SETTING_STUDY_Y_GAP_HI: float = 3000.0
STAR_SETTING_STUDY_Y_CROP_GAP_RATIO: float = 1.25
STAR_SETTING_STUDY_Y_CROP_PAD_FRAC: float = 0.1
# STAR AOD twin-axis side labels (figure coords: offset left from panel edge ax_aod_bot.x0).
# Increase BASELINE to move green "Baseline" further left; increase EXEC_TIME to move black
# "Execution time" further left (keep EXEC_TIME > BASELINE so black stays left of green).
STAR_AOD_BASELINE_LABEL_X: float = 0.06
STAR_AOD_EXEC_TIME_LABEL_X: float = 0.082
# STAR compilation-strategy legend (ncol=3): col0 Baseline/+ Routing, col1 Greedy/High,
# col2 invisible spacer (width via STAR_COMPILE_LEGEND_COL3_WIDTH_CHARS); microarch = fig.text.
STAR_COMPILE_LEGEND_COLUMNSPACING: float = 1.03
# Invisible col-3 spacer label width (non-breaking spaces) to widen the legend frame.
STAR_COMPILE_LEGEND_COL3_WIDTH_CHARS: int = 10
STAR_COMPILE_LEGEND_X_SHIFT: float = (
    0.0  # negative = shift whole legend left after centering
)
STAR_COMPILE_MICROARCH_DX: float = 0.18
STAR_COMPILE_MICROARCH_DY: float = 0.0
STAR_COMPILE_LEGEND_BORDERPAD: float = 0.55  # larger when microarch fig.text is drawn
STAR_AOD_BREAK_PAD_FRAC: float = 0.03
STAR_AOD_BREAK_NQUBIT_ANCHOR: int = 64
STAR_AOD_BREAK_ANCHOR_AOD: int = 5
# Broken-axis layout (tight gap between segments; sparse upper ticks).
STAR_BROKEN_AXIS_HEIGHT_RATIOS: tuple[int, int] = (2, 5)
STAR_BROKEN_INNER_HSPACE: float = 0.02
STAR_BROKEN_TOP_Y_NBINS: int = 3
STAR_BROKEN_Y_TOP_PAD_FRAC: float = 0.04
AOD_BOTTOM_LEGEND_BASE_Y: float = -0.028
STAR_SETTING_LEGEND_GAP_BLEND: float = 0.55
T_SETTING_LEGEND_GAP_BLEND: float = 0.48
FIG_SAVE_PAD_INCHES: float = 0.18
STAR_SETTING_AOD_PANEL_HSPACE: float = 0.52
STAR_AOD_TWIN_Y_PAD_FRAC: float = 0.10
STAR_AOD_BOTTOM_AXIS_Y_SHIFT: float = 0.055
SETTING_STUDY_N_QUBITS: tuple[int, ...] = (16, 36, 64, 100)
STAR_T_MERGED_FIGSIZE: tuple[float, float] = (18.4, 11.5)
STAR_T_MERGED_WSPACE: float = 0.34
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
# Ideal per-leg movement time: half of measured CNOT circuit time per leg.
_MOVEMENT_OPTIMAL_CNOT_LEG_FRACTION: float = 0.5

SHOW_T_CULTIVATION_D13 = False
SHOW_EXPECTED_TIME_LINE: bool = False
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
    "Greedy exec.",
    "High parallelism exec.",
    # " + Lookahead angle prep.",  # "(d) + Lookahead angle prep.",
    # "Async. execution",
]
# col_based strategies (indices 2–3) both use optimized microarchitecture.
_STAR_MICROARCH_OPT_SETTING_INDICES: frozenset[int] = frozenset({2, 3})
_STAR_COMPILE_MICROARCH_NOTE = "+ Microarch. opt."
_LEGEND_COL3_SPACER_MARKER = (
    "\u00a0"  # prefix for invisible width-reserving col-3 labels
)


def _star_legend_col3_spacer_label() -> str:
    """Invisible label text that reserves horizontal space in legend column 3."""
    n = max(
        int(STAR_COMPILE_LEGEND_COL3_WIDTH_CHARS),
        len(_STAR_COMPILE_MICROARCH_NOTE) + 2,
    )
    return f"{_LEGEND_COL3_SPACER_MARKER}{'\u00a0' * n}"


def _is_legend_col3_spacer_label(label: str) -> bool:
    return str(label).startswith(_LEGEND_COL3_SPACER_MARKER)


# T AOD line panel: emphasize AOD = 5 (light ramp color is easy to miss).
T_AOD_EMPHASIS_AOD: int = 5
T_AOD_EMPHASIS_LINESTYLE: tuple[int, tuple[int, int]] = (0, (5, 2))


def _star_setting_study_entries() -> list[tuple[int, str]]:
    """(``SETTINGS`` index, legend label) for the STAR setting-study panel."""
    n = min(len(SETTINGS), len(_STAR_ABLATION_LABELS))
    return [(i, _STAR_ABLATION_LABELS[i]) for i in range(n)]


def _star_aod_axis_label(setting_idx: int) -> str:
    """Short strategy name for the STAR AOD panel twin y-axis (no microarch suffix)."""
    if 0 <= int(setting_idx) < len(_STAR_ABLATION_LABELS):
        return _STAR_ABLATION_LABELS[int(setting_idx)]
    return "Best strategy"


def _star_setting_study_legend_handles(
    star_setting_colors: dict[int, tuple],
    star_study_entries: list[tuple[int, str]],
) -> list:
    """Legend patches for STAR compilation strategies (microarch note placed separately)."""
    return [
        _setting_study_legend_handle(
            color=star_setting_colors[setting_idx],
            label=label,
        )
        for setting_idx, label in star_study_entries
        if setting_idx in star_setting_colors
    ]


def _star_compile_strategy_legend_cells(
    star_study_entries: list[tuple[int, str]],
) -> tuple[list[tuple[int | None, str]], int, bool]:
    """Return (cells, ncol, show_microarch_note).

    Matplotlib fills each column top-to-bottom (index = col * nrows + row). For ncol=3::
        col0: Baseline, + Routing opt.
        col1: Greedy exec. (top), High parallelism exec. (bottom)
        col2: (placeholder), (placeholder)
    Handle order: [Baseline, + Routing, Greedy, High, _, _].
    """
    by_label = {label: int(idx) for idx, label in star_study_entries}
    show_microarch = bool(
        _STAR_MICROARCH_OPT_SETTING_INDICES.intersection(by_label.values())
    )

    def _cell(lbl: str | None, *, spacer: bool = False) -> tuple[int | None, str]:
        if spacer:
            return (None, _star_legend_col3_spacer_label())
        if lbl is None or not str(lbl).strip():
            return (None, _star_legend_col3_spacer_label())
        if lbl in by_label:
            return (by_label[lbl], lbl)
        return (None, _star_legend_col3_spacer_label())

    if not show_microarch:
        # ncol=2: [Baseline, + Routing, Greedy, High] → col0 | col1 (Greedy top)
        return (
            [
                _cell("Baseline"),
                _cell(" + Routing opt."),
                _cell("Greedy exec."),
                _cell("High parallelism exec."),
            ],
            2,
            False,
        )

    return (
        [
            _cell("Baseline"),
            _cell(" + Routing opt."),
            _cell("Greedy exec."),
            _cell("High parallelism exec."),
            _cell(None, spacer=True),
            _cell(None, spacer=True),
        ],
        3,
        True,
    )


def _star_compile_strategy_legend_handle(
    setting_idx: int | None,
    label: str,
    *,
    star_setting_colors: dict[int, tuple],
) -> Line2D | Patch:
    if setting_idx is None and _is_legend_col3_spacer_label(label):
        return Line2D(
            [],
            [],
            linestyle="None",
            marker=None,
            color="none",
            label=label,
            alpha=0.0,
        )
    if setting_idx is None or not label.strip():
        return Line2D(
            [],
            [],
            linestyle="None",
            marker=None,
            color="none",
            label=" ",
        )
    return _setting_study_legend_handle(
        color=star_setting_colors[int(setting_idx)],
        label=label,
    )


def _apply_star_compile_strategy_legend(
    fig,
    *,
    star_setting_colors: dict[int, tuple],
    star_study_entries: list[tuple[int, str]],
    legend_x: float,
    legend_y: float,
    legend_fs: float,
    legend_handlelength: float,
):
    """STAR strategy legend: 2 rows; microarch fig.text between Greedy and High."""
    cells, legend_ncol, show_microarch = _star_compile_strategy_legend_cells(
        star_study_entries
    )
    handles = [
        _star_compile_strategy_legend_handle(
            setting_idx,
            label,
            star_setting_colors=star_setting_colors,
        )
        for setting_idx, label in cells
    ]
    leg = fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(legend_x, legend_y),
        ncol=int(legend_ncol),
        fontsize=legend_fs,
        frameon=True,
        handlelength=legend_handlelength,
        handletextpad=0.35,
        labelspacing=0.42,
        columnspacing=float(STAR_COMPILE_LEGEND_COLUMNSPACING),
        borderpad=float(STAR_COMPILE_LEGEND_BORDERPAD) if show_microarch else 0.35,
    )
    for text in leg.get_texts():
        if _is_legend_col3_spacer_label(text.get_text()):
            text.set_alpha(0.0)
    micro_text = None
    if show_microarch:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = {t.get_text().strip(): t for t in leg.get_texts()}
        greedy_t = texts.get("Greedy exec.")
        high_t = texts.get("High parallelism exec.")
        if greedy_t is not None and high_t is not None:
            bb_g = greedy_t.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            bb_h = high_t.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            # Greedy (top) and High (bottom) share column 1; microarch sits between them.
            micro_x = 0.5 * (float(bb_g.x0) + float(bb_g.x1)) + float(
                STAR_COMPILE_MICROARCH_DX
            )
            micro_y = 0.5 * (float(bb_h.y1) + float(bb_g.y0)) + float(
                STAR_COMPILE_MICROARCH_DY
            )
            micro_text = fig.text(
                micro_x,
                micro_y,
                _STAR_COMPILE_MICROARCH_NOTE,
                ha="center",
                va="center",
                fontsize=legend_fs,
                color=_FIG_AXIS_DARK,
                zorder=25,
            )
            micro_text.set_clip_on(False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    if micro_text is not None:
        bb_txt = micro_text.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        bb = Bbox.union([bb, bb_txt])
    delta_x = (
        float(legend_x)
        - 0.5 * (float(bb.x0) + float(bb.x1))
        + float(STAR_COMPILE_LEGEND_X_SHIFT)
    )
    leg.set_bbox_to_anchor(
        (float(legend_x) + delta_x, float(legend_y)),
        transform=fig.transFigure,
    )
    leg.set_zorder(2)
    if leg.get_frame() is not None:
        leg.get_frame().set_zorder(2)
    if micro_text is not None:
        mx, my = micro_text.get_position()
        micro_text.set_position((float(mx) + delta_x, float(my)))
        fig.add_artist(micro_text)
    return leg, micro_text


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
    *,
    aod_linestyles: dict[int, str | tuple] | None = None,
) -> list[Line2D]:
    """Legend rows: colored marker + AOD number (reference figure style)."""
    handles: list[Line2D] = []
    for aod in sorted(aod_color_map):
        edge = mcolors.to_rgb(aod_color_map[int(aod)])
        face = _marker_facecolor(edge)
        ls = (aod_linestyles or {}).get(int(aod), "-")
        handles.append(
            Line2D(
                [0],
                [0],
                color=edge,
                linestyle=ls,
                linewidth=2.0 if ls != "-" else 0.0,
                marker="o",
                markersize=_LINE_MARKERSIZE - 1,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=_LINE_MARKER_EDGEWIDTH
                + (0.5 if int(aod) == int(T_AOD_EMPHASIS_AOD) else 0.0),
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
        "n_cnot",
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


def _log_scale_tick_values(y_lo: float, y_hi: float) -> list[float]:
    """Major tick positions for log axes using 1-2-5 spacing per decade."""
    if y_lo <= 0.0 or y_hi <= y_lo:
        return []
    log_lo = int(math.floor(math.log10(y_lo)))
    log_hi = int(math.ceil(math.log10(y_hi)))
    ticks: list[float] = []
    for exp in range(log_lo, log_hi + 1):
        for mult in (1, 2, 5):
            v = float(mult) * (10.0**exp)
            if float(y_lo) <= v <= float(y_hi):
                ticks.append(v)
    if len(ticks) < 4:
        for exp in range(log_lo, log_hi + 1):
            for mult in range(1, 10):
                v = float(mult) * (10.0**exp)
                if float(y_lo) <= v <= float(y_hi) and v not in ticks:
                    ticks.append(v)
    return sorted(ticks)


def _format_log_tick_value(v: float, _pos) -> str:
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:g}"


def _apply_setting_panel_log_ticks(ax) -> None:
    """Dense 1-2-5 log ticks on compilation-strategy panels."""
    y_lo, y_hi = ax.get_ylim()
    ticks = _log_scale_tick_values(float(y_lo), float(y_hi))
    if len(ticks) < 2:
        ax.yaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
        )
        return
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick_value))


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
        "n_cnot",
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


def _resolve_setting_aod_axes(
    ax_setting_parent,
    ax_aod_parent,
) -> dict:
    """Resolve broken sub-axes and twin AOD axis for one setting/AOD column."""
    setting_axes = getattr(ax_setting_parent, "_broken_setting_axes", None)
    if setting_axes:
        ax_setting_top, ax_setting_bot = setting_axes
    else:
        ax_setting_top = ax_setting_bot = ax_setting_parent

    aod_axes = getattr(ax_aod_parent, "_broken_aod_axes", None)
    if aod_axes:
        ax_aod_top, ax_aod_bot = aod_axes
    else:
        ax_aod_top = ax_aod_bot = ax_aod_parent

    return {
        "setting_parent": ax_setting_parent,
        "aod_parent": ax_aod_parent,
        "setting_top": ax_setting_top,
        "setting_bot": ax_setting_bot,
        "aod_top": ax_aod_top,
        "aod_bot": ax_aod_bot,
        "aod_broken": aod_axes is not None,
        "aod_twin_right": getattr(ax_aod_parent, "_star_aod_twin_right", None),
    }


def _style_setting_aod_column_pre_layout(
    resolved: dict,
    *,
    column_titles: tuple[str, str],
    architecture_label: str,
    xlabel: str,
    y_axis_thousands: bool,
    setting_panel_log_scale: bool = False,
) -> None:
    panel_title_fs = _FIG_FONT_SIZE - 1
    ax_setting_top = resolved["setting_top"]
    ax_setting_bot = resolved["setting_bot"]
    ax_aod_top = resolved["aod_top"]
    ax_aod_bot = resolved["aod_bot"]
    aod_axes = resolved["aod_broken"]
    aod_twin_right = resolved["aod_twin_right"]

    ax_setting_top.set_title(column_titles[0], fontsize=panel_title_fs, pad=8)
    aod_title_ax = (
        ax_aod_top
        if architecture_label == "STAR architecture" and aod_axes
        else ax_aod_bot
    )
    aod_title_ax.set_title(column_titles[1], fontsize=panel_title_fs, pad=8)
    if aod_axes and aod_title_ax is not ax_aod_bot:
        ax_aod_bot.set_title("")
    elif aod_axes and aod_title_ax is not ax_aod_top:
        ax_aod_top.set_title("")

    if setting_panel_log_scale:
        setting_y_label = "Execution time (log)"
    elif y_axis_thousands:
        setting_y_label = "Execution time (×10³)"
    else:
        setting_y_label = "Execution time"
    aod_y_label = "Execution time (×10³)" if y_axis_thousands else "Execution time"
    ax_setting_bot.set_ylabel(
        setting_y_label, fontsize=_FIG_FONT_SIZE, labelpad=2, color=_FIG_AXIS_DARK
    )
    twin_style = getattr(resolved["aod_parent"], "_star_aod_twin_style", None)
    twin_ylabel = getattr(resolved["aod_parent"], "_star_aod_twin_ylabel", None)
    if aod_twin_right is not None and twin_style is not None:
        _baseline_label, sync_rgb, best_rgb = twin_style
        ax_aod_bot.set_ylabel("")
        ax_aod_bot.tick_params(axis="y", colors=sync_rgb)
        aod_twin_right.set_ylabel(
            twin_ylabel or _baseline_label,
            fontsize=_FIG_FONT_SIZE,
            labelpad=2,
            color=best_rgb,
        )
        aod_twin_right.tick_params(axis="y", colors=best_rgb)
    elif aod_twin_right is not None:
        ax_aod_bot.set_ylabel(
            aod_y_label, fontsize=_FIG_FONT_SIZE, labelpad=2, color=_FIG_AXIS_DARK
        )
        aod_twin_right.set_ylabel("")
    else:
        ax_aod_bot.set_ylabel(
            aod_y_label, fontsize=_FIG_FONT_SIZE, labelpad=2, color=_FIG_AXIS_DARK
        )
    if y_axis_thousands:
        _apply_y_axis_thousands([ax_aod_bot])
        if not setting_panel_log_scale:
            _apply_y_axis_thousands([ax_setting_bot])
    for _ax in (ax_aod_top, ax_aod_bot):
        _apply_panel_y_ticks(_ax, nbins=T_CULTIVATION_Y_NBINS)
    if setting_panel_log_scale:
        for _ax in (ax_setting_top, ax_setting_bot):
            _apply_setting_panel_log_ticks(_ax)
    else:
        for _ax in (ax_setting_top, ax_setting_bot):
            _apply_panel_y_ticks(_ax, nbins=T_CULTIVATION_Y_NBINS)
    for _ax in (ax_setting_top, ax_setting_bot, ax_aod_top, ax_aod_bot):
        _ax.tick_params(axis="both", which="major", pad=1)
    ax_aod_bot.set_xlabel(
        xlabel, fontsize=_FIG_FONT_SIZE, labelpad=0, color=_FIG_AXIS_DARK
    )


def _finalize_setting_aod_column_post_layout(
    fig,
    resolved: dict,
    *,
    setting_legend_handles: list,
    aod_legend_handles: list | None,
    legend_fs: float,
    legend_handlelength: float,
    is_star: bool,
    legend_y_blend: float,
    setting_legend_ncol: int,
    bottom_axis_y_shift: float,
) -> None:
    ax_setting_parent = resolved["setting_parent"]
    ax_aod_parent = resolved["aod_parent"]
    ax_setting_top = resolved["setting_top"]
    ax_setting_bot = resolved["setting_bot"]
    ax_aod_top = resolved["aod_top"]
    ax_aod_bot = resolved["aod_bot"]
    aod_axes = resolved["aod_broken"]
    aod_twin_right = resolved["aod_twin_right"]

    if aod_axes:
        _apply_aod_comparison_x_ticks(ax_aod_top, ax_setting_parent)
        ax_aod_top.set_xticklabels([])

    twin_style = getattr(ax_aod_parent, "_star_aod_twin_style", None)
    if aod_twin_right is not None and is_star and twin_style is not None:
        baseline_label, sync_rgb, best_rgb = twin_style
        _apply_dark_axis_style(ax_aod_bot, include_x=False, include_y=False)
        ax_aod_bot.tick_params(axis="y", colors=sync_rgb)
        aod_twin_right.tick_params(axis="y", colors=best_rgb)
        aod_twin_right.tick_params(axis="x", bottom=False, labelbottom=False)
        aod_pos = ax_aod_bot.get_position()
        y_mid = 0.4 * (aod_pos.y0 + aod_pos.y1)
        fig.text(
            aod_pos.x0 - STAR_AOD_BASELINE_LABEL_X,
            y_mid,
            baseline_label,
            rotation=90,
            va="center",
            ha="center",
            color=sync_rgb,
            fontsize=_FIG_FONT_SIZE,
        )
        fig.text(
            aod_pos.x0 - STAR_AOD_EXEC_TIME_LABEL_X,
            y_mid,
            "Execution time",
            rotation=90,
            va="center",
            ha="center",
            color=_FIG_AXIS_DARK,
            fontsize=_FIG_FONT_SIZE,
        )
    else:
        _apply_dark_axis_style(ax_aod_bot, include_x=True, include_y=True)
        if aod_twin_right is not None:
            _apply_dark_axis_style(aod_twin_right, include_x=False, include_y=True)
            aod_twin_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_setting_bot.tick_params(axis="both", colors=_FIG_AXIS_DARK)
    ax_setting_top.tick_params(axis="both", colors=_FIG_AXIS_DARK)

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
        if aod_twin_right is not None:
            twin_pos = aod_twin_right.get_position()
            shifted_bot = ax_aod_bot.get_position()
            aod_twin_right.set_position(
                [
                    twin_pos.x0,
                    shifted_bot.y0,
                    twin_pos.width,
                    shifted_bot.height,
                ]
            )

    _commit_aod_x_axis(
        ax_aod_bot,
        ax_setting_parent,
        twin_ax=aod_twin_right,
    )

    setting_bbox = ax_setting_parent.get_position()
    aod_bbox = ax_aod_parent.get_position()
    legend_x = 0.5 * (setting_bbox.x0 + setting_bbox.x1)
    if is_star:
        legend_y = aod_bbox.y1 + STAR_SETTING_LEGEND_GAP_BLEND * (
            setting_bbox.y0 - aod_bbox.y1
        )
    else:
        top_pos = setting_bbox
        bot_pos = ax_aod_bot.get_position()
        blend = float(np.clip(legend_y_blend, 0.0, 1.0))
        legend_y = (1.0 - blend) * top_pos.y0 + blend * bot_pos.y1

    star_legend_meta = getattr(ax_setting_parent, "_star_legend_meta", None)
    if is_star and star_legend_meta is not None:
        strategy_leg, microarch_text = _apply_star_compile_strategy_legend(
            fig,
            star_setting_colors=star_legend_meta["colors"],
            star_study_entries=star_legend_meta["entries"],
            legend_x=legend_x,
            legend_y=legend_y,
            legend_fs=legend_fs,
            legend_handlelength=legend_handlelength,
        )
        if SHOW_EXPECTED_TIME_LINE:
            expected_leg = fig.legend(
                handles=[_expected_time_legend_handle()],
                loc="center",
                bbox_to_anchor=(legend_x, legend_y - 0.058),
                fontsize=legend_fs,
                frameon=True,
                handlelength=legend_handlelength,
            )
            fig.add_artist(strategy_leg)
            fig.add_artist(expected_leg)
        if microarch_text is not None:
            fig.add_artist(microarch_text)
    else:
        combined_legend = list(setting_legend_handles)
        if SHOW_EXPECTED_TIME_LINE:
            combined_legend.append(_expected_time_legend_handle())
        fig.legend(
            handles=combined_legend,
            loc="center",
            bbox_to_anchor=(legend_x, legend_y),
            ncol=setting_legend_ncol,
            fontsize=legend_fs,
            frameon=True,
            handlelength=legend_handlelength,
            handletextpad=0.35,
            labelspacing=0.35,
            columnspacing=0.6,
        )

    if aod_legend_handles:
        fig.legend(
            handles=aod_legend_handles,
            loc="upper left",
            bbox_to_anchor=(setting_bbox.x0, setting_bbox.y1 - 0.02),
            fontsize=legend_fs,
            frameon=True,
            handlelength=legend_handlelength,
            handletextpad=0.35,
            labelspacing=0.35,
        )

    aod_bbox_center = 0.5 * (aod_bbox.x0 + aod_bbox.x1)
    _apply_aod_bottom_legend(
        fig, ax_aod_parent, legend_fs=legend_fs, bbox_x=aod_bbox_center
    )


def _align_merged_t_aod_to_star(
    axes,
    star_resolved: dict,
) -> None:
    """Move T AOD panel down to match STAR AOD (keep STAR at standalone position)."""
    star_pos = axes[1, 0].get_position()
    t_pos = axes[1, 1].get_position()
    axes[1, 1].set_position([t_pos.x0, star_pos.y0, t_pos.width, star_pos.height])
    aod_twin_right = star_resolved.get("aod_twin_right")
    if aod_twin_right is not None:
        twin_pos = aod_twin_right.get_position()
        aod_twin_right.set_position(
            [twin_pos.x0, star_pos.y0, twin_pos.width, star_pos.height]
        )


def _merged_bottom_margin(star_aod_parent, t_aod_parent) -> float:
    star_rows = len(getattr(star_aod_parent, "_aod_bottom_legend_rows", None) or [])
    t_rows = len(getattr(t_aod_parent, "_aod_bottom_legend_rows", None) or [])
    max_rows = max(star_rows, t_rows)
    if max_rows >= 2:
        return 0.21
    if max_rows >= 1:
        return 0.16
    return 0.10


def _save_star_t_merged_setting_aod_figure(
    *,
    draw_star_setting,
    draw_star_aod,
    draw_t_setting,
    draw_t_aod,
    star_setting_legend_handles: list,
    t_setting_legend_handles: list,
    star_column_titles: tuple[str, str],
    t_column_titles: tuple[str, str],
    aod_legend_handles: list | None,
    out_path: str,
    xlabel: str = "Number of Qubits/Factories",
    legend_handlelength: float = 1.0,
    panel_hspace: float = STAR_SETTING_AOD_PANEL_HSPACE,
    figure_top: float = 0.90,
    setting_panel_log_scale: bool = False,
) -> None:
    """Two columns (STAR left, T-cultivation right), each with setting + AOD rows."""
    fig, axes = plt.subplots(2, 2, figsize=STAR_T_MERGED_FIGSIZE, squeeze=False)
    draw_star_setting(axes[0, 0])
    draw_star_aod(axes[1, 0])
    draw_t_setting(axes[0, 1])
    draw_t_aod(axes[1, 1])

    star_resolved = _resolve_setting_aod_axes(axes[0, 0], axes[1, 0])
    t_resolved = _resolve_setting_aod_axes(axes[0, 1], axes[1, 1])
    legend_fs = _FIG_FONT_SIZE - 4

    _style_setting_aod_column_pre_layout(
        star_resolved,
        column_titles=star_column_titles,
        architecture_label="STAR architecture",
        xlabel=xlabel,
        y_axis_thousands=False,
        setting_panel_log_scale=setting_panel_log_scale,
    )
    _style_setting_aod_column_pre_layout(
        t_resolved,
        column_titles=t_column_titles,
        architecture_label="T-cultivation",
        xlabel=xlabel,
        y_axis_thousands=True,
        setting_panel_log_scale=setting_panel_log_scale,
    )

    header_fs = _FIG_FONT_SIZE
    fig.text(
        0.25, 0.975, "STAR architecture", ha="center", va="top", fontsize=header_fs
    )
    fig.text(0.75, 0.975, "T-cultivation", ha="center", va="top", fontsize=header_fs)

    bottom_margin = _merged_bottom_margin(axes[1, 0], axes[1, 1])
    fig.subplots_adjust(
        top=figure_top,
        bottom=bottom_margin,
        left=0.07,
        right=0.97,
        hspace=panel_hspace,
        wspace=STAR_T_MERGED_WSPACE,
    )

    _finalize_setting_aod_column_post_layout(
        fig,
        star_resolved,
        setting_legend_handles=star_setting_legend_handles,
        aod_legend_handles=aod_legend_handles,
        legend_fs=legend_fs,
        legend_handlelength=legend_handlelength,
        is_star=True,
        legend_y_blend=0.72,
        setting_legend_ncol=3,
        bottom_axis_y_shift=STAR_AOD_BOTTOM_AXIS_Y_SHIFT,
    )
    _align_merged_t_aod_to_star(axes, star_resolved)
    _finalize_setting_aod_column_post_layout(
        fig,
        t_resolved,
        setting_legend_handles=t_setting_legend_handles,
        aod_legend_handles=aod_legend_handles,
        legend_fs=legend_fs,
        legend_handlelength=legend_handlelength,
        is_star=False,
        legend_y_blend=T_SETTING_LEGEND_GAP_BLEND,
        setting_legend_ncol=2,
        bottom_axis_y_shift=0.0,
    )

    _apply_aod_comparison_x_ticks(axes[1, 0], axes[0, 0])
    _commit_aod_x_axis(
        axes[1, 0],
        axes[0, 0],
        twin_ax=star_resolved.get("aod_twin_right"),
    )
    _commit_aod_x_axis(axes[1, 1], axes[0, 1])

    _save_prx_figure(fig, out_path)


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
    setting_legend_ncol: int = 2,
    figure_top: float = 0.92,
    setting_panel_log_scale: bool = False,
) -> None:
    """Two rows, one column: setting study (top) and AOD sweep (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 11.5), squeeze=False)
    draw_setting(axes[0, 0])
    draw_aod(axes[1, 0])

    resolved = _resolve_setting_aod_axes(axes[0, 0], axes[1, 0])
    legend_fs = _FIG_FONT_SIZE - 4
    _style_setting_aod_column_pre_layout(
        resolved,
        column_titles=column_titles,
        architecture_label=figure_title,
        xlabel=xlabel,
        y_axis_thousands=y_axis_thousands,
        setting_panel_log_scale=setting_panel_log_scale,
    )

    fig.suptitle(figure_title, fontsize=_FIG_FONT_SIZE + 1, y=0.98)
    aod_legend_rows = getattr(axes[1, 0], "_aod_bottom_legend_rows", None) or []
    bottom_margin = (
        0.22 if len(aod_legend_rows) >= 2 else (0.17 if aod_legend_rows else 0.12)
    )
    if resolved["aod_twin_right"] is not None and bottom_axis_y_shift:
        bottom_margin += 0.03
    right_margin = 0.84 if resolved["aod_twin_right"] is not None else 0.88
    fig.tight_layout(
        rect=(0.08, bottom_margin, right_margin + 0.04, 0.94),
        pad=0.10,
        h_pad=0.35,
    )
    fig.subplots_adjust(
        top=figure_top,
        bottom=bottom_margin,
        left=0.12,
        right=right_margin,
        hspace=panel_hspace,
    )
    _finalize_setting_aod_column_post_layout(
        fig,
        resolved,
        setting_legend_handles=setting_legend_handles,
        aod_legend_handles=aod_legend_handles,
        legend_fs=legend_fs,
        legend_handlelength=legend_handlelength,
        is_star=figure_title == "STAR architecture",
        legend_y_blend=legend_y_blend,
        setting_legend_ncol=setting_legend_ncol,
        bottom_axis_y_shift=bottom_axis_y_shift,
    )
    _commit_aod_x_axis(
        resolved["aod_bot"],
        axes[0, 0],
        twin_ax=resolved.get("aod_twin_right"),
    )

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

    _save_prx_figure(fig, out_path)


def _normalize_placement_name(placement: str) -> str:
    return str(placement).strip()


def _aggregate_total_time_by_qubits(df: pd.DataFrame) -> pd.DataFrame:
    """Per *n_qubits*: mean/min/max of ``total_time``."""
    grouped = (
        df.groupby("n_qubits")[["total_time"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    grouped["total_time_std"] = grouped["total_time_std"].fillna(0.0)
    return grouped


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
    grouped = _aggregate_total_time_by_qubits(df)
    return grouped if not grouped.empty else None


def _aggregate_t_setting_study(
    layer_df: pd.DataFrame,
    triple: tuple[int, float, int],
    setting: tuple[str, bool, bool, bool],
    code_distance: int,
    aod: int,
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
    grouped = _aggregate_total_time_by_qubits(work)
    return grouped if not grouped.empty else None


def _build_star_setting_dual_series(
    dfs_dict_micro: dict[str, pd.DataFrame],
    entries: list[tuple[int, str]],
    setting_colors: dict[int, tuple],
    code_distance: int,
    *,
    compare_aods: tuple[int, ...] = SETTING_STUDY_DUAL_AODS,
) -> list[tuple]:
    """(color, label, grouped_aod0, grouped_aod1) for the setting-study bar panel."""
    if len(compare_aods) < 2:
        return []
    aod_back, aod_front = int(compare_aods[0]), int(compare_aods[1])
    dual_series: list[tuple] = []
    for setting_idx, label in entries:
        if setting_idx >= len(SETTINGS):
            continue
        g_back = _aggregate_star_setting_study(
            dfs_dict_micro,
            SETTINGS[setting_idx],
            code_distance,
            aod_back,
        )
        g_front = _aggregate_star_setting_study(
            dfs_dict_micro,
            SETTINGS[setting_idx],
            code_distance,
            aod_front,
        )
        if g_back is None and g_front is None:
            continue
        dual_series.append((setting_colors[setting_idx], label, g_back, g_front))
    return dual_series


def _build_t_setting_dual_series(
    layer_df: pd.DataFrame,
    triple: tuple[int, float, int],
    entries: list[tuple[int, str]],
    setting_colors: dict[int, tuple],
    code_distance: int,
    *,
    compare_aods: tuple[int, ...] = SETTING_STUDY_DUAL_AODS,
) -> list[tuple]:
    """(color, label, grouped_aod0, grouped_aod1) for the T setting-study bar panel."""
    if len(compare_aods) < 2:
        return []
    aod_back, aod_front = int(compare_aods[0]), int(compare_aods[1])
    dual_series: list[tuple] = []
    for setting_idx, label in entries:
        if setting_idx >= len(_T_SETTING_ABLATION_GRID):
            continue
        placement, tr, dm, rs = _T_SETTING_ABLATION_GRID[setting_idx]
        g_back = _aggregate_t_setting_study(
            layer_df,
            triple,
            (placement, tr, dm, rs),
            code_distance,
            aod_back,
        )
        g_front = _aggregate_t_setting_study(
            layer_df,
            triple,
            (placement, tr, dm, rs),
            code_distance,
            aod_front,
        )
        if g_back is None and g_front is None:
            continue
        dual_series.append((setting_colors[setting_idx], label, g_back, g_front))
    return dual_series


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
    """Break just above non-dominant series so mean±std stays visible on bottom or top."""
    v_lo, v_hi = y_extents[vanilla_idx]
    other_hi = [
        hi for idx, (_lo, hi) in y_extents.items() if int(idx) != int(vanilla_idx)
    ]
    zoom_hi = max(other_hi) if other_hi else v_lo
    break_y = zoom_hi * (1.0 + pad_frac)
    break_y = max(break_y, v_lo * (1.0 + pad_frac * 0.5))
    break_y = min(break_y, v_hi * (1.0 - pad_frac))
    if break_y <= zoom_hi:
        break_y = (zoom_hi + v_hi) * 0.5
    return float(break_y)


def _broken_axis_top_hi(
    y_extents: list[tuple[float, float]],
    y_break: float,
    *,
    pad_frac: float = STAR_BROKEN_Y_TOP_PAD_FRAC,
) -> float:
    """Upper y-limit so every series' mean+std peak remains inside the top segment."""
    peak = max((hi for _lo, hi in y_extents), default=float(y_break))
    return float(max(peak * (1.0 + pad_frac), y_break * (1.0 + pad_frac)))


def _finalize_star_bar_broken_limits(
    y_extents: dict[int, tuple[float, float]],
    *,
    vanilla_idx: int = STAR_SETTING_STUDY_VANILLA_IDX,
    dual_top_hi: float | None = None,
) -> tuple[float, float]:
    """Legacy bar-chart break/top limits (0.985 caps on 2nd-tallest and bar tops)."""
    y_break = _compute_star_setting_break_y(y_extents, vanilla_idx=vanilla_idx)
    y_hi_sorted = sorted(
        (float(hi) for (_lo, hi) in y_extents.values()),
        reverse=True,
    )
    if len(y_hi_sorted) >= 2:
        y_break = min(y_break, y_hi_sorted[1] * 0.985)
    if dual_top_hi is not None and dual_top_hi > 0.0:
        y_break = min(y_break, float(dual_top_hi) * 0.985)
        y_top_hi = float(dual_top_hi)
    else:
        y_top_hi = float(y_extents[vanilla_idx][1])
    y_pad = 0.08 * max(y_top_hi - y_break, 1e-9)
    return float(y_break), float(y_top_hi + y_pad)


def _finalize_star_bar_fixed_gap_limits(
    y_extents: dict[int, tuple[float, float]],
    *,
    dual_top_hi: float | None = None,
    gap_lo: float = STAR_SETTING_STUDY_Y_GAP_LO,
    gap_hi: float = STAR_SETTING_STUDY_Y_GAP_HI,
    pad_frac: float = STAR_SETTING_STUDY_Y_CROP_PAD_FRAC,
) -> tuple[float, float]:
    """Broken-axis limits with hidden band [gap_lo, gap_hi]; top includes all bar peaks."""
    peak_vals = [float(hi) for _lo, hi in y_extents.values()]
    if dual_top_hi is not None and float(dual_top_hi) > 0.0:
        peak_vals.append(float(dual_top_hi))
    peak = max(peak_vals) if peak_vals else float(gap_hi)
    y_top = max(float(peak) * (1.0 + pad_frac), float(gap_hi) * (1.0 + pad_frac * 0.5))
    return float(gap_lo), float(y_top)


def _star_setting_fixed_gap_break_needed(
    y_extents: dict[int, tuple[float, float]],
    *,
    vanilla_idx: int = STAR_SETTING_STUDY_VANILLA_IDX,
    gap_hi: float = STAR_SETTING_STUDY_Y_GAP_HI,
) -> bool:
    if vanilla_idx not in y_extents:
        return False
    vanilla_hi = float(y_extents[vanilla_idx][1])
    return vanilla_hi > float(gap_hi) or _star_setting_broken_axis_needed(y_extents)


def _dual_series_positive_ymin(
    dual_series: list[tuple], *, margin_frac: float = 0.85
) -> float:
    """Smallest positive bar base (mean − std) for log-scale y limits."""
    lows: list[float] = []
    for _color, _label, g_back, g_front in dual_series:
        for g in (g_back, g_front):
            if g is None or g.empty:
                continue
            mean_vals = pd.to_numeric(g["total_time_mean"], errors="coerce")
            std_vals = pd.to_numeric(
                g.get("total_time_std", pd.Series(0.0, index=g.index)),
                errors="coerce",
            ).fillna(0.0)
            crest = (mean_vals - std_vals).clip(lower=1e-6)
            lows.extend(crest.dropna().tolist())
    if not lows:
        return 1.0
    return max(float(min(lows)) * float(margin_frac), 1e-2)


def _apply_setting_panel_style(
    ax, *, log_scale: bool = False, log_ymin: float | None = None
) -> None:
    """Grid and y limits for compilation-strategy (setting-study) panels."""
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, which="both" if log_scale else "major")
    ax.relim(visible_only=True)
    ax.autoscale_view()
    if log_scale:
        ax.set_yscale("log")
        y_lo = float(log_ymin) if log_ymin is not None and log_ymin > 0 else None
        _cur_lo, y_hi = ax.get_ylim()
        if y_lo is not None and y_hi > y_lo:
            ax.set_ylim(y_lo, y_hi)
        _apply_setting_panel_log_ticks(ax)
    else:
        _lo, y_hi = ax.get_ylim()
        if y_hi > 0.0:
            ax.set_ylim(0.0, y_hi)


def _dual_series_bar_top_hi(dual_series: list[tuple]) -> float:
    """Max bar crest (mean + std) across dual-AOD series for broken-axis top limits."""
    top_hi = 0.0
    for _color, _label, g_back, g_front in dual_series:
        for g in (g_back, g_front):
            if g is None or g.empty:
                continue
            mean_vals = pd.to_numeric(g["total_time_mean"], errors="coerce")
            std_vals = pd.to_numeric(
                g.get("total_time_std", pd.Series(0.0, index=g.index)),
                errors="coerce",
            ).fillna(0.0)
            crest = mean_vals + std_vals
            if not crest.dropna().empty:
                top_hi = max(top_hi, float(crest.max()))
    return top_hi


def _aod_point_hi(
    grouped: pd.DataFrame | None,
    *,
    n_aods: int,
    n_qubits: int,
) -> float | None:
    if grouped is None or grouped.empty:
        return None
    sub = grouped[
        (grouped["n_aods"].astype(int) == int(n_aods))
        & (grouped["n_qubits"].astype(int) == int(n_qubits))
    ]
    if sub.empty:
        return None
    row = sub.iloc[0]
    mean = float(row["total_time_mean"])
    std = float(row.get("total_time_std", 0.0) or 0.0)
    return mean + std


def _compute_star_aod_break_y(
    g_sync: pd.DataFrame | None,
    g_best: pd.DataFrame | None,
    sync_extent: tuple[float, float],
    best_extent: tuple[float, float],
    *,
    pad_frac: float = STAR_AOD_BREAK_PAD_FRAC,
) -> float:
    """AOD-only break (independent of bar chart); biased so anchor qubits stay visible."""
    _sync_lo, sync_hi = sync_extent
    _best_lo, best_hi = best_extent
    break_y = best_hi * (1.0 + pad_frac)
    anchor_hi = _aod_point_hi(
        g_sync,
        n_aods=STAR_AOD_BREAK_ANCHOR_AOD,
        n_qubits=STAR_AOD_BREAK_NQUBIT_ANCHOR,
    )
    if anchor_hi is not None:
        # Slightly lower break so sync AOD=5 @ anchor qubits stays in the bottom panel.
        break_y = min(break_y, anchor_hi * 1.08)
        break_y = max(break_y, anchor_hi * 1.02, best_hi * (1.0 + 0.5 * pad_frac))
    if sync_hi > best_hi:
        break_y = min(break_y, sync_hi * (1.0 - pad_frac))
        if break_y <= best_hi:
            break_y = best_hi + 0.35 * (sync_hi - best_hi)
    return float(break_y)


def _store_setting_n_qubit_ticks(ax, x_vals: list[int]) -> None:
    ticks = sorted({int(x) for x in x_vals})
    ax._setting_n_qubits_ticks = ticks  # type: ignore[attr-defined]
    ax._setting_n_qubit_x_pos = {int(nq): float(i) for i, nq in enumerate(ticks)}  # type: ignore[attr-defined]


def _setting_n_qubit_x_positions(anchor_ax) -> dict[int, float] | None:
    return getattr(anchor_ax, "_setting_n_qubit_x_pos", None)


def _apply_dark_axis_style(
    ax, *, include_x: bool = True, include_y: bool = True
) -> None:
    if include_x:
        ax.tick_params(axis="x", colors=_FIG_AXIS_DARK)
        ax.set_xlabel(ax.get_xlabel(), color=_FIG_AXIS_DARK)
    if include_y:
        ax.tick_params(axis="y", colors=_FIG_AXIS_DARK)
        ax.set_ylabel(ax.get_ylabel(), color=_FIG_AXIS_DARK)


def _aod_comparison_x_positions(anchor_ax) -> dict[int, float]:
    stored = getattr(anchor_ax, "_setting_n_qubit_x_pos", None)
    if stored:
        return dict(stored)
    return {int(nq): float(i) for i, nq in enumerate(SETTING_STUDY_N_QUBITS)}


def _apply_aod_comparison_x_ticks(ax, anchor_ax) -> None:
    """Fixed qubit ticks (16, 36, 64, 100) aligned with the setting-study panel."""
    x_pos = _aod_comparison_x_positions(anchor_ax)
    default_pos = {int(nq): float(i) for i, nq in enumerate(SETTING_STUDY_N_QUBITS)}
    tick_nqubits = list(SETTING_STUDY_N_QUBITS)
    x_centers = [float(x_pos.get(int(nq), default_pos[int(nq)])) for nq in tick_nqubits]
    labels = [str(int(nq)) for nq in tick_nqubits]
    ax.set_xlim(min(x_centers) - 0.65, max(x_centers) + 0.65)
    ax.xaxis.set_major_locator(FixedLocator(x_centers))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.tick_params(
        axis="x",
        colors=_FIG_AXIS_DARK,
        bottom=True,
        labelbottom=True,
        labelsize=_FIG_FONT_SIZE,
        pad=4,
    )
    for lbl in ax.get_xticklabels():
        lbl.set_visible(True)
        lbl.set_clip_on(False)
        lbl.set_color(_FIG_AXIS_DARK)
        lbl.set_fontsize(_FIG_FONT_SIZE)


def _commit_aod_x_axis(ax, anchor_ax, *, twin_ax=None) -> None:
    """Apply AOD x ticks and keep labels visible (twin-safe)."""
    if twin_ax is not None:
        twin_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    _apply_aod_comparison_x_ticks(ax, anchor_ax)
    if twin_ax is not None:
        twin_ax.set_xlim(ax.get_xlim())
        # twinx shares the x-axis: never set_xticks/set_xticklabels on the twin
        # or it clears the primary axis tick labels.
        twin_ax.tick_params(axis="x", bottom=False, labelbottom=False)


def _save_prx_figure(fig, out_path: str) -> None:
    for ax in fig.axes:
        for lbl in ax.get_xticklabels():
            if lbl.get_text():
                lbl.set_visible(True)
                lbl.set_clip_on(False)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=FIG_SAVE_PAD_INCHES)
    plt.close(fig)


def _apply_setting_n_qubit_ticks(
    ax, anchor_ax, *, round_name: str = "full_trotter"
) -> None:
    x_vals = getattr(anchor_ax, "_setting_n_qubits_ticks", None)
    if not x_vals:
        return
    x_centers = np.arange(len(x_vals), dtype=float)
    ax.set_xlim(-0.65, float(len(x_centers) - 1) + 0.65)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        _format_nqubit_ticklabels(x_vals, round_name, None),
        rotation=0,
    )
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)


def _aod_plot_x(
    n_qubits: pd.Series,
    x_positions: dict[int, float] | None,
) -> pd.Series:
    if not x_positions:
        return n_qubits
    return n_qubits.map(lambda nq: x_positions.get(int(nq), np.nan))


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


def _apply_aod_series_ylim(
    ax,
    extent: tuple[float, float],
    *,
    pad_frac: float = STAR_AOD_TWIN_Y_PAD_FRAC,
) -> None:
    """Set y-limits from mean±std extent with padding (floor at zero)."""
    _y_lo, y_hi = extent
    if y_hi <= 0.0:
        return
    span = max(float(y_hi), 1e-9)
    pad = float(pad_frac) * span
    ax.set_ylim(0.0, float(y_hi) + pad)


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
    return _create_broken_y_axes(
        ax_parent,
        height_ratios=STAR_BROKEN_AXIS_HEIGHT_RATIOS,
        hspace=STAR_BROKEN_INNER_HSPACE,
        attr_name="_broken_setting_axes",
    )


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


def _setting_study_legend_handle(*, color, label: str) -> Patch:
    return Patch(facecolor=color, edgecolor="black", label=label)


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


def _ideal_movement_leg_time(cnot_time: float) -> float:
    """Ideal forward or return movement time given measured CNOT circuit time."""
    return float(cnot_time) * _MOVEMENT_OPTIMAL_CNOT_LEG_FRACTION


def _movement_optimality_gap(
    forward_actual: float,
    return_actual: float,
    cnot_time: float,
) -> dict[str, float]:
    """Ratio of measured movement legs to ideal 0.5 × CNOT circuit time."""
    cnot_time = float(cnot_time)
    forward_actual = float(forward_actual)
    return_actual = float(return_actual)
    ideal_leg = _ideal_movement_leg_time(cnot_time)
    actual_total = forward_actual + return_actual

    def _leg_gap(actual: float) -> float:
        if ideal_leg <= 0.0:
            return float("nan")
        return actual / ideal_leg

    movement_gap = (
        float("nan") if cnot_time <= 0.0 else actual_total / cnot_time
    )

    return {
        "cnot_time": cnot_time,
        "forward_actual": forward_actual,
        "forward_ideal": ideal_leg,
        "forward_gap": _leg_gap(forward_actual),
        "return_actual": return_actual,
        "return_ideal": ideal_leg,
        "return_gap": _leg_gap(return_actual),
        "movement_actual": actual_total,
        "movement_ideal": cnot_time,
        "movement_gap": movement_gap,
    }


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


def _print_runtime_profile_values(
    panel: str,
    profiles_by_aod: dict[int, pd.DataFrame | None],
    components: list[tuple[str, str, str]],
    *,
    code_distance: int,
    aods: tuple[int, ...] = RUNTIME_PROFILE_AODS,
    y_axis_thousands: bool = False,
) -> None:
    """Log mean execution-time components used in the runtime-profile stacked bars."""
    available = {
        int(aod): prof
        for aod in aods
        if (prof := profiles_by_aod.get(int(aod))) is not None and not prof.empty
    }
    if not available:
        return

    scale_note = " (y-axis ticks shown as ×10³)" if y_axis_thousands else ""
    print(f"\n  [Runtime profile — {panel}, d={int(code_distance)}{scale_note}]")
    for aod in aods:
        prof = available.get(int(aod))
        if prof is None:
            continue
        print(f"    AOD={int(aod)}:")
        for _, row in prof.sort_values("n_qubits").iterrows():
            nq = int(row["n_qubits"])
            parts = [
                f"{label}={float(row[col]):.6g}"
                for col, label, _color in components
                if col in row.index
            ]
            total = float(row["total_mean"])
            parts.append(f"total={total:.6g}")
            print(f"      n_qubits={nq}: " + ", ".join(parts))


def _print_movement_optimality_gap(
    panel: str,
    profiles_by_aod: dict[int, pd.DataFrame | None],
    *,
    code_distance: int,
    aods: tuple[int, ...] = RUNTIME_PROFILE_AODS,
) -> None:
    """Log movement ratio vs ideal 0.5 × measured CNOT circuit time per leg."""
    available = {
        int(aod): prof
        for aod in aods
        if (prof := profiles_by_aod.get(int(aod))) is not None and not prof.empty
    }
    if not available:
        return

    frac = _MOVEMENT_OPTIMAL_CNOT_LEG_FRACTION
    print(
        f"\n  [Movement optimality gap — {panel}, d={int(code_distance)}] "
        f"(ideal leg = {frac:g} × CNOT time; "
        f"gap = actual / ({frac:g} × CNOT time); 1.0 = optimal)"
    )
    for aod in aods:
        prof = available.get(int(aod))
        if prof is None or "rus_mean" not in prof.columns:
            continue
        print(f"    AOD={int(aod)}:")
        for _, row in prof.sort_values("n_qubits").iterrows():
            nq = int(row["n_qubits"])
            metrics = _movement_optimality_gap(
                float(row["forward_move_mean"]),
                float(row["return_move_mean"]),
                float(row["rus_mean"]),
            )
            cnot = metrics["cnot_time"]
            print(
                f"      n_qubits={nq}, CNOT={cnot:.6g}: "
                f"forward={metrics['forward_actual']:.6g}, "
                f"gap={metrics['forward_gap']:.4g} "
                f"({metrics['forward_actual']:.6g}/({cnot:.6g}×{frac:g})); "
                f"return={metrics['return_actual']:.6g}, "
                f"gap={metrics['return_gap']:.4g} "
                f"({metrics['return_actual']:.6g}/({cnot:.6g}×{frac:g})); "
                f"total movement={metrics['movement_actual']:.6g}, "
                f"gap={metrics['movement_gap']:.4g} "
                f"({metrics['movement_actual']:.6g}/{cnot:.6g})"
            )


def _mean_pct_improvement_vs_previous(
    prev_grouped: pd.DataFrame | None,
    curr_grouped: pd.DataFrame | None,
    *,
    eps: float = 1e-15,
) -> float | None:
    """Mean % runtime reduction of *curr* vs *prev* on matched ``n_qubits``."""
    if prev_grouped is None or curr_grouped is None:
        return None
    if prev_grouped.empty or curr_grouped.empty:
        return None
    prev = (
        prev_grouped[["n_qubits", "total_time_mean"]]
        .dropna()
        .groupby("n_qubits", as_index=True)["total_time_mean"]
        .mean()
    )
    curr = (
        curr_grouped[["n_qubits", "total_time_mean"]]
        .dropna()
        .groupby("n_qubits", as_index=True)["total_time_mean"]
        .mean()
    )
    merged = pd.DataFrame({"prev": prev, "curr": curr}).dropna()
    if merged.empty:
        return None
    base = np.clip(merged["prev"].to_numpy(dtype=float), eps, None)
    curr_vals = np.clip(merged["curr"].to_numpy(dtype=float), eps, None)
    return float(np.mean((base - curr_vals) / base) * 100.0)


def _print_setting_study_sequential_improvements(
    panel: str,
    dual_series: list[tuple],
    *,
    code_distance: int,
    compare_aods: tuple[int, ...] = SETTING_STUDY_DUAL_AODS,
) -> None:
    """Log mean % speedup of each strategy vs the immediately previous one."""
    if not dual_series or len(dual_series) < 2 or len(compare_aods) < 2:
        return

    print(
        f"\n  [Setting study — {panel}, d={int(code_distance)}] "
        "(improvement vs previous setting; positive = faster)"
    )
    for aod_idx, aod in enumerate(compare_aods[:2]):
        print(f"    AOD={int(aod)}:")
        for i in range(1, len(dual_series)):
            _color, curr_label, grouped_back, grouped_front = dual_series[i]
            _color2, prev_label, prev_back, prev_front = dual_series[i - 1]
            grouped_by_aod = (prev_back, prev_front), (grouped_back, grouped_front)
            g_prev = grouped_by_aod[0][aod_idx]
            g_curr = grouped_by_aod[1][aod_idx]
            pct = _mean_pct_improvement_vs_previous(g_prev, g_curr)
            if pct is None:
                print(
                    f"      {prev_label} → {curr_label}: "
                    "(no overlapping n_qubits)"
                )
            else:
                print(f"      {prev_label} → {curr_label}: {pct:+.2f}%")


def _mean_pct_aod_improvement(
    grouped: pd.DataFrame | None,
    prev_aod: int,
    curr_aod: int,
    *,
    eps: float = 1e-15,
) -> float | None:
    """Mean % runtime reduction at *curr_aod* vs *prev_aod* on matched ``n_qubits``."""
    if grouped is None or grouped.empty:
        return None
    if "n_aods" not in grouped.columns or "total_time_mean" not in grouped.columns:
        return None
    work = grouped.copy()
    work["n_aods"] = pd.to_numeric(work["n_aods"], errors="coerce")
    work["n_qubits"] = pd.to_numeric(work["n_qubits"], errors="coerce")
    prev = (
        work.loc[work["n_aods"] == int(prev_aod)]
        .groupby("n_qubits", as_index=True)["total_time_mean"]
        .mean()
    )
    curr = (
        work.loc[work["n_aods"] == int(curr_aod)]
        .groupby("n_qubits", as_index=True)["total_time_mean"]
        .mean()
    )
    merged = pd.DataFrame({"prev": prev, "curr": curr}).dropna()
    if merged.empty:
        return None
    base = np.clip(merged["prev"].to_numpy(dtype=float), eps, None)
    curr_vals = np.clip(merged["curr"].to_numpy(dtype=float), eps, None)
    return float(np.mean((base - curr_vals) / base) * 100.0)


def _print_aod_comparison_improvements(
    panel: str,
    strategy_label: str,
    grouped: pd.DataFrame | None,
    *,
    code_distance: int,
    step_pairs: tuple[tuple[int, int], ...] = AOD_COMPARISON_STEP_PAIRS,
) -> None:
    """Log mean % speedup for each consecutive AOD step (2 vs 1, 3 vs 2, 5 vs 3)."""
    if grouped is None or grouped.empty:
        return

    print(
        f"\n  [AOD comparison — {panel}, d={int(code_distance)}, {strategy_label}] "
        "(improvement vs previous AOD; positive = faster)"
    )
    for prev_aod, curr_aod in step_pairs:
        pct = _mean_pct_aod_improvement(grouped, prev_aod, curr_aod)
        if pct is None:
            print(
                f"    AOD {int(curr_aod)} vs {int(prev_aod)}: "
                "(no overlapping n_qubits)"
            )
        else:
            print(f"    AOD {int(curr_aod)} vs {int(prev_aod)}: {pct:+.2f}%")


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

    if verbose:
        if _has_profile(star_profiles):
            _print_runtime_profile_values(
                "STAR",
                star_profiles,
                _RUNTIME_PROFILE_STAR_COMPONENTS,
                code_distance=cd,
                aods=aods,
            )
        if _has_profile(t_profiles):
            _print_runtime_profile_values(
                "T-cultivation",
                t_profiles,
                _RUNTIME_PROFILE_T_COMPONENTS,
                code_distance=cd,
                aods=aods,
                y_axis_thousands=True,
            )
        if _has_profile(star_profiles):
            _print_movement_optimality_gap(
                "STAR", star_profiles, code_distance=cd, aods=aods
            )
        if _has_profile(t_profiles):
            _print_movement_optimality_gap(
                "T-cultivation", t_profiles, code_distance=cd, aods=aods
            )

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
    aod_linestyles: dict[int, str | tuple] | None = None,
    emphasize_aod: int | None = None,
    y_clip: tuple[float | None, float | None] | None = None,
    x_positions: dict[int, float] | None = None,
) -> list[int]:
    x_pts: list[int] = []
    if grouped is None or grouped.empty:
        return x_pts
    clip_lo, clip_hi = (None, None) if y_clip is None else y_clip
    aod_order = sorted(int(a) for a in plot_aods)
    if emphasize_aod is not None and int(emphasize_aod) in aod_order:
        aod_order = [a for a in aod_order if a != int(emphasize_aod)] + [
            int(emphasize_aod)
        ]
    for aod in aod_order:
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
        ls = (aod_linestyles or {}).get(int(aod), linestyle)
        z = 4 if emphasize_aod is not None and int(aod) == int(emphasize_aod) else 2
        x_plot = _aod_plot_x(sub["n_qubits"], x_positions)
        _plot_styled_errorbar(
            ax,
            x_plot,
            mean_vals,
            std_vals,
            color,
            linestyle=ls,
            zorder=z,
        )
        x_pts.extend(sub["n_qubits"].dropna().astype(int).tolist())
    return x_pts


def _draw_aod_sweep_broken(
    ax_bot,
    ax_top,
    grouped: pd.DataFrame | None,
    plot_aods: list[int],
    aod_colors: dict[int, tuple[float, float, float]],
    y_break: float,
    *,
    linestyle: str = "-",
    x_positions: dict[int, float] | None = None,
) -> list[int]:
    """Draw AOD sweeps on both broken segments; points visible if mean±std crosses break."""
    x_pts: list[int] = []
    if grouped is None or grouped.empty:
        return x_pts
    y_break_f = float(y_break)
    for aod in plot_aods:
        sub = grouped[grouped["n_aods"].astype(int) == int(aod)].sort_values("n_qubits")
        if sub.empty:
            continue
        mean_vals = pd.to_numeric(sub["total_time_mean"], errors="coerce")
        std_vals = pd.to_numeric(sub["total_time_std"], errors="coerce").fillna(0.0)
        y_lo = mean_vals - std_vals
        y_hi = mean_vals + std_vals
        color = aod_colors.get(int(aod), (0.2, 0.2, 0.2))
        bot_mask = y_lo <= y_break_f
        top_mask = y_hi >= y_break_f
        if bool(bot_mask.any()):
            _plot_styled_errorbar(
                ax_bot,
                _aod_plot_x(sub.loc[bot_mask, "n_qubits"], x_positions),
                mean_vals.loc[bot_mask],
                std_vals.loc[bot_mask],
                color,
                linestyle=linestyle,
            )
        if bool(top_mask.any()):
            _plot_styled_errorbar(
                ax_top,
                _aod_plot_x(sub.loc[top_mask, "n_qubits"], x_positions),
                mean_vals.loc[top_mask],
                std_vals.loc[top_mask],
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


def _aod_legend_row(
    *,
    aod_colors: dict[int, tuple[float, float, float]],
    base_rgb: tuple[float, float, float],
    strategy_label: str,
    linestyle: str | tuple = "-",
) -> tuple[list[Line2D], list[str]]:
    handles = [_aod_row_style_handle(base_rgb, linestyle=linestyle)]
    handles.extend(_aod_column_legend_handles(aod_colors))
    labels = [strategy_label] + [str(int(a)) for a in sorted(aod_colors)]
    return handles, labels


def _build_aod_bottom_legend_rows(
    *,
    aod_colors: dict[int, tuple[float, float, float]],
    base_rgb: tuple[float, float, float] | None = None,
    strategy_label: str | None = "Sync. execution",
    linestyle: str | tuple = "-",
    aod_linestyles: dict[int, str | tuple] | None = None,
) -> list[tuple[list[Line2D], list[str]]]:
    """One legend row: optional strategy swatch + AOD 1/2/3/5 markers."""
    handles: list[Line2D] = []
    labels: list[str] = []
    if strategy_label is not None and base_rgb is not None:
        handles.append(_aod_row_style_handle(base_rgb, linestyle=linestyle))
        labels.append(strategy_label)
    handles.extend(
        _aod_column_legend_handles(aod_colors, aod_linestyles=aod_linestyles)
    )
    labels.extend(str(int(a)) for a in sorted(aod_colors))
    return [(handles, labels)]


def _attach_aod_bottom_legend_spec(
    ax,
    *,
    aod_colors: dict[int, tuple[float, float, float]],
    base_rgb: tuple[float, float, float] | None = None,
    strategy_label: str | None = "Sync. execution",
    linestyle: str | tuple = "-",
    aod_linestyles: dict[int, str | tuple] | None = None,
) -> None:
    ax._aod_bottom_legend_rows = _build_aod_bottom_legend_rows(  # type: ignore[attr-defined]
        aod_colors=aod_colors,
        base_rgb=base_rgb,
        strategy_label=strategy_label,
        linestyle=linestyle,
        aod_linestyles=aod_linestyles,
    )


def _build_star_aod_bottom_legend_rows(
    *,
    sync_aod_colors: dict[int, tuple[float, float, float]],
    best_aod_colors: dict[int, tuple[float, float, float]],
    sync_base_rgb: tuple[float, float, float],
    best_base_rgb: tuple[float, float, float],
    baseline_label: str = "Baseline",
    best_label: str,
) -> list[tuple[list[Line2D], list[str]]]:
    return [
        _aod_legend_row(
            aod_colors=sync_aod_colors,
            base_rgb=sync_base_rgb,
            strategy_label=baseline_label,
            linestyle="-",
        ),
        _aod_legend_row(
            aod_colors=best_aod_colors,
            base_rgb=best_base_rgb,
            strategy_label=best_label,
            linestyle=(0, (5, 2)),
        ),
    ]


def _attach_star_aod_bottom_legend_spec(
    ax,
    *,
    sync_aod_colors: dict[int, tuple[float, float, float]],
    best_aod_colors: dict[int, tuple[float, float, float]],
    sync_base_rgb: tuple[float, float, float],
    best_base_rgb: tuple[float, float, float],
    baseline_label: str = "Baseline",
    best_label: str,
) -> None:
    ax._aod_bottom_legend_rows = _build_star_aod_bottom_legend_rows(  # type: ignore[attr-defined]
        sync_aod_colors=sync_aod_colors,
        best_aod_colors=best_aod_colors,
        sync_base_rgb=sync_base_rgb,
        best_base_rgb=best_base_rgb,
        baseline_label=baseline_label,
        best_label=best_label,
    )


def _apply_aod_bottom_legend(
    fig,
    ax_aod_parent,
    *,
    legend_fs: float,
    bbox_x: float = 0.5,
) -> None:
    rows = getattr(ax_aod_parent, "_aod_bottom_legend_rows", None)
    if not rows:
        return
    nrows = len(rows)
    ncols = len(rows[0][0])
    grid_handles = [[rows[r][0][c] for c in range(ncols)] for r in range(nrows)]
    grid_labels = [[rows[r][1][c] for c in range(ncols)] for r in range(nrows)]
    # Matplotlib fills legend column-by-column; reorder so each setting stays on one row.
    handles: list[Line2D] = []
    labels: list[str] = []
    for col in range(ncols):
        for row in range(nrows):
            handles.append(grid_handles[row][col])
            labels.append(grid_labels[row][col])
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(bbox_x, AOD_BOTTOM_LEGEND_BASE_Y),
        ncol=ncols,
        fontsize=legend_fs,
        frameon=True,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.65,
        borderpad=0.35,
        labelspacing=0.55,
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
    setting_panel_log_scale: bool = False,
    verbose: bool = True,
) -> None:
    """STAR and T-cultivation figures: setting-study bars and AOD comparison.

    Writes ``*_d{cd}_bars.pdf`` (or ``*_bars_log.pdf`` when *setting_panel_log_scale*)
    with overlapping AOD = 1 vs 5 bars on the setting panel.
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
        # AOD sweep uses ablation rows so the best compile setting matches the setting study.
        t_aod_layer = t_layers_ablation[round_name].copy()
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
        if g_star_sync is not None and not g_star_sync.empty:
            sync_aods = sorted(
                pd.to_numeric(g_star_sync["n_aods"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            missing_sync = [
                int(a) for a in AOD_COMPARISON_AODS if int(a) not in sync_aods
            ]
            if missing_sync:
                print(
                    f"  Note: Baseline AOD panel has no profiling data for "
                    f"AOD {missing_sync} (available: {sync_aods}). "
                    f"Re-run STAR profiling for Baseline at those AOD counts."
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
    _star_setting_anchor: list = []
    _t_setting_anchor: list = []

    def _apply_theory_line(
        ax,
        x_points: list[int],
        *,
        theory: str,
        x_positions: dict[int, float] | None = None,
        y_clip: tuple[float | None, float | None] | None = None,
    ) -> None:
        if not SHOW_EXPECTED_TIME_LINE:
            return
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
                b["full_trotter"][i] for i, x in enumerate(x_u) if int(x) in x_positions
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

    def _finalize_setting_panel(
        ax,
        x_points: list[int],
        *,
        theory: str,
        x_positions: dict[int, float] | None = None,
        dual_series: list[tuple] | None = None,
    ) -> None:
        _apply_theory_line(ax, x_points, theory=theory, x_positions=x_positions)
        log_ymin = (
            _dual_series_positive_ymin(dual_series)
            if setting_panel_log_scale and dual_series
            else None
        )
        _apply_setting_panel_style(
            ax,
            log_scale=setting_panel_log_scale,
            log_ymin=log_ymin,
        )

    def _star_aod_x_positions() -> dict[int, float]:
        if not _star_setting_anchor:
            return {int(nq): float(i) for i, nq in enumerate(SETTING_STUDY_N_QUBITS)}
        return _aod_comparison_x_positions(_star_setting_anchor[0])

    def _draw_star_aod_panel(ax) -> None:
        """AOD line panel: sync on left y-axis, best strategy on right (no broken axis)."""
        x_positions = _star_aod_x_positions()
        ax_right = ax.twinx()
        ax._star_aod_twin_right = ax_right  # type: ignore[attr-defined]
        star_aod_axis_label = _star_aod_axis_label(star_aod_setting_idx)
        ax._star_aod_twin_style = (  # type: ignore[attr-defined]
            "Baseline",
            star_sync_base,
            star_best_base,
        )
        ax._star_aod_twin_ylabel = star_aod_axis_label  # type: ignore[attr-defined]

        x_pts: list[int] = []
        x_pts.extend(
            _draw_aod_sweep_on_panel(
                ax,
                g_star_sync,
                star_plot_aods,
                star_sync_aod_colors,
                linestyle="-",
                x_positions=x_positions,
            )
        )
        x_pts.extend(
            _draw_aod_sweep_on_panel(
                ax_right,
                g_star_best,
                star_plot_aods,
                star_best_aod_colors,
                linestyle=(0, (5, 2)),
                x_positions=x_positions,
            )
        )

        _apply_aod_series_ylim(ax, _grouped_aod_y_extent(g_star_sync))
        _apply_aod_series_ylim(ax_right, _grouped_aod_y_extent(g_star_best))

        _apply_theory_line(ax, x_pts, theory="star", x_positions=x_positions)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax_right.set_axisbelow(True)
        ax_right.grid(False)
        _apply_panel_y_ticks(ax, nbins=STAR_AOD_COMPARISON_Y_NBINS)
        _apply_panel_y_ticks(ax_right, nbins=STAR_AOD_COMPARISON_Y_NBINS)

        _attach_star_aod_bottom_legend_spec(
            ax,
            sync_aod_colors=star_sync_aod_colors,
            best_aod_colors=star_best_aod_colors,
            sync_base_rgb=star_sync_base,
            best_base_rgb=star_best_base,
            baseline_label="Baseline",
            best_label=star_aod_axis_label,
        )

    def _draw_star_setting_panel(ax) -> None:
        nonlocal star_shared_y_break, star_shared_panel_hi
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
            y_extents[int(setting_idx)] = _agg_y_extent(grouped)

        vanilla_idx = STAR_SETTING_STUDY_VANILLA_IDX
        dual_series = _build_star_setting_dual_series(
            dfs_dict_micro,
            star_study_entries,
            star_setting_colors,
            cd,
        )
        dual_top_hi = _dual_series_bar_top_hi(dual_series)
        use_broken_axis = (
            not setting_panel_log_scale
            and vanilla_idx in y_extents
            and _star_setting_fixed_gap_break_needed(y_extents)
        )
        if use_broken_axis:
            y_break, y_top_hi = _finalize_star_bar_fixed_gap_limits(
                y_extents,
                dual_top_hi=dual_top_hi if dual_top_hi > 0.0 else None,
            )
            ax_top, ax_bot = _create_star_setting_broken_axes(ax)
            x_pts = _draw_setting_study_dual_aod_bars(ax_top, dual_series)
            _draw_setting_study_dual_aod_bars(ax_bot, dual_series)

            ax_bot.set_ylim(0.0, y_break)
            ax_top.set_ylim(STAR_SETTING_STUDY_Y_GAP_HI, y_top_hi)
            star_shared_y_break = float(y_break)
            star_shared_panel_hi = float(y_top_hi)
            _store_setting_n_qubit_ticks(ax, x_pts)

            x_u = sorted(set(int(x) for x in x_pts))
            if SHOW_EXPECTED_TIME_LINE and x_u:
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
        _store_setting_n_qubit_ticks(ax, x_pts)
        _finalize_setting_panel(
            ax,
            x_pts,
            theory="star",
            x_positions=x_pos,
            dual_series=dual_series,
        )

    def _draw_t_setting_panel(ax) -> None:
        dual_series = _build_t_setting_dual_series(
            t_setting_layer,
            triple,
            t_study_entries,
            t_setting_colors,
            cd,
        )
        x_pts = _draw_setting_study_dual_aod_bars(ax, dual_series)
        x_pos = {int(nq): float(i) for i, nq in enumerate(x_pts)}
        _store_setting_n_qubit_ticks(ax, x_pts)
        _finalize_setting_panel(
            ax,
            x_pts,
            theory="t",
            x_positions=x_pos,
            dual_series=dual_series,
        )

    def _t_aod_x_positions() -> dict[int, float]:
        if not _t_setting_anchor:
            return {int(nq): float(i) for i, nq in enumerate(SETTING_STUDY_N_QUBITS)}
        return _aod_comparison_x_positions(_t_setting_anchor[0])

    def _draw_t_aod_panel(ax) -> None:
        x_positions = _t_aod_x_positions()
        t_aod_linestyles = {
            int(T_AOD_EMPHASIS_AOD): T_AOD_EMPHASIS_LINESTYLE,
        }
        x_pts = _draw_aod_sweep_on_panel(
            ax,
            g_t_best,
            t_plot_aods,
            t_best_aod_colors,
            linestyle="-",
            aod_linestyles=t_aod_linestyles,
            emphasize_aod=T_AOD_EMPHASIS_AOD,
            x_positions=x_positions,
        )
        _finalize_panel(ax, x_pts, theory="t", x_positions=x_positions)
        _apply_panel_y_ticks(ax, nbins=T_CULTIVATION_Y_NBINS)
        if g_t_best is not None and not g_t_best.empty:
            _apply_aod_series_ylim(ax, _grouped_aod_y_extent(g_t_best))
        _attach_aod_bottom_legend_spec(
            ax,
            aod_colors=t_best_aod_colors,
            strategy_label=None,
            base_rgb=None,
            aod_linestyles={int(T_AOD_EMPHASIS_AOD): T_AOD_EMPHASIS_LINESTYLE},
        )

    plot_star = "full_trotter" in dfs_dict_micro
    plot_t = triple is not None and not t_setting_layer.empty

    if verbose:
        if plot_star:
            star_dual_series = _build_star_setting_dual_series(
                dfs_dict_micro,
                star_study_entries,
                star_setting_colors,
                cd,
            )
            if star_dual_series:
                _print_setting_study_sequential_improvements(
                    "STAR",
                    star_dual_series,
                    code_distance=cd,
                )
        if plot_t:
            t_dual_series = _build_t_setting_dual_series(
                t_setting_layer,
                triple,
                t_study_entries,
                t_setting_colors,
                cd,
            )
            if t_dual_series:
                _print_setting_study_sequential_improvements(
                    "T-cultivation",
                    t_dual_series,
                    code_distance=cd,
                )
        if plot_star:
            if g_star_sync is not None and not g_star_sync.empty:
                _print_aod_comparison_improvements(
                    "STAR",
                    "Baseline",
                    g_star_sync,
                    code_distance=cd,
                )
            if g_star_best is not None and not g_star_best.empty:
                _print_aod_comparison_improvements(
                    "STAR",
                    star_aod_setting_label,
                    g_star_best,
                    code_distance=cd,
                )
        if plot_t and g_t_best is not None and not g_t_best.empty:
            _print_aod_comparison_improvements(
                "T-cultivation",
                t_aod_setting_label,
                g_t_best,
                code_distance=cd,
            )

    setting_column_title = "Compilation Strategies"
    star_column_titles = (setting_column_title, "AOD comparison")
    t_column_titles = (setting_column_title, "AOD comparison")
    chart_suffix = "_bars_log" if setting_panel_log_scale else "_bars"
    star_setting_legend_handles = _star_setting_study_legend_handles(
        star_setting_colors,
        star_study_entries,
    )
    t_setting_legend_handles = [
        _setting_study_legend_handle(
            color=t_setting_colors[setting_idx],
            label=label,
        )
        for setting_idx, label in t_study_entries
        if setting_idx in t_setting_colors
    ]
    aod_legend_handles = _dual_aod_bar_legend_handles("black")

    def _draw_star_setting(ax) -> None:
        _star_setting_anchor.clear()
        _star_setting_anchor.append(ax)
        ax._star_legend_meta = {  # type: ignore[attr-defined]
            "colors": star_setting_colors,
            "entries": star_study_entries,
        }
        _draw_star_setting_panel(ax)

    def _draw_t_setting(ax) -> None:
        _t_setting_anchor.clear()
        _t_setting_anchor.append(ax)
        _draw_t_setting_panel(ax)

    if plot_star and plot_t:
        merged_path = os.path.join(
            output_dir,
            f"star_t_setting_and_aod_comparison_d{cd}{chart_suffix}.pdf",
        )
        _save_star_t_merged_setting_aod_figure(
            draw_star_setting=_draw_star_setting,
            draw_star_aod=_draw_star_aod_panel,
            draw_t_setting=_draw_t_setting,
            draw_t_aod=_draw_t_aod_panel,
            star_setting_legend_handles=star_setting_legend_handles,
            t_setting_legend_handles=t_setting_legend_handles,
            star_column_titles=star_column_titles,
            t_column_titles=t_column_titles,
            aod_legend_handles=aod_legend_handles,
            out_path=merged_path,
            xlabel="Number of Qubits/Factories",
            panel_hspace=STAR_SETTING_AOD_PANEL_HSPACE,
            figure_top=0.90,
            setting_panel_log_scale=setting_panel_log_scale,
        )
        if verbose:
            print(f"Saved: {merged_path}")

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
            panel_hspace=STAR_SETTING_AOD_PANEL_HSPACE,
            bottom_axis_y_shift=STAR_AOD_BOTTOM_AXIS_Y_SHIFT,
            setting_legend_ncol=3,
            figure_top=0.90,
            column_titles=star_column_titles,
            setting_panel_log_scale=setting_panel_log_scale,
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
            legend_y_blend=T_SETTING_LEGEND_GAP_BLEND,
            column_titles=t_column_titles,
            setting_panel_log_scale=setting_panel_log_scale,
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
    setting_panel_log_scale: bool = False,
    show_expected_time_line: bool | None = None,
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

    global SHOW_T_CULTIVATION_D13, SHOW_EXPECTED_TIME_LINE
    previous_show_d13 = SHOW_T_CULTIVATION_D13
    previous_show_expected = SHOW_EXPECTED_TIME_LINE
    SHOW_T_CULTIVATION_D13 = bool(include_t_cultivation_d13)
    if show_expected_time_line is not None:
        SHOW_EXPECTED_TIME_LINE = bool(show_expected_time_line)
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
            setting_panel_log_scale=setting_panel_log_scale,
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
        SHOW_EXPECTED_TIME_LINE = previous_show_expected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot STAR vs T-cultivation setting/AOD comparison figures."
    )
    parser.add_argument(
        "--no-expected-time-line",
        action="store_true",
        default=True,
        help="Omit the expected-time dashed reference line and its legend entry.",
    )
    parser.add_argument(
        "--setting-log-y",
        action="store_true",
        help="Use log scale on compilation-strategy (setting-study) panel y-axes.",
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
            show_expected_time_line=not args.no_expected_time_line,
            setting_panel_log_scale=args.setting_log_y,
            verbose=True,
        )
    else:
        print("Missing profiling CSV(s); expected:")
        print(f"  {star_csv}")
        print(f"  {t_csv}")
