import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import os
from collections import defaultdict

from src.animator.log_view_helpers import (
    normalize_factories,
    resolve_indexed,
    resolve_move_vecs,
    entry_start,
    entry_end,
)
from src.animator.rus_round_visualization import draw_trap_grid_time_range_on_axes

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
_FIG_FONT_SIZE = 14
# Full two-column paper width (PRX/APS ~6.75–7 in); used for *_no_text* timeline PDFs.
_TCULT_EXEC_FIGURE_WIDTH = 7.0
# No-text PDFs: axis titles and legends scaled relative to with-text sizes.
_NO_TEXT_FONT_SCALE = 0.5
# Combined timeline+movement no-text: keep titles/legend larger than axis ticks.
_COMBINED_TIMELINE_MOVEMENT_NO_TEXT_TITLE_SCALE = 0.82
_COMBINED_TIMELINE_MOVEMENT_NO_TEXT_AXIS_FONT_SCALE = 0.74
# No-text PDFs: x/y tick labels can be tuned independently.
_NO_TEXT_X_TICK_FONT_SCALE = 0.5
_NO_TEXT_Y_TICK_FONT_SCALE = 0.37
_BOX_HEIGHT = 0.7
_BOX_Y_OFFSET = _BOX_HEIGHT / 2
_BOX_ALPHA = 0.9
_AOD_MOVE_PURPLE_SHADES = [
    "#b64abe",
    "#D8B4FE",
    "#C084FC",
    "#A855F7",
    "#7E22CE",
]
_AOD_RETURN_MOVE_GREEN_SHADES = [
    "#9edc6f",
    "#d9f7be",
    "#b7eb8f",
    "#73d13d",
    "#389e0d",
]
# Color mapping for operations
color_map = {
    "TUM": "#9ed76c",
    "SE": "#4681a9",
    "SE_q": "#4681a9",
    "S": "#a98546",
    "CNOT": "#e74c3c",
    "Rz": "#e7ab3c",
    "TMR": "#3CA8E7",
    "move": "#b64abe",
    "return_move": "#9edc6f",
    "Barrier": "#000000",
    "RUS_success": "#E01414",
    "RUS_fail": "#DDA413",
    "TMR_fail": "#007E15",
    "CNOT(T)": "#e74c3c",
    "T": "#9ed76c",
    "H": "#5dade2",
    "SE_stage_1": "#6E9EBE",
    "SE_stage_2": "#4F64C2",
}

_TCULT_OP_ALIASES = {
    "SE Stage 1": "SE_stage_1",
    "SE Stage 2": "SE_stage_2",
    "Move": "move",
    "Return Move": "return_move",
}

# Color mapping for AOD devices (for border colors)
aod_colors = [
    "#FF57579A",
    "#EF7432F4",
    "#F3C82DEB",
    "#python-color-picker",
    "#C540BE79",
    "#B145D9E5",
    "#4825E8CC",
]
ylim = 93
_BOX_BORDER_WIDTH = 1.0
_NO_TEXT_BOX_BORDER_WIDTH = 0.4
_AXIS_LABEL_FONT_SIZE = 24
_TITLE_FONT_SIZE = 28
_LEGEND_FONT_SIZE = 20
# STAR stacked subfigures: same size for suptitle, per-panel title, and x-axis label.
_STAR_SUBFIG_HEADING_FONT_SIZE = 21
_TRAP_GRID_TIME_WINDOW_COLOR = "#3CA8E7"
_TRAP_GRID_TIME_WINDOW_ALPHA = 0.22
# Inches scaling for total figure height (larger => taller lanes / taller boxes on screen).
_STAR_SUBFIG_DEFAULT_VERTICAL_STRETCH = 1.8


def get_aod_border_color(aod_idx: int) -> str:
    """Get border color for a given AOD index."""
    return aod_colors[aod_idx % len(aod_colors)]


def _resolve_factory_value(values, idx: int):
    return resolve_indexed(values, idx)


def _resolve_factory_move_vecs(move_vecs, idx: int):
    return resolve_move_vecs(move_vecs, idx)


def _extract_qubits_from_targets(targets) -> list[int]:
    if targets is None:
        return []
    qubits: list[int] = []
    for target in targets:
        if isinstance(target, int):
            qubits.append(target)
        elif isinstance(target, (tuple, list)) and len(target) == 2:
            q0, q1 = target
            if isinstance(q0, int):
                qubits.append(q0)
            if isinstance(q1, int):
                qubits.append(q1)
    return sorted(set(qubits))


def _extract_qubits_from_value(value) -> list[int]:
    """Extract logical qubit ids from STAR execution entry values."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    ):
        return [value[0], value[1]]
    if isinstance(value, list):
        qubits: list[int] = []
        for item in value:
            qubits.extend(_extract_qubits_from_value(item))
        return sorted(set(qubits))
    return []


def _canonical_tcult_operation(operation: str) -> str:
    return _TCULT_OP_ALIASES.get(operation, operation)


def _execution_timeline_fig_width(
    figure_width: float,
    circuit_length: int,
    *,
    show_box_text: bool,
) -> float:
    """Figure width in inches for horizontal execution timelines.

    No-text PDFs use ``_TCULT_EXEC_FIGURE_WIDTH`` so they can be included at
    full two-column width without downscaling axis labels and legends.
    """
    if not show_box_text:
        return float(_TCULT_EXEC_FIGURE_WIDTH)
    return max(float(figure_width) * 0.45, circuit_length / 10 + 4.5)


def _execution_timeline_font_scale(*, show_box_text: bool) -> float:
    """Font scale for axis labels, ticks, titles, and legends."""
    return 1.0 if show_box_text else _NO_TEXT_FONT_SCALE


def _scaled_font_size(base: float, font_scale: float) -> float:
    return base * font_scale


def _capitalize_title_words(text: str) -> str:
    """Capitalize the first letter of each word; preserve all-caps tokens (e.g. STAR)."""

    def _cap_word(word: str) -> str:
        if not word:
            return word
        if word.isupper() and len(word) > 1:
            return word
        return word[0].upper() + word[1:].lower()

    return " ".join(_cap_word(part) for part in text.split())


def _execution_timeline_tick_font_sizes(
    *,
    show_box_text: bool,
    x_tick_base: float | None = None,
    y_tick_base: float | None = None,
) -> tuple[float, float]:
    """Return ``(x_tick_labelsize, y_tick_labelsize)`` in points."""
    x_base = _FIG_FONT_SIZE if x_tick_base is None else x_tick_base
    y_base = _FIG_FONT_SIZE if y_tick_base is None else y_tick_base
    if show_box_text:
        return (x_base, y_base)
    return (
        x_base * _NO_TEXT_X_TICK_FONT_SCALE,
        y_base * _NO_TEXT_Y_TICK_FONT_SCALE,
    )


def _execution_timeline_box_border_width(
    *,
    show_box_text: bool,
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
) -> float:
    """Return operation-box edge linewidth in points."""
    if show_box_text:
        return (
            _BOX_BORDER_WIDTH if box_border_width is None else float(box_border_width)
        )
    return (
        _NO_TEXT_BOX_BORDER_WIDTH
        if no_text_box_border_width is None
        else float(no_text_box_border_width)
    )


def _subplot_index_with_most_timeline_whitespace(row_end_times: list[float]) -> int:
    """Pick the stacked row whose trace ends earliest (most empty span on a shared x-axis).

    Tie-break: prefer the top row (smallest index).
    """
    if not row_end_times:
        return 0
    return min(range(len(row_end_times)), key=lambda i: (row_end_times[i], i))


def _save_path_for_box_text_variant(save_path: str, *, show_box_text: bool) -> str:
    """With-text uses *save_path*; without-text appends ``_no_text`` before the extension."""
    if show_box_text:
        return save_path
    root, ext = os.path.splitext(save_path)
    if root.endswith("_no_text"):
        return save_path
    return f"{root}_no_text{ext}"


def _factories_key(entry: dict) -> tuple[int, ...]:
    return tuple(sorted(normalize_factories(entry.get("factories"))))


def _tmr_block_from_se_rz_entries(block: list[dict], fkey: tuple[int, ...]) -> dict:
    """Build one collapsed ``TMR`` event from same-factory SE/Rz entries."""
    rz_entries = [e for e in block if e.get("operation") == "Rz"]
    theta = None
    if rz_entries:
        targets = rz_entries[0].get("targets")
        if isinstance(targets, list) and len(targets) == 1:
            theta = targets[0]
        else:
            theta = targets
    ordered = sorted(block, key=lambda e: (entry_start(e), entry_end(e)))
    return {
        "start_time": entry_start(ordered[0]),
        "end_time": entry_end(ordered[-1]),
        "factories": list(fkey),
        "operation": "TMR",
        "aod_assignment": ordered[0].get("aod_assignment"),
        "targets": theta,
        "move_vecs": None,
    }


def _partition_fkey_tmr_rounds(
    entries: list[dict],
) -> tuple[list[list[dict]], list[dict]]:
    """Split one factory's SE/Rz timeline into TMR-like rounds and leftovers.

    We primarily recognize the canonical STAR TMR schedule:
        TMR_P × SE  →  Rz  →  TMR_Q × SE

    In asynchronous logs, the trace can end mid-round (or scheduling can leave an
    unfinished tail). For plotting, we still collapse such tails into a single
    ``TMR`` box rather than leaving dangling ``SE`` boxes at the end.
    """
    from src.star.config import TMR_P, TMR_Q

    ordered = sorted(entries, key=lambda e: (entry_start(e), entry_end(e)))
    complete: list[list[dict]] = []
    orphans: list[dict] = []
    i = 0
    n = len(ordered)
    while i < n:
        block: list[dict] = []
        pre = 0
        while i < n and ordered[i].get("operation") == "SE" and pre < TMR_P:
            block.append(ordered[i])
            pre += 1
            i += 1
        # If we don't even have a full pre-Rz preparation, keep it as a collapsed
        # tail (TMR-without-angle) for nicer plots.
        if pre != TMR_P:
            if block:
                complete.append(block)
            continue
        # Missing Rz (e.g. trace ends right after preparation): still collapse.
        if i >= n or ordered[i].get("operation") != "Rz":
            complete.append(block)
            continue
        block.append(ordered[i])
        i += 1
        post = 0
        while i < n and ordered[i].get("operation") == "SE" and post < TMR_Q:
            block.append(ordered[i])
            post += 1
            i += 1
        # If post-Rz SE is truncated, still collapse what we have.
        complete.append(block)
    return complete, orphans


def _collapse_star_tmr_blocks(execution_log: list) -> list:
    """Merge factory SE/Rz into single ``TMR`` boxes for plotting.

    Parallel logs are sorted by time, so SE/Rz from concurrent TMR rounds interleave.
    Each AOD also emits its own ``Barrier`` at slightly different times, which can
    split one logical TMR across barrier segments. Recognize the fixed
    TMR_P + Rz + TMR_Q pattern per factory set instead.
    """
    by_fkey: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for entry in execution_log:
        op = entry.get("operation")
        if op not in ("SE", "Rz"):
            continue
        fkey = _factories_key(entry)
        if fkey:
            by_fkey[fkey].append(entry)

    tmr_at_first_id: dict[int, dict] = {}
    skip_ids: set[int] = set()
    for fkey, entries in by_fkey.items():
        complete, orphans = _partition_fkey_tmr_rounds(entries)
        for block in complete:
            tmr_at_first_id[id(block[0])] = _tmr_block_from_se_rz_entries(block, fkey)
            for entry in block[1:]:
                skip_ids.add(id(entry))
    merged: list[dict] = []
    for entry in execution_log:
        eid = id(entry)
        if eid in skip_ids:
            continue
        if eid in tmr_at_first_id:
            merged.append(tmr_at_first_id[eid])
            continue
        merged.append(entry)
    return merged


def _star_box_label(operation: str, value=None, move_vecs=None) -> str:
    if operation == "TMR":
        if value is not None:
            return f"TMR\nθ:{value}"
        return "TMR"
    if operation == "Rz":
        return f"RZ\nθ:{value}"
    if operation == "S":
        return f"S\nθ:{value}"
    if operation == "SE":
        return "SE"
    if operation == "SE_q":
        return "SE"
    if operation == "CNOT":
        if value is not None:
            return f"CNOT\nQ{value}"
        return "CNOT"
    if operation in {"move", "return_move"}:
        if move_vecs and len(move_vecs) >= 2:
            return f"{move_vecs[0]}\n$\\downarrow$\n{move_vecs[1]}"
        return ""
    if value is not None:
        return f"{operation}\nQ{value}"
    return operation


def _movement_color_for_aod(operation: str, aod_assignment) -> str:
    if operation == "return_move":
        shades = _AOD_RETURN_MOVE_GREEN_SHADES
    else:
        shades = _AOD_MOVE_PURPLE_SHADES

    if not isinstance(aod_assignment, int) or aod_assignment < 0:
        return shades[0]
    return shades[aod_assignment % len(shades)]


def _resolve_star_row_layout(
    execution_log,
    n_factories: int,
    n_logical_qubits: int | None,
    show_logical_qubits: bool,
):
    """Return (resolved_n_logical, row_names, y_pos, n_rows) for horizontal STAR plots."""
    if not show_logical_qubits:
        return None, None, None, n_factories
    nl = n_logical_qubits
    if nl is None:
        max_qubit = -1
        for entry in execution_log:
            value = entry.get("targets")
            qubits = _extract_qubits_from_value(value)
            if qubits:
                max_qubit = max(max_qubit, max(qubits))
        nl = max_qubit + 1 if max_qubit >= 0 else 0
    row_names = [f"q{i}" for i in range(nl)] + [f"f{i}" for i in range(n_factories)]
    y_pos = {name: idx for idx, name in enumerate(row_names)}
    return nl, row_names, y_pos, len(row_names)


def _star_execution_legend_handles():
    return [
        mpatches.Patch(facecolor=color_map["TMR"], edgecolor="black", label="TMR"),
        # mpatches.Patch(facecolor=color_map["SE"], edgecolor="black", label="SE"),
        mpatches.Patch(facecolor=color_map["CNOT"], edgecolor="black", label="CNOT"),
        # mpatches.Patch(facecolor=color_map["S"], edgecolor="black", label="S gate"),
        mpatches.Patch(facecolor=color_map["move"], edgecolor="black", label="Move"),
        mpatches.Patch(
            facecolor=color_map["return_move"],
            edgecolor="black",
            label="Return Move",
        ),
        # mpatches.Patch(
        #     facecolor=color_map["RUS_success"],
        #     edgecolor="black",
        #     label="RUS:succsss",
        # ),
        # mpatches.Patch(
        #     facecolor=color_map["RUS_fail"],
        #     edgecolor="black",
        #     label="RUS:fail",
        # ),
        # mpatches.Patch(
        #     facecolor=color_map["TMR_fail"],
        #     edgecolor="black",
        #     label="TMR:fail",
        # ),
    ]


def _plot_circuit_execution_on_ax(
    ax,
    execution_log,
    n_factories: int,
    n_logical_qubits: int | None,
    *,
    show_box_text: bool = True,
    show_logical_qubits: bool = False,
    title: str = "STAR Execution Timeline",
    shared_xmax: float | None = None,
    show_legend: bool = True,
    unified_heading_fontsize: float | None = None,
    title_pad: float | None = None,
    suppress_box_text_ops: frozenset[str] | None = None,
    collapse_tmr: bool = True,
    font_scale: float = 1.0,
    box_border_width: float | None = None,
    show_factory_yticks: bool = True,
    factory_axis_label: str = "Factories",
    show_factory_ylabel: bool = True,
    axis_tick_font_scale: float | None = None,
) -> float:
    """Draw horizontal STAR execution on *ax*; returns max end time in the log."""
    plot_log = (
        _collapse_star_tmr_blocks(execution_log) if collapse_tmr else execution_log
    )
    _, row_names, y_pos, n_rows = _resolve_star_row_layout(
        plot_log, n_factories, n_logical_qubits, show_logical_qubits
    )
    resolved_border_width = (
        box_border_width
        if box_border_width is not None
        else _execution_timeline_box_border_width(show_box_text=show_box_text)
    )

    max_time = 0.0
    for entry in plot_log:
        start_time = entry_start(entry)
        end_time = entry_end(entry)
        factories = normalize_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        value = entry.get("targets")
        move_vecs = entry.get("move_vecs")
        max_time = max(max_time, end_time)
        duration = end_time - start_time
        if duration == 0:
            duration = 0.1
        color = color_map.get(operation, "#95a5a6")

        alpha = _BOX_ALPHA
        border_color = "black"
        border_width = resolved_border_width

        if operation == "Barrier":
            continue
            ax.axvline(
                x=start_time,
                color=color,
                linestyle="--",
                linewidth=3,
                alpha=1,
                zorder=10,
            )
        else:
            no_text_operations = {
                "RUS_success",
                "RUS_fail",
                "TMR_fail",
            }
            if suppress_box_text_ops:
                no_text_operations = no_text_operations | set(suppress_box_text_ops)
            if operation in no_text_operations:
                zorder = 15
                start_time -= 0.05
            else:
                zorder = 0

            for idx, factory_id in enumerate(factories):
                factory_value = _resolve_factory_value(value, idx)
                factory_move_vecs = _resolve_factory_move_vecs(move_vecs, idx)
                factory_aod = _resolve_factory_value(aod_assignment, idx)
                y_coord = factory_id
                if show_logical_qubits:
                    row_name = f"f{factory_id}"
                    if row_name not in y_pos:
                        continue
                    y_coord = y_pos[row_name]

                box_color = color
                if operation in ["move", "return_move"]:
                    box_color = _movement_color_for_aod(operation, factory_aod)

                rect = mpatches.Rectangle(
                    (start_time, y_coord - _BOX_Y_OFFSET),
                    duration,
                    _BOX_HEIGHT,
                    facecolor=box_color,
                    edgecolor=border_color,
                    linewidth=border_width,
                    zorder=zorder,
                    alpha=alpha,
                )
                ax.add_patch(rect)

                if show_box_text and operation not in no_text_operations:
                    text = _star_box_label(operation, factory_value, factory_move_vecs)

                    ax.text(
                        start_time + duration / 2,
                        y_coord,
                        text,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                        linespacing=0.9,
                        clip_on=True,
                    )

            if show_logical_qubits and operation == "CNOT":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                        duration,
                        _BOX_HEIGHT,
                        facecolor=color,
                        edgecolor=border_color,
                        linewidth=border_width,
                        zorder=zorder,
                        alpha=alpha,
                    )
                    ax.add_patch(rect)

            if show_logical_qubits and operation == "SE_q":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                        duration,
                        _BOX_HEIGHT,
                        facecolor=color,
                        edgecolor=border_color,
                        linewidth=border_width,
                        zorder=zorder,
                        alpha=alpha,
                    )
                    ax.add_patch(rect)
                    if show_box_text and operation not in no_text_operations:
                        ax.text(
                            start_time + duration / 2,
                            y_pos[row_name],
                            "SE",
                            ha="center",
                            va="center",
                            fontsize=10,
                            fontweight="bold",
                            color="black",
                            linespacing=0.9,
                            clip_on=True,
                        )

    xmax = shared_xmax if shared_xmax is not None else max_time * 1.01
    ax.set_xlim(0, xmax)
    if show_logical_qubits:
        ax.set_ylim(-0.5, len(row_names) - 0.5)
    else:
        ax.set_ylim(-0.5, n_factories - 0.5)
    title_fs = _scaled_font_size(
        (
            unified_heading_fontsize
            if unified_heading_fontsize is not None
            else _TITLE_FONT_SIZE
        ),
        font_scale,
    )
    xlabel_fs = _scaled_font_size(
        (
            unified_heading_fontsize
            if unified_heading_fontsize is not None
            else _AXIS_LABEL_FONT_SIZE
        ),
        font_scale,
    )
    if axis_tick_font_scale is not None:
        x_tick_fs = _scaled_font_size(_FIG_FONT_SIZE, axis_tick_font_scale)
        y_tick_fs = _scaled_font_size(_FIG_FONT_SIZE, axis_tick_font_scale)
    else:
        x_tick_fs, y_tick_fs = _execution_timeline_tick_font_sizes(
            show_box_text=show_box_text
        )
    ax.set_xlabel(
        "Time (circuit moments)",
        fontsize=xlabel_fs,
        fontweight="bold",
    )
    if show_logical_qubits:
        ax.set_ylabel("")
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
    else:
        if show_factory_yticks:
            ax.set_ylabel(
                factory_axis_label,
                fontsize=xlabel_fs,
                fontweight="bold",
            )
            ax.set_yticks(range(n_factories))
            ax.set_yticklabels([str(i) for i in range(n_factories)])
        elif show_factory_ylabel:
            ax.set_ylabel(
                factory_axis_label,
                fontsize=xlabel_fs,
                fontweight="bold",
            )
            ax.set_yticks([])
            ax.tick_params(axis="y", length=0)
        else:
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.tick_params(axis="y", length=0)
    if title:
        _tpad = 6 if title_pad is None else title_pad
        ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=_tpad)
    ax.tick_params(axis="x", labelsize=x_tick_fs)
    ax.tick_params(axis="y", labelsize=y_tick_fs)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if show_legend:
        legend_handles = _star_execution_legend_handles()
        leg = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=len(legend_handles),
            fontsize=_scaled_font_size(_LEGEND_FONT_SIZE, font_scale),
            columnspacing=1.2,
            handletextpad=0.5,
        )
        leg.set_zorder(20)
    return max_time


def _draw_timeline_time_highlight(
    ax,
    time_start: float,
    time_end: float,
    *,
    color: str = "black",
    alpha: float = 0.3,
) -> None:
    """Shade ``[time_start, time_end]`` across the full row height."""
    t0 = min(float(time_start), float(time_end))
    t1 = max(float(time_start), float(time_end))
    ylo, yhi = ax.get_ylim()
    ax.axvspan(
        t0,
        t1,
        ylo,
        yhi,
        facecolor=color,
        alpha=alpha,
        zorder=-5,
        linewidth=0,
    )


def _draw_trap_grid_time_window_highlight(
    ax,
    time_start: float,
    time_end: float,
    *,
    color: str = "black",
    alpha: float = 0.3,
) -> None:
    """Shade ``[time_start, time_end]`` on a timeline row (trap-grid spatial window)."""
    _draw_timeline_time_highlight(ax, time_start, time_end, color=color, alpha=alpha)


def _draw_timeline_shade_specs(ax, shade_specs: list[dict]) -> None:
    """Apply a list of shade specs from :func:`build_rus_move_shade_specs`."""
    for spec in shade_specs:
        _draw_timeline_time_highlight(
            ax,
            spec["time_start"],
            spec["time_end"],
            color=spec.get("color", "black"),
            alpha=float(spec.get("alpha", 0.22)),
        )


def _legend_patches_for_shade_specs(
    shade_specs: list[dict],
) -> list[mpatches.Patch]:
    """One legend patch per unique shade label (preserves first-seen order)."""
    seen: set[str] = set()
    handles: list[mpatches.Patch] = []
    for spec in shade_specs:
        label = spec.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        handles.append(
            mpatches.Patch(
                facecolor=spec.get("color", "black"),
                edgecolor="black",
                alpha=float(spec.get("alpha", 0.22)),
                label=label,
            )
        )
    return handles


def _stagger_close_xtick_labels(
    ax,
    tick_values: list[float],
    *,
    min_sep: float | None = None,
    max_offset_pts: float = 11.0,
) -> None:
    """Horizontally offset x tick labels when tick positions are very close."""
    if len(tick_values) < 2:
        return
    x_lo, x_hi = ax.get_xlim()
    span = float(x_hi) - float(x_lo)
    if span <= 1e-9:
        return
    threshold = min_sep if min_sep is not None else max(1.5, span * 0.035)
    sorted_vals = sorted({round(float(t), 9) for t in tick_values})
    offset_pts: dict[float, float] = {}
    idx = 0
    while idx < len(sorted_vals):
        cluster = [sorted_vals[idx]]
        j = idx + 1
        while j < len(sorted_vals) and sorted_vals[j] - sorted_vals[j - 1] < threshold:
            cluster.append(sorted_vals[j])
            j += 1
        if len(cluster) > 1:
            for k, tick_val in enumerate(cluster):
                frac = k / (len(cluster) - 1)
                offset_pts[tick_val] = -max_offset_pts + 2 * max_offset_pts * frac
        idx = j

    if not offset_pts:
        return
    fig = ax.figure
    for label in ax.get_xticklabels():
        try:
            tick_val = round(float(label.get_text()), 9)
        except ValueError:
            continue
        dx_pts = offset_pts.get(tick_val)
        if dx_pts is None:
            continue
        label.set_transform(
            label.get_transform()
            + ScaledTranslation(dx_pts / 72.0, 0, fig.dpi_scale_trans)
        )


def _merge_timeline_xticks(
    ax,
    extra_ticks: list[float],
    *,
    bold_ticks: list[float] | None = None,
    labeled_ticks: list[float] | None = None,
    stagger_close_ticks: bool = False,
) -> None:
    """Ensure highlight ticks appear on the shared time axis.

    *labeled_ticks* controls which positions get tick marks and labels (defaults to
    all *extra_ticks*). Use a single label (e.g. only ``17``) when the window spans
    two nearby moments (``17``–``18``).
    """
    if not extra_ticks:
        return
    label_ticks = (
        [float(t) for t in labeled_ticks]
        if labeled_ticks is not None
        else [float(t) for t in extra_ticks]
    )
    current = [float(t) for t in ax.get_xticks()]
    merged = sorted(
        {round(t, 9) for t in current} | {round(float(t), 9) for t in label_ticks}
    )
    ax.set_xticks(merged)
    ax.set_xticklabels([f"{t:g}" for t in merged])
    if bold_ticks:
        bold_set = {round(float(t), 9) for t in bold_ticks}
        for label in ax.get_xticklabels():
            try:
                tick_val = round(float(label.get_text()), 9)
            except ValueError:
                continue
            if tick_val in bold_set:
                label.set_fontweight("bold")
    if stagger_close_ticks and len(label_ticks) > 1:
        _stagger_close_xtick_labels(ax, label_ticks)


def plot_star_execution_subfigures(
    row_plots,
    *,
    show_logical_qubits: bool = True,
    figure_width: float = 50.0,
    figure_vertical_stretch: float = _STAR_SUBFIG_DEFAULT_VERTICAL_STRETCH,
    suptitle: str | None = None,
    save_path: str = "output/circuit_execution/star_execution_subfigures.pdf",
    collapse_tmr: bool = True,
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
    row_time_markers: list[list[float]] | None = None,
    marker_line_kwargs: dict | None = None,
    marker_legend_label: str = "Realtime control event",
    row_time_windows: list[tuple[float, float] | None] | None = None,
    row_time_shades: list[list[dict] | None] | None = None,
    highlight_xticks: list[float] | None = None,
    time_window_legend_label: str = "Movement window",
    time_window_shade_color: str = _TRAP_GRID_TIME_WINDOW_COLOR,
    time_window_shade_alpha: float = _TRAP_GRID_TIME_WINDOW_ALPHA,
    show_factory_yticks: bool = False,
    factory_axis_label: str = "Factories",
    subfig_heading_fontsize: float | None = None,
    legend_row_index: int | None = None,
):
    """Plot multiple horizontal STAR timelines as stacked subfigures (shared time axis).

    row_plots: list of (row_title, execution_log, n_factories, n_logical_qubits)
    figure_vertical_stretch: multiplies default height so each lane/box is taller on screen.
    row_time_markers: optional per-row x positions for marker lines (same order as row_plots)
    row_time_windows: optional per-row ``(t_start, t_end)`` shaded regions (e.g. async trap-grid)
    row_time_shades: optional per-row shade specs (e.g. RUS move/return intervals)
    highlight_xticks: extra x-axis tick positions (e.g. trap-grid window bounds)
    show_factory_yticks: when False, hide factory-index tick labels but keep *factory_axis_label*
    factory_axis_label: y-axis title when factory tick labels are hidden (default ``Factories``)
    legend_row_index: stacked-panel index for the legend; default picks the row with most
        timeline whitespace on the shared x-axis
    Writes *save_path* (with box labels) and a ``_no_text`` sibling (no box labels).
    """
    if not row_plots:
        print("No row plots to render")
        return
    if row_time_markers is not None and len(row_time_markers) != len(row_plots):
        raise ValueError("row_time_markers must have the same length as row_plots")
    if row_time_windows is not None and len(row_time_windows) != len(row_plots):
        raise ValueError("row_time_windows must have the same length as row_plots")
    if row_time_shades is not None and len(row_time_shades) != len(row_plots):
        raise ValueError("row_time_shades must have the same length as row_plots")

    row_count = len(row_plots)
    lane_counts = []
    for _, execution_log, nf, nlq in row_plots:
        _, _, _, n_rows = _resolve_star_row_layout(
            execution_log, nf, nlq, show_logical_qubits
        )
        lane_counts.append(n_rows)
    row_height_ratios = [max(0.8, lanes / 3.0) for lanes in lane_counts]

    def _logs_for_xmax():
        for _, execution_log, _, _ in row_plots:
            log = (
                _collapse_star_tmr_blocks(execution_log)
                if collapse_tmr
                else execution_log
            )
            yield log

    global_xmax = (
        max(max(entry_end(entry) for entry in log) for log in _logs_for_xmax()) * 1.01
    )

    row_max_end_times: list[float] = []
    for _, execution_log, _, _ in row_plots:
        log = (
            _collapse_star_tmr_blocks(execution_log) if collapse_tmr else execution_log
        )
        row_max_end_times.append(max(entry_end(entry) for entry in log))
    resolved_legend_row_index = (
        legend_row_index
        if legend_row_index is not None
        else _subplot_index_with_most_timeline_whitespace(row_max_end_times)
    )
    if not 0 <= resolved_legend_row_index < row_count:
        raise ValueError(
            f"legend_row_index must be in [0, {row_count}), got {resolved_legend_row_index}"
        )

    max_circuit_length = max(len(execution_log) for _, execution_log, _, _ in row_plots)
    with_text_fig_width = _execution_timeline_fig_width(
        figure_width, max_circuit_length, show_box_text=True
    )
    fig_height = 0.9 + 0.90 * float(figure_vertical_stretch) * sum(row_height_ratios)
    heading_fs = (
        _STAR_SUBFIG_HEADING_FONT_SIZE
        if subfig_heading_fontsize is None
        else float(subfig_heading_fontsize)
    )
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for show_box_text in (True, False):
        font_scale = _execution_timeline_font_scale(show_box_text=show_box_text)
        resolved_border_width = _execution_timeline_box_border_width(
            show_box_text=show_box_text,
            box_border_width=box_border_width,
            no_text_box_border_width=no_text_box_border_width,
        )
        fig_width = _execution_timeline_fig_width(
            figure_width, max_circuit_length, show_box_text=show_box_text
        )
        fig_height_plot = (
            fig_height
            if show_box_text
            else fig_height * (fig_width / with_text_fig_width)
        )
        fig, axes = plt.subplots(
            row_count,
            1,
            figsize=(fig_width, fig_height_plot),
            sharex=True,
            gridspec_kw={"height_ratios": row_height_ratios},
        )
        if row_count == 1:
            axes = [axes]

        for row_idx, (
            ax,
            (row_title, execution_log, n_factories, n_logical_qubits),
        ) in enumerate(zip(axes, row_plots)):
            _plot_circuit_execution_on_ax(
                ax,
                execution_log,
                n_factories,
                n_logical_qubits,
                show_box_text=show_box_text,
                show_logical_qubits=show_logical_qubits,
                title=row_title,
                shared_xmax=global_xmax,
                show_legend=False,
                unified_heading_fontsize=heading_fs,
                title_pad=4.0,
                collapse_tmr=collapse_tmr,
                font_scale=font_scale,
                box_border_width=resolved_border_width,
                show_factory_yticks=show_factory_yticks,
                factory_axis_label=factory_axis_label,
                show_factory_ylabel=True,
            )
            if row_time_shades is not None:
                shades = row_time_shades[row_idx]
                if shades:
                    _draw_timeline_shade_specs(ax, shades)
            if row_time_windows is not None:
                window = row_time_windows[row_idx]
                if window is not None:
                    _draw_timeline_time_highlight(
                        ax,
                        window[0],
                        window[1],
                        color=time_window_shade_color,
                        alpha=time_window_shade_alpha,
                    )
            if row_time_markers is not None:
                line_kwargs = {
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 0.8,
                    "alpha": 0.35,
                    "zorder": 25,
                }
                if marker_line_kwargs:
                    line_kwargs.update(marker_line_kwargs)
                for marker_t in row_time_markers[row_idx]:
                    t = float(marker_t)
                    if abs(t) < 1e-9:
                        continue
                    # Draw in axis-fraction Y coordinates so the marker slightly
                    # exceeds the plot frame and remains visible at the x-axis.
                    marker = ax.vlines(
                        t,
                        ymin=-0.03,
                        ymax=1.0,
                        transform=ax.get_xaxis_transform(),
                        **line_kwargs,
                    )
                    marker.set_clip_on(False)

        for ax in axes[:-1]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        if highlight_xticks:
            _merge_timeline_xticks(
                axes[-1],
                highlight_xticks,
                bold_ticks=highlight_xticks,
            )

        legend_handles = list(_star_execution_legend_handles())
        if row_time_markers is not None:
            legend_line_kwargs = {
                "color": "black",
                "linestyle": "-",
                "linewidth": 0.8,
                "alpha": 0.35,
            }
            if marker_line_kwargs:
                legend_line_kwargs.update(marker_line_kwargs)
            legend_handles.append(
                Line2D([0], [0], label=marker_legend_label, **legend_line_kwargs)
            )
        if row_time_windows is not None and any(
            w is not None for w in row_time_windows
        ):
            legend_handles.append(
                mpatches.Patch(
                    facecolor=time_window_shade_color,
                    edgecolor="black",
                    alpha=time_window_shade_alpha,
                    label=time_window_legend_label,
                )
            )
        if row_time_shades is not None:
            row_shade_labels: set[str] = set()
            for shades in row_time_shades:
                if not shades:
                    continue
                for patch in _legend_patches_for_shade_specs(shades):
                    if patch.get_label() in row_shade_labels:
                        continue
                    row_shade_labels.add(patch.get_label())
                    legend_handles.append(patch)

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98 if suptitle else 0.96))

        if suptitle:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb0 = (
                axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            )
            fig.suptitle(
                suptitle,
                fontsize=_scaled_font_size(heading_fs, font_scale),
                fontweight="bold",
                y=float(min(0.998, bb0.y1 + 0.008)),
                va="bottom",
            )

        # Legend at the original upper-right (panel with most timeline whitespace),
        # with that axes stacked above later panels so the box is not covered.
        for ax_idx, ax in enumerate(axes):
            ax.set_zorder(ax_idx + 1)
        legend_ax = axes[resolved_legend_row_index]
        legend_ax.set_zorder(len(axes) + 10)
        legend = legend_ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=1,
            fontsize=_scaled_font_size(max(11, heading_fs - 1), font_scale),
            frameon=True,
            framealpha=0.98,
        )
        legend.set_zorder(1000)
        legend.set_clip_on(False)

        out_path = _save_path_for_box_text_variant(
            save_path, show_box_text=show_box_text
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSTAR execution subfigure plot saved to: {out_path}")


def _combined_timeline_movement_fig_width(
    figure_width: float,
    circuit_length: int,
    *,
    show_box_text: bool,
    timeline_span: float | None = None,
    reference_timeline_span: float | None = None,
) -> float:
    """Provisional figure width in inches (timeline-heavy; movement is a narrow strip).

    When *timeline_span* is set, width follows the displayed time range (not log length).
    """
    ref = max(float(figure_width), 1.0)
    span = (
        float(timeline_span)
        if timeline_span is not None
        else float(max(circuit_length, 1))
    )
    ref_span = (
        float(reference_timeline_span) if reference_timeline_span is not None else span
    )
    span_scale = span / ref_span if ref_span > 1e-9 else 1.0
    if show_box_text:
        tl = max(ref * 0.45, span / 10 + 4.5)
    else:
        paper = float(_TCULT_EXEC_FIGURE_WIDTH)
        tl = paper * max(1.0, ref / 30.0) * span_scale
    return tl * 1.12


def _movement_slot_figure_width(ax_mv, row_height: float) -> float | None:
    """Natural movement-panel width in figure coordinates for *row_height*."""
    x1_lim, x2_lim = ax_mv.get_xlim()
    y1_lim, y2_lim = ax_mv.get_ylim()
    span_x = float(x2_lim) - float(x1_lim)
    span_y = float(y2_lim) - float(y1_lim)
    if span_y <= 1e-9 or span_x <= 1e-9 or row_height <= 0:
        return None
    return row_height * (span_x / span_y)


def _fit_movement_in_slot(
    ax_mv, x0: float, y0: float, height: float, width: float | None = None
) -> tuple[float, float, float, float] | None:
    """Place movement at *x0* with natural equal-aspect size (or fixed *width*).

    Returns ``(x0, y0, width, height)`` of the placed axes, or ``None``.
    """
    new_w = width if width is not None else _movement_slot_figure_width(ax_mv, height)
    if new_w is None or new_w <= 0:
        return None
    new_h = height
    new_y0 = y0 + (height - new_h) / 2.0
    ax_mv.set_position([x0, new_y0, new_w, new_h])
    return (x0, new_y0, new_w, new_h)


def _tighten_movement_axes_limits(ax_mv, *, inset_frac: float = -0.01) -> None:
    """Trim trap-grid axis limits to reduce left/right padding inside the movement panel."""
    x_lo, x_hi = ax_mv.get_xlim()
    y_lo, y_hi = ax_mv.get_ylim()
    x_span = float(x_hi) - float(x_lo)
    y_span = float(y_hi) - float(y_lo)
    if x_span > 1e-9:
        dx = x_span * inset_frac
        ax_mv.set_xlim(x_lo + dx, x_hi - dx)
    if y_span > 1e-9:
        dy = y_span * inset_frac
        ax_mv.set_ylim(y_lo + dy, y_hi - dy)


def _layout_combined_timeline_movement_rows(
    timeline_axes,
    movement_axes,
    *,
    left_margin: float = 0.065,
    right_margin: float = 0.98,
    col_gap: float = 0.010,
) -> float:
    """Timeline fills the row; movement is only as wide as its aspect needs (right side).

    Returns the rightmost figure coordinate used (for trimming excess right margin).
    """
    plot_w = right_margin - left_margin
    content_right = left_margin

    for ax_tl, ax_mv in zip(timeline_axes, movement_axes):
        pos_tl = ax_tl.get_position()
        y0 = pos_tl.y0
        h = pos_tl.height
        mv_w = _movement_slot_figure_width(ax_mv, h)
        if mv_w is None or mv_w <= 0:
            mv_w = plot_w * 0.12
        mv_w = min(mv_w, plot_w * 0.35)
        tl_w = max(plot_w - col_gap - mv_w, plot_w * 0.45)
        mv_w = plot_w - col_gap - tl_w
        mv_x0 = left_margin + tl_w + col_gap
        ax_tl.set_position([left_margin, y0, tl_w, h])
        placed = _fit_movement_in_slot(ax_mv, mv_x0, y0, h, width=mv_w)
        if placed is not None:
            content_right = max(content_right, placed[0] + placed[2])
        else:
            content_right = max(content_right, mv_x0 + mv_w)

    return content_right + 0.01


def _shrink_figure_to_content_width(
    fig, content_right: float, *, pad_frac: float = 0.02
) -> float:
    """Resize figure so horizontal inches match used content (fraction of axes span)."""
    w_in, h_in = fig.get_size_inches()
    used = min(0.99, float(content_right) + pad_frac)
    if used > 0.15 and used < 0.995:
        fig.set_size_inches(w_in * used, h_in)
    return used


def plot_star_timeline_movement_combined(
    row_plots,
    *,
    logic_qubit_locations,
    magic_state_locations,
    movement_time_start: float,
    movement_time_end: float,
    movement_code_distance: int = 3,
    movement_overlay: str = "both",
    movement_column_title: str | None = None,
    timeline_movement_width_ratios: tuple[float, float] = (1.0, 1.0),
    show_logical_qubits: bool = False,
    figure_width: float = 50.0,
    figure_vertical_stretch: float = _STAR_SUBFIG_DEFAULT_VERTICAL_STRETCH,
    suptitle: str | None = None,
    save_path: str = "output/circuit_execution/star_timeline_movement.pdf",
    collapse_tmr: bool = True,
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
    row_time_markers: list[list[float]] | None = None,
    marker_line_kwargs: dict | None = None,
    marker_legend_label: str = "Realtime control event",
    row_time_windows: list[tuple[float, float] | None] | None = None,
    row_time_shades: list[list[dict] | None] | None = None,
    highlight_xticks: list[float] | None = None,
    time_window_legend_label: str = "Movement window",
    time_window_shade_color: str = _TRAP_GRID_TIME_WINDOW_COLOR,
    time_window_shade_alpha: float = _TRAP_GRID_TIME_WINDOW_ALPHA,
    show_factory_yticks: bool = False,
    factory_axis_label: str = "Factories",
    subfig_heading_fontsize: float | None = None,
    timeline_xmax: float | None = 65.0,
    movement_axis_margin: float | None = 0.05,
    movement_axes_inset_frac: float = 0.05,
    highlight_xtick_labels: list[float] | None = None,
    marker_line_ymin: float = -0.01,
    marker_line_ymax: float = 1.0,
) -> None:
    """Timeline (left) and trap-grid movement (right) for each execution setting.

    One row per setting; writes *save_path* and a ``_no_text`` sibling.
    """
    if not row_plots:
        print("No row plots to render")
        return
    if row_time_markers is not None and len(row_time_markers) != len(row_plots):
        raise ValueError("row_time_markers must have the same length as row_plots")
    if row_time_windows is not None and len(row_time_windows) != len(row_plots):
        raise ValueError("row_time_windows must have the same length as row_plots")
    if row_time_shades is not None and len(row_time_shades) != len(row_plots):
        raise ValueError("row_time_shades must have the same length as row_plots")

    row_count = len(row_plots)
    lane_counts = []
    for _, execution_log, nf, nlq in row_plots:
        _, _, _, n_rows = _resolve_star_row_layout(
            execution_log, nf, nlq, show_logical_qubits
        )
        lane_counts.append(n_rows)
    row_height_ratios = [max(1.15, lanes / 2.2) for lanes in lane_counts]

    def _logs_for_xmax():
        for _, execution_log, _, _ in row_plots:
            log = (
                _collapse_star_tmr_blocks(execution_log)
                if collapse_tmr
                else execution_log
            )
            yield log

    full_xmax = (
        max(max(entry_end(entry) for entry in log) for log in _logs_for_xmax()) * 1.01
    )
    if timeline_xmax is not None:
        global_xmax = min(full_xmax, float(timeline_xmax))
    else:
        global_xmax = full_xmax

    max_circuit_length = max(len(execution_log) for _, execution_log, _, _ in row_plots)
    with_text_fig_width = _combined_timeline_movement_fig_width(
        figure_width,
        max_circuit_length,
        show_box_text=True,
        timeline_span=global_xmax,
        reference_timeline_span=full_xmax,
    )
    lane_sum = sum(row_height_ratios)
    fig_height = 1.2 + 1.0 * float(figure_vertical_stretch) * lane_sum
    heading_fs = (
        _STAR_SUBFIG_HEADING_FONT_SIZE
        if subfig_heading_fontsize is None
        else float(subfig_heading_fontsize)
    )
    t0 = min(float(movement_time_start), float(movement_time_end))
    t1 = max(float(movement_time_start), float(movement_time_end))
    if movement_column_title is None:
        movement_column_title = f"t={t0:g}–{t1:g}"
    movement_column_title = _capitalize_title_words(movement_column_title)
    if suptitle:
        suptitle = _capitalize_title_words(suptitle)

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for show_box_text in (True, False):
        font_scale = _execution_timeline_font_scale(show_box_text=show_box_text)
        resolved_border_width = _execution_timeline_box_border_width(
            show_box_text=show_box_text,
            box_border_width=box_border_width,
            no_text_box_border_width=no_text_box_border_width,
        )
        fig_width = _combined_timeline_movement_fig_width(
            figure_width,
            max_circuit_length,
            show_box_text=show_box_text,
            timeline_span=global_xmax,
            reference_timeline_span=full_xmax,
        )
        fig_height_plot = (
            fig_height
            if show_box_text
            else fig_height * (fig_width / with_text_fig_width)
        )
        fig = plt.figure(figsize=(fig_width, fig_height_plot))
        gs = fig.add_gridspec(
            row_count,
            2,
            width_ratios=list(timeline_movement_width_ratios),
            height_ratios=row_height_ratios,
            hspace=0.14,
            wspace=0.02,
        )
        title_font_scale = (
            font_scale
            if show_box_text
            else _COMBINED_TIMELINE_MOVEMENT_NO_TEXT_TITLE_SCALE
        )
        axis_font_scale = (
            1.08
            if show_box_text
            else _COMBINED_TIMELINE_MOVEMENT_NO_TEXT_AXIS_FONT_SCALE
        )
        row_title_fs = _scaled_font_size(heading_fs + 1, title_font_scale)
        col_title_fs = _scaled_font_size(heading_fs, title_font_scale)
        suptitle_fs = _scaled_font_size(heading_fs + 3, title_font_scale)
        legend_fs = _scaled_font_size(max(14, heading_fs + 1), title_font_scale)
        row_titles = [_capitalize_title_words(title) for title, _, _, _ in row_plots]
        timeline_axes = []
        movement_axes = []
        for row_idx, (
            _row_title,
            execution_log,
            n_factories,
            n_logical_qubits,
        ) in enumerate(row_plots):
            if row_idx == 0:
                ax_tl = fig.add_subplot(gs[row_idx, 0])
            else:
                ax_tl = fig.add_subplot(gs[row_idx, 0], sharex=timeline_axes[0])
            timeline_axes.append(ax_tl)
            ax_mv = fig.add_subplot(gs[row_idx, 1])
            movement_axes.append(ax_mv)

            _plot_circuit_execution_on_ax(
                ax_tl,
                execution_log,
                n_factories,
                n_logical_qubits,
                show_box_text=show_box_text,
                show_logical_qubits=show_logical_qubits,
                title="",
                shared_xmax=global_xmax,
                show_legend=False,
                unified_heading_fontsize=heading_fs,
                title_pad=4.0,
                collapse_tmr=collapse_tmr,
                font_scale=axis_font_scale,
                axis_tick_font_scale=axis_font_scale,
                box_border_width=resolved_border_width,
                show_factory_yticks=show_factory_yticks,
                factory_axis_label=factory_axis_label,
                show_factory_ylabel=True,
            )
            if row_time_shades is not None:
                shades = row_time_shades[row_idx]
                if shades:
                    _draw_timeline_shade_specs(ax_tl, shades)
            if row_time_windows is not None:
                window = row_time_windows[row_idx]
                if window is not None:
                    _draw_timeline_time_highlight(
                        ax_tl,
                        window[0],
                        window[1],
                        color=time_window_shade_color,
                        alpha=time_window_shade_alpha,
                    )
            if row_time_markers is not None:
                line_kwargs = {
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 0.8,
                    "alpha": 0.35,
                    "zorder": 25,
                }
                if marker_line_kwargs:
                    line_kwargs.update(marker_line_kwargs)
                for marker_t in row_time_markers[row_idx]:
                    t = float(marker_t)
                    if abs(t) < 1e-9:
                        continue
                    marker = ax_tl.vlines(
                        t,
                        ymin=marker_line_ymin,
                        ymax=marker_line_ymax,
                        transform=ax_tl.get_xaxis_transform(),
                        **line_kwargs,
                    )
                    marker.set_clip_on(False)

            if not draw_trap_grid_time_range_on_axes(
                ax_mv,
                execution_log,
                logic_qubit_locations,
                magic_state_locations,
                time_start=movement_time_start,
                time_end=movement_time_end,
                title=None,
                code_distance=movement_code_distance,
                movement_overlay=movement_overlay,  # type: ignore[arg-type]
                axis_margin=movement_axis_margin,
                arrow_label_offset=0.03,
            ):
                ax_mv.text(
                    0.5,
                    0.5,
                    "No movement in window",
                    ha="center",
                    va="center",
                    transform=ax_mv.transAxes,
                    fontsize=_scaled_font_size(heading_fs - 2, font_scale),
                )
                ax_mv.set_axis_off()
            elif movement_axes_inset_frac > 0:
                _tighten_movement_axes_limits(
                    ax_mv, inset_frac=movement_axes_inset_frac
                )

        for ax_tl in timeline_axes[:-1]:
            ax_tl.set_xlabel("")
            ax_tl.tick_params(axis="x", labelbottom=False)

        if highlight_xticks:
            tick_labels = (
                highlight_xtick_labels
                if highlight_xtick_labels is not None
                else highlight_xticks
            )
            _merge_timeline_xticks(
                timeline_axes[-1],
                highlight_xticks,
                bold_ticks=tick_labels,
                labeled_ticks=tick_labels,
            )

        legend_handles = list(_star_execution_legend_handles())
        if row_time_markers is not None:
            legend_line_kwargs = {
                "color": "black",
                "linestyle": "-",
                "linewidth": 0.8,
                "alpha": 0.35,
            }
            if marker_line_kwargs:
                legend_line_kwargs.update(marker_line_kwargs)
            legend_handles.append(
                Line2D([0], [0], label=marker_legend_label, **legend_line_kwargs)
            )
        if row_time_windows is not None and any(
            w is not None for w in row_time_windows
        ):
            legend_handles.append(
                mpatches.Patch(
                    facecolor=time_window_shade_color,
                    edgecolor="black",
                    alpha=time_window_shade_alpha,
                    label=time_window_legend_label,
                )
            )
        if row_time_shades is not None:
            row_shade_labels: set[str] = set()
            for shades in row_time_shades:
                if not shades:
                    continue
                for patch in _legend_patches_for_shade_specs(shades):
                    if patch.get_label() in row_shade_labels:
                        continue
                    row_shade_labels.add(patch.get_label())
                    legend_handles.append(patch)

        layout_top = 0.96 if suptitle else 0.97
        layout_bottom = 0.22
        fig.tight_layout(rect=(0.02, layout_bottom, 0.98, layout_top))

        content_right = _layout_combined_timeline_movement_rows(
            timeline_axes,
            movement_axes,
        )
        fig.subplots_adjust(right=min(0.98, content_right))
        content_center = (0.065 + content_right) / 2.0

        row_title_offset = 0.008
        first_row_title_y = timeline_axes[0].get_position().y1 + row_title_offset

        for row_idx, row_title in enumerate(row_titles):
            pos = timeline_axes[row_idx].get_position()
            fig.text(
                content_center,
                pos.y1 + row_title_offset,
                row_title,
                ha="center",
                va="bottom",
                fontsize=row_title_fs,
                fontweight="bold",
                transform=fig.transFigure,
            )

        if movement_axes:
            mv0 = movement_axes[0].get_position()
            fig.text(
                (mv0.x0 + mv0.x1) / 2.0,
                mv0.y1 + 0.006,
                movement_column_title,
                ha="center",
                va="bottom",
                fontsize=col_title_fs,
                fontweight="bold",
                transform=fig.transFigure,
            )

        if suptitle:
            suptitle_gap = 0.006
            suptitle_line_frac = 0.022
            fig.text(
                content_center,
                first_row_title_y + suptitle_line_frac + suptitle_gap,
                suptitle,
                ha="center",
                va="bottom",
                fontsize=suptitle_fs,
                fontweight="bold",
                transform=fig.transFigure,
            )

        n_handles = len(legend_handles)
        # legend_ncol = min(n_handles, 5) if n_handles <= 6 else (n_handles + 1) // 2
        legend_ncol = min(n_handles, 7) if n_handles <= 7 else (n_handles + 1) // 2
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(content_center, 0.0),
            bbox_transform=fig.transFigure,
            ncol=legend_ncol,
            fontsize=legend_fs,
            frameon=True,
            framealpha=0.98,
            columnspacing=1.0,
            handletextpad=0.4,
        )

        _shrink_figure_to_content_width(fig, content_right)

        out_path = _save_path_for_box_text_variant(
            save_path, show_box_text=show_box_text
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"\nSTAR timeline + movement plot saved to: {out_path}")


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================
def plot_circuit_execution(
    execution_log,
    n_factories,
    n_logical_qubits: int | None = None,
    save_path="output/circuit_execution.pdf",
    figure_width=16,
    show_logical_qubits: bool = False,
    collapse_tmr: bool = True,
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
):
    """
    Plot circuit execution timeline showing operations on each factory.

    Writes *save_path* (with box labels) and a ``_no_text`` sibling (no box labels).
    """
    if not execution_log:
        print("No execution log to plot")
        return
    circuit_length = len(execution_log)
    with_text_fig_width = _execution_timeline_fig_width(
        figure_width, circuit_length, show_box_text=True
    )
    _, _, _, n_rows = _resolve_star_row_layout(
        execution_log, n_factories, n_logical_qubits, show_logical_qubits
    )
    base_height = max(10, n_rows)
    base_path = save_path.split(".")[0]
    output_dir = os.path.dirname(base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    for show_box_text in (True, False):
        font_scale = _execution_timeline_font_scale(show_box_text=show_box_text)
        resolved_border_width = _execution_timeline_box_border_width(
            show_box_text=show_box_text,
            box_border_width=box_border_width,
            no_text_box_border_width=no_text_box_border_width,
        )
        fig_width = _execution_timeline_fig_width(
            figure_width, circuit_length, show_box_text=show_box_text
        )
        height_scale = fig_width / with_text_fig_width
        fig, ax = plt.subplots(figsize=(fig_width, base_height * height_scale))
        _plot_circuit_execution_on_ax(
            ax,
            execution_log,
            n_factories,
            n_logical_qubits,
            show_box_text=show_box_text,
            show_logical_qubits=show_logical_qubits,
            title="STAR Execution Timeline",
            shared_xmax=None,
            show_legend=True,
            collapse_tmr=collapse_tmr,
            font_scale=font_scale,
            box_border_width=resolved_border_width,
        )
        plt.tight_layout(rect=(0, 0.12, 1, 1))
        out_path = _save_path_for_box_text_variant(
            save_path, show_box_text=show_box_text
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nCircuit execution plot saved to: {out_path}")


def plot_circuit_execution_vertical(
    execution_log,
    n_factories,
    n_logical_qubits: int | None = None,
    save_path="output/circuit_execution_vertical.pdf",
    figure_height=16,
    show_logical_qubits: bool = False,
    collapse_tmr: bool = True,
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
):
    """
    Plot circuit execution timeline with vertical time axis (top to bottom).
    Factories are arranged horizontally (left to right).

    Writes *save_path* (with box labels) and a ``_no_text`` sibling (no box labels).
    """
    if not execution_log:
        print("No execution log to plot")
        return

    plot_log = (
        _collapse_star_tmr_blocks(execution_log) if collapse_tmr else execution_log
    )
    circuit_length = len(execution_log)
    base_path = save_path.split(".")[0]
    output_dir = os.path.dirname(base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    nl = n_logical_qubits
    if show_logical_qubits and nl is None:
        max_qubit = -1
        for entry in execution_log:
            value = entry.get("targets")
            qubits = _extract_qubits_from_value(value)
            if qubits:
                max_qubit = max(max_qubit, max(qubits))
        nl = max_qubit + 1 if max_qubit >= 0 else 0

    if show_logical_qubits:
        with_text_fig_width = max(6, nl * 0.8 + n_factories * 0.8)
    else:
        with_text_fig_width = max(8, n_factories * 0.8)
    base_fig_height = circuit_length / 5

    for show_box_text in (True, False):
        font_scale = _execution_timeline_font_scale(show_box_text=show_box_text)
        resolved_border_width = _execution_timeline_box_border_width(
            show_box_text=show_box_text,
            box_border_width=box_border_width,
            no_text_box_border_width=no_text_box_border_width,
        )
        if show_logical_qubits:
            row_names = [f"q{i}" for i in range(nl)] + [
                f"f{i}" for i in range(n_factories)
            ]
            y_pos = {name: idx for idx, name in enumerate(row_names)}
            fig_width = (
                with_text_fig_width
                if show_box_text
                else float(_TCULT_EXEC_FIGURE_WIDTH)
            )
            fig, ax = plt.subplots(
                figsize=(
                    fig_width,
                    base_fig_height * (fig_width / with_text_fig_width),
                )
            )
        else:
            row_names = None
            y_pos = None
            fig_width = (
                with_text_fig_width
                if show_box_text
                else float(_TCULT_EXEC_FIGURE_WIDTH)
            )
            fig, ax = plt.subplots(
                figsize=(
                    fig_width,
                    base_fig_height * (fig_width / with_text_fig_width),
                )
            )

        for entry in plot_log:
            start_time = entry_start(entry)
            end_time = entry_end(entry)
            factories = normalize_factories(entry.get("factories"))
            operation = entry.get("operation")
            aod_assignment = entry.get("aod_assignment")
            value = entry.get("targets")
            move_vecs = entry.get("move_vecs")

            duration = end_time - start_time
            if duration == 0:
                duration = 0.1
            color = color_map.get(operation, "#95a5a6")

            alpha = _BOX_ALPHA
            border_color = "black"
            border_width = resolved_border_width

            if operation == "Barrier":
                continue

            no_text_operations = {
                "RUS_success",
                "RUS_fail",
                "TMR_fail",
            }
            if operation in no_text_operations:
                zorder = 15
                start_time -= 0.05
            else:
                zorder = 0

            for idx, factory_id in enumerate(factories):
                factory_value = _resolve_factory_value(value, idx)
                factory_move_vecs = _resolve_factory_move_vecs(move_vecs, idx)
                factory_aod = _resolve_factory_value(aod_assignment, idx)
                x_coord = factory_id
                if show_logical_qubits:
                    row_name = f"f{factory_id}"
                    if row_name not in y_pos:
                        continue
                    x_coord = y_pos[row_name]

                box_color = color
                if operation in ["move", "return_move"]:
                    box_color = _movement_color_for_aod(operation, factory_aod)

                rect = mpatches.Rectangle(
                    (x_coord - _BOX_Y_OFFSET, start_time),
                    _BOX_HEIGHT,
                    duration,
                    facecolor=box_color,
                    edgecolor=border_color,
                    alpha=alpha,
                    linewidth=border_width,
                    zorder=zorder,
                )
                ax.add_patch(rect)

                if show_box_text and operation not in no_text_operations:
                    text = _star_box_label(operation, factory_value, factory_move_vecs)
                    ax.text(
                        x_coord,
                        start_time + duration / 2,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="black",
                        linespacing=0.9,
                        clip_on=True,
                    )

            if show_logical_qubits and operation == "CNOT":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (y_pos[row_name] - _BOX_Y_OFFSET, start_time),
                        _BOX_HEIGHT,
                        duration,
                        facecolor=color,
                        edgecolor=border_color,
                        alpha=alpha,
                        linewidth=border_width,
                        zorder=zorder,
                    )
                    ax.add_patch(rect)

            if show_logical_qubits and operation == "SE_q":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (y_pos[row_name] - _BOX_Y_OFFSET, start_time),
                        _BOX_HEIGHT,
                        duration,
                        facecolor=color,
                        edgecolor=border_color,
                        alpha=alpha,
                        linewidth=border_width,
                        zorder=zorder,
                    )
                    ax.add_patch(rect)
                    if show_box_text:
                        ax.text(
                            y_pos[row_name],
                            start_time + duration / 2,
                            "SE",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color="black",
                            linespacing=0.9,
                            clip_on=True,
                        )

        ax.set_ylim(max(entry_end(e) for e in plot_log) * 1.01, 0)
        if show_logical_qubits:
            ax.set_xlim(-0.5, len(row_names) - 0.5)
        else:
            ax.set_xlim(-0.5, n_factories - 0.5)
        ax.set_ylabel("")
        if show_logical_qubits:
            ax.set_xlabel(
                "Qubits and Magic State Factories",
                fontsize=_scaled_font_size(_AXIS_LABEL_FONT_SIZE, font_scale),
                fontweight="bold",
            )
            ax.set_xticks(range(len(row_names)))
            ax.set_xticklabels(row_names)
            ax.set_title(
                "STAR Execution Timeline (Vertical Time)",
                fontsize=_scaled_font_size(_TITLE_FONT_SIZE, font_scale),
                fontweight="bold",
            )
        else:
            ax.set_xlabel(
                "Magic State Factory ID",
                fontsize=_scaled_font_size(_AXIS_LABEL_FONT_SIZE, font_scale),
                fontweight="bold",
            )
            ax.set_xticks(range(n_factories))
            ax.set_title(
                "STAR Execution Timeline (Vertical Time)",
                fontsize=_scaled_font_size(_TITLE_FONT_SIZE, font_scale),
                fontweight="bold",
            )
        x_tick_fs, y_tick_fs = _execution_timeline_tick_font_sizes(
            show_box_text=show_box_text
        )
        ax.tick_params(axis="x", labelsize=x_tick_fs)
        ax.tick_params(axis="y", labelsize=y_tick_fs)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        leg = ax.legend(
            handles=_star_execution_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=len(_star_execution_legend_handles()),
            fontsize=_scaled_font_size(_LEGEND_FONT_SIZE, font_scale),
            columnspacing=1.2,
            handletextpad=0.5,
        )
        leg.set_zorder(20)
        plt.tight_layout(rect=(0, 0.12, 1, 1))
        out_path = _save_path_for_box_text_variant(
            save_path, show_box_text=show_box_text
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nCircuit execution plot (vertical) saved to: {out_path}")


def plot_t_cultivation_execution(
    execution_log,
    n_qubits: int,
    n_factories: int,
    save_path="output/circuit_execution/t_cultivation_execution.pdf",
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
):
    """Plot a single T-cultivation timeline with qubit and factory lanes.

    Writes *save_path* (with box labels) and a ``_no_text`` sibling (no box labels).
    """
    if not execution_log:
        print("No execution log to plot")
        return

    rows = [f"q{i}" for i in range(n_qubits)] + [f"f{i}" for i in range(n_factories)]
    with_text_fig_width = 10.0
    base_fig_height = max(4, 0.72 * len(rows))
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    for show_box_text in (True, False):
        font_scale = _execution_timeline_font_scale(show_box_text=show_box_text)
        resolved_border_width = _execution_timeline_box_border_width(
            show_box_text=show_box_text,
            box_border_width=box_border_width,
            no_text_box_border_width=no_text_box_border_width,
        )
        fig_width = (
            with_text_fig_width if show_box_text else float(_TCULT_EXEC_FIGURE_WIDTH)
        )
        fig, ax = plt.subplots(
            figsize=(
                fig_width,
                base_fig_height * (fig_width / with_text_fig_width),
            )
        )
        _plot_t_cultivation_execution_on_ax(
            ax,
            execution_log,
            n_qubits=n_qubits,
            n_factories=n_factories,
            title="T-Cultivation Execution Timeline",
            show_legend=True,
            show_box_text=show_box_text,
            font_scale=font_scale,
            box_border_width=resolved_border_width,
        )
        plt.tight_layout()
        out_path = _save_path_for_box_text_variant(
            save_path, show_box_text=show_box_text
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nT-cultivation plot saved to: {out_path}")


def _t_cultivation_legend_handles():
    legend_ops = [
        ("H", "H"),
        ("CNOT", "CNOT"),
        ("S", "S"),
        ("T", "T"),
        ("SE_stage_1", "Check Stage"),
        ("SE_stage_2", "Escape Stage"),
        ("move", "Move"),
        ("return_move", "Return Move"),
    ]
    return [
        mpatches.Patch(facecolor=color_map[key], edgecolor="black", label=label)
        for key, label in legend_ops
        if key in color_map
    ]


def _plot_t_cultivation_execution_on_ax(
    ax,
    execution_log,
    *,
    n_qubits: int,
    n_factories: int,
    title: str,
    show_legend: bool,
    shared_xmax: float | None = None,
    unified_heading_fontsize: float | None = None,
    title_pad: float | None = None,
    show_box_text: bool = True,
    font_scale: float = 1.0,
    box_border_width: float | None = None,
) -> None:
    rows = [f"q{i}" for i in range(n_qubits)] + [f"f{i}" for i in range(n_factories)]
    y_pos = {name: idx for idx, name in enumerate(rows)}
    max_time = max(entry_end(entry) for entry in execution_log)
    resolved_border_width = (
        box_border_width
        if box_border_width is not None
        else _execution_timeline_box_border_width(show_box_text=show_box_text)
    )

    for entry in execution_log:
        start_time = entry_start(entry)
        end_time = entry_end(entry)
        factories = normalize_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        value = entry.get("targets")
        move_vecs = entry.get("move_vecs")

        operation = _canonical_tcult_operation(operation)

        if operation in ("stage_2_success", "stage_2_fail"):
            continue

        duration = max(end_time - start_time, 0.1)
        color = color_map.get(operation, "#95a5a6")
        if operation in {"move", "return_move"}:
            box_color = _movement_color_for_aod(operation, aod_assignment)
        else:
            box_color = color

        for idx, factory_id in enumerate(factories):
            row_name = f"f{factory_id}"
            if row_name not in y_pos:
                continue
            y_coord = y_pos[row_name]
            rect = mpatches.Rectangle(
                (start_time, y_coord - _BOX_Y_OFFSET),
                duration,
                _BOX_HEIGHT,
                facecolor=box_color,
                edgecolor="black",
                linewidth=resolved_border_width,
                alpha=_BOX_ALPHA,
            )
            ax.add_patch(rect)
            if show_box_text and operation in {"move", "return_move"}:
                factory_move_vecs = _resolve_factory_move_vecs(move_vecs, idx)
                text = _star_box_label(operation, None, factory_move_vecs)
                if text:
                    ax.text(
                        start_time + duration / 2,
                        y_coord,
                        text,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                        linespacing=0.9,
                        clip_on=True,
                    )
        qubits = _extract_qubits_from_targets(value)
        is_circuit_move = (
            not factories
            and move_vecs is not None
            and operation in {"move", "return_move", "CNOT"}
        )
        is_circuit_cnot_gate = (
            not factories
            and operation == "CNOT"
            and move_vecs is None
            and qubits
        )

        if is_circuit_move and qubits:
            move_op = operation if operation in {"move", "return_move"} else "move"
            box_color = _movement_color_for_aod(move_op, aod_assignment)
            for qubit in qubits:
                row_name = f"q{qubit}"
                if row_name not in y_pos:
                    continue
                y_coord = y_pos[row_name]
                rect = mpatches.Rectangle(
                    (start_time, y_coord - _BOX_Y_OFFSET),
                    duration,
                    _BOX_HEIGHT,
                    facecolor=box_color,
                    edgecolor="black",
                    linewidth=resolved_border_width,
                    alpha=_BOX_ALPHA,
                )
                ax.add_patch(rect)
                if show_box_text:
                    pair_move_vecs = _resolve_factory_move_vecs(move_vecs, 0)
                    text = _star_box_label(move_op, None, pair_move_vecs)
                    if text:
                        ax.text(
                            start_time + duration / 2,
                            y_coord,
                            text,
                            ha="center",
                            va="center",
                            fontsize=10,
                            fontweight="bold",
                            color="black",
                            linespacing=0.9,
                            clip_on=True,
                        )
        elif is_circuit_cnot_gate:
            for qubit in qubits:
                row_name = f"q{qubit}"
                if row_name not in y_pos:
                    continue
                rect = mpatches.Rectangle(
                    (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                    duration,
                    _BOX_HEIGHT,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=resolved_border_width,
                    alpha=_BOX_ALPHA,
                )
                ax.add_patch(rect)

        if factories and operation == "CNOT" and qubits:
            t_color = color_map.get("CNOT(T)", color)
            for qubit in qubits:
                row_name = f"q{qubit}"
                if row_name not in y_pos:
                    continue
                rect = mpatches.Rectangle(
                    (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                    duration,
                    _BOX_HEIGHT,
                    facecolor=t_color,
                    edgecolor="black",
                    linewidth=resolved_border_width,
                    alpha=_BOX_ALPHA,
                )
                ax.add_patch(rect)

    ax.set_xlim(0, shared_xmax if shared_xmax is not None else max_time * 1.03)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    title_fs = _scaled_font_size(
        (
            unified_heading_fontsize
            if unified_heading_fontsize is not None
            else _TITLE_FONT_SIZE
        ),
        font_scale,
    )
    xlabel_fs = _scaled_font_size(
        (
            unified_heading_fontsize
            if unified_heading_fontsize is not None
            else _AXIS_LABEL_FONT_SIZE
        ),
        font_scale,
    )
    x_tick_fs, y_tick_fs = _execution_timeline_tick_font_sizes(
        show_box_text=show_box_text,
        y_tick_base=_FIG_FONT_SIZE - 1,
    )
    ax.set_xlabel(
        "Time (circuit moments)",
        fontsize=xlabel_fs,
        fontweight="bold",
    )
    ax.set_ylabel("")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(axis="x", labelsize=x_tick_fs)
    ax.tick_params(axis="y", labelsize=y_tick_fs)
    _tpad = 6 if title_pad is None else title_pad
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=_tpad)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    if show_legend:
        ax.legend(
            handles=_t_cultivation_legend_handles(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=_scaled_font_size(_FIG_FONT_SIZE, font_scale),
            frameon=True,
        )


def plot_t_cultivation_execution_subfigures(
    row_plots,
    save_path="output/circuit_execution/t_cultivation_execution_subfigures.pdf",
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
):
    """Plot multiple T-cultivation timelines as row-wise subfigures.

    row_plots: list of (row_title, execution_log, n_qubits, n_factories)
    Writes *save_path* (with box labels) and a ``_no_text`` sibling (no box labels).
    """
    if not row_plots:
        print("No row plots to render")
        return

    row_count = len(row_plots)
    lane_counts = [n_qubits + n_factories for _, _, n_qubits, n_factories in row_plots]
    row_height_ratios = [max(0.8, lanes / 3.0) for lanes in lane_counts]
    global_xmax = (
        max(
            max(entry_end(entry) for entry in execution_log)
            for _, execution_log, _, _ in row_plots
        )
        * 1.03
    )

    row_max_end_times = [
        max(entry_end(entry) for entry in execution_log)
        for _, execution_log, _, _ in row_plots
    ]
    legend_row_index = _subplot_index_with_most_timeline_whitespace(row_max_end_times)

    with_text_fig_width = 10.0
    fig_height = 0.9 + 0.90 * sum(row_height_ratios)
    heading_fs = _STAR_SUBFIG_HEADING_FONT_SIZE
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for show_box_text in (True, False):
        font_scale = _execution_timeline_font_scale(show_box_text=show_box_text)
        resolved_border_width = _execution_timeline_box_border_width(
            show_box_text=show_box_text,
            box_border_width=box_border_width,
            no_text_box_border_width=no_text_box_border_width,
        )
        fig_width = (
            with_text_fig_width if show_box_text else float(_TCULT_EXEC_FIGURE_WIDTH)
        )
        fig_height_plot = fig_height * (fig_width / with_text_fig_width)
        fig, axes = plt.subplots(
            row_count,
            1,
            figsize=(fig_width, fig_height_plot),
            sharex=True,
            gridspec_kw={"height_ratios": row_height_ratios},
        )
        if row_count == 1:
            axes = [axes]

        for ax, (row_title, execution_log, n_qubits, n_factories) in zip(
            axes, row_plots
        ):
            _plot_t_cultivation_execution_on_ax(
                ax,
                execution_log,
                n_qubits=n_qubits,
                n_factories=n_factories,
                title=row_title,
                show_legend=False,
                shared_xmax=global_xmax,
                unified_heading_fontsize=heading_fs,
                title_pad=4.0,
                show_box_text=show_box_text,
                font_scale=font_scale,
                box_border_width=resolved_border_width,
            )

        for ax in axes[:-1]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        fig.tight_layout(rect=(0, 0.0, 1, 0.98))

        legend_ax = axes[legend_row_index]
        legend_ax.legend(
            handles=_t_cultivation_legend_handles(),
            loc="upper right",
            bbox_to_anchor=(1.02, 1.0),
            ncol=1,
            fontsize=_scaled_font_size(max(10, _FIG_FONT_SIZE - 4), font_scale),
            frameon=True,
        )
        out_path = _save_path_for_box_text_variant(
            save_path, show_box_text=show_box_text
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nT-cultivation subfigure plot saved to: {out_path}")


def _legend_patch(color_key: str, label: str) -> mpatches.Patch:
    return mpatches.Patch(
        facecolor=color_map[color_key],
        edgecolor="black",
        label=label,
    )


def _combined_star_t_legend_handles():
    """Merged legend for STAR + T-cultivation subfigures (one entry per operation)."""
    merged: list[tuple[str, str]] = [
        ("TMR", "TMR"),
        ("S", "S gate"),
        ("CNOT", "CNOT"),
        # ("SE", "SE"),
        ("SE_stage_1", "Check Stage"),
        ("SE_stage_2", "Escape Stage"),
        ("move", "Move"),
        ("return_move", "Return Move"),
        ("RUS_success", "RUS:success"),
        ("RUS_fail", "RUS:fail"),
        ("TMR_fail", "TMR:fail"),
    ]
    return [_legend_patch(key, label) for key, label in merged if key in color_map]


def plot_star_t_cultivation_execution_subfigures(
    *,
    star_log: list,
    t_log: list,
    n_logical_qubits: int = 1,
    n_factories: int = 2,
    star_row_title: str = "STAR architecture",
    t_row_title: str = "T-cultivation architecture",
    save_path: str = (
        "output/circuit_execution/star_t_cultivation_1q2f_subfigures.pdf"
    ),
    suptitle: str | None = None,
    figure_width: float = 16.0,
    figure_vertical_stretch: float = 1.8,
    collapse_tmr: bool = True,
    box_border_width: float | None = None,
    no_text_box_border_width: float | None = None,
) -> None:
    """Stacked STAR (top) and T-cultivation (bottom) timelines on a shared time axis.

    Both rows use ``n_logical_qubits`` logical lanes and ``n_factories`` factory lanes.
    Writes *save_path* (with box labels) and a ``_no_text`` sibling (no box labels).
    """
    if not star_log or not t_log:
        print("STAR and T-cultivation execution logs are required")
        return

    n_qubits = int(n_logical_qubits)
    n_f = int(n_factories)
    _, _, _, star_lanes = _resolve_star_row_layout(
        star_log, n_f, n_qubits, show_logical_qubits=True
    )
    t_lanes = n_qubits + n_f
    row_height_ratios = [
        max(0.8, star_lanes / 3.0),
        max(0.8, t_lanes / 3.0),
    ]
    global_xmax = (
        max(
            max(entry_end(entry) for entry in star_log),
            max(entry_end(entry) for entry in t_log),
        )
        * 1.03
    )
    star_plot_log = _collapse_star_tmr_blocks(star_log) if collapse_tmr else star_log
    star_row_max = max(entry_end(entry) for entry in star_plot_log)
    t_row_max = max(entry_end(entry) for entry in t_log)
    legend_row_index = _subplot_index_with_most_timeline_whitespace(
        [star_row_max, t_row_max]
    )
    circuit_length = max(len(star_log), len(t_log))
    with_text_fig_width = _execution_timeline_fig_width(
        figure_width, circuit_length, show_box_text=True
    )
    fig_height = 0.9 + 0.90 * float(figure_vertical_stretch) * sum(row_height_ratios)

    heading_fs = _STAR_SUBFIG_HEADING_FONT_SIZE
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for show_box_text in (True, False):
        font_scale = _execution_timeline_font_scale(show_box_text=show_box_text)
        resolved_border_width = _execution_timeline_box_border_width(
            show_box_text=show_box_text,
            box_border_width=box_border_width,
            no_text_box_border_width=no_text_box_border_width,
        )
        fig_width = _execution_timeline_fig_width(
            figure_width, circuit_length, show_box_text=show_box_text
        )
        fig_height_plot = (
            fig_height
            if show_box_text
            else fig_height * (fig_width / with_text_fig_width)
        )
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(fig_width, fig_height_plot),
            sharex=True,
            gridspec_kw={"height_ratios": row_height_ratios},
        )

        _plot_circuit_execution_on_ax(
            axes[0],
            star_log,
            n_f,
            n_qubits,
            show_box_text=show_box_text,
            show_logical_qubits=True,
            title=star_row_title,
            shared_xmax=global_xmax,
            show_legend=False,
            unified_heading_fontsize=heading_fs,
            title_pad=4.0,
            suppress_box_text_ops=frozenset({"SE", "SE_q", "CNOT"}),
            collapse_tmr=collapse_tmr,
            font_scale=font_scale,
            box_border_width=resolved_border_width,
        )
        _plot_t_cultivation_execution_on_ax(
            axes[1],
            t_log,
            n_qubits=n_qubits,
            n_factories=n_f,
            title=t_row_title,
            show_legend=False,
            shared_xmax=global_xmax,
            unified_heading_fontsize=heading_fs,
            title_pad=4.0,
            show_box_text=show_box_text,
            font_scale=font_scale,
            box_border_width=resolved_border_width,
        )

        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", labelbottom=False)

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98 if suptitle else 0.96))

        if suptitle:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb0 = (
                axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            )
            fig.suptitle(
                suptitle,
                fontsize=_scaled_font_size(heading_fs, font_scale),
                fontweight="bold",
                y=float(min(0.998, bb0.y1 + 0.008)),
                va="bottom",
            )

        axes[legend_row_index].legend(
            handles=_combined_star_t_legend_handles(),
            loc="upper right",
            bbox_to_anchor=(1.06, 1.0),
            ncol=2,
            fontsize=_scaled_font_size(max(10, heading_fs - 2), font_scale),
            frameon=True,
            framealpha=0.95,
            columnspacing=0.4,
            handletextpad=0.2,
        )

        out_path = _save_path_for_box_text_variant(
            save_path, show_box_text=show_box_text
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSTAR vs T-cultivation subfigure plot saved to: {out_path}")
