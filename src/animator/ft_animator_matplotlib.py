from __future__ import annotations

import bisect
import math

from src.execution_log.event_helpers import normalize_factories

from .animator_matplotlib import Animator


class FTAnimator(Animator):
    """Animator for FT-ZAIR logical instructions.

    Timeline sampling walks *every* instruction so none are skipped. Short logical
    steps (especially SE sequences) allocate at least one slow-motion frame bucket
    each (see packing in ``execution_log_to_ft_animator_code`` plus ``ceil`` here).
    """

    FT_ZONE_COLORS = {
        "CNOT": (0.0, 0.0, 1.0, 0.30),
        "SE": (1.0, 0.55, 0.0, 0.35),
        "SE_stage_1": (0.55, 0.25, 0.85, 0.35),
        "SE_stage_2": (0.10, 0.70, 0.20, 0.35),
    }
    MARKER_EDGE_COLORS = {
        "RUS_fail": "#DDA413",
        "TMR_fail": "#007E15",
    }

    def update_init(self):
        super().update_init()
        self._ft_transient = []

    def create_schedule(self) -> int:
        """One schedule segment per FT instruction boundary; minimum one frame each."""
        self.piecewise_schedule = [(0, 0, 0)]
        instructions = list(self.code.get("instructions", []))
        runtime = float(self.code.get("runtime", 0.0))
        last_real_t = 0.0

        eps = 1e-12
        for inst in instructions[1:]:
            bt = float(inst["begin_time"])
            et = float(inst["end_time"])

            gap = bt - last_real_t
            if gap > eps:
                last_end_frame = self.piecewise_schedule[-1][0]
                n_reg = max(1, int(math.ceil(gap / self._mus_per_frm)))
                self.piecewise_schedule.append(
                    (
                        last_end_frame + n_reg,
                        0,
                        bt,
                    )
                )

            dur = max(0.0, et - bt)
            n_slow = max(1, int(math.ceil(dur / self._mus_per_frm_slow)))
            last_end_frame = self.piecewise_schedule[-1][0]
            self.piecewise_schedule.append(
                (
                    last_end_frame + n_slow,
                    1,
                    et,
                )
            )
            last_real_t = et

        if runtime > last_real_t + eps:
            last_end_frame = self.piecewise_schedule[-1][0]
            n_reg = max(1, int(math.ceil((runtime - last_real_t) / self._mus_per_frm)))
            self.piecewise_schedule.append((last_end_frame + n_reg, 0, runtime))

        return self.piecewise_schedule[-1][0]

    def _update_logical_zone(
        self, inst: dict, color: tuple[float, float, float, float]
    ) -> None:
        zone = int(inst.get("zair_inst", {}).get("zone_id", 0))
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {inst["begin_time"]:.2f}'
        self.entanglement_rect[zone].set_facecolor(color)
        self.entanglement_rect[zone].set_width(
            self.entanglement_rect_range[zone][1]
        )
        self.entanglement_rect[zone].set_height(
            self.entanglement_rect_range[zone][2]
        )
        self._clear_path_lines()

    def _draw_fail_markers(self, inst: dict) -> None:
        itype = str(inst.get("type", ""))
        color = self.MARKER_EDGE_COLORS.get(
            itype, self.MARKER_EDGE_COLORS["RUS_fail"]
        )
        self.inst_str += f' | {inst["id"]} {itype}'

        n_log = int(self.code.get("n_logic_qubits") or 0)
        xs: list[float] = []
        ys: list[float] = []
        for fid in normalize_factories(inst.get("factories")):
            pid = n_log + int(fid)
            if 0 <= pid < len(self.qubit_xs):
                xs.append(self.qubit_xs[pid])
                ys.append(self.qubit_ys[pid])
        if not xs:
            return
        sc = self.ax.scatter(
            xs,
            ys,
            s=360,
            facecolors="none",
            edgecolors=color,
            linewidths=3.5,
            zorder=15,
            marker="s",
        )
        self._ft_transient.append(sc)

    def update(self, f: int):
        true_frame = f - self._init_frm
        interval_ends = [interval[0] for interval in self.piecewise_schedule]
        index = bisect.bisect_right(interval_ends, true_frame)
        if index >= len(self.piecewise_schedule):
            index = len(self.piecewise_schedule) - 1
        elif index <= 0:
            index = 0
        tmp = self.piecewise_schedule[index]

        true_time = tmp[2] - (tmp[0] - true_frame) * (
            self._mus_per_frm_slow if tmp[1] else self._mus_per_frm
        )

        for artist in getattr(self, "_ft_transient", []) or []:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._ft_transient = []

        self.inst_str = ""
        self._reset_frame_overlays()
        if f >= self._init_frm:
            for inst in self.code["instructions"][1:]:
                if not (
                    true_time >= inst["begin_time"] and true_time < inst["end_time"]
                ):
                    continue
                itype = inst.get("type")
                if itype in ("move", "return_move"):
                    self.update_arrangement(true_time, inst["rearrange_job"])
                elif itype in ("H", "S", "Rz"):
                    self.update_1qGate(inst["gate_inst"])
                elif itype in ("CNOT", "SE", "SE_stage_1", "SE_stage_2"):
                    color = self.FT_ZONE_COLORS.get(
                        itype, self.FT_ZONE_COLORS["CNOT"]
                    )
                    self._update_logical_zone(inst, color)
                elif itype in ("RUS_fail", "TMR_fail"):
                    self._draw_fail_markers(inst)
                elif itype == "init":
                    continue
                elif itype == "Barrier":
                    continue
                else:
                    raise ValueError(f"unknown FT inst type {itype}")
            self._snap_to_settled_rearrange_layout_ft(true_time)
        self.title.set_text(self.inst_str)

    def _snap_to_settled_rearrange_layout_ft(self, true_time: float) -> None:
        for inst in self.code["instructions"][1:]:
            if inst.get("type") not in ("move", "return_move"):
                continue
            rj = inst.get("rearrange_job", {})
            b = float(rj.get("begin_time", 0.0))
            e = float(rj.get("end_time", b))
            if b <= true_time < e:
                return
        last_locs = None
        for inst in self.code["instructions"][1:]:
            if inst.get("type") not in ("move", "return_move"):
                continue
            rj = inst.get("rearrange_job", {})
            if float(rj.get("end_time", 0.0)) <= true_time:
                last_locs = rj.get("end_locs")
        if last_locs:
            self._sync_qubits_to_flat_locs(last_locs)
