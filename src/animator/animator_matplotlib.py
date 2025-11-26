from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
import bisect

from src.ds.architecture import Architecture


class Animator:

    # constants for animation
    FPS = 15  # 60 originally  # frames per second
    INIT_FRM = int(FPS / 5)  # initial empty frames, 1/5 second now
    PT_MICRON = 8  # scaling factor: points per micron
    MUS_PER_FRM = 150 / FPS  # microseconds per frame
    MUS_PER_FRM_SLOW = 7 / FPS  # in slow motion, i.e., Rydberg
    CANVAS_PADDING = 10
    RYDBERG_PADDING = 3  # around each entanglement zone

    # colors
    RYDBERG_COLOR = "b"
    SLM_COLOR = "g"
    QUBIT_COLOR = "k"
    AOD_COLORS = ["r", "c", "m", "y"]  # max 4 aods so far
    AOD_TRANS = 0.7

    def animate(
        self,
        code: dict,
        architecture: Architecture,
        output: str,
        scaling_factor: int = PT_MICRON,
        font: int = 10,
        ffmpeg: str = "ffmpeg",
    ):
        """
        Args:
            code (dict):
            output (str): filename to save output.
            scaling_factor (int, optional): the unit scaling factor between the
             animation and um. Defaults to PT_MICRON.
            font (int, optional): font size in the animation. Defaults to 10.
        """

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": font})
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg

        self.code = code
        self.architecture = architecture
        self.fig, self.ax = self.setup_canvas(scaling_factor)
        self.title = self.ax.set_title("")
        self.inst_str = ""

        num_frame = self.create_schedule()
        anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.update_init,
            frames=self.INIT_FRM + num_frame,
            # blit=True
        )
        anim.save(output, writer=FFMpegWriter(self.FPS))

    def create_schedule(self):
        """
        each frame is a sample on the time axis. There are two sampling rates
        one is regular, one is slow motion. The latter is used when Rydberg
        is happening because Rydberg is so fast that it won't appear in the
        video is using the regular sampling rate.

        The time axis looks like this:

        |_____________|...|_______________|...|_________|...|__________|

        where each | will be an entry in self.piecewise_schedule. The _ stands
        for regular period, the . stands for slow motion periods.

        In piecewise_schedule consists of 3-tuples. the first number is the
        frame at the | The second number is whether the period before the | is
        slow motion (1) or not (0). The third number is the real time at the |
        """

        self.piecewise_schedule = [
            (0, 0, 0),
        ]  # add the first trivial entry
        last_end_time = 0
        for inst in self.code["instructions"]:
            if inst["type"] == "rydberg":

                # add the entry corresponding to the regular period before
                last_end_frame = self.piecewise_schedule[-1][0]
                self.piecewise_schedule.append(
                    (
                        last_end_frame
                        + round(
                            (inst["begin_time"] - last_end_time) / self.MUS_PER_FRM
                        ),
                        0,
                        inst["begin_time"],
                    )
                )

                # add the entry corresponding to the slow period for this inst
                last_end_frame = self.piecewise_schedule[-1][0]
                self.piecewise_schedule.append(
                    (
                        last_end_frame
                        + round(
                            (inst["end_time"] - inst["begin_time"])
                            / self.MUS_PER_FRM_SLOW
                        ),
                        1,
                        inst["end_time"],
                    )
                )
                last_end_time = inst["end_time"]

        # add an entry of the left over runtime after the last rydberg
        if self.code["runtime"] > last_end_time:
            last_end_frame = self.piecewise_schedule[-1][0]
            self.piecewise_schedule.append(
                (
                    last_end_frame
                    + round((self.code["runtime"] - last_end_time) / self.MUS_PER_FRM),
                    0,
                    self.code["runtime"],
                )
            )
        return self.piecewise_schedule[-1][0]

    def setup_canvas(self, scaling_factor: int):
        """set up various objects before actually drawing."""

        # arch_range is [[bottom_left x,y], [top_right x,y]]
        # unit conversion factor from um to pt. Be defensive about rcParams
        # because some environments may provide non-numeric or zero dpi.
        dpi_val = plt.rcParams.get("figure.dpi", None)
        if dpi_val is None:
            try:
                dpi_val = matplotlib.rcParamsDefault.get("figure.dpi", 100)
            except Exception:
                dpi_val = 100
        try:
            dpi = float(dpi_val)
        except Exception:
            dpi = 100.0
        if dpi == 0:
            dpi = 100.0
        px = 1.0 / dpi * scaling_factor
        (
            fig,
            ax,
        ) = plt.subplots(
            figsize=(
                (
                    2 * self.CANVAS_PADDING
                    + self.architecture.arch_range[1][0]
                    - self.architecture.arch_range[0][0]
                )
                * px,
                (
                    2 * self.CANVAS_PADDING
                    + self.architecture.arch_range[1][1]
                    - self.architecture.arch_range[0][1]
                )
                * px,
            )
        )
        ax.set_xlim(
            [
                -self.CANVAS_PADDING + self.architecture.arch_range[0][0],
                self.CANVAS_PADDING + self.architecture.arch_range[1][0],
            ]
        )
        ax.set_ylim(
            [
                -self.CANVAS_PADDING + self.architecture.arch_range[0][1],
                self.CANVAS_PADDING + self.architecture.arch_range[1][1],
            ]
        )

        # rydberg_range is a list. Each entry is for an entanglement zone,
        # each entry is a pair [[bottom_left x, y], [top_right x,y]]
        # self.entanglement_rect_range is for matplotlib plotting. the first
        # entry is the bottom_left x,y (with padding). The second entry is
        # the width, and the third entry is the height.
        self.entanglement_rect_range = [
            (
                (
                    range_pair[0][0] - self.RYDBERG_PADDING,
                    range_pair[0][1] - self.RYDBERG_PADDING,
                ),
                range_pair[1][0] - range_pair[0][0] + 2 * self.RYDBERG_PADDING,
                range_pair[1][1] - range_pair[0][1] + 2 * self.RYDBERG_PADDING,
            )
            for range_pair in self.architecture.rydberg_range
        ]

        return fig, ax

    def update_init(self):
        # find all slms
        slm_xs = []
        slm_ys = []
        for slm_id, slm_arr in self.architecture.dict_SLM.items():
            for r in range(slm_arr.n_r):
                for c in range(slm_arr.n_c):
                    x, y = self.architecture.exact_SLM_location(slm_id, r, c)
                    slm_xs.append(x)
                    slm_ys.append(y)
        # draw slms
        self.ax.scatter(
            slm_xs, slm_ys, marker="o", s=40, facecolor="none", edgecolor=self.SLM_COLOR
        )

        # initialize qubits
        self.qubit_xs = []
        self.qubit_ys = []
        for q in self.code["instructions"][0]["init_locs"]:
            x, y = self.architecture.exact_SLM_location(q[1], q[2], q[3])
            self.qubit_xs.append(x)
            self.qubit_ys.append(y)
        # draw qubits
        self.qubit_scat = self.ax.scatter(
            self.qubit_xs, self.qubit_ys, marker=".", c=self.QUBIT_COLOR
        )

        # initialize aod cols
        self.aod_col_plots = {
            aod_id: [
                self.ax.axvline(
                    0,
                    self.architecture.arch_range[0][1],
                    self.architecture.arch_range[1][1],
                    c=(0, 0, 0, 0),
                    ls="--",
                )
                for _ in range(aod.n_c)
            ]
            for aod_id, aod in self.architecture.dict_AOD.items()
        }

        # initilize aod rows
        self.aod_row_plots = {
            aod_id: [
                self.ax.axhline(
                    0,
                    self.architecture.arch_range[0][0],
                    self.architecture.arch_range[1][0],
                    c=(0, 0, 0, 0),
                    ls="--",
                )
                for _ in range(aod.n_r)
            ]
            for aod_id, aod in self.architecture.dict_AOD.items()
        }

        # initialize Rydberg zones
        self.entanglement_rect = []
        for entangle_zone in self.entanglement_rect_range:
            rect = matplotlib.patches.Rectangle(
                entangle_zone[0],
                entangle_zone[1],
                entangle_zone[2],
                linewidth=1,
                edgecolor="none",
                # facecolor=(self.RYDBERG_COLOR, 0.3) # !
                facecolor=(0, 0, 1, 0.3),  #
            )
            self.ax.add_patch(rect)
            self.entanglement_rect.append(rect)

        # initialize single qubit gates
        self.qubit_1qGate = []

        # initialize path line
        self.path_line_collection = [
            LineCollection([], linewidths=2, capstyle="round")
            for _ in range(len(self.code["instructions"][0]["init_locs"]))
        ]
        self.path_line_loc = [[] for q in self.code["instructions"][0]["init_locs"]]
        self.path_line_color = [[] for q in self.code["instructions"][0]["init_locs"]]
        for lc in self.path_line_collection:
            self.ax.add_collection(lc)

        return

    def update(self, f: int):  # f is the frame
        true_frame = f - self.INIT_FRM  # consider the initial frozen frames

        # get which piecewise schedule f is in
        interval_ends = [interval[0] for interval in self.piecewise_schedule]
        index = bisect.bisect_right(interval_ends, true_frame)
        tmp = self.piecewise_schedule[index]
        # calculate true time of this frame: tmp[2] is the end time of this
        # period. tmp[0] is the end frame of this period. So we deduct the
        # remianing time from tmp[2]. The remaining time is calculated as the
        # product of remaining frames=tmp[0]-true_frame, and the sampling rate
        # which depends on whether this period is regular or slow motion.
        true_time = tmp[2] - (tmp[0] - true_frame) * (
            self.MUS_PER_FRM_SLOW if tmp[1] else self.MUS_PER_FRM
        )

        self.inst_str = ""
        # reset Rydberg zones to trivial
        for rect in self.entanglement_rect:
            rect.set_width(0)
            rect.set_height(0)
        # reset 1qGate to trivial
        for gate in self.qubit_1qGate:
            gate.remove()
        self.qubit_1qGate = []
        # reset AOD color to trivial
        for aod_id, aod in self.architecture.dict_AOD.items():
            for r in range(aod.n_r):
                self.aod_row_plots[aod_id][r].set_color((0, 0, 0, 0))
            for c in range(aod.n_c):
                self.aod_col_plots[aod_id][c].set_color((0, 0, 0, 0))

        if f >= self.INIT_FRM:
            for inst in self.code["instructions"][1:]:
                if true_time >= inst["begin_time"] and true_time < inst["end_time"]:
                    if inst["type"] == "rydberg":
                        self.update_rydberg(inst)
                    elif inst["type"] == "rearrangeJob":
                        self.update_arrangement(true_time, inst)
                    elif inst["type"] == "1qGate":
                        self.update_1qGate(inst)
                    else:
                        raise ValueError(f"unknown inst type {inst['type']}")
        self.title.set_text(self.inst_str)
        return

    def update_rydberg(self, inst: dict):
        self.inst_str += (
            f' | {inst["id"]} {inst["type"]} \n elapsed time: {inst["begin_time"]:.2f}'
        )
        self.entanglement_rect[inst["zone_id"]].set_width(
            self.entanglement_rect_range[inst["zone_id"]][1]
        )
        self.entanglement_rect[inst["zone_id"]].set_height(
            self.entanglement_rect_range[inst["zone_id"]][2]
        )
        # reset path line to be empty
        for lc in self.path_line_collection:
            lc.set_segments([])
        for tmp in self.path_line_loc:
            tmp.clear()
        for tmp in self.path_line_color:
            tmp.clear()

    def update_arrangement(self, time: float, inst: dict):
        self.inst_str += f' | {inst["id"]} {inst["type"]}'
        for detail_inst in inst["insts"]:
            if time >= detail_inst["begin_time"] and time < detail_inst["end_time"]:
                ratio = (time - detail_inst["begin_time"]) / (
                    detail_inst["end_time"] - detail_inst["begin_time"]
                )
                if detail_inst["type"] == "activate":
                    return self.update_activate(
                        ratio, time, detail_inst, inst["aod_id"]
                    )
                elif detail_inst["type"] == "deactivate":
                    return self.update_deactivate(
                        ratio, time, detail_inst, inst["aod_id"]
                    )
                elif detail_inst["type"].startswith("move"):
                    return self.update_move(
                        ratio,
                        time,
                        detail_inst,
                        zip(
                            detail_inst["begin_coord"],
                            detail_inst["end_coord"],
                        ),
                        inst["aod_id"],
                    )

    def update_activate(self, ratio: float, time: float, inst: dict, aod_id: int):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'
        for col_id, col_x in zip(inst["col_id"], inst["col_x"]):
            self.aod_col_plots[aod_id][col_id].set_xdata((col_x,))
            self.aod_col_plots[aod_id][col_id].set_color(
                # (self.AOD_COLORS[aod_id], ratio*self.AOD_TRANS) # !
                (1, 0, 0, ratio * self.AOD_TRANS)
            )
        for row_id, row_y in zip(inst["row_id"], inst["row_y"]):
            self.aod_row_plots[aod_id][row_id].set_ydata((row_y,))
            self.aod_row_plots[aod_id][row_id].set_color(
                # (self.AOD_COLORS[aod_id], ratio*self.AOD_TRANS) # !
                (1, 0, 0, ratio * self.AOD_TRANS)
            )

    def update_deactivate(self, ratio: float, time: float, inst: dict, aod_id: int):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'
        for col_id in inst["col_id"]:
            self.aod_col_plots[aod_id][col_id].set_color(
                # (self.AOD_COLORS[aod_id], (1-ratio)*self.AOD_TRANS) # !
                (1, 0, 0, (1 - ratio) * self.AOD_TRANS)
            )
        for row_id in inst["row_id"]:
            self.aod_row_plots[aod_id][row_id].set_color(
                # (self.AOD_COLORS[aod_id], (1-ratio)*self.AOD_TRANS) # !
                (1, 0, 0, (1 - ratio) * self.AOD_TRANS)
            )

    def update_move(
        self, ratio: float, time: float, inst: dict, qubit_coord, aod_id: int
    ):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'

        def interpolate(r: float, begin: int, end: int):
            D = end - begin
            return begin + 3 * D * (r**2) - 2 * D * (r**3)

        # update qubit
        for begin_coords_row, end_coords_row in qubit_coord:
            for begin_coords, end_coords in zip(begin_coords_row, end_coords_row):
                q_id = begin_coords["id"]
                old_xs = self.qubit_xs[q_id]
                old_ys = self.qubit_ys[q_id]

                self.qubit_xs[q_id] = interpolate(
                    ratio, begin_coords["x"], end_coords["x"]
                )
                self.qubit_ys[q_id] = interpolate(
                    ratio, begin_coords["y"], end_coords["y"]
                )
                self.qubit_scat.set_offsets(list(zip(self.qubit_xs, self.qubit_ys)))

                new_xs = self.qubit_xs[q_id]
                new_ys = self.qubit_ys[q_id]
                self.path_line_loc[q_id].append([(old_xs, old_ys), (new_xs, new_ys)])
                self.path_line_color[q_id] = []
                # print(q_id)
                # print(self.path_line_loc[q_id])
                for i in reversed(range(len(self.path_line_loc[q_id]))):
                    # self.path_line_color[q_id].append((105,105,105, 1))
                    self.path_line_color[q_id].append((128, 128, 128, pow(0.90, i)))
                # print(self.path_line_color[q_id])
                # input()

        for q, (seg, color) in enumerate(zip(self.path_line_loc, self.path_line_color)):
            self.path_line_collection[q].set_segments(seg)
            self.path_line_collection[q].set_color(color)

        # update AOD
        for row_id, row_begin_y, row_end_y in zip(
            inst["row_id"], inst["row_y_begin"], inst["row_y_end"]
        ):
            self.aod_row_plots[aod_id][row_id].set_ydata(
                (interpolate(ratio, row_begin_y, row_end_y),)
            )
            self.aod_row_plots[aod_id][row_id].set_color(self.AOD_COLORS[aod_id])
        for col_id, col_begin_x, col_end_x in zip(
            inst["col_id"], inst["col_x_begin"], inst["col_x_end"]
        ):
            self.aod_col_plots[aod_id][col_id].set_xdata(
                (interpolate(ratio, col_begin_x, col_end_x),)
            )
            self.aod_col_plots[aod_id][col_id].set_color(self.AOD_COLORS[aod_id])

    def update_1qGate(self, inst: dict):
        self.inst_str += (
            f' | {inst["id"]} {inst["type"]} \n elapsed time: {inst["end_time"]:.2f}'
        )

        for g in inst["gates"]:
            q = g["q"]
            x = self.qubit_xs[q]
            y = self.qubit_ys[q]
            self.qubit_1qGate.append(self.ax.scatter(x, y, s=300, color=(0, 1, 0, 0.5)))

        # reset path line to be empty
        for lc in self.path_line_collection:
            lc.set_segments([])
        for tmp in self.path_line_loc:
            tmp.clear()
        for tmp in self.path_line_color:
            tmp.clear()
