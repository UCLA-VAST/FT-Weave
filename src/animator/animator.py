import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.exporters import ImageExporter
import numpy as np
import bisect
import imageio
from src.ds import Architecture


class Animator:
    # constants for animation
    FPS = 15  # frames per second
    INIT_FRM = int(FPS / 5)  # initial empty frames, 1/5 second now
    PT_MICRON = 8  # scaling factor: points per micron
    MUS_PER_FRM = 150 / FPS  # microseconds per frame
    MUS_PER_FRM_SLOW = 7 / FPS  # in slow motion, i.e., Rydberg
    CANVAS_PADDING = 10
    RYDBERG_PADDING = 3  # around each entanglement zone

    # colors (RGB tuples)
    RYDBERG_COLOR = (0, 0, 255)  # blue
    SLM_COLOR = (0, 255, 0)  # green
    QUBIT_COLOR = (0, 0, 0)  # black
    AOD_COLORS = [
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]  # red, cyan, magenta, yellow
    AOD_TRANS = 0.7

    def animate(
        self,
        code: dict,
        architecture: Architecture,
        output: str,
        font: int = 10,
        ffmpeg: str = "ffmpeg",
    ):
        """
        Args:
            code (dict): instruction code
            output (str): filename to save output.
            scaling_factor (int, optional): not used in PyQtGraph
            font (int, optional): font size in the animation. Defaults to 10.
        """
        self.code = code
        self.architecture = architecture
        self.output = output
        self.font_size = font

        # Create Qt application
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # Setup the plot window
        self.setup_canvas()

        # Create schedule
        num_frame = self.create_schedule()
        self.total_frames = self.INIT_FRM + num_frame
        self.current_frame = 0

        # Initialize frame storage for video
        self.frames = []

        # Setup timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize the plot
        self.update_init()

        # Start animation
        self.timer.start(int(1000 / self.FPS))
        self.app.exec_()

        # Save video after animation completes
        self.save_video()

    def create_schedule(self):
        """Create animation schedule"""
        self.piecewise_schedule = [
            (0, 0, 0),
        ]
        last_end_time = 0

        for inst in self.code["instructions"]:
            if inst["type"] == "rydberg":
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

    def setup_canvas(self):
        """Set up PyQtGraph plot window"""
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("Quantum Animation")

        # Set white background
        self.win.setBackground("w")

        # Add a label for the title above the plot
        self.title_label = self.win.addLabel("", row=0, col=0)
        self.title_label.setText("", color="k")

        # Add the plot below the title
        self.plot = self.win.addPlot(row=1, col=0)
        self.plot.setAspectLocked(True)

        # Add border/box around the plot
        self.plot.showGrid(x=False, y=False)
        self.plot.getViewBox().setBorder(pg.mkPen("k", width=2))

        # Set axis ranges
        arch_range = self.architecture.arch_range
        x_min = -self.CANVAS_PADDING + arch_range[0][0]
        x_max = self.CANVAS_PADDING + arch_range[1][0]
        y_min = -self.CANVAS_PADDING + arch_range[0][1]
        y_max = self.CANVAS_PADDING + arch_range[1][1]

        self.plot.setXRange(x_min, x_max)
        self.plot.setYRange(y_min, y_max)

        # Restrict view to the plot area (no panning/zooming outside)
        self.plot.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)

        # Setup entanglement zones
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

    def update_init(self):
        """Initialize plot elements"""
        # Draw SLMs
        slm_xs = []
        slm_ys = []
        for slm_id, slm_arr in self.architecture.dict_SLM.items():
            for r in range(slm_arr.n_r):
                for c in range(slm_arr.n_c):
                    x, y = self.architecture.exact_SLM_location(slm_id, r, c)
                    slm_xs.append(x)
                    slm_ys.append(y)

        self.slm_scatter = pg.ScatterPlotItem(
            x=slm_xs,
            y=slm_ys,
            symbol="o",
            size=10,
            pen=pg.mkPen(color=self.SLM_COLOR, width=2),
            brush=None,
        )
        self.plot.addItem(self.slm_scatter)

        # Initialize qubits
        self.qubit_xs = []
        self.qubit_ys = []
        for q in self.code["instructions"][0]["init_locs"]:
            x, y = self.architecture.exact_SLM_location(q[1], q[2], q[3])
            self.qubit_xs.append(x)
            self.qubit_ys.append(y)

        self.qubit_scatter = pg.ScatterPlotItem(
            x=self.qubit_xs,
            y=self.qubit_ys,
            symbol="o",
            size=8,
            pen=None,
            brush=pg.mkBrush(self.QUBIT_COLOR),
        )
        self.plot.addItem(self.qubit_scatter)

        # Initialize AOD lines
        self.aod_col_lines = {}
        self.aod_row_lines = {}

        # Get plot boundaries for clipping
        arch_range = self.architecture.arch_range
        x_min = -self.CANVAS_PADDING + arch_range[0][0]
        x_max = self.CANVAS_PADDING + arch_range[1][0]
        y_min = -self.CANVAS_PADDING + arch_range[0][1]
        y_max = self.CANVAS_PADDING + arch_range[1][1]

        for aod_id, aod in self.architecture.dict_AOD.items():
            self.aod_col_lines[aod_id] = []
            for _ in range(aod.n_c):
                line = pg.InfiniteLine(
                    pos=0,
                    angle=90,
                    pen=pg.mkPen((0, 0, 0, 0), style=QtCore.Qt.PenStyle.DashLine),
                    bounds=(y_min, y_max),  # Restrict vertical line to plot height
                )
                self.plot.addItem(line)
                self.aod_col_lines[aod_id].append(line)

            self.aod_row_lines[aod_id] = []
            for _ in range(aod.n_r):
                line = pg.InfiniteLine(
                    pos=0,
                    angle=0,
                    pen=pg.mkPen((0, 0, 0, 0), style=QtCore.Qt.PenStyle.DashLine),
                    bounds=(x_min, x_max),  # Restrict horizontal line to plot width
                )
                self.plot.addItem(line)
                self.aod_row_lines[aod_id].append(line)

        # Initialize Rydberg zones
        self.entanglement_rects = []
        for zone in self.entanglement_rect_range:
            rect = QtWidgets.QGraphicsRectItem(zone[0][0], zone[0][1], 0, 0)
            rect.setPen(pg.mkPen(None))
            rect.setBrush(pg.mkBrush((0, 0, 255, 76)))  # 0.3 alpha * 255 ≈ 76
            self.plot.addItem(rect)
            self.entanglement_rects.append(rect)

        # Initialize 1q gates scatter
        self.gate_scatter = pg.ScatterPlotItem(
            x=[],
            y=[],
            symbol="o",
            size=30,
            pen=None,
            brush=pg.mkBrush((0, 255, 0, 127)),
        )
        self.plot.addItem(self.gate_scatter)

        # Initialize path lines
        self.path_lines = []
        for _ in range(len(self.code["instructions"][0]["init_locs"])):
            line = pg.PlotCurveItem(pen=pg.mkPen((128, 128, 128), width=2))
            self.plot.addItem(line)
            self.path_lines.append(line)

        self.path_line_loc = [[] for _ in self.code["instructions"][0]["init_locs"]]
        self.path_line_active = [
            False for _ in self.code["instructions"][0]["init_locs"]
        ]

    def update_frame(self):
        """Update function called by timer"""
        if self.current_frame >= self.total_frames:
            self.timer.stop()
            self.app.quit()
            return

        self.update(self.current_frame)

        # Capture frame with fixed size
        exporter = ImageExporter(self.win.scene())
        exporter.parameters()["width"] = 1920  # Fixed width
        exporter.parameters()["height"] = 1080  # Fixed height

        # Export to QImage
        img = exporter.export(toBytes=True)

        # Convert QImage to numpy array with consistent shape
        from PIL import Image
        import io

        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.ReadWrite)
        img.save(buffer, "PNG")

        data = bytes(buffer.data())
        pil_img = Image.open(io.BytesIO(data))

        # Convert to RGB to ensure consistent format
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Resize to ensure all frames have the same dimensions
        pil_img = pil_img.resize((1920, 1080), Image.Resampling.LANCZOS)

        frame_array = np.array(pil_img)
        self.frames.append(frame_array)

        self.current_frame += 1

    def update(self, f: int):
        """Update plot for frame f"""
        true_frame = f - self.INIT_FRM

        # Calculate true time
        interval_ends = [interval[0] for interval in self.piecewise_schedule]
        index = bisect.bisect_right(interval_ends, true_frame)
        tmp = self.piecewise_schedule[index]

        true_time = tmp[2] - (tmp[0] - true_frame) * (
            self.MUS_PER_FRM_SLOW if tmp[1] else self.MUS_PER_FRM
        )

        self.inst_str = ""

        # Reset Rydberg zones
        for rect in self.entanglement_rects:
            rect.setRect(rect.rect().x(), rect.rect().y(), 0, 0)

        # Reset 1q gates
        self.gate_scatter.setData(x=[], y=[])

        # Reset AOD lines
        for aod_id, aod in self.architecture.dict_AOD.items():
            for line in self.aod_row_lines[aod_id]:
                line.setPen(pg.mkPen((0, 0, 0, 0)))
            for line in self.aod_col_lines[aod_id]:
                line.setPen(pg.mkPen((0, 0, 0, 0)))

        if f >= self.INIT_FRM:
            for inst in self.code["instructions"][1:]:
                if true_time >= inst["begin_time"] and true_time < inst["end_time"]:
                    if inst["type"] == "rydberg":
                        self.update_rydberg(inst)
                    elif inst["type"] == "rearrangeJob":
                        self.update_arrangement(true_time, inst)
                    elif inst["type"] == "1qGate":
                        self.update_1qGate(inst)

        self.title_label.setText(self.inst_str)

    def update_rydberg(self, inst: dict):
        """Update Rydberg entanglement zones"""
        self.inst_str += (
            f' | {inst["id"]} {inst["type"]} \n elapsed time: {inst["begin_time"]:.2f}'
        )
        zone = self.entanglement_rect_range[inst["zone_id"]]
        self.entanglement_rects[inst["zone_id"]].setRect(
            zone[0][0], zone[0][1], zone[1], zone[2]
        )

        # Clear all path lines
        for i, line in enumerate(self.path_lines):
            line.setData(x=[], y=[])
            self.path_line_loc[i].clear()
            self.path_line_active[i] = False

    def update_arrangement(self, time: float, inst: dict):
        """Update atom arrangement"""
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
                        zip(detail_inst["begin_coord"], detail_inst["end_coord"]),
                        inst["aod_id"],
                    )

    def update_activate(self, ratio: float, time: float, inst: dict, aod_id: int):
        """Activate AOD beams"""
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'

        color = self.AOD_COLORS[aod_id]
        alpha = int(ratio * self.AOD_TRANS * 255)

        for col_id, col_x in zip(inst["col_id"], inst["col_x"]):
            self.aod_col_lines[aod_id][col_id].setPos(col_x)
            self.aod_col_lines[aod_id][col_id].setPen(
                pg.mkPen((*color, alpha), style=QtCore.Qt.PenStyle.DashLine, width=2)
            )

        for row_id, row_y in zip(inst["row_id"], inst["row_y"]):
            self.aod_row_lines[aod_id][row_id].setPos(row_y)
            self.aod_row_lines[aod_id][row_id].setPen(
                pg.mkPen((*color, alpha), style=QtCore.Qt.PenStyle.DashLine, width=2)
            )

    def update_deactivate(self, ratio: float, time: float, inst: dict, aod_id: int):
        """Deactivate AOD beams"""
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'

        color = self.AOD_COLORS[aod_id]
        alpha = int((1 - ratio) * self.AOD_TRANS * 255)

        for col_id in inst["col_id"]:
            self.aod_col_lines[aod_id][col_id].setPen(
                pg.mkPen((*color, alpha), style=QtCore.Qt.PenStyle.DashLine, width=2)
            )

        for row_id in inst["row_id"]:
            self.aod_row_lines[aod_id][row_id].setPen(
                pg.mkPen((*color, alpha), style=QtCore.Qt.PenStyle.DashLine, width=2)
            )

    def update_move(
        self, ratio: float, time: float, inst: dict, qubit_coord, aod_id: int
    ):
        """Move atoms"""
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'

        def interpolate(r: float, begin: int, end: int):
            D = end - begin
            return begin + 3 * D * (r**2) - 2 * D * (r**3)

        # Track which qubits are moving in this instruction
        moving_qubits = set()

        # Update qubit positions
        for begin_coords_row, end_coords_row in qubit_coord:
            for begin_coords, end_coords in zip(begin_coords_row, end_coords_row):
                q_id = begin_coords["id"]
                moving_qubits.add(q_id)

                # If starting a new movement, clear this qubit's path
                if not self.path_line_active[q_id]:
                    self.path_line_loc[q_id].clear()
                    self.path_line_active[q_id] = True

                # Always use exact end coordinates when movement is complete or very close
                # This ensures perfect alignment with SLM positions
                if ratio >= 0.95:  # Snap to final position in last 5% of movement
                    self.qubit_xs[q_id] = float(end_coords["x"])
                    self.qubit_ys[q_id] = float(end_coords["y"])
                else:
                    self.qubit_xs[q_id] = interpolate(
                        ratio, begin_coords["x"], end_coords["x"]
                    )
                    self.qubit_ys[q_id] = interpolate(
                        ratio, begin_coords["y"], end_coords["y"]
                    )

                self.path_line_loc[q_id].append(
                    (self.qubit_xs[q_id], self.qubit_ys[q_id])
                )

                # Update path line
                if len(self.path_line_loc[q_id]) > 1:
                    xs = [pt[0] for pt in self.path_line_loc[q_id]]
                    ys = [pt[1] for pt in self.path_line_loc[q_id]]
                    self.path_lines[q_id].setData(x=xs, y=ys)

        # Clear paths for qubits that are NOT moving in this instruction
        for i in range(len(self.path_lines)):
            if i not in moving_qubits and self.path_line_active[i]:
                self.path_line_loc[i].clear()
                self.path_lines[i].setData(x=[], y=[])
                self.path_line_active[i] = False

        self.qubit_scatter.setData(x=self.qubit_xs, y=self.qubit_ys)

        # Update AOD lines
        color = self.AOD_COLORS[aod_id]
        for row_id, row_begin_y, row_end_y in zip(
            inst["row_id"], inst["row_y_begin"], inst["row_y_end"]
        ):
            y_pos = interpolate(ratio, row_begin_y, row_end_y)
            self.aod_row_lines[aod_id][row_id].setPos(y_pos)
            self.aod_row_lines[aod_id][row_id].setPen(
                pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=2)
            )

        for col_id, col_begin_x, col_end_x in zip(
            inst["col_id"], inst["col_x_begin"], inst["col_x_end"]
        ):
            x_pos = interpolate(ratio, col_begin_x, col_end_x)
            self.aod_col_lines[aod_id][col_id].setPos(x_pos)
            self.aod_col_lines[aod_id][col_id].setPen(
                pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=2)
            )

    def update_1qGate(self, inst: dict):
        """Update single qubit gates"""
        self.inst_str += (
            f' | {inst["id"]} {inst["type"]} \n elapsed time: {inst["end_time"]:.2f}'
        )

        target_qs: list[int] = []
        if "gates" in inst:
            target_qs = [int(g["q"]) for g in inst["gates"]]
        else:
            for block in inst.get("inst", []):
                for loc in block.get("locs", []):
                    if not loc:
                        continue
                    target_qs.append(int(loc[0]))

        uniq_qs: list[int] = []
        seen_q: set[int] = set()
        for q in target_qs:
            if q in seen_q:
                continue
            seen_q.add(q)
            uniq_qs.append(q)

        gate_xs = []
        gate_ys = []
        for q in uniq_qs:
            gate_xs.append(self.qubit_xs[q])
            gate_ys.append(self.qubit_ys[q])

        self.gate_scatter.setData(x=gate_xs, y=gate_ys)

        # Clear all path lines
        for i, line in enumerate(self.path_lines):
            line.setData(x=[], y=[])
            self.path_line_loc[i].clear()
            self.path_line_active[i] = False

    def save_video(self):
        """Save captured frames to video file"""
        if self.frames:
            try:
                # Try to use imageio with ffmpeg
                imageio.mimsave(self.output, self.frames, fps=self.FPS, codec="libx264")
                print(f"Video saved to {self.output}")
            except Exception as e:
                print(f"Error saving as MP4: {e}")
                print("Trying alternative format (GIF)...")
                # Fallback to GIF if MP4 fails
                gif_output = self.output.replace(".mp4", ".gif")
                imageio.mimsave(gif_output, self.frames, fps=self.FPS, loop=0)
                print(f"Animation saved as GIF to {gif_output}")
