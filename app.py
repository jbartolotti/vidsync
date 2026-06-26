import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

try:
    from ffpyplayer.player import MediaPlayer
except ImportError:  # pragma: no cover - optional dependency
    MediaPlayer = None


class VideoPlayer(QObject):
    """ffpyplayer-backed playback that uses ffpyplayer for audio and timing."""

    frame_ready = pyqtSignal(object, float)
    playback_finished = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._player = None
        self._path: Optional[str] = None
        self._paused = True
        self._duration_seconds = None
        self._timer = None
        self._last_pts = None
        self._last_image = None

    def load_video(self, path: str) -> None:
        if MediaPlayer is None:
            raise RuntimeError("ffpyplayer is not installed. Install ffpyplayer to use this backend.")

        self.pause()
        self._close_player()
        self._path = path
        self._player = MediaPlayer(path, ff_opts={"framedrop": True, "sync": "audio"})
        self._paused = True
        self._duration_seconds = None
        self._last_pts = None
        self._last_image = None
        if self._player is not None:
            try:
                metadata = self._player.get_metadata()
            except Exception:
                metadata = {}
            if isinstance(metadata, dict):
                duration = metadata.get("duration")
                if duration is not None:
                    try:
                        self._duration_seconds = float(duration)
                    except (TypeError, ValueError):
                        self._duration_seconds = None
        if self._player is not None:
            self._player.set_pause(True)

    def _close_player(self) -> None:
        if self._player is not None:
            try:
                self._player.close_player()
            except Exception:
                pass
        self._player = None

    def set_timer(self, timer) -> None:
        self._timer = timer

    def play(self) -> None:
        if self._player is None:
            return
        self._paused = False
        self._player.set_pause(False)
        if self._timer is not None and not self._timer.isActive():
            self._timer.start(5)

    def pause(self) -> None:
        if self._player is not None:
            self._player.set_pause(True)
        self._paused = True
        if self._timer is not None:
            self._timer.stop()

    def seek(self, time_seconds: float) -> None:
        if self._player is None:
            return
        try:
            self._player.seek(float(time_seconds), relative=False)
        except Exception:
            pass
        self._last_pts = None
        self._player.set_pause(self._paused)

    def update_frame(self) -> None:
        if self._player is None or self._paused:
            return

        try:
            frame, val = self._player.get_frame()
        except Exception:
            return

        if val == "eof":
            self.pause()
            self.playback_finished.emit()
            return
        if frame is None:
            return

        # ✅ IMPORTANT: respect timing
        if isinstance(val, (int, float)) and val > 0:
            # Adjust timer interval based on ffpyplayer timing
            if self._timer is not None:
                self._timer.start(int(val * 1000))

        img, pts = frame
        if img is None:
            return
        try:
            w, h = img.get_size()
            pix_fmt = img.get_pixel_format()

            # Convert to RGB24 bytes
            raw = img.to_bytearray()[0]
            frame_array = np.frombuffer(raw, dtype=np.uint8)

            # reshape (h, w, 3)
            frame_array = frame_array.reshape((h, w, 3))

        except Exception as e:
            print("CONVERSION ERROR:", e)
            return

        if frame_array is None:
            return
        if frame_array.ndim == 2:
            frame_array = np.repeat(frame_array[:, :, None], 3, axis=2)
        elif frame_array.ndim == 3 and frame_array.shape[2] == 4:
            frame_array = frame_array[:, :, :3]
        elif frame_array.ndim == 3 and frame_array.shape[2] == 1:
            frame_array = np.repeat(frame_array, 3, axis=2)

        if frame_array.dtype != np.uint8:
            frame_array = frame_array.astype(np.uint8)
        frame_array = np.ascontiguousarray(frame_array)
        height, width = frame_array.shape[:2]
        if width == 0 or height == 0:
            return

        bytes_per_line = width * 3
        image_bytes = frame_array.tobytes()
        qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
        qimage = qimage.copy()
        self._last_image = qimage

        pts_value = float(pts) if pts is not None else (self._last_pts if self._last_pts is not None else 0.0)
        self._last_pts = pts_value
        self.frame_ready.emit(qimage, pts_value)

    def get_frame_interval_ms(self) -> int:
        return 5

    def is_paused(self) -> bool:
        return self._paused


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ffpyplayer Video Sync")
        self.resize(1400, 800)

        self.video_player = VideoPlayer(self)
        self.video_player.frame_ready.connect(self._on_frame_ready)
        self.video_player.playback_finished.connect(self._on_playback_finished)

        self.video_frame_label = QLabel("No video loaded")
        self.video_frame_label.setAlignment(Qt.AlignCenter)
        self.video_frame_label.setStyleSheet("background-color: black; color: white;")
        self.video_frame_label.setMinimumSize(640, 360)

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.video_player.update_frame)
        self.video_timer.setTimerType(Qt.PreciseTimer)
        self.video_timer.setInterval(5)
        self.video_timer.setSingleShot(False)
        self.video_player.set_timer(self.video_timer)

        self.video_backend = "none"
        self.video_path: Optional[str] = None
        self.video_duration_seconds = None
        self.video_fps = 30.0
        self.opencv_position_ms = 0
        self._opencv_frame_needs_seek = True
        self.video_capture = None

        self.data_frame: Optional[pd.DataFrame] = None
        self.processed_frame: Optional[pd.DataFrame] = None
        self.categorical_columns = []
        self.category_controls = []
        self.legend_item = None
        self._legend_visible = True
        self.params_group = None
        self.params_toggle_button = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)

        video_controls = QHBoxLayout()
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self._load_video)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self._toggle_play_pause)
        video_controls.addWidget(self.load_video_button)
        video_controls.addWidget(self.play_pause_button)

        left_layout.addLayout(video_controls)
        left_layout.addWidget(self.video_frame_label, stretch=1)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)

        self.plot_widget = pg.PlotWidget(title="Timecourse")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
        self.plot_widget.addItem(self.playhead)

        self.legend_toggle_button = QPushButton("Hide legend")
        self.legend_toggle_button.clicked.connect(self._toggle_legend_visibility)
        legend_row = QHBoxLayout()
        legend_row.addStretch()
        legend_row.addWidget(self.legend_toggle_button)
        right_layout.addLayout(legend_row)
        self.legend_item = pg.LegendItem(offset=(10, 10))
        self.legend_item.setParentItem(self.plot_widget.plotItem)
        self.legend_item.hide()
        right_layout.addWidget(self.plot_widget, stretch=1)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        file_row = QHBoxLayout()
        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.clicked.connect(self._load_data)
        file_row.addWidget(self.load_data_button)
        controls_layout.addLayout(file_row)

        self.params_toggle_button = QPushButton("Parameters")
        self.params_toggle_button.setCheckable(True)
        self.params_toggle_button.setChecked(False)
        self.params_toggle_button.clicked.connect(self._toggle_parameters_visibility)
        controls_layout.addWidget(self.params_toggle_button)

        self.params_group = QGroupBox("Parameters")
        self.params_group.setVisible(False)
        params_layout = QFormLayout(self.params_group)
        self.tr_duration_spin = QDoubleSpinBox()
        self.tr_duration_spin.setRange(0.001, 1000.0)
        self.tr_duration_spin.setValue(2.0)
        self.tr_duration_spin.setDecimals(3)
        self.tr_duration_spin.valueChanged.connect(self._refresh_plot)
        self.time_offset_spin = QDoubleSpinBox()
        self.time_offset_spin.setRange(-100000.0, 100000.0)
        self.time_offset_spin.setValue(6.0)
        self.time_offset_spin.setDecimals(3)
        self.time_offset_spin.valueChanged.connect(self._refresh_plot)
        params_layout.addRow("TR duration (s)", self.tr_duration_spin)
        params_layout.addRow("Time offset (s)", self.time_offset_spin)
        controls_layout.addWidget(self.params_group)

        self.category_group = QGroupBox("Category Filters")
        self.category_layout = QVBoxLayout(self.category_group)
        controls_layout.addWidget(self.category_group)

        right_layout.addWidget(controls_group)
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

    def _load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            os.path.expanduser("~"),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.webm)",
        )
        if not path:
            return

        self.video_path = path
        self._stop_opencv_video()
        self.video_player.pause()
        self.video_timer.stop()
        self.video_backend = "ffpyplayer"
        self._update_backend_status("ffpyplayer")
        self.video_frame_label.setText("Loading video...")
        self.play_pause_button.setText("Play")

        try:
            self.video_player.load_video(path)
        except Exception as exc:
            self._show_error(f"ffpyplayer could not open the video:\n{exc}")
            self.video_backend = "none"
            self._update_backend_status("none")
            self.video_frame_label.setText("Video not loaded")
            return

        self.video_frame_label.setText("Video loaded")
        self.play_pause_button.setText("Pause")
        self.video_timer.stop()
        self.video_player.play()

    def _start_opencv_video(self, path: str) -> None:
        if cv2 is None:
            self._show_error("OpenCV is not installed. Install opencv-python to enable fallback video playback.")
            return

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self._show_error("The selected video could not be opened by the fallback decoder.")
            return

        self.video_capture = capture
        self.video_backend = "opencv"
        self.video_frame_label.setText("Visual fallback (audio unavailable)")
        self.video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_duration_seconds = (frame_count / self.video_fps) if self.video_fps > 0 else 0.0
        self.opencv_position_ms = 0
        self._opencv_frame_needs_seek = True
        self._display_current_frame()
        self.video_timer.stop()
        self.video_timer.setSingleShot(False)
        try:
            self.video_timer.timeout.disconnect(self.video_player.update_frame)
        except TypeError:
            pass
        try:
            self.video_timer.timeout.disconnect(self._advance_opencv_frame)
        except TypeError:
            pass
        self.video_timer.timeout.connect(self._advance_opencv_frame)
        self.video_timer.start(max(10, int(1000 / max(self.video_fps, 1))))
        self._update_backend_status("opencv")

    def _stop_opencv_video(self) -> None:
        self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_timer.stop()
        try:
            self.video_timer.timeout.disconnect(self._advance_opencv_frame)
        except TypeError:
            pass
        try:
            self.video_timer.timeout.disconnect(self.video_player.update_frame)
        except TypeError:
            pass
        self.video_timer.timeout.connect(self.video_player.update_frame)
        self.video_timer.setSingleShot(False)

    def _update_backend_status(self, backend: str) -> None:
        self.video_backend = backend

    def _on_frame_ready(self, image: QImage, pts: float) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_frame_label.setPixmap(
            pixmap.scaled(self.video_frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        # Synchronize the plot cursor with the ffpyplayer presentation timestamp.
        self.playhead.setValue(pts)

    def _on_playback_finished(self) -> None:
        self.play_pause_button.setText("Play")
        self.video_timer.stop()

    def _display_current_frame(self) -> None:
        if self.video_capture is None:
            return
        if self._opencv_frame_needs_seek:
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, self.opencv_position_ms)
            self._opencv_frame_needs_seek = False
        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_frame_label.setPixmap(
            pixmap.scaled(self.video_frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _advance_opencv_frame(self) -> None:
        if self.video_capture is None:
            return
        self.opencv_position_ms += int(1000 / max(self.video_fps, 1))
        if self.video_duration_seconds and self.opencv_position_ms >= int(self.video_duration_seconds * 1000.0):
            self.opencv_position_ms = int(self.video_duration_seconds * 1000.0)
            self.video_timer.stop()
            self.play_pause_button.setText("Play")
        self._display_current_frame()
        self.playhead.setValue(self.opencv_position_ms / 1000.0)

    def _load_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Timecourse Data", os.path.expanduser("~"), "Data Files (*.csv *.tsv *.txt)")
        if not path:
            return
        try:
            self._read_timecourse_file(path)
        except Exception as exc:  # pragma: no cover - GUI error path
            self._show_error(f"Could not read timecourse data:\n{exc}")

    def _read_timecourse_file(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".tsv":
            sep = "\t"
        elif ext == ".csv":
            sep = ","
        else:
            sep = None

        if sep is None:
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path, sep=sep)

        self.data_frame = df.copy()
        self._prepare_timecourse_data()

    def _prepare_timecourse_data(self) -> None:
        if self.data_frame is None:
            return

        df = self.data_frame.copy()
        time_column = self._detect_time_column(df.columns)
        if time_column is None:
            self._show_error("Timecourse file must contain one of the required time columns: 'index', 'seconds', or 'TR'.")
            return
        if "value" not in {col.lower() for col in df.columns}:
            self._show_error("Timecourse file must contain a 'value' column.")
            return

        value_column = next(col for col in df.columns if col.lower() == "value")
        time_column_name = next(col for col in df.columns if col.lower() == time_column.lower())

        try:
            time_values = pd.to_numeric(df[time_column_name], errors="raise")
        except Exception as exc:
            self._show_error(f"The time column could not be parsed as numeric values: {exc}")
            return

        if time_column.lower() in {"index", "tr"}:
            seconds = time_values * self.tr_duration_spin.value()
        else:
            seconds = time_values

        seconds = seconds + self.time_offset_spin.value()
        df = df.copy()
        df["__seconds__"] = seconds.astype(float)
        df["__value__"] = pd.to_numeric(df[value_column], errors="coerce")
        df = df.dropna(subset=["__value__"])
        self.processed_frame = df.reset_index(drop=True)
        self._build_category_controls()
        self._refresh_plot()

    def _detect_time_column(self, columns) -> Optional[str]:
        normalized = {col.lower().strip(): col for col in columns}
        for candidate in ("seconds", "index", "tr"):
            if candidate in normalized:
                return normalized[candidate]
        return None

    def _build_category_controls(self) -> None:
        while self.category_layout.count():
            item = self.category_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.category_controls = []

        if self.processed_frame is None:
            self.category_group.setVisible(False)
            return

        self.categorical_columns = [
            col
            for col in self.processed_frame.columns
            if col not in {"__seconds__", "__value__", "value"}
            and col.lower() not in {"seconds", "index", "tr"}
        ]

        self.category_group.setVisible(bool(self.categorical_columns))
        if not self.categorical_columns:
            return

        for column in self.categorical_columns:
            combo = QComboBox(self)
            unique_values = [str(x) for x in self.processed_frame[column].dropna().unique()]
            combo.addItem("All")
            combo.addItems(unique_values)
            combo.currentTextChanged.connect(self._refresh_plot)
            self.category_layout.addWidget(QLabel(column))
            self.category_layout.addWidget(combo)
            self.category_controls.append(combo)

    def _refresh_plot(self) -> None:
        self._update_plot()

    def _update_plot(self) -> None:
        if self.processed_frame is None:
            return

        df = self.processed_frame.copy()
        for combo, column in zip(self.category_controls, self.categorical_columns):
            selected = combo.currentText()
            if selected != "All" and selected:
                df = df[df[column].astype(str) == selected]

        self.plot_widget.clear()
        self.plot_widget.addItem(self.playhead)
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Value")

        if self.legend_item is not None:
            self.legend_item.clear()

        if df.empty:
            return

        plotted_items = []
        if self.categorical_columns:
            grouping = df[self.categorical_columns].fillna("<NA>").astype(str).agg(lambda row: " | ".join(row), axis=1)
            unique_groups = grouping.unique()
            palette = [
                QColor("#1f77b4"),
                QColor("#ff7f0e"),
                QColor("#2ca02c"),
                QColor("#d62728"),
                QColor("#9467bd"),
                QColor("#8c564b"),
                QColor("#e377c2"),
                QColor("#7f7f7f"),
            ]
            for index, group_value in enumerate(unique_groups):
                subset = df[grouping == group_value]
                pen = pg.mkPen(color=palette[index % len(palette)], width=2)
                curve = self.plot_widget.plot(subset["__seconds__"].to_numpy(), subset["__value__"].to_numpy(), pen=pen, name=str(group_value))
                plotted_items.append((curve, str(group_value)))
        else:
            curve = self.plot_widget.plot(df["__seconds__"].to_numpy(), df["__value__"].to_numpy(), pen=pg.mkPen(color=QColor("#1f77b4"), width=2), name="value")
            plotted_items.append((curve, "value"))

        if self.legend_item is not None and plotted_items:
            for curve, label in plotted_items:
                self.legend_item.addItem(curve, label)
            self.legend_item.setVisible(self._legend_visible)
            self.legend_item.show()
        elif self.legend_item is not None:
            self.legend_item.hide()

    def _on_plot_clicked(self, event) -> None:
        if self.processed_frame is None:
            return
        if not self.plot_widget.sceneBoundingRect().contains(event.scenePos()):
            return
        vb = self.plot_widget.plotItem.vb
        mouse_point = vb.mapSceneToView(event.scenePos())
        x_value = mouse_point.x()
        if np.isfinite(x_value):
            if self.video_backend == "opencv":
                self.opencv_position_ms = int(x_value * 1000.0)
                self._opencv_frame_needs_seek = True
                self._display_current_frame()
                self.playhead.setValue(self.opencv_position_ms / 1000.0)
            elif self.video_backend == "ffpyplayer":
                self.video_player.seek(float(x_value))
                self.playhead.setValue(float(x_value))

    def _toggle_play_pause(self) -> None:
        if self.video_backend == "opencv":
            if self.video_timer.isActive():
                self.video_timer.stop()
                self.play_pause_button.setText("Play")
            else:
                self.video_timer.start(max(10, int(1000 / max(self.video_fps, 1))))
                self.play_pause_button.setText("Pause")
            return

        if self.video_backend != "ffpyplayer":
            return
        if self.video_player.is_paused():
            self.video_player.play()
            self.play_pause_button.setText("Pause")
        else:
            self.video_player.pause()
            self.play_pause_button.setText("Play")

    def _toggle_legend_visibility(self) -> None:
        self._legend_visible = not self._legend_visible
        if self.legend_item is not None:
            self.legend_item.setVisible(self._legend_visible)
        self.legend_toggle_button.setText("Show legend" if not self._legend_visible else "Hide legend")

    def _toggle_parameters_visibility(self) -> None:
        if self.params_group is None or self.params_toggle_button is None:
            return
        visible = self.params_toggle_button.isChecked()
        self.params_group.setVisible(visible)
        self.params_toggle_button.setText("Hide Parameters" if visible else "Parameters")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
