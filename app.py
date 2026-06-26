import json
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


def _load_vlc_module():
    """Import python-vlc and make sure the native VLC runtime can be found."""
    candidate_dirs = []
    for env_name in ("VLC_DIR", "VLC_HOME"):
        value = os.environ.get(env_name)
        if value:
            candidate_dirs.append(value)

    for base in (
        r"C:\Program Files\VideoLAN\VLC",
        r"C:\Program Files (x86)\VideoLAN\VLC",
        os.path.join(os.environ.get("ProgramFiles", ""), "VideoLAN", "VLC"),
    ):
        if base:
            candidate_dirs.append(base)

    seen = set()
    unique_dirs = []
    for directory in candidate_dirs:
        normalized = os.path.normcase(os.path.abspath(directory))
        if normalized not in seen:
            seen.add(normalized)
            unique_dirs.append(directory)

    original_cwd = os.getcwd()
    try:
        for directory in unique_dirs:
            if not directory or not os.path.isdir(directory):
                continue
            dll_path = os.path.join(directory, "libvlc.dll")
            if os.path.isfile(dll_path):
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
                os.environ["VLC_PLUGIN_PATH"] = os.path.join(directory, "plugins")
                os.chdir(directory)
                try:
                    import vlc as vlc_mod
                    return vlc_mod, None
                except Exception as exc:
                    return None, exc

        try:
            import vlc as vlc_mod
            return vlc_mod, None
        except Exception as exc:
            return None, exc
    finally:
        os.chdir(original_cwd)


vlc, vlc_error = _load_vlc_module()

try:
    from PyQt5.QtCore import Qt, QTimer
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
        QSlider,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import Qt, QTimer
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
        QSlider,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video-Timecourse Sync")
        self.resize(1400, 800)

        self.player = None
        self.vlc_instance = None
        self.vlc_media = None
        self.vlc_error = vlc_error
        self.video_widget = QWidget(self)
        self.video_widget.setAttribute(Qt.WA_NativeWindow, True)
        self.video_widget.setMinimumSize(320, 240)
        self.video_widget.setStyleSheet("background-color: black;")
        self.video_frame_label = QLabel("No video loaded")
        self.video_frame_label.setAlignment(Qt.AlignCenter)
        self.video_frame_label.setStyleSheet("background-color: black; color: white;")
        self.video_stack = QStackedWidget(self)
        self.video_stack.addWidget(self.video_widget)
        self.video_stack.addWidget(self.video_frame_label)
        self.video_stack.setCurrentWidget(self.video_frame_label)

        self.video_capture = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._advance_opencv_frame)
        self.video_timer.setTimerType(Qt.PreciseTimer)
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self._update_vlc_position)
        self.position_timer.setInterval(100)
        self.video_backend = "none"
        self.video_fps = 30.0
        self.opencv_position_ms = 0
        self._opencv_frame_needs_seek = True
        self.video_duration_ms = 0
        self.video_path: Optional[str] = None

        if vlc is not None:
            try:
                self.vlc_instance = vlc.Instance()
                self.player = self.vlc_instance.media_player_new()
                self.video_backend = "vlc"
            except Exception as exc:
                self.player = None
                self.video_backend = "none"
                self.vlc_error = exc

        self.data_frame: Optional[pd.DataFrame] = None
        self.processed_frame: Optional[pd.DataFrame] = None
        self.categorical_columns = []
        self.category_controls = []
        self.current_groupings = {}
        self.preferences_path = os.path.join(os.path.dirname(__file__), "preferences.json")

        self._build_ui()
        self._load_preferences_from_file(self.preferences_path, quiet=True)

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)

        self.video_controls = QHBoxLayout()
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self._load_video)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self._toggle_play_pause)
        self.video_controls.addWidget(self.load_video_button)
        self.video_controls.addWidget(self.play_pause_button)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(self._on_seek_slider_moved)

        self.backend_status_label = QLabel("Backend: none")
        self.backend_status_label.setStyleSheet("font-size: 10pt; color: #555;")

        left_layout.addLayout(self.video_controls)
        left_layout.addWidget(self.video_stack, stretch=1)
        left_layout.addWidget(self.backend_status_label)
        left_layout.addWidget(self.seek_slider)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)

        self.plot_widget = pg.PlotWidget(title="Timecourse")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
        self.plot_widget.addItem(self.playhead)

        right_layout.addWidget(self.plot_widget, stretch=1)

        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)

        file_row = QHBoxLayout()
        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.clicked.connect(self._load_data)
        self.load_preferences_button = QPushButton("Load Preferences")
        self.load_preferences_button.clicked.connect(self._load_preferences)
        self.save_preferences_button = QPushButton("Save Preferences")
        self.save_preferences_button.clicked.connect(self._save_preferences)
        file_row.addWidget(self.load_data_button)
        file_row.addWidget(self.load_preferences_button)
        file_row.addWidget(self.save_preferences_button)
        control_layout.addLayout(file_row)

        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        self.tr_duration_spin = QDoubleSpinBox()
        self.tr_duration_spin.setRange(0.001, 1000.0)
        self.tr_duration_spin.setValue(2.0)
        self.tr_duration_spin.setDecimals(3)
        self.tr_duration_spin.valueChanged.connect(self._refresh_plot)
        self.time_offset_spin = QDoubleSpinBox()
        self.time_offset_spin.setRange(-100000.0, 100000.0)
        self.time_offset_spin.setValue(0.0)
        self.time_offset_spin.setDecimals(3)
        self.time_offset_spin.valueChanged.connect(self._refresh_plot)
        params_layout.addRow("TR duration (s)", self.tr_duration_spin)
        params_layout.addRow("Time offset (s)", self.time_offset_spin)
        control_layout.addWidget(params_group)

        self.category_group = QGroupBox("Category Filters")
        self.category_layout = QVBoxLayout(self.category_group)
        control_layout.addWidget(self.category_group)

        right_layout.addWidget(control_group)
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

    def _load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", os.path.expanduser("~"), "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.webm)")
        if not path:
            return

        self.video_path = path
        self._stop_opencv_video()
        self._stop_vlc_video()
        self.video_stack.setCurrentWidget(self.video_widget)
        self.video_frame_label.setText("Loading video...")
        self.play_pause_button.setText("Play")
        self.seek_slider.setRange(0, 0)
        self.video_timer.stop()
        self.position_timer.stop()

        if vlc is None or self.vlc_instance is None or self.player is None:
            if cv2 is not None:
                self._start_opencv_video(path)
            else:
                self._show_error(
                    "VLC playback is not available. Install python-vlc and the VLC runtime, or ensure libvlc.dll is on your system.\n\n"
                    f"Details: {self.vlc_error}"
                )
            return

        self.video_backend = "vlc"
        self.vlc_media = self.vlc_instance.media_new(path)
        self.player.set_media(self.vlc_media)
        self._attach_vlc_window()
        self.position_timer.start()
        self.play_pause_button.setText("Play")
        self._update_backend_status("vlc")

    def _attach_vlc_window(self) -> None:
        if self.player is None:
            return
        try:
            self.player.set_hwnd(int(self.video_widget.winId()))
        except Exception:
            QTimer.singleShot(0, self._attach_vlc_window)

    def _stop_vlc_video(self) -> None:
        self.position_timer.stop()
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass
        self.vlc_media = None

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
        self.video_stack.setCurrentWidget(self.video_frame_label)
        self.video_frame_label.setText("Visual fallback (audio unavailable)")
        self.video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_duration_ms = int((frame_count / self.video_fps * 1000.0) if self.video_fps > 0 else 0)
        self.seek_slider.setRange(0, max(self.video_duration_ms, 0))
        self.opencv_position_ms = 0
        self._opencv_frame_needs_seek = True
        self._display_current_frame()
        self.video_timer.start(max(10, int(1000 / max(self.video_fps, 1))))
        self._update_backend_status("opencv")

    def _stop_opencv_video(self) -> None:
        self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def _update_backend_status(self, backend: str) -> None:
        self.video_backend = backend
        if backend == "vlc":
            self.backend_status_label.setText("Backend: python-vlc")
        elif backend == "opencv":
            self.backend_status_label.setText("Backend: opencv")
        else:
            self.backend_status_label.setText("Backend: none")

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
        if self.video_duration_ms > 0 and self.opencv_position_ms >= self.video_duration_ms:
            self.opencv_position_ms = self.video_duration_ms
            self.video_timer.stop()
            self.play_pause_button.setText("Play")
        self._display_current_frame()
        self.playhead.setValue(self.opencv_position_ms / 1000.0)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(self.opencv_position_ms)
        self.seek_slider.blockSignals(False)

    def _update_vlc_position(self) -> None:
        if self.video_backend != "vlc" or self.player is None:
            return
        length = self.player.get_length()
        if length > 0:
            self.seek_slider.setRange(0, int(length))
        time_ms = self.player.get_time()
        if time_ms >= 0:
            self.playhead.setValue(time_ms / 1000.0)
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(time_ms)
            self.seek_slider.blockSignals(False)
            self._on_position_changed(time_ms)

    def _on_media_error(self, error) -> None:
        if self.video_path:
            self._start_opencv_video(self.video_path)
            return
        self._show_error("The selected video could not be opened by the current playback backend.")

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
        self.current_groupings = {}

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

        if df.empty:
            return

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
                self.plot_widget.plot(subset["__seconds__"].to_numpy(), subset["__value__"].to_numpy(), pen=pen, name=str(group_value))
        else:
            self.plot_widget.plot(df["__seconds__"].to_numpy(), df["__value__"].to_numpy(), pen=pg.mkPen(color=QColor("#1f77b4"), width=2), name="value")

    def _on_plot_clicked(self, event) -> None:
        if self.processed_frame is None or self.player is None:
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
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(self.opencv_position_ms)
                self.seek_slider.blockSignals(False)
            else:
                self.player.setPosition(int(x_value * 1000.0))

    def _on_position_changed(self, position_ms: int) -> None:
        if self.processed_frame is None:
            return
        seconds = position_ms / 1000.0
        self.playhead.setValue(seconds)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(position_ms)
        self.seek_slider.blockSignals(False)

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.seek_slider.setRange(0, max(duration_ms, 0))

    def _on_seek_slider_moved(self, value: int) -> None:
        if self.video_backend == "opencv":
            self.opencv_position_ms = value
            self._opencv_frame_needs_seek = True
            self._display_current_frame()
            self.playhead.setValue(self.opencv_position_ms / 1000.0)
            return
        if self.video_backend == "vlc" and self.player is not None:
            self.player.set_time(value)
            self.playhead.setValue(value / 1000.0)
            return

    def _toggle_play_pause(self) -> None:
        if self.video_backend == "opencv":
            if self.video_timer.isActive():
                self.video_timer.stop()
                self.play_pause_button.setText("Play")
            else:
                self.video_timer.start(max(10, int(1000 / max(self.video_fps, 1))))
                self.play_pause_button.setText("Pause")
            return

        if self.video_backend == "vlc" and self.player is not None:
            if self.player.is_playing():
                self.player.pause()
                self.play_pause_button.setText("Play")
            else:
                self.player.play()
                self.play_pause_button.setText("Pause")
            return

    def _load_preferences(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Preferences", os.path.expanduser("~"), "JSON Files (*.json)")
        if path:
            self._load_preferences_from_file(path)

    def _load_preferences_from_file(self, path: str, quiet: bool = False) -> None:
        if not os.path.exists(path):
            if not quiet:
                self._show_error(f"Preferences file not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                prefs = json.load(handle)
            if "tr_duration" in prefs:
                self.tr_duration_spin.setValue(float(prefs["tr_duration"]))
            if "time_offset" in prefs:
                self.time_offset_spin.setValue(float(prefs["time_offset"]))
            self.preferences_path = path
            if self.processed_frame is not None:
                self._refresh_plot()
        except Exception as exc:
            self._show_error(f"Could not load preferences:\n{exc}")

    def _save_preferences(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Preferences", self.preferences_path or os.path.expanduser("~"), "JSON Files (*.json)")
        if not path:
            return
        prefs = {
            "tr_duration": self.tr_duration_spin.value(),
            "time_offset": self.time_offset_spin.value(),
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(prefs, handle, indent=2)
            self.preferences_path = path
        except Exception as exc:
            self._show_error(f"Could not save preferences:\n{exc}")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
