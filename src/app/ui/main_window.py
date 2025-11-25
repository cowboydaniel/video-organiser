"""Application main window and supporting models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..services.metadata import MetadataReader
from ..services.metadata_cache import MetadataCache
from ..services.organizer import Organizer
from ..services.rules import RuleEngine
from ..services.settings import SettingsManager


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv"}


@dataclass
class VideoItem:
    """Represents a video entry displayed in the table view."""

    path: Path
    duration: str = "Unknown"
    resolution: str = "Unknown"
    status: str = "Pending"


class VideoTableModel(QtCore.QAbstractTableModel):
    """Table model for displaying video metadata and processing state."""

    headers = ["File name", "Duration", "Resolution", "Status"]

    def __init__(self, items: Sequence[VideoItem] | None = None, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._items: List[VideoItem] = list(items) if items else []

    def rowCount(self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()) -> int:  # type: ignore[override]
        return len(self._items)

    def columnCount(self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()) -> int:  # type: ignore[override]
        return len(self.headers)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        item = self._items[index.row()]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return item.path.name
            if index.column() == 1:
                return item.duration
            if index.column() == 2:
                return item.resolution
            if index.column() == 3:
                return item.status
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if index.column() == 0:
                return str(item.path)
        if role == QtCore.Qt.ItemDataRole.DecorationRole and index.column() == 0:
            return QtGui.QIcon.fromTheme("video-x-generic")
        return None

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def set_items(self, items: Iterable[VideoItem]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()

    def item_at(self, row: int) -> VideoItem | None:
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def update_status(self, rows: Iterable[int], status: str) -> None:
        rows = list(rows)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._items):
                self._items[row].status = status
        top_left = self.index(min(rows), 3)
        bottom_right = self.index(max(rows), 3)
        self.dataChanged.emit(top_left, bottom_right, [QtCore.Qt.ItemDataRole.DisplayRole])


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with directory picker and video organiser controls."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Organiser")
        self.resize(1000, 700)
        self.setWindowIcon(self._load_icon("video.svg"))

        self.settings_manager = SettingsManager()
        self.metadata_reader = MetadataReader(cache=MetadataCache())
        self._theme = self.settings_manager.settings.theme

        self.model = VideoTableModel()

        self._setup_ui()
        self._apply_settings()
        self._apply_style()
        self._connect_signals()

    def _setup_ui(self) -> None:
        central_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        controls_layout = QtWidgets.QHBoxLayout()
        self.directory_input = QtWidgets.QLineEdit()
        self.directory_input.setPlaceholderText("Select a directory containing videos")
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.setIcon(self._load_icon("folder.svg"))
        self.load_button = QtWidgets.QPushButton("Load Videos")
        self.load_button.setIcon(self._load_icon("refresh.svg"))
        controls_layout.addWidget(self.directory_input)
        controls_layout.addWidget(self.browse_button)
        controls_layout.addWidget(self.load_button)

        rule_group = QtWidgets.QGroupBox("Rules")
        rule_layout = QtWidgets.QGridLayout(rule_group)
        self.output_input = QtWidgets.QLineEdit(str(Path.home() / "Videos" / "Organized"))
        self.folder_template_input = QtWidgets.QLineEdit("{date:%Y/%m}")
        self.filename_template_input = QtWidgets.QLineEdit("{name}_{resolution}")
        self.tags_input = QtWidgets.QLineEdit()
        self.tags_input.setPlaceholderText("project=demo, location=home")
        self.copy_checkbox = QtWidgets.QCheckBox("Copy instead of move")
        rule_layout.addWidget(QtWidgets.QLabel("Output folder"), 0, 0)
        rule_layout.addWidget(self.output_input, 0, 1)
        rule_layout.addWidget(QtWidgets.QLabel("Folder template"), 1, 0)
        rule_layout.addWidget(self.folder_template_input, 1, 1)
        rule_layout.addWidget(QtWidgets.QLabel("Filename template"), 2, 0)
        rule_layout.addWidget(self.filename_template_input, 2, 1)
        rule_layout.addWidget(QtWidgets.QLabel("Custom tags"), 3, 0)
        rule_layout.addWidget(self.tags_input, 3, 1)
        rule_layout.addWidget(self.copy_checkbox, 4, 0, 1, 2)

        transcription_group = QtWidgets.QGroupBox("Transcription")
        transcription_layout = QtWidgets.QGridLayout(transcription_group)
        self.model_size_combo = QtWidgets.QComboBox()
        self.model_size_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["cpu", "cuda", "auto"])
        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.diarization_checkbox = QtWidgets.QCheckBox("Enable diarization")
        self.language_detection_checkbox = QtWidgets.QCheckBox("Auto-detect language")
        transcription_layout.addWidget(QtWidgets.QLabel("Model size"), 0, 0)
        transcription_layout.addWidget(self.model_size_combo, 0, 1)
        transcription_layout.addWidget(QtWidgets.QLabel("Device"), 1, 0)
        transcription_layout.addWidget(self.device_combo, 1, 1)
        transcription_layout.addWidget(QtWidgets.QLabel("Confidence threshold"), 2, 0)
        transcription_layout.addWidget(self.confidence_spin, 2, 1)
        transcription_layout.addWidget(self.diarization_checkbox, 3, 0, 1, 2)
        transcription_layout.addWidget(self.language_detection_checkbox, 4, 0, 1, 2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.table_view = QtWidgets.QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.verticalHeader().setVisible(False)
        splitter.addWidget(self.table_view)

        metadata_container = QtWidgets.QWidget()
        metadata_layout = QtWidgets.QFormLayout(metadata_container)
        self.file_name_label = QtWidgets.QLabel("–")
        self.duration_label = QtWidgets.QLabel("–")
        self.resolution_label = QtWidgets.QLabel("–")
        self.status_label = QtWidgets.QLabel("–")
        metadata_layout.addRow("File:", self.file_name_label)
        metadata_layout.addRow("Duration:", self.duration_label)
        metadata_layout.addRow("Resolution:", self.resolution_label)
        metadata_layout.addRow("Status:", self.status_label)
        splitter.addWidget(metadata_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        actions_layout = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.preview_button.setIcon(self._load_icon("organize.svg"))
        self.organize_button = QtWidgets.QPushButton("Organize")
        self.organize_button.setIcon(self._load_icon("organize.svg"))
        actions_layout.addStretch(1)
        actions_layout.addWidget(self.preview_button)
        actions_layout.addWidget(self.organize_button)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(0)

        bottom_layout = QtWidgets.QVBoxLayout()
        bottom_layout.addLayout(actions_layout)
        bottom_layout.addWidget(self.progress_bar)

        main_layout.addLayout(controls_layout)
        main_layout.addWidget(rule_group)
        main_layout.addWidget(transcription_group)
        main_layout.addWidget(splitter)
        main_layout.addLayout(bottom_layout)

        self.setCentralWidget(central_widget)

        organizer_menu = self.menuBar().addMenu("Organizer")
        self.preview_action = organizer_menu.addAction("Preview rules")
        self.organize_action = organizer_menu.addAction("Run organizer")

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self._select_directory)
        self.load_button.clicked.connect(self._load_videos)
        self.preview_button.clicked.connect(self._preview_selection)
        self.organize_button.clicked.connect(self._organize_selection)
        self.preview_action.triggered.connect(self._preview_selection)
        self.organize_action.triggered.connect(self._organize_selection)
        self.table_view.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._persist_settings()
        super().closeEvent(event)

    def _select_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select video directory", self.directory_input.text())
        if directory:
            self.directory_input.setText(directory)
            self.settings_manager.update_last_directory(Path(directory))
            self._load_videos()

    def _load_videos(self) -> None:
        directory = Path(self.directory_input.text()).expanduser()
        if not directory.exists() or not directory.is_dir():
            QtWidgets.QMessageBox.warning(self, "Directory not found", "Please select a valid directory containing video files.")
            return

        self.settings_manager.update_last_directory(directory)

        video_files = [
            self._make_video_item(path)
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        self.model.set_items(video_files)
        if video_files:
            self.table_view.selectRow(0)
        else:
            self._clear_metadata()

    def _on_selection_changed(self, selected: QtCore.QItemSelection, _deselected: QtCore.QItemSelection) -> None:
        indexes = selected.indexes()
        if not indexes:
            self._clear_metadata()
            return
        row = indexes[0].row()
        item = self.model.item_at(row)
        if not item:
            self._clear_metadata()
            return
        self.file_name_label.setText(item.path.name)
        self.duration_label.setText(item.duration)
        self.resolution_label.setText(item.resolution)
        self.status_label.setText(item.status)

    def _organize_selection(self) -> None:
        items = self._selected_items()
        if not items:
            QtWidgets.QMessageBox.information(self, "No selection", "Please select a video to organize.")
            return

        organizer = self._build_organizer()
        plans = organizer.preview([item.path for item in items], self._parse_custom_tags())
        if not plans:
            QtWidgets.QMessageBox.information(self, "Organizer", "No destinations were generated.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm organize",
            f"Apply rules to {len(plans)} file(s) and {'copy' if self.copy_checkbox.isChecked() else 'move'} them?",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self._start_progress(len(plans))

        def _progress(processed: int, total: int) -> None:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(processed)

        reports = organizer.commit(
            plans,
            copy_files=self.copy_checkbox.isChecked(),
            progress_callback=_progress,
        )

        failed = [report for report in reports if not report.success]
        if failed:
            errors = "\n".join(f"{r.plan.source.name}: {r.error}" for r in failed)
            QtWidgets.QMessageBox.critical(self, "Organizer failed", errors)
        else:
            QtWidgets.QMessageBox.information(self, "Organizer", "All files processed successfully.")

        self._finish_progress()

        row_lookup = {item.path: index for index, item in enumerate(self.model._items)}
        status_text = "Copied" if self.copy_checkbox.isChecked() else "Moved"
        for report in reports:
            row_index = row_lookup.get(report.plan.source)
            if row_index is None:
                continue
            status = status_text if report.success else "Failed"
            self.model.update_status([row_index], status)
        if items:
            if item := self.model.item_at(self.table_view.currentIndex().row()):
                self.status_label.setText(item.status)

    def _preview_selection(self) -> None:
        paths = self._selected_paths()
        if not paths:
            QtWidgets.QMessageBox.information(self, "No selection", "Please select a video to preview.")
            return

        organizer = self._build_organizer()
        plans = organizer.preview(paths, self._parse_custom_tags())
        lines = [f"{plan.source.name} -> {plan.destination}" for plan in plans]
        message = "\n".join(lines) or "No plans created."
        QtWidgets.QMessageBox.information(self, "Preview", message)

    def _clear_metadata(self) -> None:
        self.file_name_label.setText("–")
        self.duration_label.setText("–")
        self.resolution_label.setText("–")
        self.status_label.setText("–")

    def _selected_paths(self) -> List[Path]:
        return [item.path for item in self._selected_items()]

    def _selected_items(self) -> List[VideoItem]:
        selection = self.table_view.selectionModel().selectedRows()
        items: List[VideoItem] = []
        for index in selection:
            item = self.model.item_at(index.row())
            if item:
                items.append(item)
        return items

    def _parse_custom_tags(self) -> Dict[str, str]:
        text = self.tags_input.text().strip()
        if not text:
            return {}
        tags: Dict[str, str] = {}
        for chunk in text.split(','):
            if "=" in chunk:
                key, value = chunk.split("=", 1)
                tags[key.strip()] = value.strip()
        return tags

    def _build_organizer(self) -> Organizer:
        destination_root = Path(self.output_input.text()).expanduser()
        rule_engine = RuleEngine(
            destination_root=destination_root,
            folder_template=self.folder_template_input.text() or "{date:%Y/%m}",
            filename_template=self.filename_template_input.text() or "{name}_{resolution}",
        )
        return Organizer(rule_engine=rule_engine, metadata_reader=self.metadata_reader)

    def _make_video_item(self, path: Path) -> VideoItem:
        try:
            metadata = self.metadata_reader.read(path)
        except Exception:
            return VideoItem(path=path)

        duration = f"{metadata.duration_seconds:.2f}s" if metadata.duration_seconds else "Unknown"
        resolution = f"{metadata.resolution[0]}x{metadata.resolution[1]}" if metadata.resolution else "Unknown"
        return VideoItem(path=path, duration=duration, resolution=resolution)

    def _start_progress(self, total: int) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(0)

    def _finish_progress(self) -> None:
        self.progress_bar.setVisible(False)

    def _apply_settings(self) -> None:
        settings = self.settings_manager.settings
        if settings.last_directory:
            self.directory_input.setText(settings.last_directory)
        presets = settings.rule_presets
        self.output_input.setText(presets.destination_root)
        self.folder_template_input.setText(presets.folder_template)
        self.filename_template_input.setText(presets.filename_template)
        self._theme = settings.theme
        transcription = settings.transcription
        self.model_size_combo.setCurrentText(transcription.model_size)
        self.device_combo.setCurrentText(transcription.device)
        self.confidence_spin.setValue(transcription.confidence_threshold)
        self.diarization_checkbox.setChecked(transcription.diarization)
        self.language_detection_checkbox.setChecked(transcription.language_detection)

    def _apply_style(self) -> None:
        if self._theme == "dark":
            palette = (
                "QWidget { background-color: #1e1e1e; color: #f0f0f0; font-size: 12px; }"
                "QLineEdit { padding: 6px; background-color: #2a2a2a; border: 1px solid #3a3a3a; }"
                "QPushButton { padding: 6px 12px; background-color: #333333; }"
                "QTableView::item:selected { background-color: #4a90e2; color: white; }"
                "QHeaderView::section { padding: 6px; background: #2d2d2d; }"
            )
        else:
            palette = (
                "QWidget { font-size: 12px; }"
                "QLineEdit { padding: 6px; }"
                "QPushButton { padding: 6px 12px; }"
                "QTableView::item:selected { background-color: #4a90e2; color: white; }"
                "QHeaderView::section { padding: 6px; background: #f0f0f0; }"
            )
        self.setStyleSheet(palette)

    def _persist_settings(self) -> None:
        self.settings_manager.update_rule_presets(
            destination_root=self.output_input.text(),
            folder_template=self.folder_template_input.text(),
            filename_template=self.filename_template_input.text(),
        )
        self.settings_manager.update_last_directory(Path(self.directory_input.text()) if self.directory_input.text() else None)
        self.settings_manager.set_theme(self._theme)
        self.settings_manager.update_transcription_settings(
            model_size=self.model_size_combo.currentText(),
            device=self.device_combo.currentText(),
            confidence_threshold=self.confidence_spin.value(),
            diarization=self.diarization_checkbox.isChecked(),
            language_detection=self.language_detection_checkbox.isChecked(),
        )

    def _load_icon(self, name: str) -> QtGui.QIcon:
        assets_dir = Path(__file__).resolve().parent.parent / "assets"
        icon_path = assets_dir / name
        if icon_path.exists():
            return QtGui.QIcon(str(icon_path))
        return QtGui.QIcon()
