"""Application main window and supporting models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from PySide6 import QtCore, QtGui, QtWidgets


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

        self.model = VideoTableModel()

        self._setup_ui()
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
        self.organize_button = QtWidgets.QPushButton("Organize")
        self.organize_button.setIcon(self._load_icon("organize.svg"))
        actions_layout.addStretch(1)
        actions_layout.addWidget(self.organize_button)

        main_layout.addLayout(controls_layout)
        main_layout.addWidget(splitter)
        main_layout.addLayout(actions_layout)

        self.setCentralWidget(central_widget)

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self._select_directory)
        self.load_button.clicked.connect(self._load_videos)
        self.organize_button.clicked.connect(self._organize_selection)
        self.table_view.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def _select_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select video directory", self.directory_input.text())
        if directory:
            self.directory_input.setText(directory)
            self._load_videos()

    def _load_videos(self) -> None:
        directory = Path(self.directory_input.text()).expanduser()
        if not directory.exists() or not directory.is_dir():
            QtWidgets.QMessageBox.warning(self, "Directory not found", "Please select a valid directory containing video files.")
            return

        video_files = [
            VideoItem(path=path)
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
        selection = self.table_view.selectionModel().selectedRows()
        rows = [index.row() for index in selection]
        if not rows:
            QtWidgets.QMessageBox.information(self, "No selection", "Please select a video to organize.")
            return
        self.model.update_status(rows, "Organized")
        if item := self.model.item_at(rows[0]):
            self.status_label.setText(item.status)

    def _clear_metadata(self) -> None:
        self.file_name_label.setText("–")
        self.duration_label.setText("–")
        self.resolution_label.setText("–")
        self.status_label.setText("–")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-size: 12px; }
            QLineEdit { padding: 6px; }
            QPushButton { padding: 6px 12px; }
            QTableView::item:selected { background-color: #4a90e2; color: white; }
            QHeaderView::section { padding: 6px; background: #f0f0f0; }
            """
        )

    def _load_icon(self, name: str) -> QtGui.QIcon:
        assets_dir = Path(__file__).resolve().parent.parent / "assets"
        icon_path = assets_dir / name
        if icon_path.exists():
            return QtGui.QIcon(str(icon_path))
        return QtGui.QIcon()
