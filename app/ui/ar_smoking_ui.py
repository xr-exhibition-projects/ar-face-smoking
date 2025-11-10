from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Optional
import uuid

import numpy as np
import qdarktheme
from PySide6 import QtCore, QtWidgets, QtGui
import shiboken6

from app.ui import main_ui
from app.ui.core.proxy_style import ProxyStyle
from app.ui.widgets.actions import (
    card_actions,
    list_view_actions,
    video_control_actions,
)
from app.ui.widgets import ui_workers, widget_components
from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS


class ControlOptionsWindow(QtWidgets.QMainWindow):
    closed = QtCore.Signal()

    def __init__(self, content_widget: QtWidgets.QWidget, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Controls Panel")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setCentralWidget(content_widget)
        self.resize(960, 720)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        # Keep the window reusable. By default close() hides the widget, so just emit signal.
        event.accept()
        self.closed.emit()


class ARSmokingWindow(main_ui.MainWindow):
    """Упрощённый интерфейс для live faceswap через веб-камеру."""

    def __init__(self) -> None:
        self._auto_target_selected = False
        self._auto_face_selected = False
        self._target_ready = False
        self._input_ready = False
        self._pending_input_button = None
        self._default_images_dir = self._resolve_default_images_dir()
        self._display_frame_size: tuple[int, int] = (0, 0)
        self.welcomeOverlay: Optional[QtWidgets.QWidget] = None
        self.welcomeLabel: Optional[QtWidgets.QLabel] = None
        self.restart_pixmap: Optional[QtGui.QPixmap] = None
        self.restartButton: Optional[QtWidgets.QPushButton] = None
        super().__init__()
        self._webcam_backend_candidates = self._build_webcam_backend_candidates()
        self._webcam_button: Optional[widget_components.TargetMediaCardButton] = None
        self._fade_progress: float = 1.0
        self._fade_timer: Optional[QtCore.QTimer] = None
        self._fade_step: float = 0.0
        self._swap_active: bool = False
        self._awaiting_second_click: bool = False
        self._second_press_triggered: bool = False
        self._button_icon_state: str = "start"
        self._current_button_pixmap: Optional[QtGui.QPixmap] = None
        self.fade_duration_ms: int = 15_000
        self.impossible_delay_ms: int = 2_200
        self._control_options_widget: Optional[QtWidgets.QWidget] = None
        self.control_window: Optional[ControlOptionsWindow] = None

        self.setWindowTitle("AR Smoking UI")
        self._setup_ar_smoking_ui()
        self._connect_listeners()

        self._load_default_input_faces()
        self._request_webcam_listing()
        self._show_welcome_overlay()

    # ------------------------------------------------------------------ #
    #  MainWindow overrides
    # ------------------------------------------------------------------ #
    def load_last_workspace(self) -> None:  # type: ignore[override]
        """Отключаем автозагрузку рабочего пространства."""
        return

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._position_uporotsya_button()
        self._position_impossible_label()
        if self.deathOverlay.isVisible():
            self.deathOverlay.setGeometry(self.rect())
            self._refresh_death_overlay_graphics()
        self._position_config_button()

    # ------------------------------------------------------------------ #
    #  UI setup helpers
    # ------------------------------------------------------------------ #
    def _setup_ar_smoking_ui(self) -> None:
        self.menuBar().hide()

        self.centralwidget.setStyleSheet("background-color: #000000;")
        self.graphicsViewFrame.setStyleSheet("background-color: #000000; border: none;")
        self.graphicsViewFrame.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphicsViewFrame.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphicsViewFrame.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.graphicsViewFrame.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.graphicsViewFrame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        if hasattr(self, "input_Target_DockWidget") and isinstance(self.input_Target_DockWidget, QtWidgets.QDockWidget):
            self.input_Target_DockWidget.hide()

        if hasattr(self, "verticalLayout"):
            keep_graphics_view = self.graphicsViewFrame
            layout = self.verticalLayout
            for i in reversed(range(layout.count())):
                item = layout.takeAt(i)
                widget = item.widget()
                child_layout = item.layout()
                if widget and widget is keep_graphics_view:
                    keep_graphics_view.setParent(None)
                else:
                    if widget:
                        widget.setParent(None)
                    if child_layout:
                        self._clear_layout(child_layout)
            layout.addWidget(keep_graphics_view)

        self._ensure_control_panel_widget()

        # Добавляем собственную кнопку поверх видео
        self._start_pixmap = QtGui.QPixmap(self._resource_path("ui/start.png"))
        self._finish_pixmap = QtGui.QPixmap(self._resource_path("ui/finish.png"))
        self._welcome_pixmap = QtGui.QPixmap(self._resource_path("ui/not_museum.png"))
        self._welcome_timer: Optional[QtCore.QTimer] = None
        self._welcome_duration_ms: int = 45_000
        self._welcome_active: bool = False

        self.buttonUport = QtWidgets.QPushButton("", parent=self.graphicsViewFrame.viewport())
        self.buttonUport.setObjectName("buttonUport")
        self.buttonUport.setCheckable(True)
        self.buttonUport.setEnabled(False)
        self.buttonUport.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.buttonUport.setFlat(True)
        self.buttonUport.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:pressed {
                transform: scale(0.98);
            }
            """
        )
        self._update_uporotsya_button_icon(start=True)
        self.buttonUport.toggled.connect(self._on_uporotsya_toggled)
        self.buttonUport.clicked.connect(self._on_uporotsya_clicked)
        self._position_uporotsya_button()
        self.graphicsViewFrame.viewport().installEventFilter(self)

        self._impossible_pixmap = QtGui.QPixmap(self._resource_path("ui/impossible.png"))
        self.messageLabel = QtWidgets.QLabel("", parent=self.graphicsViewFrame.viewport())
        self.messageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.messageLabel.setWordWrap(False)
        self.messageLabel.setStyleSheet("background-color: transparent; border: none;")
        self.messageLabel.hide()
        self.messageLabel.setObjectName("messageLabel")

        self.deathOverlay = QtWidgets.QWidget(self)
        self.deathOverlay.setStyleSheet("background-color: #000000;")
        self.deathOverlay.hide()
        death_layout = QtWidgets.QVBoxLayout(self.deathOverlay)
        death_layout.setContentsMargins(40, 40, 40, 40)
        death_layout.addStretch()
        self.deathLabel = QtWidgets.QLabel(self.deathOverlay)
        self.deathLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.deathLabel.setStyleSheet("background-color: transparent; border: none;")
        self.death_pixmap = QtGui.QPixmap(self._resource_path("ui/death.png"))
        if self.death_pixmap.isNull():
            self.deathLabel.setText("СМЕРТЬ\nНЕИЗБЕЖНА")
        death_layout.addWidget(self.deathLabel, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        death_layout.addSpacing(32)

        restart_pixmap = QtGui.QPixmap(self._resource_path("ui/restart.png"))
        self.restartButton = QtWidgets.QPushButton("", self.deathOverlay)
        self.restartButton.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.restartButton.setFlat(True)
        self.restartButton.setStyleSheet("border: none; background: transparent;")
        self.restart_pixmap = restart_pixmap if not restart_pixmap.isNull() else None
        if self.restart_pixmap is not None:
            self.restartButton.setIcon(QtGui.QIcon(self.restart_pixmap))
            self.restartButton.setIconSize(self.restart_pixmap.size())
            self.restartButton.setFixedSize(self.restart_pixmap.size())
        else:
            self.restartButton.setText("НАЧАТЬ СНАЧАЛА")
            self.restartButton.setMinimumWidth(260)
            self.restartButton.setFixedHeight(64)
        self.restartButton.clicked.connect(self._restart_from_death_screen)
        restart_wrapper = QtWidgets.QHBoxLayout()
        restart_wrapper.addStretch()
        restart_wrapper.addWidget(self.restartButton, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        restart_wrapper.addStretch()
        death_layout.addLayout(restart_wrapper)
        death_layout.addSpacing(32)
        death_layout.addStretch()

        # Welcome overlay
        self.welcomeOverlay = QtWidgets.QWidget(self)
        self.welcomeOverlay.setStyleSheet("background-color: #000000;")
        self.welcomeOverlay.hide()
        welcome_layout = QtWidgets.QVBoxLayout(self.welcomeOverlay)
        welcome_layout.setContentsMargins(0, 0, 0, 0)
        welcome_layout.addStretch()
        self.welcomeLabel = QtWidgets.QLabel(self.welcomeOverlay)
        self.welcomeLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.welcomeLabel.setStyleSheet("background-color: transparent; border: none;")
        welcome_layout.addWidget(self.welcomeLabel, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addStretch()
        self.welcomeOverlay.installEventFilter(self)

        self.configButton = QtWidgets.QPushButton("⚙", self)
        self.configButton.setFixedSize(42, 42)
        self.configButton.setToolTip("Настройки")
        self.configButton.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(0,0,0,160);
                color: white;
                border-radius: 21px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(30,30,30,220);
            }
            """
        )
        self.configButton.clicked.connect(self._open_control_options_window)

        # Убираем лишние отступы
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        if hasattr(self, "verticalLayout"):
            self.verticalLayout.setContentsMargins(0, 0, 0, 0)
            self.verticalLayout.setSpacing(0)

        self.setMinimumSize(960, 600)
        self.deathOverlay.setGeometry(self.rect())
        self._position_config_button()

    def _connect_listeners(self) -> None:
        self.inputFacesList.model().rowsInserted.connect(self._on_input_rows_inserted)

    # ------------------------------------------------------------------ #
    #  Initialization helpers
    # ------------------------------------------------------------------ #
    def _resolve_default_images_dir(self) -> Optional[str]:
        project_root = Path(__file__).resolve().parents[2]
        images_dir = project_root / "images"
        if images_dir.is_dir():
            return str(images_dir)
        return None

    def _request_webcam_listing(self) -> None:
        QtCore.QTimer.singleShot(150, lambda: self._ensure_webcam_entry())

    def _load_default_input_faces(self) -> None:
        if not self._default_images_dir:
            QtWidgets.QMessageBox.warning(
                self,
                "Папка с изображениями не найдена",
                "Не удалось найти папку `images`. Добавьте туда изображение для подмены лица.",
            )
            return

        list_view_actions.clear_stop_loading_input_media(self)
        card_actions.clear_input_faces(self)

        self.last_input_media_folder_path = self._default_images_dir
        self.labelInputFacesPath.setText(self._default_images_dir)
        self.labelInputFacesPath.setToolTip(self._default_images_dir)

        self.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(
            main_window=self,
            folder_name=self._default_images_dir,
        )
        self.input_faces_loader_worker.thumbnail_ready.connect(
            partial(list_view_actions.add_media_thumbnail_to_source_faces_list, self)
        )
        self.input_faces_loader_worker.finished.connect(self._on_input_faces_finished)
        self.input_faces_loader_worker.start()

    # ------------------------------------------------------------------ #
    #  Auto-selection callbacks
    # ------------------------------------------------------------------ #
    def _on_input_rows_inserted(self, _parent: QtCore.QModelIndex, first: int, last: int) -> None:
        if self._auto_face_selected:
            return

        for row in range(first, last + 1):
            item = self.inputFacesList.item(row)
            button = self.inputFacesList.itemWidget(item)
            if button:
                self._assign_input_face(button)
                break

    def _on_input_faces_finished(self) -> None:
        if not self.input_faces:
            QtWidgets.QMessageBox.warning(
                self,
                "Нет исходных лиц",
                "В папке `images` не найдено ни одного лица. Добавьте изображение и перезапустите UI.",
            )
            return

        if not self._auto_face_selected:
            first_face = next(iter(self.input_faces.values()), None)
            if first_face:
                self._assign_input_face(first_face)

    # ------------------------------------------------------------------ #
    #  Preparation helpers
    # ------------------------------------------------------------------ #
    def _prepare_target_faces(self, retries: int = 0) -> None:
        if not self.selected_video_button:
            if retries < 10:
                QtCore.QTimer.singleShot(300, lambda: self._prepare_target_faces(retries + 1))
            return

        card_actions.find_target_faces(self)
        if self.target_faces:
            list(self.target_faces.values())[0].click()
            self._target_ready = True
            if self._pending_input_button:
                self._assign_input_face(self._pending_input_button)
            self._try_enable_uporotsya()
            if not self.buttonMediaPlay.isChecked():
                self.buttonMediaPlay.setChecked(True)
        elif retries < 10:
            QtCore.QTimer.singleShot(500, lambda: self._prepare_target_faces(retries + 1))
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Лицо не найдено",
                "Не удалось обнаружить лицо в видеопотоке. Попробуйте осветить сцену и перезапустить UI.",
            )

    def _assign_input_face(self, button) -> None:
        if self._auto_face_selected:
            return

        if not self.cur_selected_target_face_button:
            self._pending_input_button = button
            return

        self._pending_input_button = None
        self._auto_face_selected = True
        QtCore.QTimer.singleShot(0, button.click)
        self._input_ready = True
        self._try_enable_uporotsya()

    def _try_enable_uporotsya(self) -> None:
        if self._welcome_active:
            self.buttonUport.setEnabled(False)
            return
        ready = self._target_ready and (
            self._input_ready or bool(self.cur_selected_target_face_button.assigned_input_faces)
        )
        self.buttonUport.setEnabled(ready)

    # ------------------------------------------------------------------ #
    #  UI actions
    # ------------------------------------------------------------------ #
    def _on_uporotsya_toggled(self, checked: bool) -> None:
        if not self.selected_video_button:
            self.buttonUport.setChecked(False)
            return

        if checked:
            self._update_uporotsya_button_icon(start=False)
            self.swapfacesButton.setChecked(True)
            self._stop_face_fade(reset_progress=True)
            self._start_face_fade_in()
            self._swap_active = True
            self._awaiting_second_click = True
            self._second_press_triggered = False
            self.buttonUport.setCheckable(False)
            self.messageLabel.hide()
        else:
            self._update_uporotsya_button_icon(start=True)
            self.swapfacesButton.setChecked(False)
            self._stop_face_fade(reset_progress=True)
            self._swap_active = False
            self.buttonUport.setCheckable(True)

        video_control_actions.process_swap_faces(self)

        if not self.buttonMediaPlay.isChecked():
            self.buttonMediaPlay.setChecked(True)

    def _on_uporotsya_clicked(self) -> None:
        if not self._swap_active or self._button_icon_state != "finish":
            return
        if self._awaiting_second_click:
            self._awaiting_second_click = False
            return
        if self._second_press_triggered:
            return
        self._second_press_triggered = True
        self.buttonUport.setEnabled(False)
        self._stop_face_fade()
        self._show_impossible_message()

    # ------------------------------------------------------------------ #
    #  Webcam helpers
    # ------------------------------------------------------------------ #
    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        if not layout:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget:
                widget.setParent(None)
            if child_layout:
                self._clear_layout(child_layout)

    def _position_uporotsya_button(self) -> None:
        if not hasattr(self, "buttonUport") or self.buttonUport is None:
            return

        viewport = self.graphicsViewFrame.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            return

        frame_width, frame_height = getattr(self, "_display_frame_size", (width, height))
        frame_width = max(1, frame_width)
        frame_height = max(1, frame_height)

        aspect_ratio = frame_width / frame_height
        if aspect_ratio <= 0:
            aspect_ratio = width / height if height else 1.0
        display_width = min(width, int(height * aspect_ratio))
        display_width = max(1, display_width)

        self._refresh_button_size()
        button_width = self.buttonUport.width()
        button_height = self.buttonUport.height()
        if button_width <= 0 or button_height <= 0:
            button_width = max(200, int(display_width * 0.9))
            button_height = 48
        pos_x = max(0, (width - button_width) // 2)
        pos_y = max(0, height - button_height - 24)
        self.buttonUport.setGeometry(pos_x, pos_y, button_width, button_height)
        self.buttonUport.raise_()

    def _compute_available_button_width(self) -> int:
        viewport = self.graphicsViewFrame.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            return max(width, 0)

        frame_width, frame_height = getattr(self, "_display_frame_size", (width, height))
        if frame_width <= 1 or frame_height <= 1:
            frame_width = width
            frame_height = height

        aspect_ratio = frame_width / frame_height if frame_height else 1.0
        if aspect_ratio <= 0:
            aspect_ratio = width / height if height else 1.0
        display_width = min(width, int(height * aspect_ratio))
        display_width = max(1, display_width)
        margin = 40
        return max(1, min(display_width - margin, width - margin))

    def _compute_frame_display_width(self) -> int:
        viewport = self.graphicsViewFrame.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            return max(width, 0)

        frame_width, frame_height = getattr(self, "_display_frame_size", (width, height))
        frame_width = max(1, frame_width)
        frame_height = max(1, frame_height)

        aspect_ratio = frame_width / frame_height if frame_height else 1.0
        if aspect_ratio <= 0:
            aspect_ratio = width / height if height else 1.0
        display_width = min(width, int(height * aspect_ratio))
        display_width = max(1, display_width)
        return display_width

    def _scaled_button_pixmap(self, pixmap: QtGui.QPixmap, available_width: Optional[int] = None) -> QtGui.QPixmap:
        if pixmap.isNull():
            return pixmap

        if available_width is None:
            available_width = self._compute_available_button_width()
        if available_width <= 0 or pixmap.width() <= available_width:
            return pixmap

        scale_factor = available_width / pixmap.width()
        target_width = max(1, int(pixmap.width() * scale_factor))
        target_height = max(1, int(pixmap.height() * scale_factor))
        return pixmap.scaled(
            target_width,
            target_height,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    def _position_config_button(self) -> None:
        if not hasattr(self, "configButton") or self.configButton is None:
            return
        margin = 16
        y_offset = self.menuBar().height() + margin
        self.configButton.move(self.width() - self.configButton.width() - margin, y_offset)
        self.configButton.raise_()

    def _position_impossible_label(self) -> None:
        if not hasattr(self, "messageLabel") or self.messageLabel is None or not self.messageLabel.isVisible():
            return

        viewport = self.graphicsViewFrame.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            return

        frame_width, frame_height = getattr(self, "_display_frame_size", (width, height))
        frame_width = max(1, min(frame_width, width))
        frame_height = max(1, frame_height)

        aspect_ratio = frame_width / frame_height
        display_width = min(width, int(height * aspect_ratio))
        display_width = max(1, display_width)

        label_width = min(display_width - 60, width - 60) if display_width > 120 else display_width
        label_width = max(180, label_width)
        if self.messageLabel.pixmap():
            label_width = min(label_width, self.messageLabel.pixmap().width())
            label_height = self.messageLabel.pixmap().height()
        else:
            label_height = self.messageLabel.sizeHint().height()
        button_geom = None
        if hasattr(self, "buttonUport") and self.buttonUport:
            button_geom = self.buttonUport.geometry()
            if not button_geom.isNull():
                label_width = button_geom.width()
                label_height = max(label_height, button_geom.height())

        pos_x = max(0, (width - label_width) // 2)
        if button_geom and not button_geom.isNull():
            pos_x = button_geom.x()
            pos_y = button_geom.bottom() - label_height
        else:
            pos_y = height - label_height - 24

        pos_y = max(0, min(height - label_height, pos_y))
        self.messageLabel.setGeometry(pos_x, pos_y, label_width, label_height)
        self.messageLabel.raise_()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj == self.graphicsViewFrame.viewport() and event.type() == QtCore.QEvent.Resize:
            QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
            QtCore.QTimer.singleShot(0, self._position_impossible_label)
            QtCore.QTimer.singleShot(0, self._position_config_button)
            QtCore.QTimer.singleShot(0, self._refresh_welcome_overlay_graphics)
        if obj == getattr(self, "welcomeOverlay", None) and event.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonDblClick):
            self._dismiss_welcome_overlay()
            return True
        return super().eventFilter(obj, event)

    def _start_face_fade_in(self) -> None:
        self._fade_progress = 0.0
        if self._fade_timer:
            self._fade_timer.stop()
            self._fade_timer.deleteLater()
        self._fade_timer = QtCore.QTimer(self)
        interval_ms = 50
        total = float(max(1, self.fade_duration_ms))
        self._fade_step = interval_ms / total
        self._fade_timer.timeout.connect(self._update_fade_progress)
        self._fade_timer.start(interval_ms)

    def _update_fade_progress(self) -> None:
        self._fade_progress = min(1.0, self._fade_progress + self._fade_step)
        if self._fade_progress >= 1.0 and self._fade_timer:
            self._fade_timer.stop()
            self._fade_timer.deleteLater()
            self._fade_timer = None

    def _stop_face_fade(self, reset_progress: bool = False) -> None:
        if self._fade_timer:
            self._fade_timer.stop()
            self._fade_timer.deleteLater()
            self._fade_timer = None
        if reset_progress:
            self._fade_progress = 0.0

    def _show_impossible_message(self) -> None:
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.hide()
        if self._impossible_pixmap and not self._impossible_pixmap.isNull():
            scaled_pixmap = self._scaled_button_pixmap(self._impossible_pixmap)
            self.messageLabel.setPixmap(scaled_pixmap)
            self.messageLabel.setFixedSize(scaled_pixmap.size())
            self.messageLabel.setText("")
        else:
            self.messageLabel.setText("НЕВОЗМОЖНО")
        self.messageLabel.show()
        self._position_impossible_label()
        self.messageLabel.raise_()
        QtCore.QTimer.singleShot(int(self.impossible_delay_ms), self._show_death_screen)

    def _show_death_screen(self) -> None:
        self.messageLabel.hide()
        self.buttonUport.hide()
        self._stop_face_fade(reset_progress=False)
        try:
            self.video_processor.stop_processing()
        except Exception:
            pass
        if self.video_processor.media_capture:
            try:
                self.video_processor.media_capture.release()
            except Exception:
                pass
            self.video_processor.media_capture = None
        self.video_processor.media_path = False
        self.video_processor.file_type = None
        self.video_processor.current_frame = []
        self.swapfacesButton.setChecked(False)
        self._swap_active = False
        if self.control_window and self.control_window.isVisible():
            self.control_window.hide()
        self.deathOverlay.setGeometry(self.rect())
        self._refresh_death_overlay_graphics()
        self.deathOverlay.show()
        self.deathOverlay.raise_()
        self.configButton.hide()
        self._prepare_next_session()

    def _restart_from_death_screen(self) -> None:
        self.deathOverlay.hide()
        self.buttonUport.hide()
        self.configButton.hide()
        self._show_welcome_overlay()

    def _prepare_next_session(self) -> None:
        self._stop_face_fade(reset_progress=True)
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.setEnabled(False)
            self.buttonUport.setCheckable(True)
            self.buttonUport.setChecked(False)
            self._update_uporotsya_button_icon(start=True)
        self._swap_active = False
        self._awaiting_second_click = False
        self._second_press_triggered = False
        self.messageLabel.hide()
        self._auto_target_selected = False
        self._auto_face_selected = False
        self._pending_input_button = None
        self._input_ready = False
        self._target_ready = False
        self.selected_video_button = False
        self.target_videos = {}
        self.targetVideosList.clear()
        self.inputFacesList.clear()
        if getattr(self, "scene", None):
            self.scene.clear()
        if self.control_window:
            self.control_window.hide()
        self.video_processor.current_frame = []
        self.video_processor.media_path = False
        self.video_processor.file_type = None
        QtCore.QTimer.singleShot(100, self._request_webcam_listing)
        QtCore.QTimer.singleShot(150, self._load_default_input_faces)

    def _update_uporotsya_button_icon(self, start: bool) -> None:
        pixmap = self._start_pixmap if start else self._finish_pixmap
        self._button_icon_state = "start" if start else "finish"
        self._current_button_pixmap = pixmap if pixmap and not pixmap.isNull() else None
        tooltip = "Запустить режим" if start else "Остановить режим"
        self.buttonUport.setToolTip(tooltip)
        self.buttonUport.setAccessibleName(tooltip)
        self.buttonUport.setAccessibleDescription(tooltip)
        self._refresh_button_size()
        QtCore.QTimer.singleShot(0, self._position_uporotsya_button)

    def _refresh_button_size(self) -> None:
        available_width = self._compute_available_button_width()
        if self._current_button_pixmap and not self._current_button_pixmap.isNull():
            scaled_pixmap = self._scaled_button_pixmap(self._current_button_pixmap, available_width)
            icon = QtGui.QIcon(scaled_pixmap)
            self.buttonUport.setIcon(icon)
            self.buttonUport.setIconSize(scaled_pixmap.size())
            self.buttonUport.setFixedSize(max(1, scaled_pixmap.width()), max(1, scaled_pixmap.height()))
            self.buttonUport.setText("")
        else:
            self.buttonUport.setIcon(QtGui.QIcon())
            text = "УПОРОТЬСЯ" if self._button_icon_state == "start" else "СЛЕЗТЬ"
            self.buttonUport.setText(text)
            if available_width <= 0:
                min_width = 200
            else:
                min_width = max(200, min(available_width, self.width()))
            self.buttonUport.setMinimumSize(min_width, 48)

    def _refresh_death_overlay_graphics(self) -> None:
        available_width = self._compute_available_button_width()

        if self.death_pixmap and not self.death_pixmap.isNull():
            scaled = self._scaled_button_pixmap(self.death_pixmap, available_width)
            self.deathLabel.setPixmap(scaled)
            self.deathLabel.setFixedSize(scaled.size())
            self.deathLabel.setText("")
        else:
            self.deathLabel.setFixedSize(self.deathLabel.sizeHint())

        if self.restart_pixmap and not self.restart_pixmap.isNull():
            scaled_restart = self._scaled_button_pixmap(self.restart_pixmap, available_width)
            self.restartButton.setIcon(QtGui.QIcon(scaled_restart))
            self.restartButton.setIconSize(scaled_restart.size())
            self.restartButton.setFixedSize(scaled_restart.size())
            self.restartButton.setText("")
        elif self.restartButton and not self.restartButton.text():
            self.restartButton.setIcon(QtGui.QIcon())
            self.restartButton.setText("НАЧАТЬ СНАЧАЛА")
            self.restartButton.setMinimumWidth(260)
            self.restartButton.setFixedHeight(64)

    def _show_welcome_overlay(self) -> None:
        if self._welcome_active or not self.welcomeOverlay or not self.welcomeLabel:
            return

        self._welcome_active = True
        self.welcomeOverlay.setGeometry(self.rect())
        self._refresh_welcome_overlay_graphics()
        self.welcomeOverlay.show()
        self.welcomeOverlay.raise_()

        if self._welcome_timer:
            self._welcome_timer.stop()
            self._welcome_timer.deleteLater()

        self._welcome_timer = QtCore.QTimer(self)
        self._welcome_timer.setSingleShot(True)
        self._welcome_timer.timeout.connect(self._dismiss_welcome_overlay)
        self._welcome_timer.start(self._welcome_duration_ms)

    def _refresh_welcome_overlay_graphics(self) -> None:
        if not self._welcome_active or not self.welcomeOverlay or not self.welcomeLabel:
            return
        self.welcomeOverlay.setGeometry(self.rect())
        available_width = self._compute_frame_display_width()
        if self._welcome_pixmap and not self._welcome_pixmap.isNull():
            scaled = self._welcome_pixmap.scaled(
                available_width,
                self.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.welcomeLabel.setPixmap(scaled)
            self.welcomeLabel.setFixedSize(scaled.size())
        else:
            self.welcomeLabel.setText("")

    def _dismiss_welcome_overlay(self) -> None:
        if not self._welcome_active:
            return
        self._welcome_active = False
        if self._welcome_timer:
            self._welcome_timer.stop()
            self._welcome_timer.deleteLater()
            self._welcome_timer = None
        if self.welcomeOverlay:
            self.welcomeOverlay.hide()
        self._initialize_post_welcome_state()
        self._try_enable_uporotsya()

    def _initialize_post_welcome_state(self) -> None:
        if self.welcomeOverlay:
            self.welcomeOverlay.hide()
        self.buttonUport.show()
        self.configButton.show()
        self._position_config_button()
        QtCore.QTimer.singleShot(0, self._position_uporotsya_button)

    def _open_control_options_window(self) -> None:
        if not self._ensure_control_panel_widget():
            QtWidgets.QMessageBox.information(self, "Control Options", "Нет доступных параметров.")
            return

        if self.control_window is None:
            self.control_window = ControlOptionsWindow(self._control_options_widget, self)
            self.control_window.closed.connect(self._on_control_window_closed)
        else:
            central = self.control_window.centralWidget()
            if central is not self._control_options_widget:
                if central:
                    central.setParent(None)
                self.control_window.setCentralWidget(self._control_options_widget)

        self._control_options_widget.show()
        self.control_window.show()
        self.control_window.raise_()
        self.control_window.activateWindow()

    def _on_control_window_closed(self) -> None:
        if not self.control_window:
            return
        panel_widget = self.control_window.takeCentralWidget()
        if panel_widget:
            panel_widget.setParent(None)
            self._control_options_widget = panel_widget
        self.control_window.hide()
        self.control_window.deleteLater()
        self.control_window = None

    def _ensure_control_panel_widget(self) -> bool:
        if self._control_options_widget and shiboken6.isValid(self._control_options_widget):
            return True

        if hasattr(self, "controlOptionsDockWidget") and self.controlOptionsDockWidget:
            panel_widget = self.controlOptionsDockWidget.widget()
            if panel_widget and shiboken6.isValid(panel_widget):
                self.controlOptionsDockWidget.setWidget(None)
                self.removeDockWidget(self.controlOptionsDockWidget)
                self.controlOptionsDockWidget.setParent(None)
                panel_widget.setParent(None)
                self._control_options_widget = panel_widget
            self.controlOptionsDockWidget = None

        return bool(self._control_options_widget and shiboken6.isValid(self._control_options_widget))

    def preprocess_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            self._display_frame_size = (width, height)
            return frame

        desired_ratio = 9.0 / 16.0
        current_ratio = width / height

        if abs(current_ratio - desired_ratio) < 1e-3:
            self._display_frame_size = (width, height)
            QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
            return frame

        if current_ratio > desired_ratio:
            new_width = int(height * desired_ratio)
            new_width = max(1, min(width, new_width))
            start_x = max(0, (width - new_width) // 2)
            cropped = frame[:, start_x:start_x + new_width]
        else:
            new_height = int(width / desired_ratio)
            new_height = max(1, min(height, new_height))
            start_y = max(0, (height - new_height) // 2)
            cropped = frame[start_y:start_y + new_height, :]

        self._display_frame_size = (cropped.shape[1], cropped.shape[0])
        QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
        return np.ascontiguousarray(cropped)

    def _build_webcam_backend_candidates(self) -> list[str]:
        preferred = self.control.get("WebcamBackendSelection", "Default")
        candidates = [preferred, "DirectShow", "MSMF", "Default"]
        seen: set[str] = set()
        ordered: list[str] = []
        for name in candidates:
            if name in CAMERA_BACKENDS and name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered or ["Default"]

    def _ensure_webcam_entry(self, attempt: int = 0) -> None:
        capture = self.video_processor.media_capture
        if capture and capture.isOpened():
            if not self._auto_target_selected:
                self._auto_target_selected = True
                QtCore.QTimer.singleShot(120, self._ensure_playing)
                QtCore.QTimer.singleShot(360, self._prepare_target_faces)
                QtCore.QTimer.singleShot(400, self._position_uporotsya_button)
            return

        if attempt >= len(self._webcam_backend_candidates):
            QtWidgets.QMessageBox.critical(
                self,
                "Веб-камера недоступна",
                "Не удалось получить изображение с камеры. Проверьте, что камера не занята другим приложением. "
                "Также попробуйте изменить backend на DirectShow или MSMF.",
            )
            return

        backend_name = self._webcam_backend_candidates[attempt]
        backend_flag = CAMERA_BACKENDS[backend_name]
        if self._load_webcam_direct(backend_flag, backend_name):
            self._auto_target_selected = True
            QtCore.QTimer.singleShot(120, self._ensure_playing)
            QtCore.QTimer.singleShot(360, self._prepare_target_faces)
            return

        QtCore.QTimer.singleShot(250, lambda: self._ensure_webcam_entry(attempt + 1))

    def _ensure_playing(self) -> None:
        if self.selected_video_button and not self.buttonMediaPlay.isChecked():
            self.buttonMediaPlay.setChecked(True)

    def _load_webcam_direct(self, backend_flag: int, backend_name: str) -> bool:
        if self._webcam_button:
            self._webcam_button.deleteLater()
            self._webcam_button = None

        media_id = str(uuid.uuid4())
        self._webcam_button = widget_components.TargetMediaCardButton(
            media_path=f"Webcam 0 ({backend_name})",
            file_type="webcam",
            media_id=media_id,
            is_webcam=True,
            webcam_index=0,
            webcam_backend=backend_flag,
            main_window=self,
        )
        self._webcam_button.hide()
        self._webcam_button.load_media()

        capture = self.video_processor.media_capture
        if capture and capture.isOpened():
            self.target_videos = {media_id: self._webcam_button}
            return True

        self._webcam_button.deleteLater()
        self._webcam_button = None
        return False

    def _resource_path(self, relative_path: str) -> str:
        base_path = Path(__file__).resolve().parents[2]
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            base_path = Path(sys._MEIPASS)
        return str(base_path / relative_path)


def _apply_style(app: QtWidgets.QApplication) -> None:
    app.setStyle(ProxyStyle())
    with open("app/ui/styles/dark_styles.qss", "r", encoding="utf-8") as style_file:
        style_sheet = style_file.read()
    style_sheet = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"}) + "\n" + style_sheet
    app.setStyleSheet(style_sheet)


def run() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    _apply_style(app)
    window = ARSmokingWindow()
    window.show()
    if QtWidgets.QApplication.instance() is app:
        app.exec()


if __name__ == "__main__":
    run()

