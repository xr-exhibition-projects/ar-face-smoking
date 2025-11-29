from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path
import time
from typing import Optional, Callable, Tuple
import uuid

import numpy as np
import qdarktheme
from PySide6 import QtCore, QtWidgets, QtGui, QtSvg
import shiboken6

from app.ui import main_ui
from app.ui.core.proxy_style import ProxyStyle
from app.ui.widgets.actions import (
    card_actions,
    layout_actions,
    list_view_actions,
    video_control_actions,
    common_actions,
)
from app.ui.widgets import ui_workers, widget_components
from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
from app.ui.widgets.effect_params_layout_data import EFFECT_PARAMS_LAYOUT_DATA

import cv2


# Asset paths
ASSETS_BASE_DIR = "assets"
ASSETS_IMAGES_DIR = f"{ASSETS_BASE_DIR}/images"
ASSETS_UI_DIR = f"{ASSETS_BASE_DIR}/ui"
ASSETS_VIDEOS_DIR = f"{ASSETS_BASE_DIR}/videos"

# UI image paths
PATH_UI_START = f"{ASSETS_UI_DIR}/start.png"
PATH_UI_FINISH = f"{ASSETS_UI_DIR}/finish.png"
PATH_UI_WELCOME = f"{ASSETS_UI_DIR}/not_museum.png"
PATH_UI_IMPOSSIBLE = f"{ASSETS_UI_DIR}/impossible.png"
PATH_UI_DEATH = f"{ASSETS_UI_DIR}/death.png"
PATH_UI_BORDER_FRAME = f"{ASSETS_UI_DIR}/border_frame.svg"

# Overlay image paths
PATH_UI_OVERLAY_1 = f"{ASSETS_UI_DIR}/overlay/1.png"
PATH_UI_OVERLAY_2 = f"{ASSETS_UI_DIR}/overlay/2.png"
PATH_UI_OVERLAY_3 = f"{ASSETS_UI_DIR}/overlay/3.png"

# Death animation paths
DEATH_FRAME_PATHS = [
    f"{ASSETS_UI_DIR}/death/1.png",
    f"{ASSETS_UI_DIR}/death/2.png",
    f"{ASSETS_UI_DIR}/death/3.png",
    f"{ASSETS_UI_DIR}/death/4.png",
]

# Video paths
PATH_DEMO_VIDEO = f"{ASSETS_VIDEOS_DIR}/demo.mp4"

# Animation config path
ANIMATION_CONFIG_PATH = "animation_config.json"


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

    _BUTTON_MIN_WIDTH: int = 280
    _BUTTON_MAX_WIDTH: int = 560
    _BUTTON_FRAME_RATIO: float = 0.9
    _BUTTON_MARGIN: int = 40
    _LABEL_FRAME_RATIO: float = 0.9
    _VIDEO_FADE_TARGET_OPACITY: float = 0.3

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
        self.mediaToggleButton: Optional[QtWidgets.QPushButton] = None
        self._media_mode: str = "webcam"
        super().__init__()
        self._webcam_backend_candidates = self._build_webcam_backend_candidates()
        self._webcam_button: Optional[widget_components.TargetMediaCardButton] = None
        self._demo_video_button: Optional[widget_components.TargetMediaCardButton] = None
        demo_path = Path(self._resource_path(PATH_DEMO_VIDEO))
        self._demo_video_path: Optional[Path] = demo_path if demo_path.is_file() else None
        self._fade_progress: float = 1.0
        self._fade_timer: Optional[QtCore.QTimer] = None
        self._fade_step: float = 0.0
        self._swap_active: bool = False
        self._second_press_triggered: bool = False
        self._button_icon_state: str = "start"
        self._current_button_pixmap: Optional[QtGui.QPixmap] = None
        self._control_options_widget: Optional[QtWidgets.QWidget] = None
        self.control_window: Optional[ControlOptionsWindow] = None
        
        # Animation stage tracking
        self._current_stage: int = 0  # 0 = not started, 1 = stage1, 2 = stage2, 3 = stage3
        self._stage1_timer: Optional[QtCore.QTimer] = None
        self._stage2_timer: Optional[QtCore.QTimer] = None
        self._stage3_timer: Optional[QtCore.QTimer] = None
        self._stage_start_time: float = 0.0
        self._stage_duration_ms: int = 0
        
        # Load animation config
        self._animation_config = self._load_animation_config()
        self._animation_stages = self._animation_config.get("animation_stages", {})
        self._death_delay_timer: Optional[QtCore.QTimer] = None
        self._death_auto_return_timer: Optional[QtCore.QTimer] = None  # Таймер для автоматического возврата на welcome через 30 сек
        self._sles_auto_return_timer: Optional[QtCore.QTimer] = None  # Таймер для автоматического возврата на welcome, если не нажали "Слезть"
        self._welcome_uporotsya_timer: Optional[QtCore.QTimer] = None  # Таймер для автоматического возврата на welcome, если не нажали "УПОРОТЬСЯ"
        
        # Cached UI sizes to keep controls stable between restarts
        self._button_size_dirty: bool = True
        self._button_width_cache: Optional[int] = None
        self._button_text_min_width: Optional[int] = None
        self._message_label_size_dirty: bool = True
        self._impossible_label_pixmap_cache: Optional[QtGui.QPixmap] = None
        self._impossible_label_text_size_cache: Optional[QtCore.QSize] = None
        self._last_border_frame_size: Optional[tuple[int, int]] = None
        self.videoFadeOverlay: Optional[QtWidgets.QWidget] = None
        self._video_fade_effect: Optional[QtWidgets.QGraphicsOpacityEffect] = None
        self._video_fade_anim: Optional[QtCore.QPropertyAnimation] = None
        self._viewport_mask_active: bool = False
        self._border_frame_renderer: Optional[QtSvg.QSvgRenderer] = None
        self._border_frame_pixmap: Optional[QtGui.QPixmap] = None
        
        # Face loading state
        self._faces_loading_in_progress: bool = False

        import time
        self._startup_start_time = time.time()
        print(f"[Startup] ARSmokingWindow.__init__ started at {time.strftime('%H:%M:%S', time.localtime(self._startup_start_time))}")
        
        self.setWindowTitle("AR Smoking UI")
        t0 = time.time()
        self._setup_ar_smoking_ui()
        print(f"[Startup] _setup_ar_smoking_ui completed in {time.time() - t0:.2f}s")
        
        t0 = time.time()
        self._connect_listeners()
        print(f"[Startup] _connect_listeners completed in {time.time() - t0:.2f}s")

        t0 = time.time()
        self._model_warmup_worker = ui_workers.ModelWarmupWorker(self)
        self._model_warmup_worker.finished.connect(self._on_warmup_finished)
        # Запускаем загрузку лиц только после завершения warmup, чтобы избежать дублирования загрузки моделей
        self._model_warmup_worker.finished.connect(self._load_default_input_faces)
        print(f"[Startup] ModelWarmupWorker created in {time.time() - t0:.2f}s")
        
        t0 = time.time()
        self._model_warmup_worker.start()
        print(f"[Startup] ModelWarmupWorker.start() called in {time.time() - t0:.2f}s (worker runs in background)")
        # НЕ вызываем _load_default_input_faces() здесь - дождёмся завершения warmup
        
        # Откладываем медленные операции - выполняем их асинхронно после показа UI
        # _request_webcam_listing() и _show_welcome_overlay() выполняются быстро,
        # но загрузка изображений в welcome overlay может быть отложена
        t0 = time.time()
        self._request_webcam_listing()
        print(f"[Startup] _request_webcam_listing() completed in {time.time() - t0:.2f}s")
        
        t0 = time.time()
        self._show_welcome_overlay()
        print(f"[Startup] _show_welcome_overlay() completed in {time.time() - t0:.2f}s")
        print(f"[Startup] ARSmokingWindow.__init__ total time: {time.time() - self._startup_start_time:.2f}s")

    # ------------------------------------------------------------------ #
    #  MainWindow overrides
    # ------------------------------------------------------------------ #
    def initialize_widgets(self) -> None:  # type: ignore[override]
        """Переопределяем initialize_widgets чтобы убрать обработчик клика по видео."""
        from functools import partial
        from app.ui.widgets.actions import (
            card_actions,
            layout_actions,
            list_view_actions,
            video_control_actions,
        )
        from app.ui.widgets.event_filters import ListWidgetEventFilter, VideoSeekSliderEventFilter, videoSeekSliderLineEditEventFilter
        
        # Initialize QListWidget for target media
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Set up Menu Actions
        layout_actions.set_up_menu_actions(self)

        # Set up placeholder texts in ListWidgets (Target Videos and Input Faces)
        list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
        list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)

        # Set up click to select and drop action on ListWidgets
        self.targetVideosList.setAcceptDrops(True)
        self.targetVideosList.viewport().setAcceptDrops(False)
        self.inputFacesList.setAcceptDrops(True)
        self.inputFacesList.viewport().setAcceptDrops(False)
        list_widget_event_filter = ListWidgetEventFilter(self, self)
        self.targetVideosList.installEventFilter(list_widget_event_filter)
        self.targetVideosList.viewport().installEventFilter(list_widget_event_filter)
        self.inputFacesList.installEventFilter(list_widget_event_filter)
        self.inputFacesList.viewport().installEventFilter(list_widget_event_filter)

        # Set up folder open buttons for Target and Input
        self.buttonTargetVideosPath.clicked.connect(partial(list_view_actions.select_target_medias, self, 'folder'))
        self.buttonInputFacesPath.clicked.connect(partial(list_view_actions.select_input_face_images, self, 'folder'))

        # Initialize graphics frame to view frames
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        # НЕ устанавливаем GraphicsViewEventFilter - убираем обработчик клика по видео

        video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)

        video_slider_event_filter = VideoSeekSliderEventFilter(self, self.videoSeekSlider)
        self.videoSeekSlider.installEventFilter(video_slider_event_filter)
        self.videoSeekSlider.valueChanged.connect(partial(video_control_actions.on_change_video_seek_slider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(video_control_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(video_control_actions.on_slider_released, self))
        video_control_actions.set_up_video_seek_slider(self)
        self.frameAdvanceButton.clicked.connect(partial(video_control_actions.advance_video_slider_by_n_frames, self))
        self.frameRewindButton.clicked.connect(partial(video_control_actions.rewind_video_slider_by_n_frames, self))

        self.addMarkerButton.clicked.connect(partial(video_control_actions.add_video_slider_marker, self))
        self.removeMarkerButton.clicked.connect(partial(video_control_actions.remove_video_slider_marker, self))
        self.nextMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_next_nearest_marker, self))
        self.previousMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_previous_nearest_marker, self))

        self.viewFullScreenButton.clicked.connect(partial(video_control_actions.view_fullscreen, self))
        # Set up videoSeekLineEdit and add the event filter to handle changes
        video_control_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)

        # Connect the Play/Stop button to the play_video method
        self.buttonMediaPlay.toggled.connect(partial(video_control_actions.play_video, self))
        self.buttonMediaRecord.toggled.connect(partial(video_control_actions.record_video, self))
        
        # Инициализируем control значениями по умолчанию из SETTINGS_LAYOUT_DATA
        # (без создания виджетов, так как они не нужны в ARSmokingWindow)
        from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
        from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
        from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
        from app.ui.widgets.actions import common_actions
        
        def convert_default_value(setting_name, default_value):
            """Конвертирует значение по умолчанию в правильный тип."""
            # Конвертируем строковые значения слайдеров в числа
            if 'Slider' in setting_name and isinstance(default_value, str):
                try:
                    # Пробуем конвертировать в int, если не получается - в float
                    if '.' in default_value:
                        return float(default_value)
                    else:
                        return int(default_value)
                except (ValueError, TypeError):
                    return default_value
            # Конвертируем строковые значения Toggle в bool
            elif 'Toggle' in setting_name and isinstance(default_value, str):
                return default_value.lower() in ('true', '1', 'yes', 'on')
            return default_value
        
        # Инициализируем control из SETTINGS_LAYOUT_DATA
        for setting_group in SETTINGS_LAYOUT_DATA.values():
            for setting_name, setting_data in setting_group.items():
                if 'default' in setting_data:
                    default_value = convert_default_value(setting_name, setting_data['default'])
                    common_actions.create_control(self, setting_name, default_value)
        
        # Инициализируем default_parameters из COMMON_LAYOUT_DATA
        for param_group in COMMON_LAYOUT_DATA.values():
            for param_name, param_data in param_group.items():
                if 'default' in param_data:
                    default_value = convert_default_value(param_name, param_data['default'])
                    common_actions.create_default_parameter(self, param_name, default_value)
        
        # Инициализируем default_parameters из SWAPPER_LAYOUT_DATA
        for param_group in SWAPPER_LAYOUT_DATA.values():
            for param_name, param_data in param_group.items():
                if 'default' in param_data:
                    default_value = convert_default_value(param_name, param_data['default'])
                    common_actions.create_default_parameter(self, param_name, default_value)
        
        # Также инициализируем OutputMediaFolder
        common_actions.create_control(self, 'OutputMediaFolder', '')
        
        # Инициализируем current_widget_parameters с default_parameters
        import copy
        from app.helpers.miscellaneous import ParametersDict
        self.current_widget_parameters = ParametersDict(copy.deepcopy(self.default_parameters), self.default_parameters)
    
    def load_last_workspace(self) -> None:  # type: ignore[override]
        """Отключаем автозагрузку рабочего пространства."""
        return

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        """Переопределяем closeEvent чтобы не сохранять workspace (не нужен в ARSmokingWindow)."""
        # Останавливаем обработку видео
        if hasattr(self, "video_processor") and self.video_processor:
            self.video_processor.stop_processing()
        # НЕ вызываем save_current_workspace - не нужно в ARSmokingWindow
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        was_processing = getattr(self, "video_processor", None) and self.video_processor.processing
        super().resizeEvent(event)
        self._invalidate_overlay_sizes()
        if hasattr(self, "overlayLayer") and self.overlayLayer:
            self.overlayLayer.setGeometry(self.rect())
        self._position_uporotsya_button()
        self._position_impossible_label()
        self._update_video_fade_overlay_geometry()
        self._position_border_frame()
        if self.deathOverlay.isVisible():
            self.deathOverlay.setGeometry(self.rect())
            self._refresh_death_overlay_graphics()
        self._position_config_button()
        if (
            self._welcome_active
            and self.welcomeAnimationLabel
            and self.welcomeAnimationLabel.isVisible()
            and self.welcomeLabel
        ):
            self.welcomeAnimationLabel.setFixedSize(self.welcomeLabel.size())
            self.welcomeAnimationLabel.move(
                (self.width() - self.welcomeAnimationLabel.width()) // 2,
                (self.height() - self.welcomeAnimationLabel.height()) // 2,
            )
        # Перемасштабируем изображение при изменении размера окна
        from app.ui.widgets.actions import layout_actions
        QtCore.QTimer.singleShot(0, lambda: layout_actions.fit_image_to_view_onchange(self))
        if was_processing and not self.video_processor.processing:
            if hasattr(self, "buttonMediaPlay"):
                self.buttonMediaPlay.blockSignals(True)
                self.buttonMediaPlay.setChecked(True)
                self.buttonMediaPlay.blockSignals(False)
                video_control_actions.set_play_button_icon_to_stop(self)
            self.video_processor.process_video()

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

        # Загружаем только критичные изображения сразу (кнопки)
        # Остальные изображения загружаем асинхронно после показа UI
        self._start_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_START))
        self._finish_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_FINISH))
        
        # Откладываем загрузку welcome и death изображений - они не нужны сразу
        self._welcome_pixmap: Optional[QtGui.QPixmap] = None
        self._overlay_frames: list[QtGui.QPixmap] = []
        self._overlay_static: Optional[QtGui.QPixmap] = None
        self._overlay_mask: Optional[QtGui.QPixmap] = None
        self._overlay_frame_index: int = 0
        self._overlay_timer: Optional[QtCore.QTimer] = None
        
        # Death frames загружаем асинхронно
        self._death_frames: list[QtGui.QPixmap] = []
        self._death_intro_frame: Optional[QtGui.QPixmap] = None
        self._death_frame_index: int = 0
        self._death_anim_timer: Optional[QtCore.QTimer] = None
        self._current_death_frame: Optional[QtGui.QPixmap] = None
        
        # Загружаем изображения асинхронно после показа UI
        QtCore.QTimer.singleShot(100, self._load_deferred_images)
        self._welcome_timer: Optional[QtCore.QTimer] = None
        self._welcome_duration_ms: int = 45_000
        self._welcome_active: bool = False

        self.overlayLayer = QtWidgets.QWidget(self)
        # Оставляем слой интерактивным, чтобы дочерние кнопки продолжали получать события мыши
        self.overlayLayer.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.overlayLayer.setStyleSheet("background-color: transparent;")
        self.overlayLayer.setGeometry(self.rect())
        self.overlayLayer.raise_()

        self.videoFadeOverlay = QtWidgets.QWidget(self.overlayLayer)
        self.videoFadeOverlay.setObjectName("videoFadeOverlay")
        self.videoFadeOverlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.videoFadeOverlay.setStyleSheet("background-color: #000000;")
        self.videoFadeOverlay.hide()
        self._video_fade_effect = QtWidgets.QGraphicsOpacityEffect(self.videoFadeOverlay)
        self._video_fade_effect.setOpacity(0.0)
        self.videoFadeOverlay.setGraphicsEffect(self._video_fade_effect)
        self._update_video_fade_overlay_geometry()
        self.videoFadeOverlay.lower()

        self.buttonUport = QtWidgets.QPushButton("", parent=self.overlayLayer)
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

        self._impossible_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_IMPOSSIBLE))
        self.messageLabel = QtWidgets.QLabel("", parent=self.overlayLayer)
        self.messageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.messageLabel.setWordWrap(False)
        self.messageLabel.setStyleSheet("background-color: transparent; border: none;")
        self.messageLabel.hide()
        self.messageLabel.setObjectName("messageLabel")

        # Border frame - отображается поверх видео, но под кнопками и надписями
        self.borderFrameLabel = QtWidgets.QLabel(self.overlayLayer)
        self.borderFrameLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.borderFrameLabel.setStyleSheet("background-color: transparent; border: none;")
        self.borderFrameLabel.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        border_frame_path = self._resource_path(PATH_UI_BORDER_FRAME)
        if QtCore.QFileInfo(border_frame_path).suffix().lower() == "svg":
            renderer = QtSvg.QSvgRenderer(border_frame_path)
            if renderer.isValid():
                self._border_frame_renderer = renderer
            else:
                self._border_frame_pixmap = QtGui.QPixmap(border_frame_path)
                if not self._border_frame_pixmap.isNull():
                    self.borderFrameLabel.setPixmap(self._border_frame_pixmap)
        else:
            self._border_frame_pixmap = QtGui.QPixmap(border_frame_path)
            if not self._border_frame_pixmap.isNull():
                self.borderFrameLabel.setPixmap(self._border_frame_pixmap)
        self.borderFrameLabel.hide()
        self._update_viewport_mask(clear=True)

        self.deathOverlay = QtWidgets.QWidget(self)
        self.deathOverlay.setStyleSheet("background-color: #000000;")
        self.deathOverlay.hide()
        self.deathOverlay.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.deathOverlay.installEventFilter(self)
        death_layout = QtWidgets.QVBoxLayout(self.deathOverlay)
        death_layout.setContentsMargins(40, 40, 40, 40)
        death_layout.addStretch()
        self.deathLabel = QtWidgets.QLabel(self.deathOverlay)
        self.deathLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.deathLabel.setStyleSheet("background-color: transparent; border: none;")
        self.deathLabel.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        fallback_death = QtGui.QPixmap(self._resource_path(PATH_UI_DEATH))
        self.death_pixmap = (
            self._current_death_frame
            if self._current_death_frame
            else (fallback_death if not fallback_death.isNull() else QtGui.QPixmap())
        )
        if self.death_pixmap and not self.death_pixmap.isNull():
            self._set_death_label_frame(self.death_pixmap)
        else:
            self.deathLabel.setText("СМЕРТЬ\nНЕИЗБЕЖНА")
        death_layout.addWidget(self.deathLabel, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
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

        self.welcomeAnimationLabel = QtWidgets.QLabel(self.welcomeOverlay)
        self.welcomeAnimationLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.welcomeAnimationLabel.setStyleSheet("background-color: transparent; border: none;")
        self.welcomeAnimationLabel.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.welcomeAnimationLabel.hide()
        self.welcomeAnimationStaticLabel = QtWidgets.QLabel(self.welcomeOverlay)
        self.welcomeAnimationStaticLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.welcomeAnimationStaticLabel.setStyleSheet("background-color: transparent; border: none;")
        self.welcomeAnimationStaticLabel.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.welcomeAnimationStaticLabel.hide()
        self.welcomeAnimationMaskLabel = QtWidgets.QLabel(self.welcomeOverlay)
        self.welcomeAnimationMaskLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.welcomeAnimationMaskLabel.setStyleSheet("background-color: transparent; border: none;")
        self.welcomeAnimationMaskLabel.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.welcomeAnimationMaskLabel.hide()
        self._overlay_opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.welcomeAnimationLabel)
        self._overlay_opacity_effect.setOpacity(1.0)
        self.welcomeAnimationLabel.setGraphicsEffect(self._overlay_opacity_effect)

        self.configButton = QtWidgets.QPushButton("⚙", self)
        self.configButton.setFixedSize(42, 42)
        self.configButton.setToolTip("Настройки (Ctrl+Alt+D)")
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
        # Скрываем кнопку настроек - используем горячую клавишу Ctrl+Alt+D
        self.configButton.hide()

        self.mediaToggleButton = QtWidgets.QPushButton(self)
        self.mediaToggleButton.setCheckable(True)
        self.mediaToggleButton.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        # Скрываем кнопку переключения камера/видео
        self.mediaToggleButton.hide()
        self.mediaToggleButton.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(0,0,0,140);
                color: white;
                border-radius: 18px;
                font-size: 14px;
                font-weight: 600;
                padding: 6px 14px;
            }
            QPushButton:checked {
                background-color: rgba(79,172,201,200);
                color: black;
            }
            QPushButton:hover {
                background-color: rgba(30,30,30,220);
            }
            """
        )
        self.mediaToggleButton.toggled.connect(self._on_media_toggle)
        # Кнопка всегда скрыта
        self.mediaToggleButton.hide()
        self._update_media_toggle_button()
        QtCore.QTimer.singleShot(0, self._position_border_frame)
        QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
        QtCore.QTimer.singleShot(0, self._position_impossible_label)

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

    def _on_warmup_finished(self) -> None:
        import time
        if hasattr(self, '_startup_start_time'):
            elapsed = time.time() - self._startup_start_time
            print(f"[Startup] ModelWarmupWorker finished in {elapsed:.2f}s (total since init)")
        self._model_warmup_worker = None
        # _load_default_input_faces() будет вызван автоматически через сигнал finished.connect()

    def _load_animation_config(self) -> dict:
        """Загружает конфигурацию анимации из JSON файла."""
        default_config = {
            "timings": {
                "stage1": {
                    "duration_ms": 4000,
                },
                "stage2": {
                    "duration_ms": 2000,
                },
                "stage3": {
                    "duration_ms": 5000,
                },
            },
            "animation_stages": {
                "stage1": {
                    "start_intensity": 0.0,
                    "end_intensity": 1.0,
                },
                "stage2": {
                    "start_intensity": 1.0,
                    "end_intensity": 1.0,
                },
                "stage3": {
                    "start_intensity": 1.0,
                    "end_intensity": 1.0,
                },
            },
            "zombie_overlay": {
                "stage1": {
                    "color_start": 0.0,
                    "color_end": 0.0,
                    "texture_start": 0.0,
                    "texture_end": 0.0,
                },
                "stage2": {
                    "color_start": 0.0,
                    "color_end": 0.8,
                    "texture_start": 0.0,
                    "texture_end": 0.8,
                },
                "stage3": {
                    "color_start": 0.8,
                    "color_end": 1.0,
                    "texture_start": 0.8,
                    "texture_end": 1.0,
                },
            },
        }
        
        # Для frozen приложений ищем рядом с exe, иначе в корне проекта
        if getattr(sys, "frozen", False):
            # Для frozen приложения ищем рядом с exe
            base_path = Path(sys.executable).parent
            config_path = base_path / ANIMATION_CONFIG_PATH
        else:
            # Для обычного запуска используем корень проекта
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / ANIMATION_CONFIG_PATH
        
        if not config_path.exists():
            # Создаём файл с дефолтными значениями
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                print(f"Создан файл конфигурации анимации: {config_path}")
            except Exception as e:
                print(f"Не удалось создать файл конфигурации: {e}")
            return default_config
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Валидация и применение дефолтных значений для отсутствующих ключей
            if "animation_stages" not in config:
                config["animation_stages"] = default_config["animation_stages"]
            else:
                for stage_name, stage_defaults in default_config["animation_stages"].items():
                    stage_cfg = config["animation_stages"].setdefault(stage_name, {})
                    for key, value in stage_defaults.items():
                        stage_cfg.setdefault(key, value)

            if "timings" not in config:
                config["timings"] = default_config["timings"]
            else:
                # Валидация timings
                timings_default = default_config.get("timings", {})
                for key, value in timings_default.items():
                    if key in ("stage1", "stage2", "stage3"):
                        # stage1, stage2, stage3 - это объекты с параметрами
                        if key not in config["timings"]:
                            config["timings"][key] = value
                        else:
                            for subkey, subvalue in value.items():
                                config["timings"][key].setdefault(subkey, subvalue)
                    else:
                        # Остальные тайминги (death_screen_delay_ms и т.д.)
                        config["timings"].setdefault(key, value)

            if "zombie_overlay" not in config:
                config["zombie_overlay"] = default_config["zombie_overlay"]
            else:
                for stage_name, stage_defaults in default_config["zombie_overlay"].items():
                    stage_cfg = config["zombie_overlay"].setdefault(stage_name, {})
                    # Backwards compatibility: migrate color_multiplier -> start/end
                    legacy_color = stage_cfg.pop("color_multiplier", None)
                    legacy_texture = stage_cfg.pop("texture_multiplier", None)
                    for key, value in stage_defaults.items():
                        stage_cfg.setdefault(key, value)
                    if legacy_color is not None:
                        stage_cfg["color_start"] = stage_cfg["color_end"] = float(legacy_color)
                    if legacy_texture is not None:
                        stage_cfg["texture_start"] = stage_cfg["texture_end"] = float(legacy_texture)
            
            # Ensure stage3 exists for backwards compatibility
            if "stage3" not in config.get("animation_stages", {}):
                config.setdefault("animation_stages", {})["stage3"] = default_config["animation_stages"]["stage3"]
            if "stage3" not in config.get("zombie_overlay", {}):
                config.setdefault("zombie_overlay", {})["stage3"] = default_config["zombie_overlay"]["stage3"]
            if "stage3" not in config.get("timings", {}):
                config.setdefault("timings", {})["stage3"] = default_config["timings"]["stage3"]
            
            print(f"Загружена конфигурация анимации из: {config_path}")
            return config
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON в конфигурации анимации: {e}. Используются значения по умолчанию.")
            return default_config
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации анимации: {e}. Используются значения по умолчанию.")
            return default_config

    def _compute_stage_progress(self) -> float:
        """Вычисляет прогресс текущего этапа (0.0 - 1.0) на основе таймеров.
        
        Возвращает значение от 0.0 до 1.0, где:
        - 0.0 = начало этапа (start_intensity)
        - 1.0 = конец этапа (end_intensity)
        """
        if self._current_stage <= 0:
            return 0.0
        
        # Если таймер не установлен, возвращаем 0.0 (этап еще не начался)
        if self._stage_duration_ms <= 0 or self._stage_start_time <= 0:
            return 0.0
        
        current_time = time.monotonic()
        elapsed_ms = (current_time - self._stage_start_time) * 1000.0
        progress = elapsed_ms / float(self._stage_duration_ms)
        
        # Ограничиваем прогресс от 0.0 до 1.0
        progress = max(0.0, min(1.0, progress))
        
        return progress

    def get_aging_factors(self) -> Tuple[float, float]:
        """Возвращает множители для эффекта старения (old_face) исходя из текущего этапа анимации.
        Использует fade-анимацию для плавного проявления."""
        if self._current_stage <= 0:
            return 0.0, 0.0

        # Используем актуальные значения из конфига (могут быть обновлены слайдерами)
        animation_stages = self._animation_config.get("animation_stages", {})
        stage_key = f"stage{self._current_stage}"
        stage_cfg = animation_stages.get(stage_key, {})
        
        if not stage_cfg:
            return 0.0, 0.0
        
        start_intensity = float(stage_cfg.get("start_intensity", 0.0))
        end_intensity = float(stage_cfg.get("end_intensity", start_intensity))
        
        # Убеждаемся, что значения в диапазоне 0.0-1.0 (на случай, если они в процентах)
        if start_intensity > 1.0:
            start_intensity = start_intensity / 100.0
        if end_intensity > 1.0:
            end_intensity = end_intensity / 100.0
        
        # Используем fade-анимацию для плавного проявления
        # _fade_progress обновляется через _update_fade_progress() в fade-таймере
        # Интерполируем между start_intensity и end_intensity на основе _fade_progress
        current_intensity = start_intensity + (end_intensity - start_intensity) * self._fade_progress
        current_intensity = max(0.0, min(1.0, current_intensity))
        
        return current_intensity, current_intensity

    def get_zombie_overlay_factors(self) -> Tuple[float, float]:
        """Возвращает множители для эффекта зомби (zombie_texture) исходя из текущего этапа анимации."""
        overlay_cfg = self._animation_config.get("zombie_overlay", {})
        if self._current_stage <= 0:
            return 0.0, 0.0

        stage_key = f"stage{self._current_stage}"
        stage_cfg = overlay_cfg.get(stage_key, {})

        if not stage_cfg:
            return 0.0, 0.0

        # Получаем start и end значения для плавной интерполяции
        color_start = float(stage_cfg.get("color_start", 0.0))
        color_end = float(stage_cfg.get("color_end", color_start))
        texture_start = float(stage_cfg.get("texture_start", 0.0))
        texture_end = float(stage_cfg.get("texture_end", texture_start))

        # Убеждаемся, что значения в диапазоне 0.0-1.0 (на случай, если они в процентах)
        if color_start > 1.0:
            color_start = color_start / 100.0
        if color_end > 1.0:
            color_end = color_end / 100.0
        if texture_start > 1.0:
            texture_start = texture_start / 100.0
        if texture_end > 1.0:
            texture_end = texture_end / 100.0

        # Вычисляем прогресс этапа (0.0 - 1.0) на основе прошедшего времени
        stage_progress = self._compute_stage_progress()

        # Плавно интерполируем между start и end значениями
        # stage_progress = 0.0 -> current = start
        # stage_progress = 1.0 -> current = end
        current_color = color_start + (color_end - color_start) * stage_progress
        current_texture = texture_start + (texture_end - texture_start) * stage_progress

        # Ограничиваем значения от 0.0 до 1.0
        current_color = max(0.0, min(1.0, current_color))
        current_texture = max(0.0, min(1.0, current_texture))

        return current_color, current_texture

    # ------------------------------------------------------------------ #
    #  Initialization helpers
    # ------------------------------------------------------------------ #
    def _resolve_default_images_dir(self) -> Optional[str]:
        """Определяет путь к папке с изображениями для подмены лица."""
        # Для frozen приложений ищем в нескольких местах
        if getattr(sys, "frozen", False):
            base_path = Path(sys.executable).parent
            # Пробуем рядом с exe
            images_dir = base_path / ASSETS_IMAGES_DIR
            if images_dir.is_dir():
                return str(images_dir)
            # Пробуем в _internal
            images_dir = base_path / "_internal" / ASSETS_IMAGES_DIR
            if images_dir.is_dir():
                return str(images_dir)
            return None
        else:
            # Для обычного запуска используем корень проекта
            project_root = Path(__file__).resolve().parents[2]
            images_dir = project_root / ASSETS_IMAGES_DIR
            if images_dir.is_dir():
                return str(images_dir)
            return None

    def _request_webcam_listing(self) -> None:
        QtCore.QTimer.singleShot(150, lambda: self._ensure_webcam_entry())
        if self._demo_video_path:
            QtCore.QTimer.singleShot(1200, self._fallback_to_demo_video)

    def _load_default_input_faces(self) -> None:
        import time
        t0 = time.time()
        print(f"[Startup] _load_default_input_faces() started")
        
        # Защита от повторного вызова
        if self._faces_loading_in_progress:
            print(f"[Startup] _load_default_input_faces() skipped (already in progress)")
            return
        
        if self._default_images_dir:
            print(f"[Startup] Found images directory: {self._default_images_dir}")
        
        if not self._default_images_dir:
            QtWidgets.QMessageBox.warning(
                self,
                "Папка с изображениями не найдена",
                f"Не удалось найти папку `{ASSETS_IMAGES_DIR}`. Добавьте туда изображение для подмены лица.",
            )
            print(f"[Startup] _load_default_input_faces() skipped (no images dir) in {time.time() - t0:.2f}s")
            print(f"[Startup] Searched path: {ASSETS_IMAGES_DIR}")
            return

        # Устанавливаем флаг, чтобы предотвратить повторный вызов
        self._faces_loading_in_progress = True
        
        list_view_actions.clear_stop_loading_input_media(self)
        card_actions.clear_input_faces(self)

        self.last_input_media_folder_path = self._default_images_dir
        self.labelInputFacesPath.setText(self._default_images_dir)
        self.labelInputFacesPath.setToolTip(self._default_images_dir)

        t1 = time.time()
        self.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(
            main_window=self,
            folder_name=self._default_images_dir,
        )
        print(f"[Startup] InputFacesLoaderWorker created in {time.time() - t1:.2f}s")
        
        self.input_faces_loader_worker.thumbnail_ready.connect(
            partial(list_view_actions.add_media_thumbnail_to_source_faces_list, self)
        )
        self.input_faces_loader_worker.finished.connect(self._on_input_faces_finished)
        # Сбрасываем флаг при завершении загрузки
        self.input_faces_loader_worker.finished.connect(lambda: setattr(self, '_faces_loading_in_progress', False))
        
        t1 = time.time()
        self.input_faces_loader_worker.start()
        print(f"[Startup] InputFacesLoaderWorker.start() called in {time.time() - t1:.2f}s (worker runs in background)")
        print(f"[Startup] _load_default_input_faces() setup completed in {time.time() - t0:.2f}s")

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
        import time
        t0 = time.time()
        if hasattr(self, '_startup_start_time'):
            elapsed = time.time() - self._startup_start_time
            print(f"[Startup] InputFacesLoaderWorker finished in {elapsed:.2f}s (total since init)")
        
        if not self.input_faces:
            images_path = self._default_images_dir or ASSETS_IMAGES_DIR
            QtWidgets.QMessageBox.warning(
                self,
                "Нет исходных лиц",
                f"В папке `{images_path}` не найдено ни одного лица. Добавьте изображение и перезапустите UI.",
            )
            return

        if not self._auto_face_selected:
            first_face = next(iter(self.input_faces.values()), None)
            if first_face:
                t1 = time.time()
                self._assign_input_face(first_face)
                print(f"[Startup] _assign_input_face() completed in {time.time() - t1:.2f}s")
        print(f"[Startup] _on_input_faces_finished() completed in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------ #
    #  Preparation helpers
    # ------------------------------------------------------------------ #
    def _prepare_target_faces(self, retries: int = 0) -> None:
        import time
        if retries == 0:
            t_start = time.time()
            print(f"[Startup] _prepare_target_faces() started (retry {retries})")
        
        if not self.selected_video_button:
            if retries < 10:
                QtCore.QTimer.singleShot(100, lambda: self._prepare_target_faces(retries + 1))  # Уменьшено с 300ms до 100ms
            return

        was_playing = self.buttonMediaPlay.isChecked()
        t0 = time.time()
        card_actions.find_target_faces(self)
        print(f"[Startup] find_target_faces() completed in {time.time() - t0:.2f}s")
        
        if was_playing and not self.video_processor.processing:
            QtCore.QTimer.singleShot(0, self._ensure_playing)

        if self.target_faces:
            list(self.target_faces.values())[0].click()
            self._target_ready = True
            if self._pending_input_button:
                self._assign_input_face(self._pending_input_button)
            self._try_enable_uporotsya()
            if not self.buttonMediaPlay.isChecked():
                self.buttonMediaPlay.setChecked(True)
            if retries == 0:
                elapsed = time.time() - t_start
                print(f"[Startup] _prepare_target_faces() completed in {elapsed:.2f}s")
                if hasattr(self, '_startup_start_time'):
                    total_elapsed = time.time() - self._startup_start_time
                    print(f"[Startup] Target faces prepared in {total_elapsed:.2f}s (total since init)")
        elif retries < 10:
            QtCore.QTimer.singleShot(200, lambda: self._prepare_target_faces(retries + 1))  # Уменьшено с 500ms до 200ms
        # Убрали показ ошибки здесь - ошибка показывается только при нажатии на кнопку "УПОРОТЬСЯ"

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
        if not ready and self.selected_video_button:
            ready = True
        self.buttonUport.setEnabled(ready)

    # ------------------------------------------------------------------ #
    #  UI actions
    # ------------------------------------------------------------------ #
    def _on_uporotsya_toggled(self, checked: bool) -> None:
        if not self.selected_video_button:
            self.buttonUport.setChecked(False)
            return

        if checked:
            # Отменяем таймер автоматического возврата (пользователь нажал "УПОРОТЬСЯ")
            self._cancel_welcome_uporotsya_timer()
            if not self.target_faces:
                card_actions.find_target_faces(self)
                if not self.target_faces:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Лицо не найдено",
                        "Не удалось обнаружить лицо. Убедитесь, что камера направлена на лицо и попробуйте ещё раз.",
                    )
                    self.buttonUport.setChecked(False)
                    return
            
            self._current_stage = 1
            self._start_stage1_animation()
            
            if self._stage_start_time <= 0 or self._stage_duration_ms <= 0:
                timings = self._animation_config.get("timings", {})
                stage1_timings = timings.get("stage1", {})
                duration_ms = stage1_timings.get("duration_ms", 4000)
                self._stage_start_time = time.monotonic()
                self._stage_duration_ms = duration_ms
            
            if self._current_stage != 1:
                self._current_stage = 1
            
            self.swapfacesButton.setChecked(True)
            self._swap_active = True
            self._second_press_triggered = False
            self.buttonUport.setCheckable(False)
            self.buttonUport.hide()
            self.messageLabel.hide()
        else:
            self._update_uporotsya_button_icon(start=True)
            self.swapfacesButton.setChecked(False)
            self._stop_face_fade(reset_progress=True)
            self._swap_active = False

        video_control_actions.process_swap_faces(self)

        if not self.buttonMediaPlay.isChecked():
            self.buttonMediaPlay.setChecked(True)

    def _on_uporotsya_clicked(self) -> None:
        if not self._swap_active or self._button_icon_state != "finish":
            return
        if self._second_press_triggered:
            return
        if self._current_stage != 1:
            return

        # Отменяем таймер автоматического возврата (пользователь нажал "Слезть")
        self._cancel_sles_auto_return_timer()

        # Переход ко второму этапу
        self._second_press_triggered = True
        # НЕ меняем _current_stage здесь - он уже установлен в _start_stage2_animation
        # self._current_stage = 2
        # НЕ вызываем _stop_face_fade() здесь, так как он может сбросить _stage_start_time
        # self._stop_face_fade()
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.hide()
        self._start_stage2_animation()

    def _reset_swap_state(self) -> None:
        self._swap_active = False
        self._second_press_triggered = False
        # Отменяем таймер "Слезть" при сбросе состояния
        self._cancel_sles_auto_return_timer()
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.setEnabled(False)
            self.buttonUport.setCheckable(True)
            self.buttonUport.setChecked(False)
            self._update_uporotsya_button_icon(start=True)
        if hasattr(self, "swapfacesButton"):
            self.swapfacesButton.setChecked(False)
        if hasattr(self, "faceMaskCheckBox"):
            self.faceMaskCheckBox.setChecked(False)
        if hasattr(self, "faceCompareCheckBox"):
            self.faceCompareCheckBox.setChecked(False)

    def _ensure_default_input_face(self) -> None:
        if not self.input_faces:
            return
        first_face = next(iter(self.input_faces.values()), None)
        if not first_face:
            return
        self._pending_input_button = first_face
        QtCore.QTimer.singleShot(0, lambda button=first_face: self._assign_input_face(button))

    def _on_media_toggle(self, checked: bool) -> None:
        target_mode = "webcam" if checked else "video"
        if target_mode == self._media_mode:
            return

        success = self._activate_webcam_mode() if target_mode == "webcam" else self._activate_video_mode()

        if not success:
            self.mediaToggleButton.blockSignals(True)
            self.mediaToggleButton.setChecked(self._media_mode == "webcam")
            self.mediaToggleButton.blockSignals(False)
            return

        self._media_mode = target_mode
        self._update_media_toggle_button()

    def _activate_video_mode(self) -> bool:
        self.buttonMediaPlay.setChecked(False)
        self.video_processor.stop_processing()

        self._reset_swap_state()
        self._auto_target_selected = False
        self._target_ready = False
        self._input_ready = False
        self._auto_face_selected = False
        self._pending_input_button = None
        self._ensure_default_input_face()

        if not self._demo_video_path or not self._demo_video_path.is_file():
            QtWidgets.QMessageBox.warning(
                self,
                "Демо-видео недоступно",
                "Файл демо-видео нельзя найти. Добавьте файл или выберите веб-камеру.",
            )
            return False

        self._auto_target_selected = False
        self._target_ready = False
        self._input_ready = False

        success = self._load_demo_video()
        if not success:
            QtWidgets.QMessageBox.warning(
                self,
                "Не удалось загрузить видео",
                "Попробуйте выбрать другой файл или переключиться на веб-камеру.",
            )
        return success

    def _activate_webcam_mode(self) -> bool:
        self.buttonMediaPlay.setChecked(False)
        self.video_processor.stop_processing()

        self._reset_swap_state()
        self._auto_target_selected = False
        self._target_ready = False
        self._input_ready = False
        self._auto_face_selected = False
        self._pending_input_button = None
        self._ensure_default_input_face()

        success = False
        for backend_name in self._webcam_backend_candidates:
            backend_flag = CAMERA_BACKENDS[backend_name]
            if self._load_webcam_direct(backend_flag, backend_name):
                success = True
                break

        if not success:
            QtWidgets.QMessageBox.critical(
                self,
                "Веб-камера недоступна",
                "Не удалось подключиться к веб-камере. Проверьте устройство и попробуйте снова.",
            )
            return False

        QtCore.QTimer.singleShot(120, self._ensure_playing)
        QtCore.QTimer.singleShot(360, self._prepare_target_faces)
        QtCore.QTimer.singleShot(400, self._try_enable_uporotsya)
        return True

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

    def _invalidate_overlay_sizes(self) -> None:
        """Marks cached button/label sizes for recompute on next resize-aware update."""
        self._button_size_dirty = True
        self._button_width_cache = None
        self._button_text_min_width = None
        self._message_label_size_dirty = True
        self._impossible_label_pixmap_cache = None
        self._impossible_label_text_size_cache = None

    def _position_uporotsya_button(self) -> None:
        if not hasattr(self, "buttonUport") or self.buttonUport is None:
            return

        viewport = self.graphicsViewFrame.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            return

        viewport_pos = viewport.mapTo(self, QtCore.QPoint(0, 0))
        vx = viewport_pos.x()
        vy = viewport_pos.y()

        # Получаем ширину рамки, если она видима
        border_frame_width = self._get_border_frame_width()
        
        frame_width, frame_height = getattr(self, "_display_frame_size", (width, height))
        frame_width = max(1, frame_width)
        frame_height = max(1, frame_height)

        aspect_ratio = frame_width / frame_height
        if aspect_ratio <= 0:
            aspect_ratio = width / height if height else 1.0
        display_width = min(width, int(height * aspect_ratio))
        display_width = max(1, display_width)
        
        # Если рамка видима, используем её ширину как максимальную ширину кнопки
        if border_frame_width is not None and border_frame_width > 0:
            max_button_width = int(border_frame_width * self._BUTTON_FRAME_RATIO)
            display_width = min(display_width, max_button_width)

        self._refresh_button_size()
        button_width = self.buttonUport.width()
        button_height = self.buttonUport.height()
        if button_width <= 0 or button_height <= 0:
            button_width = max(
                self._BUTTON_MIN_WIDTH, min(display_width, self._BUTTON_MAX_WIDTH)
            )
            button_height = 48
        
        # Ограничиваем ширину кнопки шириной рамки, если рамка видима
        if border_frame_width is not None and border_frame_width > 0:
            button_width = min(button_width, int(border_frame_width * self._BUTTON_FRAME_RATIO))
        else:
            button_width = min(button_width, self._BUTTON_MAX_WIDTH)
        button_width = max(self._BUTTON_MIN_WIDTH, button_width)

        frame_height = self._get_border_frame_height()
        vertical_offset = frame_height if frame_height and frame_height > 0 else height
        vertical_offset = int(vertical_offset * 0.15)
        pos_x = vx + max(0, (width - button_width) // 2)
        pos_y = vy + max(0, height - button_height - vertical_offset)
        self.buttonUport.setGeometry(pos_x, pos_y, button_width, button_height)
        # Кнопка всегда должна быть поверх рамки
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
        
        # Получаем ширину рамки, если она видима
        border_frame_width = self._get_border_frame_width()
        if border_frame_width is not None and border_frame_width > 0:
            # Используем ширину рамки как максимальную ширину
            max_button_width = int(border_frame_width * self._BUTTON_FRAME_RATIO)
        else:
            max_button_width = self._BUTTON_MAX_WIDTH
        display_width = min(display_width, max_button_width)
        
        margin = self._BUTTON_MARGIN
        base_width = max(1, min(display_width - margin, width - margin))
        return max(self._BUTTON_MIN_WIDTH, min(base_width, max_button_width))

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

    def _get_impossible_label_pixmap(self, force: bool = False) -> Optional[QtGui.QPixmap]:
        if force:
            self._message_label_size_dirty = True
        if (
            not self._message_label_size_dirty
            and self._impossible_label_pixmap_cache is not None
            and not self._impossible_label_pixmap_cache.isNull()
        ):
            return self._impossible_label_pixmap_cache
        if not self._impossible_pixmap or self._impossible_pixmap.isNull():
            return None
        max_label_width = self._compute_available_button_width()
        if max_label_width <= 0:
            max_label_width = self._impossible_pixmap.width()
        if self._impossible_pixmap.width() <= max_label_width:
            scaled_pixmap = QtGui.QPixmap(self._impossible_pixmap)
        else:
            scale_factor = max_label_width / self._impossible_pixmap.width()
            target_height = max(1, int(self._impossible_pixmap.height() * scale_factor))
            scaled_pixmap = self._impossible_pixmap.scaled(
                max_label_width,
                target_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        self._impossible_label_pixmap_cache = scaled_pixmap
        self._message_label_size_dirty = False
        return scaled_pixmap

    def _apply_impossible_text_size(self) -> None:
        self.messageLabel.setPixmap(QtGui.QPixmap())
        text = "НЕВОЗМОЖНО"
        self.messageLabel.setText(text)
        available_width = self._compute_available_button_width()
        if available_width <= 0:
            available_width = self._BUTTON_MAX_WIDTH
        if self._message_label_size_dirty or self._impossible_label_text_size_cache is None:
            fm = self.messageLabel.fontMetrics()
            width = min(available_width, max(self._BUTTON_MIN_WIDTH, fm.horizontalAdvance(text) + 24))
            button_height = getattr(self, "buttonUport", None).height() if hasattr(self, "buttonUport") and self.buttonUport else 48
            height = max(button_height, fm.height() + 12)
            self._impossible_label_text_size_cache = QtCore.QSize(max(1, width), max(1, height))
            self._message_label_size_dirty = False
        if self._impossible_label_text_size_cache:
            width = min(self._impossible_label_text_size_cache.width(), available_width)
            self.messageLabel.setFixedSize(width, self._impossible_label_text_size_cache.height())

    def _position_config_button(self) -> None:
        if not hasattr(self, "configButton") or self.configButton is None:
            return
        margin = 16
        y_offset = self.menuBar().height() + margin
        x_config = self.width() - self.configButton.width() - margin
        if getattr(self, "mediaToggleButton", None) and self.mediaToggleButton.isVisible():
            toggle_width = self.mediaToggleButton.width()
            spacing = 12
            x_toggle = max(margin, x_config - toggle_width - spacing)
            self.mediaToggleButton.move(x_toggle, y_offset)
            self.mediaToggleButton.raise_()
            x_config = self.width() - self.configButton.width() - margin
        self.configButton.move(x_config, y_offset)
        self.configButton.raise_()
        if getattr(self, "mediaToggleButton", None) and self.mediaToggleButton.isVisible():
            self.mediaToggleButton.raise_()

    def _position_impossible_label(self) -> None:
        if not hasattr(self, "messageLabel") or self.messageLabel is None or not self.messageLabel.isVisible():
            return

        viewport = self.graphicsViewFrame.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            return

        viewport_pos = viewport.mapTo(self, QtCore.QPoint(0, 0))
        vx = viewport_pos.x()
        vy = viewport_pos.y()

        # Получаем ширину рамки, если она видима
        border_frame_width = self._get_border_frame_width()

        frame_width, frame_height = getattr(self, "_display_frame_size", (width, height))
        frame_width = max(1, frame_width)
        frame_height = max(1, frame_height)

        aspect_ratio = frame_width / frame_height
        if aspect_ratio <= 0:
            aspect_ratio = width / height if height else 1.0
        display_width = min(width, int(height * aspect_ratio))
        display_width = max(1, display_width)

        available_width = self._compute_available_button_width()
        label_pixmap = self.messageLabel.pixmap()
        if label_pixmap and not label_pixmap.isNull():
            if self._message_label_size_dirty:
                scaled = self._get_impossible_label_pixmap(force=True)
            else:
                scaled = self._get_impossible_label_pixmap()
            if scaled and not scaled.isNull():
                self.messageLabel.setPixmap(scaled)
                self.messageLabel.setFixedSize(scaled.size())
        else:
            self._apply_impossible_text_size()

        label_width = min(self.messageLabel.width(), available_width if available_width > 0 else self.messageLabel.width())
        label_height = self.messageLabel.height()

        frame_height_value = self._get_border_frame_height()
        vertical_offset = frame_height_value if frame_height_value and frame_height_value > 0 else height
        vertical_offset = int(vertical_offset * 0.15)

        pos_x = vx + max(0, (width - label_width) // 2)
        pos_y = vy + max(0, height - label_height - vertical_offset)
        
        self.messageLabel.setGeometry(pos_x, pos_y, label_width, label_height)
        # Надпись всегда должна быть поверх рамки
        self.messageLabel.raise_()
        self._update_video_fade_overlay_geometry()

    def _update_video_fade_overlay_geometry(self) -> None:
        if not self.videoFadeOverlay:
            return
        viewport = self.graphicsViewFrame.viewport()
        if not viewport:
            self.videoFadeOverlay.hide()
            return
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            self.videoFadeOverlay.hide()
            return
        viewport_pos = viewport.mapTo(self, QtCore.QPoint(0, 0))
        self.videoFadeOverlay.setGeometry(viewport_pos.x(), viewport_pos.y(), width, height)
        # Держим затемнение под кнопками и надписями, но над видео
        self.videoFadeOverlay.lower()
        # Показываем overlay если анимация активна или opacity > 0
        if self._video_fade_anim and self._video_fade_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self.videoFadeOverlay.show()
        elif self._video_fade_effect and self._video_fade_effect.opacity() > 0.001:
            self.videoFadeOverlay.show()
        else:
            self.videoFadeOverlay.hide()

    def _start_video_darkening(self, duration_ms: int) -> None:
        if not self.videoFadeOverlay or not self._video_fade_effect:
            return
        # Обновляем геометрию перед показом
        viewport = self.graphicsViewFrame.viewport()
        if viewport:
            width = viewport.width()
            height = viewport.height()
            if width > 0 and height > 0:
                viewport_pos = viewport.mapTo(self, QtCore.QPoint(0, 0))
                self.videoFadeOverlay.setGeometry(viewport_pos.x(), viewport_pos.y(), width, height)
        # Показываем overlay и устанавливаем его под кнопками
        self.videoFadeOverlay.show()
        self.videoFadeOverlay.lower()
        if self._video_fade_anim:
            self._video_fade_anim.stop()
            self._video_fade_anim.deleteLater()
        animation = QtCore.QPropertyAnimation(self._video_fade_effect, b"opacity", self)
        animation.setDuration(max(250, duration_ms))
        animation.setStartValue(self._video_fade_effect.opacity())
        animation.setEndValue(self._VIDEO_FADE_TARGET_OPACITY)
        animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)

        current_anim = animation

        def on_finished() -> None:
            if self._video_fade_effect and self.videoFadeOverlay:
                if self._video_fade_effect.opacity() <= 0.001:
                    self.videoFadeOverlay.hide()
                else:
                    self.videoFadeOverlay.show()
            if self._video_fade_anim is current_anim:
                current_anim.deleteLater()
                self._video_fade_anim = None
            else:
                current_anim.deleteLater()

        animation.finished.connect(on_finished)
        animation.start()
        self._video_fade_anim = animation

    def _reset_video_darkening(self) -> None:
        if self._video_fade_anim:
            self._video_fade_anim.stop()
            self._video_fade_anim.deleteLater()
            self._video_fade_anim = None
        if self._video_fade_effect:
            self._video_fade_effect.setOpacity(0.0)
        if self.videoFadeOverlay:
            self.videoFadeOverlay.hide()

    def _update_viewport_mask(self, clear: bool = False) -> None:
        viewport_widget = getattr(self, "graphicsViewFrame", None)
        viewport = viewport_widget.viewport() if viewport_widget else None
        if not viewport:
            return
        border_label = getattr(self, "borderFrameLabel", None)
        if (
            clear
            or border_label is None
            or not border_label.isVisible()
            or border_label.width() <= 0
            or border_label.height() <= 0
        ):
            if self._viewport_mask_active:
                viewport.clearMask()
                self._viewport_mask_active = False
            return

        top_left_global = border_label.mapToGlobal(QtCore.QPoint(0, 0))
        bottom_right_global = border_label.mapToGlobal(border_label.rect().bottomRight())
        top_left = viewport.mapFromGlobal(top_left_global)
        bottom_right = viewport.mapFromGlobal(bottom_right_global)
        mask_rect = QtCore.QRect(top_left, bottom_right).normalized()
        mask_rect = mask_rect.intersected(QtCore.QRect(0, 0, viewport.width(), viewport.height()))
        if mask_rect.isEmpty():
            if self._viewport_mask_active:
                viewport.clearMask()
                self._viewport_mask_active = False
            return

        viewport.setMask(QtGui.QRegion(mask_rect))
        self._viewport_mask_active = True

    def _get_border_frame_width(self) -> Optional[int]:
        """Возвращает ширину рамки или прогноз, если рамка пока скрыта."""
        if not hasattr(self, "borderFrameLabel") or self.borderFrameLabel is None:
            return self._estimate_border_frame_width()
        if self.borderFrameLabel.isVisible():
            return self.borderFrameLabel.width()
        if self._last_border_frame_size:
            return self._last_border_frame_size[0]
        return self._estimate_border_frame_width()

    def _estimate_border_frame_width(self) -> Optional[int]:
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return None
        frame_height = height
        frame_width = int(frame_height * 9 / 16)
        if frame_width > width:
            frame_width = width
            frame_height = int(frame_width * 16 / 9)
        return frame_width

    def _get_border_frame_height(self) -> Optional[int]:
        """Возвращает высоту рамки или прогноз, если рамка пока скрыта."""
        if not hasattr(self, "borderFrameLabel") or self.borderFrameLabel is None:
            return self._estimate_border_frame_height()
        if self.borderFrameLabel.isVisible():
            return self.borderFrameLabel.height()
        if self._last_border_frame_size:
            return self._last_border_frame_size[1]
        return self._estimate_border_frame_height()

    def _estimate_border_frame_height(self) -> Optional[int]:
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return None
        frame_height = height
        frame_width = int(frame_height * 9 / 16)
        if frame_width > width:
            frame_width = width
            frame_height = int(frame_width * 16 / 9)
        return frame_height
    
    def _position_border_frame(self) -> None:
        """Позиционирует рамку поверх видеопотока, по центру, с теми же пропорциями, что и welcome/death экраны."""
        if not hasattr(self, "borderFrameLabel") or self.borderFrameLabel is None:
            return
        
        # Используем ту же логику позиционирования, что и для welcome/death экранов
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return
        
        # Пропорции 9:16 (как у welcome/death экранов)
        frame_height = height
        frame_width = int(frame_height * 9 / 16)
        if frame_width > width:
            frame_width = width
            frame_height = int(frame_width * 16 / 9)
        
        # Центрируем
        offset_x = (width - frame_width) // 2
        offset_y = (height - frame_height) // 2
        
        # Масштабируем изображение рамки
        rendered_pixmap = None
        if self._border_frame_renderer and self._border_frame_renderer.isValid():
            rendered_pixmap = QtGui.QPixmap(frame_width, frame_height)
            rendered_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(rendered_pixmap)
            self._border_frame_renderer.render(painter)
            painter.end()
        elif self._border_frame_pixmap and not self._border_frame_pixmap.isNull():
            rendered_pixmap = self._border_frame_pixmap.scaled(
                frame_width,
                frame_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

        if rendered_pixmap and not rendered_pixmap.isNull():
            self.borderFrameLabel.setPixmap(rendered_pixmap)
            self.borderFrameLabel.setFixedSize(rendered_pixmap.size())
            self.borderFrameLabel.move(
                offset_x + (frame_width - rendered_pixmap.width()) // 2,
                offset_y + (frame_height - rendered_pixmap.height()) // 2,
            )
        else:
            self.borderFrameLabel.setFixedSize(frame_width, frame_height)
            self.borderFrameLabel.move(offset_x, offset_y)
        
        # Держим рамку ниже остальных элементов
        self.borderFrameLabel.lower()
        self._update_viewport_mask()
        new_border_size = (self.borderFrameLabel.width(), self.borderFrameLabel.height())
        if self._last_border_frame_size != new_border_size:
            self._last_border_frame_size = new_border_size
            self._invalidate_overlay_sizes()
            QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
            QtCore.QTimer.singleShot(0, self._position_impossible_label)

    def _update_media_toggle_button(self) -> None:
        if not self.mediaToggleButton:
            return
        is_webcam = self._media_mode == "webcam"
        self.mediaToggleButton.blockSignals(True)
        self.mediaToggleButton.setChecked(is_webcam)
        self.mediaToggleButton.blockSignals(False)
        if is_webcam:
            self.mediaToggleButton.setText("Камера")
            self.mediaToggleButton.setToolTip("Переключить на видео")
        else:
            self.mediaToggleButton.setText("Видео")
            self.mediaToggleButton.setToolTip("Переключить на веб-камеру")
        self.mediaToggleButton.adjustSize()
        QtCore.QTimer.singleShot(0, self._position_config_button)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj == self.graphicsViewFrame.viewport() and event.type() == QtCore.QEvent.Resize:
            self._invalidate_overlay_sizes()
            QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
            QtCore.QTimer.singleShot(0, self._position_impossible_label)
            QtCore.QTimer.singleShot(0, self._position_border_frame)
            QtCore.QTimer.singleShot(0, self._position_config_button)
            QtCore.QTimer.singleShot(0, self._update_video_fade_overlay_geometry)
            QtCore.QTimer.singleShot(0, self._refresh_welcome_overlay_graphics)
        if obj == getattr(self, "welcomeOverlay", None) and event.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonDblClick):
            self._dismiss_welcome_overlay()
            return True
        if obj == getattr(self, "deathOverlay", None) and event.type() == QtCore.QEvent.MouseButtonRelease:
            if self.deathOverlay.isVisible():
                # При клике на экран "Смерть неизбежна" возвращаемся на стартовый экран "НЕМУЗЕЙ"
                self._restart_scenario(force_welcome=True)
            return True
        return super().eventFilter(obj, event)

    def _start_stage1_animation(self) -> None:
        """Запускает первый этап анимации: от "УПОРОТЬСЯ" до появления кнопки "СЛЕЗТЬ" """
        # Загружаем конфигурацию анимации
        timings = self._animation_config.get("timings", {})
        stage1_timings = timings.get("stage1", {})
        duration_ms = stage1_timings.get("duration_ms", 4000)
        
        # Загружаем параметры интенсивности
        animation_stages = self._animation_config.get("animation_stages", {})
        stage1_cfg = animation_stages.get("stage1", {})
        start_intensity = float(stage1_cfg.get("start_intensity", 0.0))
        end_intensity = float(stage1_cfg.get("end_intensity", 1.0))
        
        # Убеждаемся, что значения в диапазоне 0.0-1.0
        if start_intensity > 1.0:
            start_intensity = start_intensity / 100.0
        if end_intensity > 1.0:
            end_intensity = end_intensity / 100.0
        
        # Устанавливаем текущий этап
        self._current_stage = 1
        
        # Инициализируем fade-прогресс с начальной интенсивностью
        self._fade_progress = start_intensity
        
        # Запускаем fade-анимацию для плавного проявления эффекта старения
        self._start_fade_timer(end_intensity, duration_ms)
        
        # Запускаем таймер для показа кнопки "слезть"
        self._stage_start_time = time.monotonic()
        self._stage_duration_ms = duration_ms
        
        # Останавливаем предыдущий таймер, если он был запущен
        if self._stage1_timer:
            self._stage1_timer.stop()
            self._stage1_timer.deleteLater()
        
        # Создаем и запускаем новый таймер
        self._stage1_timer = QtCore.QTimer(self)
        self._stage1_timer.setSingleShot(True)
        self._stage1_timer.timeout.connect(self._show_finish_button)
        self._stage1_timer.start(duration_ms)
        
    
    def _start_stage2_animation(self) -> None:
        """Запускает второй этап анимации: от "СЛЕЗТЬ" до "НЕВОЗМОЖНО" """
        timings = self._animation_config.get("timings", {})
        stage2_timings = timings.get("stage2", {})
        duration_ms = stage2_timings.get("duration_ms", 2000)
        
        # Загружаем параметры интенсивности
        animation_stages = self._animation_config.get("animation_stages", {})
        stage2_cfg = animation_stages.get("stage2", {})
        start_intensity = float(stage2_cfg.get("start_intensity", 1.0))
        end_intensity = float(stage2_cfg.get("end_intensity", 1.0))
        
        # Убеждаемся, что значения в диапазоне 0.0-1.0
        if start_intensity > 1.0:
            start_intensity = start_intensity / 100.0
        if end_intensity > 1.0:
            end_intensity = end_intensity / 100.0
        
        # Устанавливаем текущий этап
        self._current_stage = 2
        
        # Инициализируем fade-прогресс с начальной интенсивностью этапа 2
        self._fade_progress = start_intensity
        
        # Запускаем fade-анимацию для плавного проявления эффекта старения
        self._start_fade_timer(end_intensity, duration_ms)
        self._start_video_darkening(duration_ms)
        
        # Запускаем таймер для показа надписи "Невозможно" и перехода к stage3
        self._stage_start_time = time.monotonic()
        self._stage_duration_ms = duration_ms
        if self._stage2_timer:
            self._stage2_timer.stop()
            self._stage2_timer.deleteLater()
        self._stage2_timer = QtCore.QTimer(self)
        self._stage2_timer.setSingleShot(True)
        self._stage2_timer.timeout.connect(self._on_stage2_complete)
        self._stage2_timer.start(duration_ms)
    
    def _start_stage3_animation(self) -> None:
        """Запускает третий этап анимации: от "НЕВОЗМОЖНО" до "СМЕРТЬ НЕИЗБЕЖНА" """
        timings = self._animation_config.get("timings", {})
        stage3_timings = timings.get("stage3", {})
        duration_ms = stage3_timings.get("duration_ms", 5000)
        
        # Загружаем параметры интенсивности
        animation_stages = self._animation_config.get("animation_stages", {})
        stage3_cfg = animation_stages.get("stage3", {})
        start_intensity = float(stage3_cfg.get("start_intensity", 1.0))
        end_intensity = float(stage3_cfg.get("end_intensity", 1.0))
        
        # Убеждаемся, что значения в диапазоне 0.0-1.0
        if start_intensity > 1.0:
            start_intensity = start_intensity / 100.0
        if end_intensity > 1.0:
            end_intensity = end_intensity / 100.0
        
        # Устанавливаем текущий этап
        self._current_stage = 3
        
        # Инициализируем fade-прогресс с начальной интенсивностью этапа 3
        self._fade_progress = start_intensity
        
        # Запускаем fade-анимацию для плавного проявления эффекта старения
        self._start_fade_timer(end_intensity, duration_ms)
        # Запускаем затемнение видео
        self._start_video_darkening(duration_ms)
        
        # Запускаем таймер для показа экрана смерти
        self._stage_start_time = time.monotonic()
        self._stage_duration_ms = duration_ms
        if self._stage3_timer:
            self._stage3_timer.stop()
            self._stage3_timer.deleteLater()
        self._stage3_timer = QtCore.QTimer(self)
        self._stage3_timer.setSingleShot(True)
        self._stage3_timer.timeout.connect(self._on_stage3_complete)
        self._stage3_timer.start(duration_ms)

    def _cancel_death_delay_timer(self) -> None:
        if self._death_delay_timer:
            self._death_delay_timer.stop()
            self._death_delay_timer.deleteLater()
            self._death_delay_timer = None
    
    def _cancel_death_auto_return_timer(self) -> None:
        """Отменяет таймер автоматического возврата на welcome экран."""
        if self._death_auto_return_timer:
            self._death_auto_return_timer.stop()
            self._death_auto_return_timer.deleteLater()
            self._death_auto_return_timer = None
    
    def _start_death_auto_return_timer(self) -> None:
        """Запускает таймер для автоматического возврата на welcome экран,
        если пользователь не кликнул на экран смерти.
        Длительность таймера читается из animation_config.json (death_auto_return_ms)."""
        # Отменяем предыдущий таймер, если он был запущен
        self._cancel_death_auto_return_timer()
        
        # Читаем длительность из конфига (по умолчанию 30 секунд = 30000 мс)
        timings = self._animation_config.get("timings", {})
        auto_return_duration_ms = timings.get("death_auto_return_ms", 30000)
        
        # Создаем новый таймер
        self._death_auto_return_timer = QtCore.QTimer(self)
        self._death_auto_return_timer.setSingleShot(True)
        self._death_auto_return_timer.timeout.connect(self._on_death_auto_return_timeout)
        self._death_auto_return_timer.start(auto_return_duration_ms)
    
    def _on_death_auto_return_timeout(self) -> None:
        """Вызывается по истечении времени, заданного в конфиге (death_auto_return_ms).
        Автоматически возвращает на welcome экран (НЕМУЗЕЙ)."""
        if self.deathOverlay.isVisible():
            # Автоматически возвращаемся на welcome экран (НЕМУЗЕЙ), всегда показывая welcome overlay
            self._restart_scenario(force_welcome=True)
    
    def _cancel_sles_auto_return_timer(self) -> None:
        """Отменяет таймер автоматического возврата на welcome экран для кнопки 'Слезть'."""
        if self._sles_auto_return_timer:
            self._sles_auto_return_timer.stop()
            self._sles_auto_return_timer.deleteLater()
            self._sles_auto_return_timer = None
    
    def _start_sles_auto_return_timer(self) -> None:
        """Запускает таймер для автоматического возврата на welcome экран,
        если пользователь не нажал на кнопку 'Слезть'.
        Длительность таймера читается из animation_config.json (sles_auto_return_ms)."""
        # Отменяем предыдущий таймер, если он был запущен
        self._cancel_sles_auto_return_timer()
        
        # Читаем длительность из конфига (по умолчанию 2 минуты = 120000 мс)
        timings = self._animation_config.get("timings", {})
        auto_return_duration_ms = timings.get("sles_auto_return_ms", 120000)
        
        # Создаем новый таймер
        self._sles_auto_return_timer = QtCore.QTimer(self)
        self._sles_auto_return_timer.setSingleShot(True)
        self._sles_auto_return_timer.timeout.connect(self._on_sles_auto_return_timeout)
        self._sles_auto_return_timer.start(auto_return_duration_ms)
    
    def _on_sles_auto_return_timeout(self) -> None:
        """Вызывается по истечении времени, заданного в конфиге (sles_auto_return_ms).
        Автоматически возвращает на welcome экран (НЕМУЗЕЙ), если пользователь не нажал 'Слезть'."""
        # Проверяем, что мы все еще на этапе 1 и кнопка "Слезть" видна
        if self._current_stage == 1 and self._button_icon_state == "finish":
            # Останавливаем анимацию и возвращаемся на welcome экран
            self._stop_face_fade(reset_progress=True)
            self.swapfacesButton.setChecked(False)
            self._swap_active = False
            self._second_press_triggered = False
            self._current_stage = 0  # Сбрасываем этап
            if hasattr(self, "buttonUport") and self.buttonUport:
                self.buttonUport.hide()
                # Сбрасываем состояние кнопки на "Упороться"
                self._update_uporotsya_button_icon(start=True)
                self.buttonUport.setCheckable(True)
                self.buttonUport.setChecked(False)
            try:
                self.video_processor.stop_processing()
            except Exception:
                pass
            # Возвращаемся на welcome экран
            self._restart_scenario(force_welcome=True)
    
    def _cancel_welcome_uporotsya_timer(self) -> None:
        """Отменяет таймер автоматического возврата на welcome экран для кнопки 'УПОРОТЬСЯ'."""
        if self._welcome_uporotsya_timer:
            self._welcome_uporotsya_timer.stop()
            self._welcome_uporotsya_timer.deleteLater()
            self._welcome_uporotsya_timer = None
    
    def _start_welcome_uporotsya_timer(self) -> None:
        """Запускает таймер для автоматического возврата на welcome экран,
        если пользователь не нажал на кнопку 'УПОРОТЬСЯ'.
        Длительность таймера читается из animation_config.json (welcome_uporotsya_timeout_ms)."""
        # Отменяем предыдущий таймер, если он был запущен
        self._cancel_welcome_uporotsya_timer()
        
        # Читаем длительность из конфига (по умолчанию 45 секунд = 45000 мс)
        timings = self._animation_config.get("timings", {})
        timeout_duration_ms = timings.get("welcome_uporotsya_timeout_ms", 45000)
        
        print(f"[Timer] Запуск таймера welcome_uporotsya: {timeout_duration_ms} мс")
        
        # Создаем новый таймер
        self._welcome_uporotsya_timer = QtCore.QTimer(self)
        self._welcome_uporotsya_timer.setSingleShot(True)
        self._welcome_uporotsya_timer.timeout.connect(self._on_welcome_uporotsya_timeout)
        self._welcome_uporotsya_timer.start(timeout_duration_ms)
    
    def _on_welcome_uporotsya_timeout(self) -> None:
        """Вызывается по истечении времени, заданного в конфиге (welcome_uporotsya_timeout_ms).
        Автоматически возвращает на welcome экран (НЕМУЗЕЙ), если пользователь не нажал 'УПОРОТЬСЯ'."""
        print(f"[Timer] Таймер welcome_uporotsya истек! _welcome_active={self._welcome_active}")
        # Проверяем, что welcome экран НЕ активен (т.е. мы на экране с видеопотоком)
        # и кнопка "УПОРОТЬСЯ" видна (т.е. пользователь еще не нажал на нее)
        button_visible = hasattr(self, "buttonUport") and self.buttonUport and self.buttonUport.isVisible()
        if not self._welcome_active and button_visible:
            print("[Timer] Выполняем возврат на welcome экран (пользователь не нажал УПОРОТЬСЯ)")
            # Останавливаем обработку видео, если она запущена
            try:
                self.video_processor.stop_processing()
            except Exception:
                pass
            
            # Сбрасываем состояние кнопки
            if hasattr(self, "buttonUport") and self.buttonUport:
                self.buttonUport.hide()
                self._update_uporotsya_button_icon(start=True)
                self.buttonUport.setCheckable(True)
                self.buttonUport.setChecked(False)
            
            # Сбрасываем состояние
            self.swapfacesButton.setChecked(False)
            self._swap_active = False
            self._second_press_triggered = False
            self._current_stage = 0
            
            # Перезапускаем welcome экран (просто скрываем и показываем заново)
            self._welcome_active = False
            if self.welcomeOverlay:
                self.welcomeOverlay.hide()
            self._stop_overlay_animation()
            # Показываем welcome экран заново
            self._show_welcome_overlay()
    
    def _start_fade_timer(self, target_intensity: float, duration_ms: int) -> None:
        """Запускает fade-анимацию для плавного проявления эффекта старения."""
        if self._fade_timer:
            self._fade_timer.stop()
            self._fade_timer.deleteLater()
        
        self._fade_timer = QtCore.QTimer(self)
        interval_ms = 16  # Обновляем каждые 16ms (~60 FPS) для более быстрой и плавной анимации
        total = float(max(1, duration_ms))
        
        # Вычисляем шаг для fade-анимации
        intensity_range = target_intensity - self._fade_progress
        self._fade_step = (interval_ms / total) * intensity_range

        def update_progress():
            self._update_fade_progress(target_intensity)

        self._fade_timer.timeout.connect(update_progress)
        self._fade_timer.start(interval_ms)
    
    def _update_fade_progress(self, target_intensity: float) -> None:
        """Обновляет прогресс fade-анимации до целевой интенсивности."""
        if self._fade_step > 0:
            self._fade_progress = min(target_intensity, self._fade_progress + self._fade_step)
        elif self._fade_step < 0:
            self._fade_progress = max(target_intensity, self._fade_progress + self._fade_step)
        
        # Ограничиваем значение в допустимом диапазоне
        self._fade_progress = max(0.0, min(1.0, self._fade_progress))
        
        # Проверяем, достигли ли мы целевой интенсивности
        if abs(self._fade_progress - target_intensity) < abs(self._fade_step) or \
           (self._fade_step > 0 and self._fade_progress >= target_intensity) or \
           (self._fade_step < 0 and self._fade_progress <= target_intensity):
            self._fade_progress = target_intensity
            if self._fade_timer:
                self._fade_timer.stop()
                self._fade_timer.deleteLater()
                self._fade_timer = None
    
    def _show_finish_button(self) -> None:
        """Показывает кнопку 'Слезть' после задержки в первом этапе"""
        if self._current_stage == 1:
            # Обновляем иконку кнопки на "СЛЕЗТЬ"
            self._update_uporotsya_button_icon(start=False)
            if hasattr(self, "buttonUport") and self.buttonUport:
                # Позиционируем кнопку перед показом
                QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
                # Показываем кнопку
                self.buttonUport.show()
                # Кнопка всегда должна быть поверх рамки
                self.buttonUport.raise_()
            
            # Запускаем таймер для автоматического возврата на welcome экран, если не нажали "Слезть"
            self._start_sles_auto_return_timer()
    
    def _on_stage2_complete(self) -> None:
        """Вызывается после завершения второго этапа - показываем "НЕВОЗМОЖНО" и запускаем stage3"""
        if self._current_stage == 2:
            self._show_impossible_message()
            # НЕ меняем _current_stage здесь - он уже установлен в _start_stage3_animation
            # self._current_stage = 3
            self._start_stage3_animation()
    
    def _on_stage3_complete(self) -> None:
        """Вызывается после завершения третьего этапа - показываем экран смерти"""
        if self._current_stage == 3:
            self._show_death_screen()
    
    def _stop_face_fade(self, reset_progress: bool = False) -> None:
        if self._fade_timer:
            self._fade_timer.stop()
            self._fade_timer.deleteLater()
            self._fade_timer = None
        # НЕ останавливаем таймеры этапов, если этап активен (они нужны для показа кнопок и сообщений)
        # Останавливаем только если reset_progress=True (полный сброс)
        if reset_progress:
            if self._stage1_timer:
                self._stage1_timer.stop()
                self._stage1_timer.deleteLater()
                self._stage1_timer = None
            if self._stage2_timer:
                self._stage2_timer.stop()
                self._stage2_timer.deleteLater()
                self._stage2_timer = None
            if self._stage3_timer:
                self._stage3_timer.stop()
                self._stage3_timer.deleteLater()
                self._stage3_timer = None
        self._cancel_death_delay_timer()
        self._cancel_sles_auto_return_timer()  # Отменяем таймер "Слезть" при остановке
        if reset_progress:
            # Сбрасываем на стартовую интенсивность первого этапа
            stage1_config = self._animation_stages.get("stage1", {})
            start_intensity = max(0.0, min(1.0, stage1_config.get("start_intensity", 0.0)))
            self._fade_progress = start_intensity
            self._current_stage = 0
            # Сбрасываем таймеры этапов
            self._stage_start_time = 0.0
            self._stage_duration_ms = 0
            self._reset_video_darkening()

    def _show_impossible_message(self) -> None:
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.hide()
        scaled_pixmap = None
        if self._impossible_pixmap and not self._impossible_pixmap.isNull():
            scaled_pixmap = self._get_impossible_label_pixmap()
        if scaled_pixmap and not scaled_pixmap.isNull():
            self.messageLabel.setPixmap(scaled_pixmap)
            self.messageLabel.setFixedSize(scaled_pixmap.size())
            self.messageLabel.setText("")
        else:
            self._apply_impossible_text_size()
        self.messageLabel.show()
        self._position_impossible_label()
        # Надпись всегда должна быть поверх рамки
        self.messageLabel.raise_()
        # Экран смерти появится после завершения анимации stage2
        if self.mediaToggleButton:
            self.mediaToggleButton.hide()

    def _show_death_screen(self) -> None:
        self._cancel_death_delay_timer()
        self._reset_video_darkening()
        self.messageLabel.hide()
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.hide()
        self._stop_face_fade(reset_progress=True)
        try:
            self.video_processor.stop_processing()
        except Exception:
            pass
        self.swapfacesButton.setChecked(False)
        self._swap_active = False
        if self.control_window and self.control_window.isVisible():
            self.control_window.hide()
        self.deathOverlay.setGeometry(self.rect())
        self._refresh_death_overlay_graphics()
        self.deathOverlay.show()
        self.deathOverlay.raise_()
        self._start_death_animation()
        self.configButton.hide()
        if self.mediaToggleButton:
            self.mediaToggleButton.hide()
        # Скрываем рамку, когда показывается death overlay
        if hasattr(self, "borderFrameLabel") and self.borderFrameLabel:
            self.borderFrameLabel.hide()
            self._update_viewport_mask(clear=True)
        
        # Запускаем таймер для автоматического возврата на welcome экран через 30 секунд
        self._start_death_auto_return_timer()

    def _restart_scenario(self, force_welcome: bool = False) -> None:
        """Перезапускает сценарий после экрана смерти.
        Видео/камера запускается заново, faceswap не выгружается, но не применяется до нажатия на 'упороться'.
        
        Args:
            force_welcome: Если True, всегда показывает welcome экран, даже если есть выбранное видео/камера.
        """
        # Отменяем таймер автоматического возврата (пользователь кликнул вручную или таймер истек)
        self._cancel_death_auto_return_timer()
        
        # Останавливаем death анимацию
        self._stop_death_animation()
        
        # Скрываем death overlay
        self.deathOverlay.hide()
        
        # Сбрасываем состояние анимации
        self._stop_face_fade(reset_progress=True)
        
        # Выключаем swapfacesButton (faceswap не применяется до нажатия на "упороться")
        self.swapfacesButton.setChecked(False)
        self._swap_active = False
        self._second_press_triggered = False
        
        # Скрываем сообщения
        self.messageLabel.hide()
        
        # Скрываем кнопку "упороться" (она появится после welcome overlay)
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.hide()
        
        # Скрываем control window если открыт
        if self.control_window and self.control_window.isVisible():
            self.control_window.hide()
        
        # ВАЖНО: НЕ выгружаем selected_video_button и target_faces - они остаются для повторного использования
        # НЕ очищаем target_videos, input_faces и т.д.
        
        # Если force_welcome=True (автоматический возврат), всегда показываем welcome экран
        if force_welcome:
            # Сбрасываем кнопку "упороться" в начальное состояние (иконка "start")
            self._update_uporotsya_button_icon(start=True)
            if hasattr(self, "buttonUport") and self.buttonUport:
                self.buttonUport.setCheckable(True)
                self.buttonUport.setChecked(False)
            # Сбрасываем этап
            self._current_stage = 0
            self._show_welcome_overlay()
            return
        
        # Проверяем, есть ли выбранное видео/камера
        if not self.selected_video_button:
            # Если нет выбранного видео, показываем welcome overlay
            self._show_welcome_overlay()
            return
        
        # Скрываем welcome overlay если он активен
        if self._welcome_active:
            if self.welcomeOverlay:
                self.welcomeOverlay.hide()
            if self._welcome_timer:
                self._welcome_timer.stop()
                self._welcome_timer.deleteLater()
                self._welcome_timer = None
            self._stop_overlay_animation()
            self._welcome_active = False
        
        # Если есть выбранное видео/камера, проверяем состояние
        capture = self.video_processor.media_capture
        
        # Если камера/видео закрыта, перезагружаем
        if not capture or not capture.isOpened():
            if hasattr(self.selected_video_button, 'load_media'):
                self.selected_video_button.load_media()
                # Ждем немного перед запуском обработки
                QtCore.QTimer.singleShot(300, self._ensure_playing)
        else:
            # Камера/видео уже открыта - просто запускаем обработку
            QtCore.QTimer.singleShot(100, self._ensure_playing)
        
        # Сбрасываем кнопку "упороться" в начальное состояние (иконка "start")
        self._update_uporotsya_button_icon(start=True)
        self.buttonUport.setCheckable(True)
        self.buttonUport.setChecked(False)
        
        # Показываем кнопку "упороться" после небольшой задержки (чтобы видео успело запуститься)
        def show_uporotsya_button():
            self.buttonUport.show()
            self._position_uporotsya_button()
            self._try_enable_uporotsya()
            # Показываем рамку после скрытия death overlay
            if hasattr(self, "borderFrameLabel") and self.borderFrameLabel:
                self._position_border_frame()
                self.borderFrameLabel.show()
                self._update_viewport_mask()
        
        QtCore.QTimer.singleShot(200, show_uporotsya_button)
        
        # Кнопка настроек скрыта - используем горячую клавишу Ctrl+Alt+Shift+D
        # self.configButton.show()

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

    def _refresh_button_size(self, force: bool = False) -> None:
        if force:
            self._button_size_dirty = True
        if self._button_width_cache is None or self._button_size_dirty:
            available_width = self._compute_available_button_width()
            if available_width <= 0:
                available_width = self._BUTTON_MIN_WIDTH
            self._button_width_cache = available_width
            self._button_text_min_width = (
                max(self._BUTTON_MIN_WIDTH, min(available_width, self.width()))
                if available_width > 0
                else self._BUTTON_MIN_WIDTH
            )
            self._button_size_dirty = False
        available_width = self._button_width_cache or self._BUTTON_MIN_WIDTH
        min_width = self._button_text_min_width or self._BUTTON_MIN_WIDTH
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
            self.buttonUport.setMinimumSize(min_width, 48)

    def _refresh_death_overlay_graphics(self) -> None:
        if not self.deathOverlay or not self.deathLabel:
            return
        
        self.deathOverlay.setGeometry(self.rect())
        overlay_width = max(1, self.deathOverlay.width())
        overlay_height = max(1, self.deathOverlay.height())
        
        # Используем ту же логику пропорций, что и для welcome экрана (9:16)
        frame_height = overlay_height
        frame_width = int(frame_height * 9 / 16)
        if frame_width > overlay_width:
            frame_width = overlay_width
            frame_height = int(frame_width * 16 / 9)
        
        offset_x = (self.deathOverlay.width() - frame_width) // 2
        offset_y = (self.deathOverlay.height() - frame_height) // 2
        
        frame = self._current_death_frame or self.death_pixmap
        if frame and not frame.isNull():
            pixmap = frame
            scaled = pixmap.scaled(
                frame_width,
                frame_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.deathLabel.setPixmap(scaled)
            self.deathLabel.setFixedSize(scaled.size())
            self.deathLabel.move(
                offset_x + (frame_width - scaled.width()) // 2,
                offset_y + (frame_height - scaled.height()) // 2,
            )
            self.deathLabel.setText("")
        else:
            self.deathLabel.setPixmap(QtGui.QPixmap())
            self.deathLabel.setFixedSize(frame_width, frame_height)
            self.deathLabel.move(offset_x, offset_y)
            self.deathLabel.setText("СМЕРТЬ\nНЕИЗБЕЖНА")

    def _load_deferred_images(self) -> None:
        """Загружает изображения, которые не нужны сразу при старте."""
        import time
        t0 = time.time()
        
        # Загружаем welcome изображения
        if self._welcome_pixmap is None:
            self._welcome_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_WELCOME))
        if not self._overlay_frames:
            self._overlay_frames = [
                QtGui.QPixmap(self._resource_path(PATH_UI_OVERLAY_1)),
            ]
        if self._overlay_static is None:
            self._overlay_static = QtGui.QPixmap(self._resource_path(PATH_UI_OVERLAY_2))
        if self._overlay_mask is None:
            self._overlay_mask = QtGui.QPixmap(self._resource_path(PATH_UI_OVERLAY_3))
        
        # Загружаем death frames
        if not self._death_frames:
            death_frame_paths = DEATH_FRAME_PATHS
            self._death_frames = [
                QtGui.QPixmap(self._resource_path(path)) for path in death_frame_paths
            ]
            self._death_frames = [frame for frame in self._death_frames if not frame.isNull()]
            self._death_intro_frame = self._death_frames[0] if self._death_frames else None
            self._current_death_frame = (
                self._death_intro_frame if self._death_intro_frame and not self._death_intro_frame.isNull() else None
            )
        
        print(f"[Startup] _load_deferred_images() completed in {time.time() - t0:.2f}s")
        
        # Обновляем welcome overlay если он уже показан
        if self._welcome_active:
            self._refresh_welcome_overlay_graphics()

    def _show_welcome_overlay(self) -> None:
        # Если welcome уже активен, просто обновляем графику
        if self._welcome_active:
            if self.welcomeOverlay and self.welcomeLabel:
                self.welcomeOverlay.setGeometry(self.rect())
                self._refresh_welcome_overlay_graphics()
            return
        
        if not self.welcomeOverlay or not self.welcomeLabel:
            return

        print("[Welcome] Показываем welcome overlay")
        self._welcome_active = True
        self.welcomeOverlay.setGeometry(self.rect())
        # Если изображения еще не загружены, загружаем их синхронно
        if self._welcome_pixmap is None:
            self._load_deferred_images()
        self._refresh_welcome_overlay_graphics()
        self.welcomeOverlay.show()
        self.welcomeOverlay.raise_()
        self._start_overlay_animation()
        # Скрываем рамку, когда показывается welcome overlay
        if hasattr(self, "borderFrameLabel") and self.borderFrameLabel:
            self.borderFrameLabel.hide()
            self._update_viewport_mask(clear=True)

    def _refresh_welcome_overlay_graphics(self) -> None:
        if not self._welcome_active or not self.welcomeOverlay or not self.welcomeLabel:
            return
        self.welcomeOverlay.setGeometry(self.rect())
        overlay_width = max(1, self.welcomeOverlay.width())
        overlay_height = max(1, self.welcomeOverlay.height())
        frame_height = overlay_height
        frame_width = int(frame_height * 9 / 16)
        if frame_width > overlay_width:
            frame_width = overlay_width
            frame_height = int(frame_width * 16 / 9)
        offset_x = (self.welcomeOverlay.width() - frame_width) // 2
        offset_y = (self.welcomeOverlay.height() - frame_height) // 2
        if self._welcome_pixmap and not self._welcome_pixmap.isNull():
            pixmap = self._welcome_pixmap
            scaled = pixmap.scaled(
                frame_width,
                frame_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.welcomeLabel.setPixmap(scaled)
            self.welcomeLabel.setFixedSize(scaled.size())
            self.welcomeLabel.move(
                offset_x + (frame_width - scaled.width()) // 2,
                offset_y + (frame_height - scaled.height()) // 2,
            )
        else:
            self.welcomeLabel.setPixmap(QtGui.QPixmap())
            self.welcomeLabel.setFixedSize(frame_width, frame_height)
            self.welcomeLabel.move(offset_x, offset_y)

        if self.welcomeAnimationLabel and self.welcomeAnimationLabel.isVisible():
            self.welcomeAnimationLabel.setFixedSize(self.welcomeLabel.size())
            self.welcomeAnimationLabel.move(
                offset_x + (frame_width - self.welcomeAnimationLabel.width()) // 2,
                offset_y + (frame_height - self.welcomeAnimationLabel.height()) // 2,
            )
            self._apply_overlay_animation_frame()
        if self.welcomeAnimationStaticLabel and self.welcomeAnimationStaticLabel.isVisible():
            self.welcomeAnimationStaticLabel.setFixedSize(self.welcomeLabel.size())
            self.welcomeAnimationStaticLabel.move(
                offset_x + (frame_width - self.welcomeAnimationStaticLabel.width()) // 2,
                offset_y + (frame_height - self.welcomeAnimationStaticLabel.height()) // 2,
            )
        if self.welcomeAnimationMaskLabel and self.welcomeAnimationMaskLabel.isVisible():
            self._update_overlay_mask_pixmap(self.welcomeLabel.size(), offset_x, offset_y)

    def _set_death_label_frame(self, frame: Optional[QtGui.QPixmap], available_width: Optional[int] = None) -> None:
        if not hasattr(self, "deathLabel") or self.deathLabel is None:
            return
        if not frame or frame.isNull():
            self.deathLabel.clear()
            self._current_death_frame = None
            return
        # Сохраняем кадр и обновляем графику через _refresh_death_overlay_graphics
        # чтобы использовать те же пропорции, что и welcome экран
        self._current_death_frame = frame
        self._refresh_death_overlay_graphics()

    def _start_death_animation(self) -> None:
        if hasattr(self, "deathLabel") and self.deathLabel:
            self.deathLabel.show()
        if self._death_intro_frame and not self._death_intro_frame.isNull():
            self._set_death_label_frame(self._death_intro_frame)
        elif self.death_pixmap and not self.death_pixmap.isNull():
            self._set_death_label_frame(self.death_pixmap)

        self._death_frame_index = 0
        if not self._death_frames:
            return
        if self._death_anim_timer is None:
            self._death_anim_timer = QtCore.QTimer(self)
            self._death_anim_timer.timeout.connect(self._advance_death_animation)
        self._death_anim_timer.start(250)

    def _stop_death_animation(self) -> None:
        if self._death_anim_timer and self._death_anim_timer.isActive():
            self._death_anim_timer.stop()
        if self._death_intro_frame and not self._death_intro_frame.isNull():
            self._set_death_label_frame(self._death_intro_frame)
        elif self.death_pixmap and not self.death_pixmap.isNull():
            self._set_death_label_frame(self.death_pixmap)

    def _advance_death_animation(self) -> None:
        if not self._death_frames:
            return
        self._death_frame_index = (self._death_frame_index + 1) % len(self._death_frames)
        self._set_death_label_frame(self._death_frames[self._death_frame_index])

    def _start_overlay_animation(self) -> None:
        valid_frames = [frame for frame in self._overlay_frames if not frame.isNull()]
        if not valid_frames or not self.welcomeAnimationLabel:
            return
        self._overlay_frames = valid_frames
        if self._overlay_timer is None:
            self._overlay_timer = QtCore.QTimer(self)
            self._overlay_timer.timeout.connect(self._advance_overlay_animation)

        overlay_width = max(1, self.welcomeOverlay.width())
        overlay_height = max(1, self.welcomeOverlay.height())
        frame_height = overlay_height
        frame_width = int(frame_height * 9 / 16)
        if frame_width > overlay_width:
            frame_width = overlay_width
            frame_height = int(frame_width * 16 / 9)
        offset_x = (self.welcomeOverlay.width() - frame_width) // 2
        offset_y = (self.welcomeOverlay.height() - frame_height) // 2

        animation_size = QtCore.QSize(frame_width, frame_height)
        self.welcomeAnimationLabel.setFixedSize(animation_size)
        self.welcomeAnimationLabel.move(offset_x, offset_y)

        if self._overlay_static and not self._overlay_static.isNull():
            self.welcomeAnimationStaticLabel.setFixedSize(animation_size)
            self.welcomeAnimationStaticLabel.move(offset_x, offset_y)
            static_scaled = self._overlay_static.scaled(
                animation_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.welcomeAnimationStaticLabel.setPixmap(static_scaled)
            self.welcomeAnimationStaticLabel.show()
            self.welcomeAnimationStaticLabel.raise_()
        else:
            self.welcomeAnimationStaticLabel.hide()

        if self._overlay_mask and not self._overlay_mask.isNull():
            self._update_overlay_mask_pixmap(animation_size, offset_x, offset_y)
        else:
            self.welcomeAnimationMaskLabel.hide()

        self.welcomeAnimationLabel.setGraphicsEffect(self._overlay_opacity_effect)
        self.welcomeAnimationLabel.show()
        self.welcomeAnimationLabel.raise_()
        if self.welcomeAnimationMaskLabel and not self.welcomeAnimationMaskLabel.isHidden():
            self.welcomeAnimationMaskLabel.raise_()

        self._overlay_phase = 0.0
        if not self._overlay_timer.isActive():
            self._overlay_timer.start(33)  # ~30 fps

        self._apply_overlay_animation_frame()

    def _stop_overlay_animation(self) -> None:
        if self._overlay_timer and self._overlay_timer.isActive():
            self._overlay_timer.stop()
        self._overlay_phase = 0.0
        if self.welcomeAnimationLabel:
            self.welcomeAnimationLabel.hide()
        if self.welcomeAnimationStaticLabel:
            self.welcomeAnimationStaticLabel.hide()
        if self.welcomeAnimationMaskLabel:
            self.welcomeAnimationMaskLabel.hide()

    def _advance_overlay_animation(self) -> None:
        if not self._overlay_frames or not self.welcomeAnimationLabel or not self.welcomeAnimationLabel.isVisible():
            return
        self._overlay_phase = (self._overlay_phase + 0.033) % 2.0
        if self._overlay_phase <= 1.0:
            opacity = self._overlay_phase
        else:
            opacity = 2.0 - self._overlay_phase
        self._overlay_opacity_effect.setOpacity(opacity)

    def _apply_overlay_animation_frame(self, animate: bool = False) -> None:
        if not self._overlay_frames or not self.welcomeAnimationLabel or not self.welcomeAnimationLabel.isVisible():
            return
        frame = self._overlay_frames[self._overlay_frame_index]
        if frame.isNull():
            return
        target_size = self.welcomeAnimationLabel.size()
        scaled = frame.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        image = QtGui.QImage(target_size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        x_offset = (scaled.width() - target_size.width()) // 2
        y_offset = (scaled.height() - target_size.height()) // 2
        painter.drawPixmap(-x_offset, -y_offset, scaled)
        painter.end()
        self.welcomeAnimationLabel.setPixmap(QtGui.QPixmap.fromImage(image))
        if not animate:
            self._overlay_opacity_effect.setOpacity(1.0)

    def _update_overlay_mask_pixmap(self, target_size: QtCore.QSize, offset_x: int, offset_y: int) -> None:
        if not self.welcomeAnimationMaskLabel:
            return
        if not self._overlay_mask or self._overlay_mask.isNull():
            self.welcomeAnimationMaskLabel.hide()
            return

        mask_original = self._overlay_mask
        mask_scaled = mask_original.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        crop_width = target_size.width()
        crop_height = target_size.height()
        crop_width = min(crop_width, mask_scaled.width())
        crop_height = min(crop_height, mask_scaled.height())
        start_x = max(0, (mask_scaled.width() - crop_width) // 2)
        start_y = max(0, (mask_scaled.height() - crop_height) // 2)
        mask_cropped = mask_scaled.copy(start_x, start_y, crop_width, crop_height)
        if mask_original.devicePixelRatioF() != mask_cropped.devicePixelRatioF():
            mask_cropped.setDevicePixelRatio(mask_original.devicePixelRatioF())

        self.welcomeAnimationMaskLabel.setGraphicsEffect(None)
        self.welcomeAnimationMaskLabel.setFixedSize(target_size)
        self.welcomeAnimationMaskLabel.move(offset_x, offset_y)
        self.welcomeAnimationMaskLabel.setPixmap(mask_cropped)
        self.welcomeAnimationMaskLabel.show()
        self.welcomeAnimationMaskLabel.raise_()

    def _dismiss_welcome_overlay(self) -> None:
        if not self._welcome_active:
            return
        self._welcome_active = False
        # Отменяем таймер автоматического возврата (пользователь кликнул на welcome экран)
        self._cancel_welcome_uporotsya_timer()
        if self._welcome_timer:
            self._welcome_timer.stop()
            self._welcome_timer.deleteLater()
            self._welcome_timer = None
        if self.welcomeOverlay:
            self.welcomeOverlay.hide()
        self._stop_overlay_animation()
        self._initialize_post_welcome_state()
        # Показываем рамку после скрытия welcome overlay
        if hasattr(self, "borderFrameLabel") and self.borderFrameLabel:
            self._position_border_frame()
            self.borderFrameLabel.show()
            self._update_viewport_mask()
        # Активируем кнопку "Упороться" с небольшой задержкой, чтобы дать время инициализироваться
        QtCore.QTimer.singleShot(100, self._try_enable_uporotsya)
        # Запускаем таймер для автоматического возврата на welcome экран, если пользователь не нажмет "УПОРОТЬСЯ"
        # Таймер запускается после того, как показан видеопоток и кнопка "УПОРОТЬСЯ"
        QtCore.QTimer.singleShot(200, self._start_welcome_uporotsya_timer)

    def _initialize_post_welcome_state(self) -> None:
        if self.welcomeOverlay:
            self.welcomeOverlay.hide()
        # Убеждаемся, что кнопка в правильном состоянии
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.setCheckable(True)
            self.buttonUport.setChecked(False)
            self._update_uporotsya_button_icon(start=True)
        self.buttonUport.show()
        # Кнопка настроек скрыта - используем горячую клавишу Ctrl+Alt+Shift+D
        # self.configButton.show()
        # Кнопка переключения камера/видео скрыта
        # if self.mediaToggleButton:
        #     self.mediaToggleButton.show()
        #     self._update_media_toggle_button()
        # Кнопка настроек скрыта, поэтому не позиционируем её
        # self._position_config_button()
        QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
        QtCore.QTimer.singleShot(0, self._ensure_playing)
        QtCore.QTimer.singleShot(0, lambda: layout_actions.fit_image_to_view_onchange(self))
        # Показываем рамку после скрытия welcome overlay
        if hasattr(self, "borderFrameLabel") and self.borderFrameLabel:
            QtCore.QTimer.singleShot(0, self._position_border_frame)
            self.borderFrameLabel.show()
            QtCore.QTimer.singleShot(0, self._update_viewport_mask)

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

    def keyPressEvent(self, event):
        """Обработка горячих клавиш. Переопределяет метод из MainWindow."""
        key = event.key()
        
        # F11 - переключение полноэкранного режима
        if key == QtCore.Qt.Key_F11:
            from app.ui.widgets.actions import video_control_actions
            video_control_actions.view_fullscreen(self)
            return
        
        # Проверяем комбинацию Ctrl+Alt+D для открытия ControlPanel
        # Работает независимо от раскладки клавиатуры
        modifiers = event.modifiers()
        has_ctrl = modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        has_alt = modifiers & QtCore.Qt.KeyboardModifier.AltModifier
        
        if has_ctrl and has_alt:
            # Получаем физический код клавиши (nativeVirtualKey для Windows)
            # Это позволяет работать независимо от раскладки
            native_key = event.nativeVirtualKey() if hasattr(event, 'nativeVirtualKey') else None
            
            # Проверяем как виртуальный код (Key_D), так и физический (68 для D на Windows)
            # Также проверяем Key_V, так как в русской раскладке D может быть представлена как V
            is_d_key = (key == QtCore.Qt.Key_D or 
                       key == QtCore.Qt.Key_V or  # В русской раскладке D -> В (Key_V)
                       (native_key and native_key == 68))  # Физический код D на Windows
            
            if is_d_key:
                self._open_control_options_window()
                return
        
        # Вызываем родительский метод для обработки остальных горячих клавиш
        super().keyPressEvent(event)

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

        # Add Effect Params tab if not exists
        if self._control_options_widget and hasattr(self, "tabWidget"):
            self._add_effect_params_tab()

        return bool(self._control_options_widget and shiboken6.isValid(self._control_options_widget))
    
    def _add_effect_params_tab(self) -> None:
        """Добавляет вкладку Effect Params в Control Panel."""
        if not hasattr(self, "tabWidget"):
            return
        
        # Check if tab already exists
        for i in range(self.tabWidget.count()):
            if self.tabWidget.tabText(i) == "Effect Params":
                return
        
        # Create new tab
        effect_params_tab = QtWidgets.QWidget()
        effect_params_tab.setObjectName("effect_params_tab")
        effect_params_layout = QtWidgets.QVBoxLayout(effect_params_tab)
        effect_params_layout.setObjectName("effect_params_layout")
        effect_params_widgets_layout = QtWidgets.QVBoxLayout()
        effect_params_widgets_layout.setObjectName("effectParamsWidgetsLayout")
        effect_params_layout.addLayout(effect_params_widgets_layout)
        
        # Add widgets to the tab
        layout_actions.add_widgets_to_tab_layout(
            self, 
            LAYOUT_DATA=EFFECT_PARAMS_LAYOUT_DATA, 
            layoutWidget=effect_params_widgets_layout, 
            data_type='control'
        )
        
        # Add tab to tabWidget
        self.tabWidget.addTab(effect_params_tab, "Effect Params")
        
        # Connect sliders to update config
        self._connect_effect_params_sliders()
        
        # Create and connect Save button manually
        save_group_box = None
        for i in range(effect_params_widgets_layout.count()):
            item = effect_params_widgets_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, widget_components.FormGroupBox) and widget.title() == "Save Config":
                    save_group_box = widget
                    break
        
        if save_group_box:
            save_layout = save_group_box.layout()
            if save_layout:
                # Remove existing widgets if any
                while save_layout.count():
                    item = save_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # Create save button
                save_button = QtWidgets.QPushButton("Save Animation Config")
                save_button.setToolTip("Save current animation parameters to animation_config.json file.")
                save_button.clicked.connect(self._save_animation_config)
                save_layout.addWidget(save_button)
        
        # Load current values from config
        self._load_effect_params_from_config()
    
    def _connect_effect_params_sliders(self) -> None:
        """Подключает слайдеры к обновлению конфига в реальном времени."""
        slider_mappings = {
            'AgingStage1StartSlider': ('animation_stages', 'stage1', 'start_intensity'),
            'AgingStage1EndSlider': ('animation_stages', 'stage1', 'end_intensity'),
            'AgingStage2StartSlider': ('animation_stages', 'stage2', 'start_intensity'),
            'AgingStage2EndSlider': ('animation_stages', 'stage2', 'end_intensity'),
            'AgingStage3StartSlider': ('animation_stages', 'stage3', 'start_intensity'),
            'AgingStage3EndSlider': ('animation_stages', 'stage3', 'end_intensity'),
            'ZombieStage1StartSlider': ('zombie_overlay', 'stage1', 'texture_start'),
            'ZombieStage1EndSlider': ('zombie_overlay', 'stage1', 'texture_end'),
            'ZombieStage2StartSlider': ('zombie_overlay', 'stage2', 'texture_start'),
            'ZombieStage2EndSlider': ('zombie_overlay', 'stage2', 'texture_end'),
            'ZombieStage3StartSlider': ('zombie_overlay', 'stage3', 'texture_start'),
            'ZombieStage3EndSlider': ('zombie_overlay', 'stage3', 'texture_end'),
            'Stage1DurationSlider': ('timings', 'stage1', 'duration_ms'),
            'Stage2DurationSlider': ('timings', 'stage2', 'duration_ms'),
            'Stage3DurationSlider': ('timings', 'stage3', 'duration_ms'),
        }
        
        for slider_name, (config_section, stage_key, param_key) in slider_mappings.items():
            if slider_name in self.parameter_widgets:
                widget = self.parameter_widgets[slider_name]
                # Connect value changed signal
                if hasattr(widget, 'valueChanged'):
                    def make_handler(section, stage_key, param_key, is_intensity):
                        def handler(value):
                            normalized_value = value / 100.0 if is_intensity else value
                            self._update_animation_config_value(section, stage_key, param_key, normalized_value)
                        return handler
                    widget.valueChanged.connect(
                        make_handler(config_section, stage_key, param_key, 'Intensity' in slider_name)
                    )
    
    def _update_animation_config_value(self, section: str, stage_key: str, param_key: str, value: float) -> None:
        """Обновляет значение в конфиге анимации."""
        if section not in self._animation_config:
            self._animation_config[section] = {}
        if stage_key not in self._animation_config[section]:
            self._animation_config[section][stage_key] = {}
        self._animation_config[section][stage_key][param_key] = value
        
        # Обновляем кэш _animation_stages, если изменяется animation_stages
        if section == "animation_stages":
            self._animation_stages = self._animation_config.get("animation_stages", {})
        
        # For zombie_overlay, also update color_start/color_end to match texture
        if section == "zombie_overlay" and param_key in ("texture_start", "texture_end"):
            color_key = param_key.replace("texture", "color")
            self._animation_config[section][stage_key][color_key] = value
        
        # Reload stages
        if section == "animation_stages":
            self._animation_stages = self._animation_config.get("animation_stages", {})
    
    def _load_effect_params_from_config(self) -> None:
        """Загружает значения из конфига в слайдеры."""
        # Aging effect
        for stage_num in [1, 2, 3]:
            stage_key = f"stage{stage_num}"
            stage_cfg = self._animation_stages.get(stage_key, {})
            start_val = int(float(stage_cfg.get("start_intensity", 0.0)) * 100)
            end_val = int(float(stage_cfg.get("end_intensity", 0.0)) * 100)
            
            if f'AgingStage{stage_num}StartSlider' in self.parameter_widgets:
                self.parameter_widgets[f'AgingStage{stage_num}StartSlider'].set_value(start_val)
            if f'AgingStage{stage_num}EndSlider' in self.parameter_widgets:
                self.parameter_widgets[f'AgingStage{stage_num}EndSlider'].set_value(end_val)
        
        # Zombie effect
        zombie_overlay = self._animation_config.get("zombie_overlay", {})
        for stage_num in [1, 2, 3]:
            stage_key = f"stage{stage_num}"
            stage_cfg = zombie_overlay.get(stage_key, {})
            start_val = int(float(stage_cfg.get("texture_start", 0.0)) * 100)
            end_val = int(float(stage_cfg.get("texture_end", 0.0)) * 100)
            
            if f'ZombieStage{stage_num}StartSlider' in self.parameter_widgets:
                self.parameter_widgets[f'ZombieStage{stage_num}StartSlider'].set_value(start_val)
            if f'ZombieStage{stage_num}EndSlider' in self.parameter_widgets:
                self.parameter_widgets[f'ZombieStage{stage_num}EndSlider'].set_value(end_val)
        
        # Stage timings
        timings = self._animation_config.get("timings", {})
        for stage_num in [1, 2, 3]:
            stage_key = f"stage{stage_num}"
            stage_timings = timings.get(stage_key, {})
            duration_val = int(stage_timings.get("duration_ms", 4000))
            
            if f'Stage{stage_num}DurationSlider' in self.parameter_widgets:
                self.parameter_widgets[f'Stage{stage_num}DurationSlider'].set_value(duration_val)
    
    def _save_animation_config(self) -> None:
        """Сохраняет текущую конфигурацию анимации в файл."""
        # Determine config path
        if getattr(sys, "frozen", False):
            base_path = Path(sys.executable).parent
            config_path = base_path / ANIMATION_CONFIG_PATH
        else:
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / ANIMATION_CONFIG_PATH
        
        try:
            # Update config with current values from sliders
            self._update_config_from_sliders()
            
            # Save to file
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._animation_config, f, indent=2, ensure_ascii=False)
            
            print(f"Конфигурация анимации сохранена в: {config_path}")
            QtWidgets.QMessageBox.information(
                self, 
                "Сохранено", 
                f"Конфигурация анимации сохранена в:\n{config_path}"
            )
        except Exception as e:
            print(f"Ошибка при сохранении конфигурации: {e}")
            QtWidgets.QMessageBox.warning(
                self, 
                "Ошибка", 
                f"Не удалось сохранить конфигурацию:\n{e}"
            )
    
    def _update_config_from_sliders(self) -> None:
        """Обновляет конфиг из значений слайдеров."""
        # Aging effect
        for stage_num in [1, 2, 3]:
            stage_key = f"stage{stage_num}"
            if stage_key not in self._animation_config["animation_stages"]:
                self._animation_config["animation_stages"][stage_key] = {}
            
            if f'AgingStage{stage_num}StartSlider' in self.parameter_widgets:
                start_val = self.parameter_widgets[f'AgingStage{stage_num}StartSlider'].value() / 100.0
                self._animation_config["animation_stages"][stage_key]["start_intensity"] = start_val
            if f'AgingStage{stage_num}EndSlider' in self.parameter_widgets:
                end_val = self.parameter_widgets[f'AgingStage{stage_num}EndSlider'].value() / 100.0
                self._animation_config["animation_stages"][stage_key]["end_intensity"] = end_val
        
        # Zombie effect
        for stage_num in [1, 2, 3]:
            stage_key = f"stage{stage_num}"
            if stage_key not in self._animation_config["zombie_overlay"]:
                self._animation_config["zombie_overlay"][stage_key] = {}
            
            if f'ZombieStage{stage_num}StartSlider' in self.parameter_widgets:
                start_val = self.parameter_widgets[f'ZombieStage{stage_num}StartSlider'].value() / 100.0
                self._animation_config["zombie_overlay"][stage_key]["texture_start"] = start_val
                self._animation_config["zombie_overlay"][stage_key]["color_start"] = start_val
            if f'ZombieStage{stage_num}EndSlider' in self.parameter_widgets:
                end_val = self.parameter_widgets[f'ZombieStage{stage_num}EndSlider'].value() / 100.0
                self._animation_config["zombie_overlay"][stage_key]["texture_end"] = end_val
                self._animation_config["zombie_overlay"][stage_key]["color_end"] = end_val
        
        # Stage timings
        for stage_num in [1, 2, 3]:
            stage_key = f"stage{stage_num}"
            if stage_key not in self._animation_config["timings"]:
                self._animation_config["timings"][stage_key] = {}
            
            if f'Stage{stage_num}DurationSlider' in self.parameter_widgets:
                duration_val = self.parameter_widgets[f'Stage{stage_num}DurationSlider'].value()
                self._animation_config["timings"][stage_key]["duration_ms"] = duration_val
        
        # Reload stages
        self._animation_stages = self._animation_config.get("animation_stages", {})

    def preprocess_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            self._display_frame_size = (width, height)
            return frame

        # Для веб-камеры используем родное разрешение без обрезки
        if hasattr(self, 'video_processor') and self.video_processor.file_type == 'webcam':
            self._display_frame_size = (width, height)
            QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
            return frame

        # Для видео обрезаем до соотношения 9:16
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
        candidates = [preferred, "OBS Virtual Camera", "DirectShow", "MSMF", "Default"]
        seen: set[str] = set()
        ordered: list[str] = []
        for name in candidates:
            if name in CAMERA_BACKENDS and name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered or ["Default"]

    def _ensure_webcam_entry(self, attempt: int = 0) -> None:
        import time
        if attempt == 0:
            t_start = time.time()
            print(f"[Startup] _ensure_webcam_entry() started (attempt {attempt})")
        
        capture = self.video_processor.media_capture
        if capture and capture.isOpened():
            if not self._auto_target_selected:
                self._auto_target_selected = True
                if attempt == 0 and hasattr(self, '_startup_start_time'):
                    elapsed = time.time() - self._startup_start_time
                    print(f"[Startup] Webcam already opened in {elapsed:.2f}s (total since init)")
                QtCore.QTimer.singleShot(50, self._ensure_playing)
                QtCore.QTimer.singleShot(100, self._prepare_target_faces)  # Уменьшено с 360ms до 100ms
                QtCore.QTimer.singleShot(150, self._position_uporotsya_button)
                QtCore.QTimer.singleShot(200, self._try_enable_uporotsya)
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
        t0 = time.time()
        if self._load_webcam_direct(backend_flag, backend_name):
            elapsed = time.time() - t0
            print(f"[Startup] _load_webcam_direct() succeeded with {backend_name} in {elapsed:.2f}s")
            if hasattr(self, '_startup_start_time'):
                total_elapsed = time.time() - self._startup_start_time
                print(f"[Startup] Webcam opened in {total_elapsed:.2f}s (total since init)")
            self._auto_target_selected = True
            QtCore.QTimer.singleShot(120, self._ensure_playing)
            QtCore.QTimer.singleShot(360, self._prepare_target_faces)
            QtCore.QTimer.singleShot(400, self._try_enable_uporotsya)
            return

        QtCore.QTimer.singleShot(250, lambda: self._ensure_webcam_entry(attempt + 1))

    def _fallback_to_demo_video(self) -> None:
        capture = self.video_processor.media_capture
        if capture and capture.isOpened():
            return
        if not self._demo_video_path:
            return
        if self._load_demo_video():
            QtCore.QTimer.singleShot(0, self._ensure_playing)

    def _ensure_playing(self) -> None:
        import time
        t_start = time.time()
        print(f"[Startup] _ensure_playing() started")
        
        if not self.selected_video_button:
            return

        if not self.buttonMediaPlay.isChecked():
            self.buttonMediaPlay.setChecked(True)
            return

        if not self.video_processor.processing:
            video_control_actions.set_play_button_icon_to_stop(self)
            t0 = time.time()
            self.video_processor.process_video()
            print(f"[Startup] video_processor.process_video() called in {time.time() - t0:.2f}s")
            elapsed = time.time() - t_start
            print(f"[Startup] _ensure_playing() completed in {elapsed:.2f}s")
            if hasattr(self, '_startup_start_time'):
                total_elapsed = time.time() - self._startup_start_time
                print(f"[Startup] Video processing initiated in {total_elapsed:.2f}s (total since init)")

    def _load_webcam_direct(self, backend_flag: int, backend_name: str) -> bool:
        import time
        t_start = time.time()
        print(f"[Startup] _load_webcam_direct() started with backend: {backend_name}")
        
        t0 = time.time()
        if self._webcam_button:
            self._webcam_button.deleteLater()
            self._webcam_button = None
        print(f"[Startup] Previous webcam button cleanup in {time.time() - t0:.2f}s")

        media_id = str(uuid.uuid4())
        webcam_index: int | str = 0
        
        # Определяем webcam_index для OBS Virtual Camera (если нужно)
        # VideoCapture будет создан в load_media(), чтобы избежать двойного создания
        t0 = time.time()
        if backend_name == "OBS Virtual Camera":
            # Проверяем доступность OBS камеры, но не создаём VideoCapture здесь
            obs_device_names = ["video=OBS Virtual Camera", "video=OBS Virtual Camera (1)", "video=OBS Virtual Camera (2)"]
            webcam_index = 0  # По умолчанию
            backend_flag = cv2.CAP_DSHOW
            # Попробуем найти доступную OBS камеру (быстрая проверка без создания capture)
            for device_name in obs_device_names:
                temp_capture = cv2.VideoCapture(device_name, cv2.CAP_DSHOW)
                if temp_capture.isOpened():
                    webcam_index = device_name
                    temp_capture.release()
                    break
                temp_capture.release()
        print(f"[Startup] OBS camera check in {time.time() - t0:.2f}s")

        t0 = time.time()
        self._webcam_button = widget_components.TargetMediaCardButton(
            media_path=f"Webcam ({backend_name})",
            file_type="webcam",
            media_id=media_id,
            is_webcam=True,
            webcam_index=webcam_index,
            webcam_backend=backend_flag,
            main_window=self,
        )
        print(f"[Startup] TargetMediaCardButton created in {time.time() - t0:.2f}s")
        
        t0 = time.time()
        self._webcam_button.hide()
        print(f"[Startup] Button hide() in {time.time() - t0:.2f}s")
        
        t0 = time.time()
        self._webcam_button.load_media()
        load_media_time = time.time() - t0
        print(f"[Startup] load_media() completed in {load_media_time:.2f}s")

        t0 = time.time()
        capture = self.video_processor.media_capture
        if capture and capture.isOpened():
            self.target_videos = {media_id: self._webcam_button}
            self._media_mode = "webcam"
            t1 = time.time()
            self._update_media_toggle_button()
            print(f"[Startup] _update_media_toggle_button() in {time.time() - t1:.2f}s")
            elapsed = time.time() - t_start
            print(f"[Startup] Post-load_media operations in {time.time() - t0:.2f}s")
            print(f"[Startup] _load_webcam_direct() succeeded in {elapsed:.2f}s")
            return True

        self._webcam_button.deleteLater()
        self._webcam_button = None
        elapsed = time.time() - t_start
        print(f"[Startup] _load_webcam_direct() failed in {elapsed:.2f}s (capture not opened)")
        return False

    def _load_demo_video(self) -> bool:
        if not self._demo_video_path or not self._demo_video_path.is_file():
            return False

        if self._webcam_button:
            self._webcam_button.deleteLater()
            self._webcam_button = None

        if self._demo_video_button:
            self._demo_video_button.deleteLater()
            self._demo_video_button = None

        media_id = str(uuid.uuid4())
        demo_button = widget_components.TargetMediaCardButton(
            media_path=str(self._demo_video_path),
            file_type="video",
            media_id=media_id,
            main_window=self,
        )
        demo_button.hide()
        demo_button.load_media()

        self._demo_video_button = demo_button
        self.target_videos = {media_id: demo_button}
        self.selected_video_button = demo_button

        self._media_mode = "video"
        self._update_media_toggle_button()

        QtCore.QTimer.singleShot(120, self._ensure_playing)
        QtCore.QTimer.singleShot(360, self._prepare_target_faces)
        QtCore.QTimer.singleShot(400, self._try_enable_uporotsya)
        return True

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

