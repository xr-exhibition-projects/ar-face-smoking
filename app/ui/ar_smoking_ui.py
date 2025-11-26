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
from PySide6 import QtCore, QtWidgets, QtGui
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
        self._prevent_video_pause = True
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
        self._awaiting_second_click: bool = False
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
    def load_last_workspace(self) -> None:  # type: ignore[override]
        """Отключаем автозагрузку рабочего пространства."""
        return

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        was_processing = getattr(self, "video_processor", None) and self.video_processor.processing
        super().resizeEvent(event)
        self._position_uporotsya_button()
        self._position_impossible_label()
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

        # Добавляем собственную кнопку поверх видео
        self._start_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_START))
        self._finish_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_FINISH))
        self._welcome_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_WELCOME))
        self._overlay_frames: list[QtGui.QPixmap] = [
            QtGui.QPixmap(self._resource_path(PATH_UI_OVERLAY_1)),
        ]
        self._overlay_static = QtGui.QPixmap(self._resource_path(PATH_UI_OVERLAY_2))
        self._overlay_mask = QtGui.QPixmap(self._resource_path(PATH_UI_OVERLAY_3))
        self._overlay_frame_index: int = 0
        self._overlay_timer: Optional[QtCore.QTimer] = None
        death_frame_paths = DEATH_FRAME_PATHS
        self._death_frames: list[QtGui.QPixmap] = [
            QtGui.QPixmap(self._resource_path(path)) for path in death_frame_paths
        ]
        self._death_frames = [frame for frame in self._death_frames if not frame.isNull()]
        self._death_intro_frame = self._death_frames[0] if self._death_frames else None
        self._death_frame_index: int = 0
        self._death_anim_timer: Optional[QtCore.QTimer] = None
        self._current_death_frame: Optional[QtGui.QPixmap] = (
            self._death_intro_frame if self._death_intro_frame and not self._death_intro_frame.isNull() else None
        )
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

        self._impossible_pixmap = QtGui.QPixmap(self._resource_path(PATH_UI_IMPOSSIBLE))
        self.messageLabel = QtWidgets.QLabel("", parent=self.graphicsViewFrame.viewport())
        self.messageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.messageLabel.setWordWrap(False)
        self.messageLabel.setStyleSheet("background-color: transparent; border: none;")
        self.messageLabel.hide()
        self.messageLabel.setObjectName("messageLabel")

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
        self._overlay_fade_animation: Optional[QtCore.QPropertyAnimation] = None

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
        """Возвращает множители для эффекта старения (old_face) исходя из текущего этапа анимации."""
        if self._current_stage <= 0:
            return 0.0, 0.0

        # Используем актуальные значения из конфига (могут быть обновлены слайдерами)
        animation_stages = self._animation_config.get("animation_stages", {})
        stage_key = f"stage{self._current_stage}"
        stage_cfg = animation_stages.get(stage_key, {})
        
        if not stage_cfg:
            print(f"[get_aging_factors] No config for {stage_key}, returning 0.0")
            return 0.0, 0.0
        
        start_intensity = float(stage_cfg.get("start_intensity", 0.0))
        end_intensity = float(stage_cfg.get("end_intensity", start_intensity))
        
        # Убеждаемся, что значения в диапазоне 0.0-1.0 (на случай, если они в процентах)
        if start_intensity > 1.0:
            start_intensity = start_intensity / 100.0
        if end_intensity > 1.0:
            end_intensity = end_intensity / 100.0
        
        # Вычисляем прогресс этапа (0.0 - 1.0) на основе прошедшего времени
        # 0.0 = начало этапа, 1.0 = конец этапа
        stage_progress = self._compute_stage_progress()
        
        # Плавно интерполируем между start_intensity и end_intensity
        # stage_progress = 0.0 -> current_intensity = start_intensity
        # stage_progress = 1.0 -> current_intensity = end_intensity
        current_intensity = start_intensity + (end_intensity - start_intensity) * stage_progress
        current_intensity = max(0.0, min(1.0, current_intensity))
        
        # Логируем каждые 30 кадров для отладки
        if not hasattr(self, '_aging_log_counter'):
            self._aging_log_counter = 0
        self._aging_log_counter += 1
        if self._aging_log_counter % 30 == 0:
            print(f"[get_aging_factors] Stage: {self._current_stage}, progress: {stage_progress:.3f}, intensity: {current_intensity:.3f} (start: {start_intensity:.3f}, end: {end_intensity:.3f}), start_time: {self._stage_start_time}, duration: {self._stage_duration_ms}")
        
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
                QtCore.QTimer.singleShot(300, lambda: self._prepare_target_faces(retries + 1))
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
            QtCore.QTimer.singleShot(500, lambda: self._prepare_target_faces(retries + 1))
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
        print(f"[_on_uporotsya_toggled] checked={checked}, selected_video_button={self.selected_video_button is not None}")
        if not self.selected_video_button:
            self.buttonUport.setChecked(False)
            print(f"[_on_uporotsya_toggled] No video button selected, returning")
            return

        if checked:
            print(f"[_on_uporotsya_toggled] Button checked, target_faces={len(self.target_faces) if self.target_faces else 0}")
            if not self.target_faces:
                card_actions.find_target_faces(self)
                if not self.target_faces:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Лицо не найдено",
                        "Не удалось обнаружить лицо. Убедитесь, что камера направлена на лицо и попробуйте ещё раз.",
                    )
                    self.buttonUport.setChecked(False)
                    print(f"[_on_uporotsya_toggled] No target faces found, returning")
                    return
            # Начинаем первый этап анимации
            # ВАЖНО: устанавливаем _current_stage и запускаем анимацию ДО включения swapfacesButton,
            # чтобы таймер начал отсчет как можно раньше
            print(f"[_on_uporotsya_toggled] Starting stage 1 animation, current_stage before: {self._current_stage}")
            
            # ВАЖНО: Устанавливаем _current_stage = 1 ПЕРЕД вызовом _start_stage1_animation,
            # чтобы при первом вызове swap_core current_stage уже был > 0
            self._current_stage = 1
            print(f"[_on_uporotsya_toggled] Set _current_stage = {self._current_stage}")
            
            self._start_stage1_animation()
            
            # ВАЖНО: убеждаемся, что _stage_start_time и _stage_duration_ms установлены
            # (они устанавливаются в _start_stage1_animation, но на всякий случай проверяем)
            if self._stage_start_time <= 0 or self._stage_duration_ms <= 0:
                timings = self._animation_config.get("timings", {})
                stage1_timings = timings.get("stage1", {})
                duration_ms = stage1_timings.get("duration_ms", 4000)
                self._stage_start_time = time.monotonic()
                self._stage_duration_ms = duration_ms
                print(f"[_on_uporotsya_toggled] Fixed stage timing: start_time={self._stage_start_time}, duration={self._stage_duration_ms}")
            
            # ВАЖНО: Убеждаемся, что _current_stage все еще = 1 перед включением swapfacesButton
            if self._current_stage != 1:
                print(f"[_on_uporotsya_toggled] WARNING: _current_stage changed to {self._current_stage}, resetting to 1")
                self._current_stage = 1
            
            # Включаем swapfacesButton после запуска анимации
            print(f"[_on_uporotsya_toggled] Setting swapfacesButton checked=True, current_stage={self._current_stage}")
            self.swapfacesButton.setChecked(True)
            # НЕ вызываем _stop_face_fade здесь, так как он может остановить таймер stage1
            # self._stop_face_fade(reset_progress=True)
            self._swap_active = True
            self._awaiting_second_click = True
            self._second_press_triggered = False
            self.buttonUport.setCheckable(False)
            # Скрываем кнопку - она появится через таймер в _show_finish_button()
            self.buttonUport.hide()
            self.messageLabel.hide()
            print(f"[_on_uporotsya_toggled] Stage 1 initialized: _current_stage={self._current_stage}, _stage_start_time={self._stage_start_time}, _stage_duration_ms={self._stage_duration_ms}, swapfacesButton.isChecked()={self.swapfacesButton.isChecked()}")
        else:
            print(f"[_on_uporotsya_toggled] Button unchecked, stopping animation")
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
        self._awaiting_second_click = False
        self._second_press_triggered = False
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

        frame_width, frame_height = getattr(self, "_display_frame_size", (width, height))
        frame_width = max(1, frame_width)
        frame_height = max(1, frame_height)

        aspect_ratio = frame_width / frame_height
        if aspect_ratio <= 0:
            aspect_ratio = width / height if height else 1.0
        display_width = min(width, int(height * aspect_ratio))
        display_width = max(1, display_width)

        # Определяем размеры надписи
        if self.messageLabel.pixmap():
            label_width = self.messageLabel.pixmap().width()
            label_height = self.messageLabel.pixmap().height()
        else:
            label_width = self.messageLabel.sizeHint().width()
            label_height = self.messageLabel.sizeHint().height()
        
        # Используем ту же логику позиционирования, что и для кнопки
        pos_x = max(0, (width - label_width) // 2)
        pos_y = max(0, height - label_height - 24)
        
        self.messageLabel.setGeometry(pos_x, pos_y, label_width, label_height)
        self.messageLabel.raise_()

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
            QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
            QtCore.QTimer.singleShot(0, self._position_impossible_label)
            QtCore.QTimer.singleShot(0, self._position_config_button)
            QtCore.QTimer.singleShot(0, self._refresh_welcome_overlay_graphics)
        if obj == getattr(self, "welcomeOverlay", None) and event.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonDblClick):
            self._dismiss_welcome_overlay()
            return True
        if obj == getattr(self, "deathOverlay", None) and event.type() == QtCore.QEvent.MouseButtonRelease:
            if self.deathOverlay.isVisible():
                self._restart_from_death_screen()
            return True
        return super().eventFilter(obj, event)

    def _start_stage1_animation(self) -> None:
        """Запускает первый этап анимации: от "УПОРОТЬСЯ" до появления кнопки "СЛЕЗТЬ" """
        # Загружаем конфигурацию анимации
        timings = self._animation_config.get("timings", {})
        stage1_timings = timings.get("stage1", {})
        duration_ms = stage1_timings.get("duration_ms", 4000)
        
        # Загружаем параметры интенсивности для логирования
        animation_stages = self._animation_config.get("animation_stages", {})
        stage1_cfg = animation_stages.get("stage1", {})
        start_intensity = stage1_cfg.get("start_intensity", 0.0)
        end_intensity = stage1_cfg.get("end_intensity", 1.0)
        
        # Устанавливаем текущий этап
        self._current_stage = 1
        
        # Запускаем таймер для показа кнопки "слезть"
        # ВАЖНО: устанавливаем start_time ДО начала обработки кадров, чтобы прогресс вычислялся правильно
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
        
        print(f"[_start_stage1_animation] Stage 1 started: duration={duration_ms}ms, start_time={self._stage_start_time}, current_stage={self._current_stage}, intensity: {start_intensity} -> {end_intensity}")
    
    def _start_stage2_animation(self) -> None:
        """Запускает второй этап анимации: от "СЛЕЗТЬ" до "НЕВОЗМОЖНО" """
        timings = self._animation_config.get("timings", {})
        stage2_timings = timings.get("stage2", {})
        duration_ms = stage2_timings.get("duration_ms", 2000)
        
        # Устанавливаем текущий этап
        self._current_stage = 2
        
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
        
        # Устанавливаем текущий этап
        self._current_stage = 3
        
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

    def _start_fade_timer(self, target_intensity: float, duration_ms: int, on_complete: Optional[Callable[[], None]] = None) -> None:
        if self._fade_timer:
            self._fade_timer.stop()
            self._fade_timer.deleteLater()
        self._fade_timer = QtCore.QTimer(self)
        interval_ms = 50
        total = float(max(1, duration_ms))
        current_stage_key = f"stage{self._current_stage}"
        stage_cfg = self._animation_stages.get(current_stage_key, {})
        start_intensity = float(stage_cfg.get("start_intensity", self._fade_progress))
        end_intensity = float(stage_cfg.get("end_intensity", target_intensity))
        intensity_range = end_intensity - start_intensity
        self._fade_step = (interval_ms / total) * intensity_range

        def update_progress():
            self._update_fade_progress(target_intensity, on_complete=on_complete)

        self._fade_timer.timeout.connect(update_progress)
        self._fade_timer.start(interval_ms)

    def _start_death_delay_timer(self) -> None:
        self._cancel_death_delay_timer()
        delay = max(0, int(self._death_delay_ms))
        self._death_delay_timer = QtCore.QTimer(self)
        self._death_delay_timer.setSingleShot(True)
        self._death_delay_timer.timeout.connect(self._show_death_screen)
        self._death_delay_timer.start(delay)

    def _cancel_death_delay_timer(self) -> None:
        if self._death_delay_timer:
            self._death_delay_timer.stop()
            self._death_delay_timer.deleteLater()
            self._death_delay_timer = None
    
    def _update_fade_progress(self, target_intensity: float, on_complete: Optional[Callable[[], None]] = None) -> None:
        """Обновляет прогресс анимации до целевой интенсивности"""
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
            if on_complete:
                on_complete()
    
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
                self.buttonUport.raise_()  # Поднимаем кнопку на передний план
    
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
        if reset_progress:
            # Сбрасываем на стартовую интенсивность первого этапа
            stage1_config = self._animation_stages.get("stage1", {})
            start_intensity = max(0.0, min(1.0, stage1_config.get("start_intensity", 0.0)))
            self._fade_progress = start_intensity
            self._current_stage = 0
            # Сбрасываем таймеры этапов
            self._stage_start_time = 0.0
            self._stage_duration_ms = 0

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
        # Экран смерти появится после завершения анимации stage2
        if self.mediaToggleButton:
            self.mediaToggleButton.hide()

    def _show_death_screen(self) -> None:
        self._cancel_death_delay_timer()
        self.messageLabel.hide()
        if hasattr(self, "buttonUport") and self.buttonUport:
            self.buttonUport.hide()
        # Сбрасываем состояние анимации при показе экрана смерти
        self._stop_face_fade(reset_progress=True)
        # Останавливаем обработку видео, но НЕ освобождаем media_capture для веб-камеры
        # чтобы при повторном запуске можно было продолжить использовать тот же источник
        try:
            self.video_processor.stop_processing()
        except Exception:
            pass
        # НЕ освобождаем media_capture для веб-камеры, чтобы при повторном запуске можно было использовать тот же источник
        # if self.video_processor.media_capture:
        #     try:
        #         self.video_processor.media_capture.release()
        #     except Exception:
        #         pass
        #     self.video_processor.media_capture = None
        # НЕ сбрасываем video_processor состояние, чтобы при повторном запуске можно было использовать тот же источник
        # self.video_processor.media_path = False
        # self.video_processor.file_type = None
        # self.video_processor.current_frame = []
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
        # Подготавливаем следующую сессию (сбрасывает состояние для повторного запуска)
        self._prepare_next_session()

    def _restart_from_death_screen(self) -> None:
        """Перезапускает сессию после экрана смерти, сохраняя состояние для работы эффектов."""
        print(f"[_restart_from_death_screen] Restarting from death screen")
        self._stop_death_animation()
        self.deathOverlay.hide()
        self.buttonUport.hide()
        self.configButton.hide()
        if self.mediaToggleButton:
            self.mediaToggleButton.hide()
        
        # ВАЖНО: НЕ сбрасываем состояние анимации здесь, так как оно уже было сброшено в _prepare_next_session()
        # Просто показываем welcome overlay и готовим кнопку к новому запуску
        if hasattr(self, "buttonUport") and self.buttonUport:
            # Включаем кнопку для повторного использования
            self.buttonUport.setEnabled(True)
            self.buttonUport.setCheckable(True)
            self.buttonUport.setChecked(False)
            self._update_uporotsya_button_icon(start=True)
            print(f"[_restart_from_death_screen] Button enabled: {self.buttonUport.isEnabled()}, checkable: {self.buttonUport.isCheckable()}")
        
        # Убеждаемся, что swapfacesButton выключен (будет включен при нажатии на "упороться")
        self.swapfacesButton.setChecked(False)
        self._swap_active = False
        print(f"[_restart_from_death_screen] swapfacesButton checked: {self.swapfacesButton.isChecked()}, swap_active: {self._swap_active}")
        
        # Убеждаемся, что состояние анимации сброшено (для нового запуска)
        self._current_stage = 0
        self._stage_start_time = 0
        self._stage_duration_ms = 0
        print(f"[_restart_from_death_screen] Animation state reset: current_stage={self._current_stage}, start_time={self._stage_start_time}, duration={self._stage_duration_ms}")
        
        # Проверяем наличие необходимых данных для повторного запуска
        print(f"[_restart_from_death_screen] selected_video_button: {self.selected_video_button is not None}, target_faces: {len(self.target_faces) if self.target_faces else 0}")
        
        # Показываем welcome overlay
        self._show_welcome_overlay()
        
        # Если видео было остановлено, перезапускаем его (для веб-камеры)
        if self.selected_video_button and hasattr(self.selected_video_button, 'is_webcam') and self.selected_video_button.is_webcam:
            if not self.buttonMediaPlay.isChecked():
                # Не запускаем автоматически, пользователь сам нажмет кнопку
                pass

    def _prepare_next_session(self) -> None:
        """Подготавливает следующую сессию, сбрасывая состояние анимации, но сохраняя данные для повторного использования."""
        # Сбрасываем состояние анимации
        self._stop_face_fade(reset_progress=True)
        
        # Сбрасываем состояние кнопок
        if hasattr(self, "buttonUport") and self.buttonUport:
            # НЕ отключаем кнопку здесь - она будет включена в _restart_from_death_screen
            # self.buttonUport.setEnabled(False)
            self.buttonUport.setCheckable(True)
            self.buttonUport.setChecked(False)
            self._update_uporotsya_button_icon(start=True)
        
        # Сбрасываем состояние свапа
        self._swap_active = False
        self._awaiting_second_click = False
        self._second_press_triggered = False
        self.messageLabel.hide()
        
        # Сбрасываем флаги автоматического выбора
        self._auto_target_selected = False
        self._auto_face_selected = False
        self._pending_input_button = None
        self._input_ready = False
        self._target_ready = False
        
        # НЕ сбрасываем selected_video_button, чтобы при повторном запуске можно было использовать тот же источник
        # self.selected_video_button = False
        
        # НЕ очищаем target_videos и списки, чтобы при повторном запуске можно было использовать те же лица
        # self.target_videos = {}
        # self.targetVideosList.clear()
        # self.inputFacesList.clear()
        
        # Очищаем сцену
        if getattr(self, "scene", None):
            self.scene.clear()
        
        # Скрываем окна
        if self.control_window:
            self.control_window.hide()
        if self.mediaToggleButton:
            self.mediaToggleButton.hide()
        
        # НЕ сбрасываем video_processor состояние, чтобы при повторном запуске можно было использовать тот же источник
        # self.video_processor.current_frame = []
        # self.video_processor.media_path = False
        # self.video_processor.file_type = None
        
        # ВАЖНО: НЕ освобождаем media_capture, чтобы при повторном запуске можно было использовать тот же источник
        # (это делается в _show_death_screen, но мы не делаем это здесь)
        # Сбрасываем флаг загрузки лиц, чтобы разрешить повторную загрузку
        self._faces_loading_in_progress = False
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

    def _show_welcome_overlay(self) -> None:
        if self._welcome_active or not self.welcomeOverlay or not self.welcomeLabel:
            return

        self._welcome_active = True
        self.welcomeOverlay.setGeometry(self.rect())
        self._refresh_welcome_overlay_graphics()
        self.welcomeOverlay.show()
        self.welcomeOverlay.raise_()
        self._start_overlay_animation()

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
        if self._welcome_timer:
            self._welcome_timer.stop()
            self._welcome_timer.deleteLater()
            self._welcome_timer = None
        if self.welcomeOverlay:
            self.welcomeOverlay.hide()
        self._stop_overlay_animation()
        self._initialize_post_welcome_state()
        self._try_enable_uporotsya()

    def _initialize_post_welcome_state(self) -> None:
        if self.welcomeOverlay:
            self.welcomeOverlay.hide()
        self.buttonUport.show()
        self.configButton.show()
        # Кнопка переключения камера/видео скрыта
        # if self.mediaToggleButton:
        #     self.mediaToggleButton.show()
        #     self._update_media_toggle_button()
        self._position_config_button()
        QtCore.QTimer.singleShot(0, self._position_uporotsya_button)
        QtCore.QTimer.singleShot(0, self._ensure_playing)
        QtCore.QTimer.singleShot(0, lambda: layout_actions.fit_image_to_view_onchange(self))

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
                QtCore.QTimer.singleShot(120, self._ensure_playing)
                QtCore.QTimer.singleShot(360, self._prepare_target_faces)
                QtCore.QTimer.singleShot(400, self._position_uporotsya_button)
                QtCore.QTimer.singleShot(500, self._try_enable_uporotsya)
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

