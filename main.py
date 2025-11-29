from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtWidgets
import qdarktheme

from app.ui import main_ui, ar_smoking_ui
from app.ui.core.proxy_style import ProxyStyle


def _resolve_base_path() -> Path:
    """Определяет базовый путь для frozen приложения или обычного запуска.
    Для frozen приложения возвращает папку с exe, а не временную папку PyInstaller.
    """
    if getattr(sys, "frozen", False):
        # Для frozen приложения возвращаем папку с exe
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _prepare_environment() -> None:
    base_path = _resolve_base_path()
    os.chdir(base_path)

    # Добавляем пути для поиска DLL файлов
    add_dll_directory = getattr(os, "add_dll_directory", None)
    
    # При frozen приложении (PyInstaller) библиотеки находятся в _internal
    if getattr(sys, "frozen", False):
        internal_dir = base_path / "_internal"
        if internal_dir.exists():
            os.environ["PATH"] = str(internal_dir) + os.pathsep + os.environ.get("PATH", "")
            if callable(add_dll_directory):
                try:
                    add_dll_directory(str(internal_dir))
                except OSError:
                    pass
    
    # Также добавляем dependencies, если он существует (для портативной версии)
    dependencies_dir = base_path / "dependencies"
    if dependencies_dir.exists():
        os.environ["PATH"] = str(dependencies_dir) + os.pathsep + os.environ.get("PATH", "")
        if callable(add_dll_directory):
            try:
                add_dll_directory(str(dependencies_dir))
            except OSError:
                pass

    model_assets_dir = base_path / "model_assets"
    os.environ.setdefault("VISOMASTER_MODELS_DIR", str(model_assets_dir))


def main() -> None:
    _prepare_environment()
    base_path = _resolve_base_path()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle())

    # Загружаем стили - если файл не найден, используем только qdarktheme
    _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})
    
    # Пробуем загрузить дополнительные стили из файла (опционально)
    possible_paths = []
    if getattr(sys, "frozen", False):
        possible_paths = [
            base_path / "_internal" / "app" / "ui" / "styles" / "dark_styles.qss",
            base_path / "app" / "ui" / "styles" / "dark_styles.qss",
        ]
    else:
        possible_paths = [
            base_path / "app" / "ui" / "styles" / "dark_styles.qss",
            base_path / "_internal" / "app" / "ui" / "styles" / "dark_styles.qss",
        ]
    
    style_path = None
    for path in possible_paths:
        if path.exists():
            style_path = path
            break
    
    if style_path is not None:
        try:
            with style_path.open("r", encoding="utf-8") as f:
                custom_style = f.read()
            _style = _style + "\n" + custom_style
        except Exception:
            # Если не удалось загрузить дополнительные стили, используем только qdarktheme
            pass
    
    app.setStyleSheet(_style)

    window = ar_smoking_ui.ARSmokingWindow() #main_ui.MainWindow()
    window.show()
    # Запускаем приложение сразу в полноэкранном режиме
    window.showFullScreen()
    # Устанавливаем флаг полноэкранного режима, чтобы F11 работал правильно
    window.is_full_screen = True
    app.exec()


if __name__ == "__main__":
    main()