from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtWidgets
import qdarktheme

from app.ui.ar_smoking_ui import ARSmokingWindow
from app.ui.core.proxy_style import ProxyStyle


def _resolve_base_path() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _prepare_environment() -> None:
    base_path = _resolve_base_path()
    os.chdir(base_path)

    dependencies_dir = base_path / "dependencies"
    if dependencies_dir.exists():
        os.environ["PATH"] = str(dependencies_dir) + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            try:
                add_dll_directory(str(dependencies_dir))
            except OSError:
                pass

    model_assets_dir = base_path / "model_assets"
    os.environ.setdefault("VISOMASTER_MODELS_DIR", str(model_assets_dir))


def main() -> None:
    _prepare_environment()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle())

    style_path = Path("app/ui/styles/dark_styles.qss")
    with style_path.open("r", encoding="utf-8") as f:
        _style = f.read()
    _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"}) + "\n" + _style
    app.setStyleSheet(_style)

    window = ARSmokingWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()


