import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

spec_path = Path(sys.argv[0]).resolve()
project_root = spec_path.parent.parent

datas = collect_data_files("qdarktheme")
# Собираем данные для PySide6
try:
    datas += collect_data_files("PySide6")
except Exception:
    pass
try:
    datas += collect_data_files("shiboken6")
except Exception:
    pass

datas += [
    (str(project_root / "app"), "app"),
    # model_assets исключён из сборки - должен быть размещён рядом с exe
    (str(project_root / "dependencies"), "dependencies"),
    (str(project_root / "assets"), "assets"),
    # animation_config.json исключён из сборки - должен быть размещён рядом с exe для редактирования
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "README.md"), "."),
]

binaries = []
# DLL файлы TensorRT будут скопированы в корень сборки скриптом build_dist.py

hiddenimports = set()
# Добавляем основные модули
hiddenimports.add("PySide6")
hiddenimports.add("shiboken6")
hiddenimports.add("qdarktheme")

# Собираем подмодули для основных библиотек
for module_name in ("onnxruntime", "skimage", "torchvision"):
    try:
        hiddenimports.update(collect_submodules(module_name))
    except ModuleNotFoundError:
        pass

# PySide6 должен быть обнаружен автоматически, но добавляем основные модули явно
try:
    import PySide6
    # Добавляем все подмодули PySide6
    hiddenimports.update(collect_submodules("PySide6"))
except (ModuleNotFoundError, ImportError):
    # Если PySide6 не найден, добавляем базовые модули
    hiddenimports.add("PySide6.QtCore")
    hiddenimports.add("PySide6.QtWidgets")
    hiddenimports.add("PySide6.QtGui")

hiddenimports = list(hiddenimports)

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ARFaceEffect",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ARFaceEffect",
)

