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

# Добавляем DLL файлы TensorRT в binaries, чтобы PyInstaller разместил их в _internal
# где Python сможет их найти при импорте
binaries = []
tensorrt_dlls = [
    ('dependencies/nvinfer_10.dll', '.'),
    ('dependencies/nvinfer_builder_resource_10.dll', '.'),
    ('dependencies/nvinfer_plugin_10.dll', '.'),
    ('dependencies/nvonnxparser_10.dll', '.'),
]
# Проверяем существование файлов перед добавлением
for dll_path, dest_dir in tensorrt_dlls:
    full_path = project_root / dll_path
    if full_path.exists():
        binaries.append((str(full_path), dest_dir))
    else:
        print(f"Warning: TensorRT DLL not found: {full_path}")

hiddenimports = set()
# Добавляем основные модули
hiddenimports.add("PySide6")
hiddenimports.add("shiboken6")
hiddenimports.add("qdarktheme")

# Собираем подмодули для основных библиотек
# ВНИМАНИЕ: collect_submodules может быть медленным для больших библиотек
# Если сборка всё ещё медленная, можно закомментировать эти строки
# и полагаться на автоматическое обнаружение PyInstaller
for module_name in ("onnxruntime", "skimage", "torchvision"):
    try:
        hiddenimports.update(collect_submodules(module_name))
    except ModuleNotFoundError:
        pass

# PySide6 должен быть обнаружен автоматически, но добавляем основные модули явно
try:
    import PySide6
    # Добавляем все подмодули PySide6
    # ВНИМАНИЕ: это может быть медленно, но необходимо для корректной работы
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
    upx=False,  # Отключено для ускорения сборки (UPX очень медленный на больших проектах)
    upx_exclude=[],
    name="ARFaceEffect",
)

