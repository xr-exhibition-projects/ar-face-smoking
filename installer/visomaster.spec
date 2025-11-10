# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

spec_path = Path(sys.argv[0]).resolve()
project_root = spec_path.parent.parent

datas = collect_data_files("qdarktheme")
datas += [
    (str(project_root / "app"), "app"),
    (str(project_root / "model_assets"), "model_assets"),
    (str(project_root / "dependencies"), "dependencies"),
    (str(project_root / "images"), "images"),
    (str(project_root / "ui"), "ui"),
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "README.md"), "."),
]

binaries = []

hiddenimports = set()
for module_name in ("onnxruntime", "skimage", "torchvision"):
    try:
        hiddenimports.update(collect_submodules(module_name))
    except ModuleNotFoundError:
        pass

hiddenimports = list(hiddenimports)

a = Analysis(
    ["main.py"],
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VisoMaster",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    name="VisoMaster",
)

