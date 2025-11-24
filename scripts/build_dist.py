from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VisoMaster distributable with PyInstaller.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous build and dist directories before building.",
    )
    parser.add_argument(
        "--spec",
        default="installer/visomaster.spec",
        help="Path to the PyInstaller spec file to use (default: %(default)s).",
    )
    parser.add_argument(
        "--dist-dir",
        default=None,
        help="Custom dist directory name (defaults to the name defined in the spec).",
    )
    parser.add_argument(
        "--pyinstaller",
        default="pyinstaller",
        help="PyInstaller executable to invoke (default: %(default)s).",
    )
    return parser.parse_args()


def ensure_pyinstaller_available(executable: str) -> None:
    try:
        subprocess.run([executable, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "PyInstaller is required to build the project. Install it with `pip install pyinstaller` "
            "or pass a custom executable via --pyinstaller."
        ) from exc


def clean_directories(root: Path) -> None:
    # Очищаем build полностью
    build_path = root / "build"
    if build_path.exists():
        shutil.rmtree(build_path)
    
    # Очищаем dist, но сохраняем model_assets в ARFaceEffect если он есть
    dist_path = root / "dist"
    if dist_path.exists():
        ar_face_effect_path = dist_path / "ARFaceEffect"
        if ar_face_effect_path.exists():
            # Сохраняем model_assets если он существует
            model_assets_backup = None
            model_assets_path = ar_face_effect_path / "model_assets"
            if model_assets_path.exists():
                # Создаем временную копию
                backup_path = root / ".model_assets_backup"
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.copytree(model_assets_path, backup_path)
                model_assets_backup = backup_path
            
            # Удаляем ARFaceEffect
            shutil.rmtree(ar_face_effect_path)
            
            # Восстанавливаем model_assets если был
            if model_assets_backup is not None:
                ar_face_effect_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(model_assets_backup, model_assets_path)
                shutil.rmtree(model_assets_backup)
        else:
            # Если нет ARFaceEffect, удаляем весь dist
            shutil.rmtree(dist_path)


def build(pyinstaller_exe: str, spec_path: Path) -> None:
    cmd = [
        pyinstaller_exe,
        "--clean",
        "--noconfirm",
        str(spec_path),
    ]
    subprocess.check_call(cmd)


def copy_tensorrt_dlls(dist_dir: Path, project_root: Path) -> None:
    """Копирует DLL файлы TensorRT из dependencies в корень сборки (рядом с exe)."""
    dependencies_dir = project_root / "dependencies"
    tensorrt_dlls = [
        'nvinfer_10.dll',
        'nvinfer_builder_resource_10.dll',
        'nvinfer_plugin_10.dll',
        'nvonnxparser_10.dll',
    ]
    
    copied = []
    for dll_name in tensorrt_dlls:
        src = dependencies_dir / dll_name
        if src.exists():
            dst = dist_dir / dll_name
            shutil.copy2(src, dst)
            copied.append(dll_name)
    
    if copied:
        print(f"Copied TensorRT DLLs to dist root: {', '.join(copied)}")


def copy_animation_config(dist_dir: Path, project_root: Path) -> None:
    """Копирует animation_config.json в корень сборки (рядом с exe) для редактирования."""
    config_src = project_root / "animation_config.json"
    if config_src.exists():
        config_dst = dist_dir / "animation_config.json"
        shutil.copy2(config_src, config_dst)
        print(f"Copied animation_config.json to dist root")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    spec_path = (project_root / args.spec).resolve()

    if not spec_path.exists():
        raise SystemExit(f"Unable to locate spec file at {spec_path}")

    ensure_pyinstaller_available(args.pyinstaller)

    if args.clean:
        clean_directories(project_root)

    build(args.pyinstaller, spec_path)
    if args.dist_dir:
        dist_dir = project_root / "dist" / args.dist_dir
    else:
        # Guess dist folder name from spec file name
        if "ar" in spec_path.stem.lower():
            dist_name = "ARFaceEffect"
        else:
            dist_name = "VisoMaster"
        dist_dir = project_root / "dist" / dist_name
    
    # Копируем DLL TensorRT в корень сборки (рядом с exe)
    copy_tensorrt_dlls(dist_dir, project_root)
    
    # Копируем animation_config.json в корень сборки (рядом с exe)
    copy_animation_config(dist_dir, project_root)
    
    print(f"\nBuild finished. Distributable folder: {dist_dir}")


if __name__ == "__main__":
    main()

