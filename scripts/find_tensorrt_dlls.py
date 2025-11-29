"""
Скрипт для поиска и копирования DLL файлов TensorRT в папку dependencies.

Использование:
    python scripts/find_tensorrt_dlls.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

try:
    import tensorrt as trt
    import site
except ImportError:
    print("TensorRT не установлен. Установите его через:")
    print("  pip install tensorrt==10.6.0 --extra-index-url https://pypi.nvidia.com")
    print("  pip install tensorrt-cu12_libs==10.6.0")
    sys.exit(1)


def find_tensorrt_dlls() -> list[Path]:
    """Ищет DLL файлы TensorRT в системе."""
    dll_names = [
        'nvinfer_10.dll',
        'nvinfer_builder_resource_10.dll',
        'nvinfer_plugin_10.dll',
        'nvonnxparser_10.dll',
    ]
    
    found_dlls = []
    
    # 1. Проверяем в site-packages/tensorrt_libs
    for site_packages_dir in site.getsitepackages():
        tensorrt_libs_dir = Path(site_packages_dir) / "tensorrt_libs"
        if tensorrt_libs_dir.exists():
            for dll_name in dll_names:
                dll_path = tensorrt_libs_dir / dll_name
                if dll_path.exists():
                    found_dlls.append(dll_path)
                    print(f"Найден: {dll_path}")
    
    # 2. Проверяем в site-packages/nvidia
    for site_packages_dir in site.getsitepackages():
        nvidia_dir = Path(site_packages_dir) / "nvidia"
        if nvidia_dir.exists():
            for dll_name in dll_names:
                # Ищем рекурсивно
                for dll_path in nvidia_dir.rglob(dll_name):
                    if dll_path not in found_dlls:
                        found_dlls.append(dll_path)
                        print(f"Найден: {dll_path}")
    
    # 3. Проверяем в site-packages/tensorrt
    for site_packages_dir in site.getsitepackages():
        tensorrt_dir = Path(site_packages_dir) / "tensorrt"
        if tensorrt_dir.exists():
            for dll_name in dll_names:
                # Ищем рекурсивно
                for dll_path in tensorrt_dir.rglob(dll_name):
                    if dll_path not in found_dlls:
                        found_dlls.append(dll_path)
                        print(f"Найден: {dll_path}")
    
    return found_dlls


def copy_dlls_to_dependencies(found_dlls: list[Path], project_root: Path) -> None:
    """Копирует найденные DLL файлы в папку dependencies."""
    dependencies_dir = project_root / "dependencies"
    dependencies_dir.mkdir(exist_ok=True)
    
    dll_names = [
        'nvinfer_10.dll',
        'nvinfer_builder_resource_10.dll',
        'nvinfer_plugin_10.dll',
        'nvonnxparser_10.dll',
    ]
    
    copied = []
    for dll_name in dll_names:
        # Ищем соответствующий DLL
        dll_path = None
        for found_dll in found_dlls:
            if found_dll.name == dll_name:
                dll_path = found_dll
                break
        
        if dll_path:
            dst = dependencies_dir / dll_name
            shutil.copy2(dll_path, dst)
            copied.append(dll_name)
            print(f"Скопирован: {dll_name} -> {dst}")
        else:
            print(f"ВНИМАНИЕ: {dll_name} не найден!")
    
    if copied:
        print(f"\n✓ Успешно скопировано {len(copied)} из {len(dll_names)} DLL файлов в {dependencies_dir}")
    else:
        print("\n✗ Не удалось найти DLL файлы TensorRT")
        print("\nАльтернативные способы:")
        print("1. Скачайте DLL с GitHub: https://github.com/visomaster/visomaster-assets/releases/tag/v0.1.0_dp")
        print("2. Скачайте TensorRT с официального сайта NVIDIA: https://developer.nvidia.com/tensorrt")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    
    print("=" * 70)
    print("Поиск DLL файлов TensorRT...")
    print("=" * 70)
    
    # Показываем информацию о Python и conda
    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    
    # Проверяем, используется ли conda
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    conda_prefix = os.environ.get("CONDA_PREFIX", None)
    if conda_env:
        print(f"Conda environment: {conda_env}")
        if conda_prefix:
            print(f"Conda prefix: {conda_prefix}")
            conda_site_packages = Path(conda_prefix) / "Lib" / "site-packages"
            print(f"Conda site-packages: {conda_site_packages}")
            if conda_site_packages.exists():
                tensorrt_libs_path = conda_site_packages / "tensorrt_libs"
                print(f"Ожидаемый путь к tensorrt_libs: {tensorrt_libs_path}")
                if tensorrt_libs_path.exists():
                    print(f"✓ Папка tensorrt_libs найдена!")
                else:
                    print(f"✗ Папка tensorrt_libs не найдена")
    
    print(f"\nВерсия TensorRT: {trt.__version__ if hasattr(trt, '__version__') else 'неизвестна'}")
    
    # Показываем все site-packages
    print("\nВсе пути site-packages:")
    for sp in site.getsitepackages():
        print(f"  - {sp}")
    
    print("\n" + "=" * 70)
    found_dlls = find_tensorrt_dlls()
    
    if not found_dlls:
        print("\n✗ DLL файлы TensorRT не найдены в стандартных местах.")
        print("\nПопробуйте:")
        print("1. Убедитесь, что установлен tensorrt-cu12_libs:")
        print("   pip install tensorrt-cu12_libs==10.6.0")
        print("\n2. Проверьте путь вручную:")
        if conda_prefix:
            manual_path = Path(conda_prefix) / "Lib" / "site-packages" / "tensorrt_libs"
            print(f"   {manual_path}")
        print("\n3. Или скачайте DLL вручную с:")
        print("   https://github.com/visomaster/visomaster-assets/releases/tag/v0.1.0_dp")
        return
    
    print(f"\n✓ Найдено {len(found_dlls)} DLL файлов")
    copy_dlls_to_dependencies(found_dlls, project_root)


if __name__ == "__main__":
    main()

