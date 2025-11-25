"""
Utility to pre-build TensorRT engines for LivePortrait models.

Run from the project root:
    conda activate env50  # или другая среда
    python scripts/build_trt_engines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.processors.models_processor import ModelsProcessor  # noqa: E402


class _SilentSignal:
    def emit(self, *_args, **_kwargs) -> None:  # pragma: no cover - простая заглушка
        return


class _DummyMainWindow:
    """Минимальная заглушка MainWindow для ModelsProcessor."""

    def __init__(self) -> None:
        self.control = {}
        self.model_loading_signal = _SilentSignal()
        self.model_loaded_signal = _SilentSignal()


def _resolve_plugin_path() -> str | None:
    """Возвращает путь к grid_sample_3d_plugin, если он существует."""
    primary = ROOT_DIR / "model_assets" / "liveportrait_onnx" / "grid_sample_3d_plugin.dll"
    fallback = ROOT_DIR / "model_assets" / "grid_sample_3d_plugin.dll"
    if primary.exists():
        return str(primary)
    if fallback.exists():
        return str(fallback)
    return None


def build_trt_engines() -> None:
    dummy_window = _DummyMainWindow()
    models_processor = ModelsProcessor(dummy_window, device="cuda")
    models_processor.switch_providers_priority("TensorRT")

    plugin_path = _resolve_plugin_path()

    models_to_build = [
        ("LivePortraitMotionExtractor", "fp32", None),
        ("LivePortraitAppearanceFeatureExtractor", "fp16", None),
        ("LivePortraitStitchingEye", "fp16", None),
        ("LivePortraitStitchingLip", "fp16", None),
        ("LivePortraitStitching", "fp16", None),
        ("LivePortraitWarpingSpadeFix", "fp16", plugin_path),
    ]

    for name, precision, plugin in models_to_build:
        print(f"[TensorRT] Building {name} (precision={precision}) ...")
        models_processor.load_model_trt(name, custom_plugin_path=plugin, precision=precision)
        print(f"[TensorRT] {name} ready.")

    print("All TensorRT engines are ready.")


if __name__ == "__main__":
    build_trt_engines()

