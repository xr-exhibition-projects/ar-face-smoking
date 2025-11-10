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
    for dirname in ("build", "dist"):
        path = root / dirname
        if path.exists():
            shutil.rmtree(path)


def build(pyinstaller_exe: str, spec_path: Path) -> None:
    cmd = [
        pyinstaller_exe,
        "--clean",
        "--noconfirm",
        str(spec_path),
    ]
    subprocess.check_call(cmd)


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
            dist_name = "VisoMasterAR"
        else:
            dist_name = "VisoMaster"
        dist_dir = project_root / "dist" / dist_name
    print(f"\nBuild finished. Distributable folder: {dist_dir}")


if __name__ == "__main__":
    main()

