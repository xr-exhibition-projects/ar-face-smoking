# Build and Packaging Guide

This document explains how to produce a standalone build of **VisoMaster** and wrap it in a Windows installer. All commands below assume you execute them from the project root (`VisoMaster/`).

## 1. Prepare the Environment

1. **Activate the Conda environment** (required for proper dependency resolution):
   ```powershell
   conda activate visomaster
   ```
   Make sure you have the `visomaster` conda environment set up as described in `README.md`.

2. Install the build-only tools (if not already installed):
   ```powershell
   pip install pyinstaller
   ```
   Inno Setup compilation requires the [Inno Setup](https://jrsoftware.org/isinfo.php) toolchain (version 6.2 or newer) to be installed on Windows.

## 2. Generate the Standalone Build

Run the helper script which wraps `pyinstaller` and uses the supplied spec file:

```powershell
conda activate visomaster
python scripts/build_dist.py --clean
```

The script:
- Clears previous `build/` and `dist/` folders when `--clean` is provided.
- **Preserves `model_assets`** in `dist/VisoMasterAR/` if it exists (won't be deleted during cleanup).
- Invokes PyInstaller with `installer/visomaster.spec`.
- Produces the distributable folder at `dist/VisoMaster/`.
- **Automatically copies TensorRT DLL files** from `dependencies/` to the dist root (next to the exe).

You can verify the build by launching:

```powershell
dist\VisoMaster\VisoMaster.exe
```

### Building the AR Smoking UI Variant

To build the dedicated AR interface, point the helper script at the alternate spec:

```powershell
conda activate visomaster
python scripts/build_dist.py --clean --spec installer/visomaster_ar.spec --dist-dir ARFaceEffect
```

This produces `dist/ARFaceEffect/ARFaceEffect.exe`, which launches `ARSmokingWindow`.

**Important files that are excluded from the build** (to allow editing after build):
- The `model_assets` folder is **excluded** from the build to reduce size. You must manually copy it to `dist/ARFaceEffect/model_assets/` after building:
  ```powershell
  Copy-Item -Path "model_assets" -Destination "dist\ARFaceEffect\model_assets" -Recurse
  ```
- The `animation_config.json` file is **automatically copied** to the dist root (next to exe) during build, so you can edit it after building.
- The `assets/images/` folder is included in the build, but you can also place images (like `old_face.jpg`) in `dist/ARFaceEffect/assets/images/` after building - the application will use files from there if they exist.

## 3. Create the Windows Installer

1. Open `installer/VisoMaster.iss` in the **Inno Setup Compiler**.
2. Adjust `MyAppVersion` (and other metadata) at the top if required.  
   For the AR build, set `#define MyDistDir "..\dist\VisoMasterAR"` (and update the output filename if you want a different installer name).
3. Compile the script (`Build > Compile` or `Ctrl+F9`).

The installer executable will be generated in `installer/output/VisoMaster_Setup.exe`.

## 4. Updating the Bundled Assets

PyInstaller bundles the following directories and files:

- `app/` (all application code)
- `dependencies/` (DLL files and other dependencies)
- `assets/` (UI images, videos, etc.)
- `animation_config.json` (animation configuration)
- `README.md`, `LICENSE`

**Note**: The `model_assets/` folder is **excluded** from the build to reduce executable size. You must place it manually in the dist folder:

```powershell
# After building, copy model_assets to the dist folder
Copy-Item -Path "model_assets" -Destination "dist\VisoMasterAR\model_assets" -Recurse
```

The build script automatically:
- Copies TensorRT DLL files (`nvinfer_10.dll`, `nvinfer_builder_resource_10.dll`, `nvinfer_plugin_10.dll`, `nvonnxparser_10.dll`) from `dependencies/` to the dist root (next to the exe).
- Preserves `model_assets` in `dist/VisoMasterAR/` during cleanup (won't be deleted when using `--clean`).

Make sure these assets are up to date before rebuilding. If you add new data files at runtime, include them either by updating `installer/visomaster.spec` or copying them manually into `dist/VisoMaster/` prior to running Inno Setup.

## 5. Troubleshooting

- **DLL load errors** when launching the packaged build:
  - Ensure TensorRT DLL files are present in `dependencies/` before building (they will be automatically copied to the dist root).
  - Make sure you're building with the `visomaster` conda environment activated.
  - Check that `dependencies/` folder exists and contains all required DLL files.

- **TensorRT import errors**:
  - TensorRT DLL files are automatically copied to the dist root during build. They should be next to the exe file.
  - If errors persist, ensure the DLL files are in `dependencies/` and rebuild.

- **Missing model_assets errors**:
  - Remember that `model_assets` is excluded from the build. Copy it manually to `dist/ARFaceEffect/model_assets/` after building.

- **Editing configuration and images after build**:
  - `animation_config.json` is automatically copied to the dist root during build. You can edit it directly in `dist/ARFaceEffect/animation_config.json`.
  - To change the face image (`old_face.jpg`), place it in `dist/ARFaceEffect/assets/images/old_face.jpg` - it will be used instead of the one in `_internal`.

- **PyInstaller module errors**:
  - If PyInstaller fails due to missing modules, add them to the `hiddenimports` list in `installer/visomaster.spec`.
  - Make sure you're using the correct conda environment (`conda activate visomaster`).

- **Custom PyInstaller**:
  - Use `python scripts/build_dist.py --pyinstaller <path-to-pyinstaller>` to point to a custom PyInstaller executable if the default is unavailable.

- **Build script preserves model_assets**:
  - When using `--clean`, the build script will preserve `model_assets` in `dist/ARFaceEffect/` if it exists, so you don't need to re-copy it after each rebuild.

