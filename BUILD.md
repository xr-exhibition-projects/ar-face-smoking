# Build and Packaging Guide

This document explains how to produce a standalone build of **VisoMaster** and wrap it in a Windows installer. All commands below assume you execute them from the project root (`VisoMaster/`).

## 1. Prepare the Environment

1. Create and activate the existing Conda environment (see `README.md` for details) or any Python 3.10+ virtual environment.
2. Install the runtime dependencies:
   ```powershell
   pip install -r requirements_cu124.txt
   ```
3. Install the build-only tools:
   ```powershell
   pip install pyinstaller
   ```
   Inno Setup compilation requires the [Inno Setup](https://jrsoftware.org/isinfo.php) toolchain (version 6.2 or newer) to be installed on Windows.

## 2. Generate the Standalone Build

Run the helper script which wraps `pyinstaller` and uses the supplied spec file:

```powershell
python scripts/build_dist.py --clean
```

The script:
- Clears previous `build/` and `dist/` folders when `--clean` is provided.
- Invokes PyInstaller with `installer/visomaster.spec`.
- Produces the distributable folder at `dist/VisoMaster/`.

You can verify the build by launching:

```powershell
dist\VisoMaster\VisoMaster.exe
```

### Building the AR Smoking UI Variant

To build the dedicated AR interface, point the helper script at the alternate spec:

```powershell
python scripts/build_dist.py --clean --spec installer/visomaster_ar.spec --dist-dir VisoMasterAR
```

This produces `dist/VisoMasterAR/VisoMasterAR.exe`, which launches `ARSmokingWindow`.

## 3. Create the Windows Installer

1. Open `installer/VisoMaster.iss` in the **Inno Setup Compiler**.
2. Adjust `MyAppVersion` (and other metadata) at the top if required.  
   For the AR build, set `#define MyDistDir "..\dist\VisoMasterAR"` (and update the output filename if you want a different installer name).
3. Compile the script (`Build > Compile` or `Ctrl+F9`).

The installer executable will be generated in `installer/output/VisoMaster_Setup.exe`.

## 4. Updating the Bundled Assets

PyInstaller bundles the following directories and files:

- `app/`
- `model_assets/`
- `dependencies/`
- `images/`
- `README.md`, `LICENSE`

Make sure these assets are up to date before rebuilding. If you add new data files at runtime, include them either by updating `installer/visomaster.spec` or copying them manually into `dist/VisoMaster/` prior to running Inno Setup.

## 5. Troubleshooting

- **DLL load errors** when launching the packaged build usually mean `dependencies/` is missing required binaries. Re-run the build script and ensure the folder exists before packaging.
- If PyInstaller fails due to missing modules, add them to the `hiddenimports` list in `installer/visomaster.spec`.
- Use `python scripts/build_dist.py --pyinstaller <path-to-pyinstaller>` to point to a custom PyInstaller executable if the default is unavailable.

