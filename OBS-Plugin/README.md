# Face Captions – OBS Plugin (C++)

Native OBS filter: face-following captions with real-time speech-to-text on your camera source.

---

# Step-by-step build (Windows)

Replace these with **your** paths where noted:

| What | Example (yours may differ) |
|------|-----------------------------|
| Project folder | `C:\Users\Pramod Tiwari\Downloads\tiktok projects\face-captions` |
| OBS-Plugin folder | `C:\Users\Pramod Tiwari\Downloads\tiktok projects\face-captions\OBS-Plugin` |
| OBS build folder | `C:\obs-studio\build_x64` |
| vcpkg folder | `C:\vcpkg` |
| Where to install OBS plugin | `C:\obs-studio\build_x64\obs-plugins\64bit` or `C:\Program Files\obs-studio\obs-plugins\64bit` |

---

## Step 1: Install Visual Studio

1. Download **Visual Studio 2022** (Community is free): https://visualstudio.microsoft.com/downloads/
2. Run the installer. In **Workloads**, enable **"Desktop development with C++"**.
3. Install.

---

## Step 2: Install vcpkg (package manager)

1. Open **PowerShell** (or Command Prompt).
2. Go to a folder where you want vcpkg (e.g. `C:\`):

   ```powershell
   cd C:\
   ```

3. Clone vcpkg:

   ```powershell
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   ```

4. Install **OpenCV** and **SIMDe** (64-bit). SIMDe is required by libobs when building the plugin. **Run these commands from the vcpkg folder** (e.g. `C:\vcpkg`), not from the project:

   ```powershell
   cd C:\vcpkg
   .\vcpkg install opencv4:x64-windows simde:x64-windows
   ```

   Or from any folder using the full path: `C:\vcpkg\vcpkg install opencv4:x64-windows simde:x64-windows`

   Note the path where vcpkg is installed (e.g. `C:\vcpkg`). You will use it as `CMAKE_TOOLCHAIN_FILE`.

---

## Step 3: Build OBS Studio from source

OBS must be **built from source** because the installer does not include the files needed to build plugins.

1. Install **CMake**: https://cmake.org/download/ (e.g. "Windows x64 Installer"). Add CMake to PATH during install.

2. Clone OBS:

   ```powershell
   cd C:\
   git clone --recursive https://github.com/obsproject/obs-studio.git
   cd obs-studio
   ```

3. Create a build folder and configure. **Do not use vcpkg here** — OBS uses its own dependencies (obs-deps). Run:

   ```powershell
   mkdir build_x64
   cd build_x64
   cmake .. -G "Visual Studio 17 2022" -A x64
   ```

   (Leave out `-DCMAKE_TOOLCHAIN_FILE`. vcpkg is only for building the Face Captions plugin later.)

4. Build OBS:

   ```powershell
   cmake --build . --config RelWithDebInfo
   ```

   When it finishes, your OBS build is in `C:\obs-studio\build_x64`. Remember this path as **OBS_BUILD_DIR**.

---

## Step 4: Download Vosk API (for speech-to-text)

1. Go to: https://github.com/alphacep/vosk-api/releases  
2. Download the **Windows** zip, e.g. **`vosk-win64-0.3.45.zip`** (or latest).
3. Extract it to a folder, e.g. `C:\vosk-api` (so you have `C:\vosk-api\vosk_api.h` and `C:\vosk-api\libvosk.dll` or similar).
4. Remember this folder as **VOSK_ROOT** (e.g. `C:\vosk-api`).

---

## Step 5: Download face detection and speech models (inside the plugin)

**Where:** `OBS-Plugin` folder inside your face-captions project.

```powershell
cd "C:\Users\Pramod Tiwari\Downloads\tiktok projects\face-captions\OBS-Plugin"
```

Run these **one by one**:

1. **Haar cascade** (face detection):

   ```powershell
   .\scripts\download_haarcascade.ps1
   ```

   This creates `OBS-Plugin\data\haarcascade_frontalface_default.xml`.

2. **Vosk small model** (~40 MB, required for STT):

   ```powershell
   .\scripts\download_vosk_model.ps1
   ```

   This creates `OBS-Plugin\models\vosk-model-small-en-us-0.15\`.

3. **Vosk large model** (optional, ~1.8 GB, better accuracy):

   ```powershell
   .\scripts\download_vosk_model_large.ps1
   ```

   This creates `OBS-Plugin\models\vosk-model-en-us-0.22\`. The plugin will use it automatically if you leave the model path empty.

---

## Step 6: Configure and build the Face Captions plugin

**Where:** a **build** folder **inside** `OBS-Plugin`.

1. Go to the plugin folder and create `build`:

   ```powershell
   cd "C:\Users\Pramod Tiwari\Downloads\tiktok projects\face-captions\OBS-Plugin"
   mkdir build
   cd build
   ```

2. Run **CMake** (replace paths with yours):

   ```powershell
   cmake .. -G "Visual Studio 17 2022" -A x64 `
     -DOBS_BUILD_DIR="C:/obs-studio/build_x64" `
     -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
     -DVOSK_ROOT="C:/vosk-api"
   ```

   - `OBS_BUILD_DIR` = folder where you built OBS (Step 3).  
   - `CMAKE_TOOLCHAIN_FILE` = path to vcpkg’s `scripts/buildsystems/vcpkg.cmake`.  
   - `VOSK_ROOT` = folder where you extracted Vosk API (Step 4).  
   - If you get **“Could NOT find SIMDe”**, add: `-DOBS_DEPS_DIR="C:/obs-studio/.deps/obs-deps-2025-08-23-x64"` (adjust the `obs-deps-*-x64` folder name to match what’s under `C:\obs-studio\.deps`).

3. Build the plugin:

   ```powershell
   cmake --build . --config RelWithDebInfo
   ```

   The plugin DLL is created at:  
   `OBS-Plugin\build\RelWithDebInfo\obs-face-captions.dll`

---

## Step 7: Deploy the plugin into OBS

**Where:** `OBS-Plugin` folder (not inside `build`).

1. Go back to the plugin root:

   ```powershell
   cd "C:\Users\Pramod Tiwari\Downloads\tiktok projects\face-captions\OBS-Plugin"
   ```

2. Run the deploy script.

   - If you use **OBS built from source** (Step 3) and run it from its build folder:

     ```powershell
     .\scripts\deploy-to-obs.ps1
     ```

     By default this copies to `C:\obs-studio\build_x64\obs-plugins\64bit`.

   - If you use **installed OBS** (e.g. from obsproject.com):

     ```powershell
     .\scripts\deploy-to-obs.ps1 -ObsPlugins "C:\Program Files\obs-studio\obs-plugins\64bit"
     ```

   The script copies:
   - `obs-face-captions.dll` into the OBS plugins folder.
   - **Dependency DLLs** (Vosk and OpenCV from vcpkg) next to the plugin so OBS can load it (avoids "module could not be found" / error 126). Default paths: Vosk `C:\vosk-api`, vcpkg bins `C:\vcpkg\installed\x64-windows\bin`. If yours differ: `-VoskRoot "C:\path\to\vosk-api"` and `-VcpkgBin "C:\vcpkg\installed\x64-windows\bin"`.
   - The `models` folder and `haarcascade_frontalface_default.xml` into OBS’s plugin data folder so the plugin can find them.

3. **Start OBS from the build** (required for the plugin to load): from the `OBS-Plugin` folder run:
   ```powershell
   .\scripts\run-obs-from-build.ps1
   ```
   Do **not** start OBS from the Start Menu or an installer shortcut — that runs the installed OBS, which does not have the plugin or its DLLs, and you will get "Plugin Load Error" (126).

---

## Step 8: Use the filter in OBS

1. Add a source: **Video Capture Device** (your camera).
2. Right-click that source → **Filters** → **+** → **Face Captions**.
3. In the filter settings, leave **Vosk Model Path** empty so the plugin uses the best model (large if you downloaded it, otherwise small).
4. Speak into the mic; captions should appear above your face. If you see “Loading model...”, wait a few seconds for the model to load.

---

## Summary: what you downloaded and where

| Item | Where you got it | Where it ends up |
|------|------------------|-------------------|
| Visual Studio 2022 | microsoft.com | Installed on PC |
| vcpkg | github.com/Microsoft/vcpkg | e.g. `C:\vcpkg` |
| OpenCV | via `vcpkg install opencv4:x64-windows` | Inside vcpkg |
| OBS Studio source | github.com/obsproject/obs-studio | e.g. `C:\obs-studio`, built in `C:\obs-studio\build_x64` |
| Vosk API (Windows zip) | github.com/alphacep/vosk-api/releases | e.g. `C:\vosk-api` |
| Haar cascade | `.\scripts\download_haarcascade.ps1` | `OBS-Plugin\data\` |
| Vosk small model | `.\scripts\download_vosk_model.ps1` | `OBS-Plugin\models\` |
| Vosk large model | `.\scripts\download_vosk_model_large.ps1` | `OBS-Plugin\models\` |
| Plugin DLL | `cmake --build .` in `OBS-Plugin\build` | `OBS-Plugin\build\RelWithDebInfo\obs-face-captions.dll` → then copied by deploy script to OBS’s `obs-plugins\64bit\` |

---

## Troubleshooting

- **“Cannot find libobs” or CMake fails:** Check `OBS_BUILD_DIR` points to the folder that contains `rundir`, `RelWithDebInfo`, etc. (your OBS build folder).
- **“Could NOT find SIMDe”:** Install SIMDe from the **vcpkg folder** (e.g. `cd C:\vcpkg` then `.\vcpkg install simde:x64-windows`). Then reconfigure the plugin. If you don’t use vcpkg for the plugin, set `-DOBS_DEPS_DIR="C:/obs-studio/.deps/obs-deps-2025-08-23-x64"` to point at OBS’s prebuilt deps.
- **“Cannot find OpenCV”:** Make sure you ran `vcpkg install opencv4:x64-windows` and use the same vcpkg path in `CMAKE_TOOLCHAIN_FILE`.
- **“Cannot find Vosk”:** Set `VOSK_ROOT` to the folder that contains `vosk_api.h` and the `.dll`/`.lib` files.
- **OBS crashes when adding the filter:** Ensure the Vosk DLL (e.g. `libvosk.dll`) is next to `obs-face-captions.dll` in the OBS plugins folder. The build copies it when `VOSK_ROOT` is set; you can also copy it manually from your VOSK_ROOT folder.
- **“obs-face-captions failed to load” / “compiled with newer libobs 32.1”:** You are running the **installed** OBS (e.g. 32.0.4) while the plugin was built against your **source-built** OBS (libobs 32.1). OBS will not load a plugin built with a newer libobs. **Fix:** Run OBS from your build folder instead of the installed one. Set the **working directory** to the rundir, then run the exe (so OBS finds `data/obs-studio/locale/`):  
  `Set-Location "C:\obs-studio\build_x64\rundir\RelWithDebInfo"; .\bin\64bit\obs64.exe`  
  (Note the **64bit** subfolder inside `bin`.) If it’s not there, see the next item (AJA build failure).
- **“zlib.dll was not found” when running obs64.exe from build:** The build’s copy step can miss some deps. Copy it from obs-deps:  
  `Copy-Item "C:\obs-studio\.deps\obs-deps-2025-08-23-x64\bin\zlib.dll" -Destination "C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit\" -Force`  
  `Copy-Item "C:\obs-studio\.deps\obs-deps-2025-08-23-x64\bin\zlib.dll" -Destination "C:\obs-studio\build_x64\rundir\RelWithDebInfo\bin\64bit\zlib1.dll" -Force`  
  (Adjust paths if your obs-deps or build folder differ.) For zlib1.dll: the deps bundle only has `zlib.dll`; copy it again as `zlib1.dll` (second command above).   For other missing DLLs, copy from the same deps `bin` or from vcpkg's `installed\x64-windows\bin`.
- **"Failed to find locale/en-US.ini" or "Failed to load locale":** OBS looks for `data/obs-studio/locale/` relative to the **current working directory**. Start OBS from the rundir folder so the app finds `data` there. In PowerShell:  
  `Set-Location "C:\obs-studio\build_x64\rundir\RelWithDebInfo"; .\bin\64bit\obs64.exe`  
  Or: `cd C:\obs-studio\build_x64\rundir\RelWithDebInfo` then run `.\bin\64bit\obs64.exe`.
- **"Failed to initialize video" / "GPU may not be supported":** Run OBS from a **non-Administrator** terminal (e.g. a normal PowerShell window). If it still fails, update your graphics drivers (Intel/NVIDIA/AMD) from the manufacturer’s site.
- **No obs64.exe / OBS build fails on AJA (LNK2019 `__std_find_first_of_trivial_pos_1`):** The AJA plugin can fail to link on some MSVC versions. You can still get the OBS frontend: (1) Try building only the frontend:  
  `cmake --build . --config RelWithDebInfo --target obs-studio`  
  in `C:\obs-studio\build_x64`. If that produces `rundir\RelWithDebInfo\bin\64bit\obs64.exe`, use it. (2) If not, disable AJA: in `C:\obs-studio\plugins\CMakeLists.txt` comment out the block  
  `add_obs_plugin(aja ...)`  
  (the 4 lines for aja), reconfigure (`cmake .. -G "Visual Studio 17 2022" -A x64` in `build_x64`), then run  
  `cmake --build . --config RelWithDebInfo`  
  again. You don’t need AJA for the Face Captions plugin.
- **Face Captions does not appear / Plugin Load Error / error 126:** Run `.\scripts\deploy-to-obs.ps1` from `OBS-Plugin` (it copies the plugin, Vosk, OpenCV, and VC++ runtime into `rundir\RelWithDebInfo\bin\64bit`). Start OBS from the rundir. If it still fails, run `.\scripts\list-plugin-dependencies.ps1` to see which DLLs are missing and copy them into `bin\64bit`. If it still doesn’t show, copy `libvosk.dll` from your `VOSK_ROOT` (or use `-VoskRoot` in the deploy script). Copy into the same folder as `obs-face-captions.dll` (e.g. `rundir\RelWithDebInfo\obs-plugins\64bit`), then restart OBS again.
- **No face detected:** Run `.\scripts\download_haarcascade.ps1` from `OBS-Plugin`, then rebuild and deploy again so the cascade is in the plugin data folder.
- **No speech / “Listening...” forever:** Run at least `.\scripts\download_vosk_model.ps1`, then deploy again. Leave **Vosk Model Path** empty in the filter.
