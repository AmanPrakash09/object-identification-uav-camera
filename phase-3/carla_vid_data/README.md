# CARLA 0.9.10 — dataset & traffic capture

Scripts for **CARLA 0.9.10** that record RGB frames, optional 2D boxes / COCO-style JSON, speed logs, and weather-controlled scenes. Cameras can use different **heights** (fixed, stepped cycle, or smooth oscillation) and **move at a set speed** over the map via `sensor_trajectory.py` (`OverheadSensorTrajectory`).

## Layout

| Path | Purpose |
|------|---------|
| `config/carla_api.example.json` | Copy to `config/carla_api.local.json` (gitignored) to set `dist_dir` or `egg` for the CARLA Python API. |
| `carla_api_paths.py` | Resolves the CARLA `.egg` and runs `ensure_carla_on_path()` before `import carla`. |
| `sensor_trajectory.py` | Shared overhead / patrol motion for RGB sensors. |
| `outputs/ego_vehicle/` | Default run directory for `record_ego_vehicle.py`. |
| `outputs/ai_traffic/` | Default run directory for `record_ai_traffic.py`. |
| `outputs/legacy/` | Older scenario scripts write here. |
| `requirements.txt` | Minimal pip deps for this repo’s scripts (NumPy, OpenCV, Pygame). |
| `requirements-lock.txt` | Full `pip freeze` from a working 3.7 env — use for identical reinstalls (see §2). |

## Requirements

- **OS:** Windows 10 or 11  
- **Python:** **3.7.9** (CARLA 0.9.10 eggs target 3.7; newer interpreters usually cannot load them.)

## 1. Install CARLA 0.9.10

The API and assets target **0.9.10** only; newer CARLA releases are not drop-in compatible.

1. Download the Windows package from the [CARLA 0.9.10 release](https://github.com/carla-simulator/carla/releases/tag/0.9.10) (`CARLA_0.9.10.zip`).
2. Extract it somewhere permanent (any folder is fine).

## 2. Python environment

Use **Python 3.7.x** (64-bit). The CARLA 0.9.10 client is loaded from the official **`.egg`** via `carla_api_paths.py`; it is **not** installed with pip.

1. Install [Python 3.7.9](https://www.python.org/downloads/release/python-379/) (or any 3.7.x) and confirm `python --version` shows 3.7 in a new terminal.
2. From this project directory, create and activate a virtual environment (name is up to you; `.venv` is common):

   ```powershell
   py -3.7 -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install -U pip setuptools wheel
   ```

3. Install dependencies — pick one:
   - **Minimal (recommended for these scripts only):**  
     `pip install -r requirements.txt`
   - **Locked env (matches your saved freeze):**  
     `pip install -r requirements-lock.txt`  
     Use this when you want the same package set and versions you recorded with `pip freeze > requirements-lock.txt`. That file may include extra libraries from the machine where it was generated; that is fine if you rely on those tools.

4. Point `carla_api_paths` at your CARLA install (see §3).

**Refreshing the lockfile** after upgrading packages in an activated venv:

```powershell
pip freeze > requirements-lock.txt
```

**`requirements.txt`** intentionally lists only what the recording scripts import besides the stdlib and `carla`: **NumPy**, **OpenCV** (`opencv-python-headless` by default), and **Pygame**, with version ranges compatible with Python 3.7.

### Optional — CARLA’s *example* dependencies

Only if you run scripts under CARLA’s `PythonAPI\examples` folder — that directory has its **own** `requirements.txt` (not the one in this repo):

```powershell
cd <your-CARLA-extract>\WindowsNoEditor\PythonAPI\examples
pip install -r requirements.txt
```

## 3. CARLA Python API path

Call `ensure_carla_on_path()` from `carla_api_paths.py`. Resolution order:

1. **`CARLA_PYTHON_EGG`** — full path to the `.egg` file.  
2. **`CARLA_DIST_DIR`** — directory containing the matching `carla-*.egg`.  
3. **`config/carla_api.local.json`** (preferred) or **`config/carla_api.json`**, or legacy root **`carla_paths.json`** — `{"egg": "..."}` or `{"dist_dir": "..."}`.  
4. **Default** — `%USERPROFILE%\Downloads\CARLA_0.9.10\WindowsNoEditor\...`.

Copy `config/carla_api.example.json` to `config/carla_api.local.json` and edit paths when CARLA is not in the default location.

## 4. Run the simulator and recorders

**Simulator:**

```powershell
cd <your-CARLA-extract>\WindowsNoEditor
.\CarlaUE4.exe
```

**Recorders** (venv activated, from this directory, with `CarlaUE4.exe` running):

- **`record_ego_vehicle.py`** — One ego vehicle (autopilot or manual), RGB + boxes + `coco_annotations.json`, `speed_log.csv`, optional MP4. Weather and camera motion at the top of the file (`CAMERA_VELOCITY_MPS`, `CAMERA_BOUND_HALF_XY`, `CAMERA_HEIGHT_MODE`, etc.).
- **`record_ai_traffic.py`** — Many AI vehicles; outputs under `outputs/ai_traffic/`, same camera trajectory options. COCO annotations include **`carla_actor_id`** so each box matches **`actors[].id`** in `speeds.jsonl` for that frame (COCO’s own `"id"` is still the annotation counter).

```powershell
python .\record_ego_vehicle.py
python .\record_ai_traffic.py
```

Older scripts in subfolders still use `carla_api_paths` and write to `outputs/legacy/`.
