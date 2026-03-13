# Exports Folder

This folder stores structured outputs from the detection + tracking + speed-estimation pipeline.

## Purpose

The notebook produces more than just an annotated video.  
This folder keeps tabular and summary artifacts that make it easier to:

- inspect speed-estimation results
- debug tracks
- compare runs
- analyze trajectories later without rerunning the notebook
- export results into other tools

## Files

### `*_run_config.json`
Stores the configuration used for a specific run.

Example:
- `single_average_passenger_car_bev_run_run_config.json`

Typical contents:
- run name
- representative point mode
- border margin
- smoothing settings
- speed filtering settings
- reference vehicle dimensions
- calibration name

Use this file to understand how a run was produced.

---

### `*_track_observations.csv`
Stores one row per saved track observation.

Example:
- `single_average_passenger_car_bev_run_track_observations.csv`

Typical columns include:
- `track_id`
- `obs_index`
- `frame_idx`
- `class_id`
- `confidence`
- `image_x`
- `image_y`
- `world_x_m`
- `world_y_m`
- `raw_speed_kmh`
- `smoothed_speed_kmh`
- `windowed_raw_speed_kmh`
- `windowed_smoothed_speed_kmh`
- `final_filtered_speed_kmh`
- `final_display_speed_kmh`

Use this file when you want frame-level detail for each track.

---

### `*_track_summary.json`
Stores compact per-track summaries.

Example:
- `single_average_passenger_car_bev_run_track_summary.json`

Typical contents for each track:
- track ID
- class ID
- confidence
- number of observations
- first frame
- last frame
- total traveled distance
- mean final speed
- max final speed
- latest final speed

Use this file for quick run-level inspection.

## Interpretation notes

### Observation rows
Each row in the CSV corresponds to one accepted observation stored in the track history.

That means:
- not every video frame is guaranteed to appear
- border-rejected frames may be missing
- only frames used by the speed-estimation logic are exported

### Speeds
The exported files may contain multiple speed versions:

- `raw_speed_kmh`  
  Direct speed from consecutive observations

- `smoothed_speed_kmh`  
  Trailing average of raw speeds

- `windowed_*`  
  Speeds computed using a wider frame gap for better stability

- `final_*`  
  Speeds after additional filtering and display stabilization

In most cases, `final_display_speed_kmh` is the value intended for final reporting or overlay.

## Current export workflow

A typical run generates these outputs after:

1. detection
2. tracking
3. representative-point extraction
4. homography projection
5. speed computation
6. stabilization / filtering
7. export to CSV + JSON

## Naming convention

Recommended pattern:

`<run_name>_run_config.json`  
`<run_name>_track_observations.csv`  
`<run_name>_track_summary.json`

Example:
- `single_average_passenger_car_bev_run_run_config.json`
- `single_average_passenger_car_bev_run_track_observations.csv`
- `single_average_passenger_car_bev_run_track_summary.json`

## Suggested usage

Use this folder for:

- checking whether a speed spike came from a specific frame
- comparing different representative-point modes
- comparing different smoothing settings
- tracking run-to-run calibration changes
- building plots outside the notebook

## Summary

This folder stores machine-readable outputs from each speed-estimation run so results can be inspected, compared, and reused without rerunning the full notebook.