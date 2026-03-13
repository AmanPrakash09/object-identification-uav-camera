# Speed Estimation Runs Folder

This folder stores full output bundles from end-to-end speed-estimation runs.

## Purpose

Each subfolder in this directory corresponds to one complete pipeline run using:

- object detection
- multi-frame tracking
- homography-based projection
- speed estimation
- final annotated video export
- structured CSV / JSON results

This folder is meant to keep each run self-contained so results from different videos, calibration settings, or parameter choices can be compared easily.

## Structure

Each run should be stored in its own subdirectory.

Example:
- `custom_video_bev_e2e/`

A typical run directory may contain files such as:

- annotated output video
- per-track observation CSV
- per-track summary JSON
- run configuration JSON

Example pattern:
- `<run_name>_annotated.mp4`
- `<run_name>_track_observations.csv`
- `<run_name>_track_summary.json`
- `<run_name>_run_config.json`

## Why use per-run subfolders

Grouping outputs by run makes it easier to:

- compare experiments
- avoid overwriting previous results
- trace a result back to its configuration
- keep videos and structured outputs together
- organize runs by dataset, view type, or calibration choice

## Recommended naming

Use a descriptive run name that captures the scenario.

Suggested pattern:
- `<video_or_scene_name>_<view_type>_<pipeline_tag>`

Examples:
- `custom_video_bev_e2e`
- `single_average_passenger_car_bev_e2e`
- `uav0000120_oblique_e2e`

Where:
- `bev` = bird’s-eye view
- `e2e` = end-to-end pipeline run

## Typical workflow

A run folder is usually created after:

1. selecting input source  
   - video path or frames directory

2. loading calibration  
   - saved homography / saved reference points

3. running the detection + tracking + speed-estimation pipeline

4. exporting final outputs into a dedicated run directory

## Relationship to other folders

### `calibration/`
Stores reusable calibration artifacts such as:
- reference image points
- world points
- homography matrices

These are reused across runs when the camera/setup is unchanged.

### `exports/`
Stores standalone exported CSV / JSON artifacts from notebook experiments.

### `videos/speed-estimation-runs/`
Stores complete run bundles, usually including:
- annotated video
- exported observations
- exported summaries
- run configuration

So this folder is the most complete record of a finished speed-estimation experiment.

## Suggested contents for each run folder

Each run folder should ideally include:

- final annotated video
- track observations CSV
- track summary JSON
- run configuration JSON
- optional notes file if the run is experimental

Optional extras:
- debug plots
- trajectory visualizations
- alternative output videos
- screenshots of calibration frame

## Best practices

- Create a new subfolder for each new run instead of overwriting old outputs.
- Keep run names short but descriptive.
- Reuse saved calibration only when camera geometry is unchanged.
- If parameters change significantly, save a new run instead of replacing the old one.
- Use the run config JSON to make results reproducible.

## Current contents

Example:
- `custom_video_bev_e2e/`

This indicates an end-to-end run for a custom bird’s-eye-view video.

## Summary

This folder is the main storage location for complete speed-estimation pipeline runs.  
Each subfolder acts as a reproducible experiment record containing both visual outputs and structured data.