# Calibration Folder

This folder stores calibration artifacts used by the homography-based speed estimation pipeline.

## Purpose

The speed-estimation workflow needs a mapping from image coordinates to a local ground-plane coordinate system.  
That mapping is created from a manually selected reference vehicle and saved here so calibration does not need to be repeated every run.

In this project, calibration is based on:

- one selected reference vehicle in a calibration frame
- four manually chosen tire / road-contact corner points
- assumed real-world dimensions for that reference vehicle
- a homography that maps image points into local metric coordinates

## Files

### `*.json`
Human-readable calibration metadata.

Example:
- `single_average_passenger_car_bev.json`

This file stores:
- calibration name
- selected image points
- destination world points
- image-to-world homography
- world-to-image homography
- reference vehicle dimensions
- optional metadata such as calibration frame index

Use this file when you want to inspect or version-control calibration information.

### `*.npz`
Compact NumPy archive containing the same core arrays.

Example:
- `single_average_passenger_car_bev.npz`

This file stores:
- `image_points`
- `world_points`
- `H_img_to_world`
- `H_world_to_img`

Use this file for faster programmatic loading in notebooks or scripts.

## Current calibration workflow

The calibration is created from the notebook roughly as follows:

1. choose a calibration frame
2. manually click 4 tire-corner / road-contact points of one reference vehicle
3. define the assumed real-world dimensions of that vehicle
4. compute the homography
5. save the calibration artifacts into this folder

## Point ordering convention

The saved point order must stay consistent between image points and world points.

Current convention:

1. front-left tire location
2. front-right tire location
3. rear-right tire location
4. rear-left tire location

If the click order changes, the saved calibration will become invalid.

## Important notes

- A calibration is only valid for the same camera setup or same visual geometry assumptions.
- If the viewpoint, altitude, zoom, crop, or scene geometry changes significantly, create a new calibration.
- For bird’s-eye-view footage, clicking the vehicle footprint is more stable than clicking visible roof corners.
- Calibration quality strongly affects speed quality.

## Naming convention

Recommended pattern:

`<scene_or_video_name>_<view_type>.json`
`<scene_or_video_name>_<view_type>.npz`

Example:
- `single_average_passenger_car_bev.json`
- `single_average_passenger_car_bev.npz`

## When to create a new calibration

Create a new calibration when:

- the video comes from a different camera position
- the scene changes
- the field of view changes
- the old calibration produces implausible speeds
- you want separate calibrations for different datasets

## Summary

This folder contains reusable homography calibration artifacts that allow the pipeline to convert tracked image-space motion into approximate ground-plane motion in meters.