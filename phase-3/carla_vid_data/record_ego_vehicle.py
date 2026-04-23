"""
Single-ego vehicle dataset capture: RGB, 2D boxes, COCO JSON, speed CSV, optional MP4.

Configure weather, lighting, ``sensor_trajectory.OverheadSensorTrajectory``, and autopilot vs. manual WASD at the top of this file.
"""
from __future__ import annotations

import json
import math
import os
import queue
import sys

from carla_api_paths import ensure_carla_on_path

ensure_carla_on_path()

import carla
import cv2
import numpy as np
import pygame
from sensor_trajectory import OverheadSensorTrajectory
from pygame.locals import K_ESCAPE, K_SPACE, K_a, K_d, K_s, K_w

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join("outputs", "ego_vehicle")
RGB_DIR = os.path.join(LOG_DIR, "rgb")
LABELED_DIR = os.path.join(LOG_DIR, "labeled")

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOV = 90.0

# Seconds per CARLA tick (must match video FPS below if you use MP4 export)
FIXED_DELTA = 0.05
VIDEO_FPS = 1.0 / FIXED_DELTA

# Ego vehicle blueprint (first match). Examples:
#   'vehicle.volkswagen.t2'
#   'vehicle.carlamotors.carlacola'
#   'vehicle.tesla.model3'
VEHICLE_BLUEPRINT_FILTER = "vehicle.volkswagen.t2"

# If True, Traffic Manager drives the ego; WASD is ignored. If False, click the Pygame
# window and drive manually (useful when collecting a specific trajectory).
EGO_USE_AUTOPILOT = True
TRAFFIC_MANAGER_PORT = 8000
# Positive values make AI slower (percent above limit)
GLOBAL_SPEED_DIFF_PERCENT = 30.0

# --- Recording camera (RGB sensor): height profile + optional patrol velocity ---
# World-space velocity in m/s (CARLA X/Y). (0, 0, 0) keeps the sensor over the spawn point.
CAMERA_VELOCITY_MPS = (5.0, 3.0, 0.0)
# Keep the rig within a square of this half-extent (m) around the spawn; None = no clamp.
CAMERA_BOUND_HALF_XY = 180.0
CAMERA_REFLECT_AT_BOUNDS = True
# "fixed" | "cycle" | "oscillate" — altitude is added on top of the spawn point's Z.
CAMERA_HEIGHT_MODE = "oscillate"
CAMERA_HEIGHT_FIXED_M = 28.0
CAMERA_HEIGHT_CYCLE_M = (22.0, 45.0, 75.0)
CAMERA_HEIGHT_CYCLE_FRAMES = 90
CAMERA_HEIGHT_OSC_MIN_M = 22.0
CAMERA_HEIGHT_OSC_MAX_M = 52.0
CAMERA_HEIGHT_OSC_PERIOD_S = 18.0
CAMERA_PITCH_DEG = -90.0
CAMERA_YAW_DEG = 0.0
# If True, sensor yaw aligns with horizontal motion (only affects non-nadir shots).
CAMERA_YAW_FOLLOWS_VELOCITY = False

# --- Weather & lighting (CARLA 0.9.10 WeatherParameters) ---
# Option A: set USE_WEATHER_PRESET to a name from carla.WeatherParameters (string attr).
# Option B: leave USE_WEATHER_PRESET None and edit CUSTOM_WEATHER_* fields.
USE_WEATHER_PRESET = None  # e.g. "HardRainSunset" or "ClearNoon"

CUSTOM_WEATHER = dict(
    cloudiness=40.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=10.0,
    sun_azimuth_angle=45.0,
    sun_altitude_angle=83.0,  # ~noon bright; 15 golden hour; -90 night
    fog_density=2.0,
    fog_distance=0.75,
    wetness=0.0,
)


def apply_weather(world: carla.World) -> None:
    if USE_WEATHER_PRESET:
        preset = getattr(carla.WeatherParameters, USE_WEATHER_PRESET, None)
        if preset is None:
            raise ValueError(f"Unknown weather preset: {USE_WEATHER_PRESET}")
        world.set_weather(preset)
        return
    w = world.get_weather()
    for key, value in CUSTOM_WEATHER.items():
        if hasattr(w, key):
            setattr(w, key, value)
    world.set_weather(w)


# ---------------------------------------------------------------------------
# Geometry helpers (same convention as previous bus/semi_truck scripts)
# ---------------------------------------------------------------------------


def build_projection_matrix(w: int, h: int, fov: float) -> np.ndarray:
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def get_image_point(loc: carla.Location, k: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_c = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    point_2d = np.dot(k, point_c)
    if point_2d[2] > 0.0:
        point_2d = np.array([point_2d[0] / point_2d[2], point_2d[1] / point_2d[2], point_2d[2]])
    return point_2d


def get_bbox_and_draw(
    image: np.ndarray,
    vehicle: carla.Vehicle,
    camera: carla.Actor,
    k: np.ndarray,
    label: str,
) -> tuple[np.ndarray, list[int] | None]:
    bounding_box = vehicle.bounding_box
    transform = vehicle.get_transform()
    w2c = np.array(camera.get_transform().get_inverse_matrix())

    points_2d = []
    for vert in bounding_box.get_world_vertices(transform):
        p2d = get_image_point(vert, k, w2c)
        if p2d[2] > 0:
            points_2d.append((int(p2d[0]), int(p2d[1])))

    if not points_2d:
        return image, None

    xs = [p[0] for p in points_2d]
    ys = [p[1] for p in points_2d]
    x_min = max(0, min(xs))
    x_max = min(IMAGE_WIDTH, max(xs))
    y_min = max(0, min(ys))
    y_max = min(IMAGE_HEIGHT, max(ys))
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return image, None

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(
        image,
        label,
        (x_min, max(15, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return image, [int(x_min), int(y_min), int(width), int(height)]


def main() -> None:
    for directory in (LOG_DIR, RGB_DIR, LABELED_DIR):
        os.makedirs(directory, exist_ok=True)

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((500, 230))
    pygame.display.set_caption("CARLA dataset logger")
    font = pygame.font.SysFont("Arial", 22)

    ego_label = VEHICLE_BLUEPRINT_FILTER.split(".")[-1]
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": ego_label, "supercategory": "vehicle"}],
    }
    annotation_id = 1
    speed_history: list[float] = []

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    apply_weather(world)

    traffic_manager = None
    if EGO_USE_AUTOPILOT:
        traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.global_percentage_speed_difference(GLOBAL_SPEED_DIFF_PERCENT)

    actor_list: list[carla.Actor] = []
    speed_log_file = None
    frame_num = 0

    try:
        speed_log_path = os.path.join(LOG_DIR, "speed_log.csv")
        speed_log_file = open(speed_log_path, "w", encoding="utf-8")
        speed_log_file.write("frame,speed_kmh\n")

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        bps = blueprint_library.filter(VEHICLE_BLUEPRINT_FILTER)
        if not bps:
            raise RuntimeError(f"No blueprint matched filter: {VEHICLE_BLUEPRINT_FILTER!r}")
        vehicle_bp = bps[0]

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        spawn_point.location.z += 1.0

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        if EGO_USE_AUTOPILOT and traffic_manager is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())

        spectator = world.get_spectator()

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(FOV))

        cam_rig = OverheadSensorTrajectory(
            spawn_point.location,
            velocity_mps=CAMERA_VELOCITY_MPS,
            bound_half_extent_xy=CAMERA_BOUND_HALF_XY,
            reflect_at_bounds=CAMERA_REFLECT_AT_BOUNDS,
            pitch_deg=CAMERA_PITCH_DEG,
            yaw_deg=CAMERA_YAW_DEG,
            yaw_follow_velocity=CAMERA_YAW_FOLLOWS_VELOCITY,
            height_mode=CAMERA_HEIGHT_MODE,
            height_fixed_m=CAMERA_HEIGHT_FIXED_M,
            height_cycle_m=CAMERA_HEIGHT_CYCLE_M,
            height_cycle_frames=CAMERA_HEIGHT_CYCLE_FRAMES,
            height_oscillate_min_m=CAMERA_HEIGHT_OSC_MIN_M,
            height_oscillate_max_m=CAMERA_HEIGHT_OSC_MAX_M,
            height_oscillate_period_s=CAMERA_HEIGHT_OSC_PERIOD_S,
        )
        camera = world.spawn_actor(camera_bp, cam_rig.initial_transform(FIXED_DELTA))
        actor_list.append(camera)

        image_queue: queue.Queue = queue.Queue()
        camera.listen(image_queue.put)
        k = build_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV)

        mode = "Autopilot (Traffic Manager)" if EGO_USE_AUTOPILOT else "Manual WASD"
        print(f"\nLogging to {os.path.abspath(LOG_DIR)}")
        print(f"Ego mode: {mode}. Focus the small Pygame window; ESC quits.\n")

        running = True
        while running:
            camera.set_transform(cam_rig.step(FIXED_DELTA))
            world.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    running = False

            if not EGO_USE_AUTOPILOT:
                keys = pygame.key.get_pressed()
                control = carla.VehicleControl()
                control.throttle = 1.0 if keys[K_w] else 0.0
                control.steer = -0.5 if keys[K_a] else (0.5 if keys[K_d] else 0.0)
                control.brake = 1.0 if keys[K_SPACE] else 0.0
                control.reverse = bool(keys[K_s])
                if keys[K_s]:
                    control.throttle = 1.0
                vehicle.apply_control(control)

            car_transform = vehicle.get_transform()
            spectator_offset = (
                -15.0 * car_transform.rotation.get_forward_vector() + carla.Location(z=7.0)
            )
            spectator_tf = carla.Transform(
                car_transform.location + spectator_offset, car_transform.rotation
            )
            spectator_tf.rotation.pitch -= 15.0
            spectator.set_transform(spectator_tf)

            image = image_queue.get()
            img_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            img_bgr = img_array[:, :, :3]
            img_draw = img_bgr.copy()
            img_labeled, bbox = get_bbox_and_draw(img_draw, vehicle, camera, k, ego_label)

            fname = f"frame_{frame_num:05d}.png"
            cv2.imwrite(os.path.join(RGB_DIR, fname), img_bgr)
            cv2.imwrite(os.path.join(LABELED_DIR, fname), img_labeled)

            coco_data["images"].append(
                {
                    "id": frame_num,
                    "file_name": fname,
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                }
            )
            if bbox is not None:
                coco_data["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": frame_num,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

            vel = vehicle.get_velocity()
            speed_kmh = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) * 3.6
            speed_log_file.write(f"{frame_num},{speed_kmh:.2f}\n")
            speed_history.append(speed_kmh)

            display.fill((30, 30, 30))
            display.blit(font.render(mode, True, (200, 200, 200)), (12, 12))
            display.blit(
                font.render("ESC: stop logging", True, (160, 160, 160)),
                (12, 42),
            )
            display.blit(
                font.render(f"Speed: {speed_kmh:.1f} km/h", True, (120, 255, 140)),
                (12, 100),
            )
            cam_tf = camera.get_transform()
            display.blit(
                font.render(
                    f"Cam Z: {cam_tf.location.z:.1f} m  |  XY speed: {math.hypot(CAMERA_VELOCITY_MPS[0], CAMERA_VELOCITY_MPS[1]):.1f} m/s",
                    True,
                    (140, 180, 220),
                ),
                (12, 132),
            )
            pygame.display.flip()
            frame_num += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pygame.quit()

        with open(os.path.join(LOG_DIR, "coco_annotations.json"), "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()

        if speed_log_file:
            speed_log_file.close()

        if frame_num > 0:
            print(f"Writing MP4 ({frame_num} frames)...")
            video_path = os.path.join(LOG_DIR, "telemetry_replay.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (IMAGE_WIDTH, IMAGE_HEIGHT))
            cv2_font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(frame_num):
                p = os.path.join(RGB_DIR, f"frame_{i:05d}.png")
                if not os.path.isfile(p):
                    continue
                frame = cv2.imread(p)
                text = f"{speed_history[i]:.1f} km/h"
                tw, _ = cv2.getTextSize(text, cv2_font, 1, 2)[0]
                tx = IMAGE_WIDTH - tw - 20
                ty = IMAGE_HEIGHT - 20
                cv2.putText(frame, text, (tx, ty), cv2_font, 1, (0, 0, 0), 4)
                cv2.putText(frame, text, (tx, ty), cv2_font, 1, (255, 255, 255), 2)
                out.write(frame)
            out.release()
            print(f"Video: {video_path}")

        print("Done.")


if __name__ == "__main__":
    main()
