"""
Multi-vehicle Traffic Manager capture: bird's-eye RGB, optional motion and varying altitude.

Writes under ``LOG_DIR`` (default ``outputs/ai_traffic``):
  rgb/               — raw PNGs
  labeled/           — same frames with 2D boxes + type names
  coco_annotations.json — COCO-style (multiple vehicles per frame)
  speeds.jsonl       — one JSON object per line: frame + per-actor speeds
"""
from __future__ import annotations

import json
import math
import os
import queue
import random
import sys

from carla_api_paths import ensure_carla_on_path

ensure_carla_on_path()

import carla
import cv2
import numpy as np
import pygame
from sensor_trajectory import OverheadSensorTrajectory
from pygame.locals import K_ESCAPE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join("outputs", "ai_traffic")
RGB_DIR = os.path.join(LOG_DIR, "rgb")
LABELED_DIR = os.path.join(LOG_DIR, "labeled")

NUM_VEHICLES = 22
SPAWN_BUBBLE = 12.0
FIXED_DELTA = 0.05
VIDEO_FPS = 1.0 / FIXED_DELTA

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOV = 75.0

# Recording camera: patrol velocity (m/s world X/Y) and altitude profile (see sensor_trajectory.py).
CAMERA_VELOCITY_MPS = (10.0, 0.0, 0.0)
CAMERA_BOUND_HALF_XY = 220.0
CAMERA_REFLECT_AT_BOUNDS = True
CAMERA_HEIGHT_MODE = "cycle"
CAMERA_HEIGHT_FIXED_M = 55.0
CAMERA_HEIGHT_CYCLE_M = (38.0, 55.0, 85.0, 120.0)
CAMERA_HEIGHT_CYCLE_FRAMES = 80
CAMERA_HEIGHT_OSC_MIN_M = 40.0
CAMERA_HEIGHT_OSC_MAX_M = 95.0
CAMERA_HEIGHT_OSC_PERIOD_S = 24.0
CAMERA_PITCH_DEG = -90.0
CAMERA_YAW_DEG = 0.0
CAMERA_YAW_FOLLOWS_VELOCITY = False

TRAFFIC_MANAGER_PORT = 8000
GLOBAL_SPEED_DIFF_PERCENT = 75.0  # slow crawl at intersection

USE_WEATHER_PRESET = None
CUSTOM_WEATHER = dict(
    cloudiness=55.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=15.0,
    sun_azimuth_angle=200.0,
    sun_altitude_angle=35.0,
    fog_density=5.0,
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


def bbox_for_actor(
    image: np.ndarray,
    actor: carla.Actor,
    camera: carla.Actor,
    k: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> list[int] | None:
    bb = actor.bounding_box
    transform = actor.get_transform()
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    pts = []
    for vert in bb.get_world_vertices(transform):
        p2d = get_image_point(vert, k, w2c)
        if p2d[2] > 0:
            pts.append((int(p2d[0]), int(p2d[1])))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min = max(0, min(xs))
    x_max = min(IMAGE_WIDTH, max(xs))
    y_min = max(0, min(ys))
    y_max = min(IMAGE_HEIGHT, max(ys))
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(
        image,
        label[:18],
        (x_min, max(14, y_min - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return [int(x_min), int(y_min), int(w), int(h)]


def random_vehicle_blueprint(library: carla.BlueprintLibrary) -> carla.ActorBlueprint:
    all_vehs = list(library.filter("vehicle.*"))
    skip = ("bike", "motor", "yamaha", "harley", "kawasaki", "vespa", "razor", "diamond", "gopher", "omafiets")
    pool = [bp for bp in all_vehs if not any(s in bp.id.lower() for s in skip)]
    if not pool:
        pool = all_vehs
    return random.choice(pool)


def main() -> None:
    for d in (LOG_DIR, RGB_DIR, LABELED_DIR):
        os.makedirs(d, exist_ok=True)

        pygame.init()
        display = pygame.display.set_mode((460, 150))
    pygame.display.set_caption("AI swarm recorder — ESC to stop")
    font = pygame.font.SysFont("Arial", 20)

    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    apply_weather(world)

    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_synchronous_mode(True)
    tm.global_percentage_speed_difference(GLOBAL_SPEED_DIFF_PERCENT)

    actor_list: list[carla.Actor] = []
    vehicles: list[carla.Vehicle] = []
    type_to_cat: dict[str, int] = {}
    categories: list[dict] = []
    coco_images: list[dict] = []
    coco_ann: list[dict] = []
    ann_id = 1

    speeds_file = open(os.path.join(LOG_DIR, "speeds.jsonl"), "w", encoding="utf-8")
    frame_num = 0

    def register_type(type_id: str) -> int:
        if type_id not in type_to_cat:
            cid = len(type_to_cat) + 1
            type_to_cat[type_id] = cid
            short = type_id.split(".")[-1]
            categories.append({"id": cid, "name": short, "supercategory": "vehicle"})
        return type_to_cat[type_id]

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        epicenter = spawn_points[len(spawn_points) // 2]

        ordered = sorted(
            spawn_points,
            key=lambda p: p.location.distance(epicenter.location),
        )

        for sp in ordered:
            if len(vehicles) >= NUM_VEHICLES:
                break
            too_close = any(
                sp.location.distance(v.get_location()) < SPAWN_BUBBLE for v in vehicles
            )
            if too_close:
                continue
            bp = random_vehicle_blueprint(bp_lib)
            veh = world.try_spawn_actor(bp, sp)
            if veh is None:
                continue
            vehicles.append(veh)
            actor_list.append(veh)
            veh.set_autopilot(True, tm.get_port())
            tm.ignore_lights_percentage(veh, 100)
            register_type(veh.type_id)

        if not vehicles:
            raise RuntimeError("Failed to spawn any vehicles.")

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        cam_bp.set_attribute("fov", str(FOV))
        cam_rig = OverheadSensorTrajectory(
            epicenter.location,
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
        camera = world.spawn_actor(cam_bp, cam_rig.initial_transform(FIXED_DELTA))
        actor_list.append(camera)

        spec = world.get_spectator()
        spec.set_transform(cam_rig.initial_transform(FIXED_DELTA))

        q: queue.Queue = queue.Queue()
        camera.listen(q.put)
        k = build_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV)

        colors = [
            (0, 255, 100),
            (255, 128, 0),
            (0, 200, 255),
            (255, 0, 180),
            (220, 220, 50),
            (180, 100, 255),
        ]

        print(f"\nSpawned {len(vehicles)} AI vehicles. Logging to {os.path.abspath(LOG_DIR)}")
        print("Focus the Pygame window; press ESC to stop.\n")

        running = True
        while running:
            tf = cam_rig.step(FIXED_DELTA)
            camera.set_transform(tf)
            spec.set_transform(tf)
            world.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    running = False

            img = q.get()
            arr = np.reshape(np.copy(img.raw_data), (img.height, img.width, 4))
            rgb = arr[:, :, :3]
            labeled = rgb.copy()

            actors_payload = []
            for i, veh in enumerate(vehicles):
                if not veh.is_alive:
                    continue
                short = veh.type_id.split(".")[-1]
                col = colors[i % len(colors)]
                bbox = bbox_for_actor(labeled, veh, camera, k, col, short)
                cid = register_type(veh.type_id)
                vel = veh.get_velocity()
                skm = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) * 3.6
                actors_payload.append(
                    {"id": veh.id, "type_id": veh.type_id, "speed_kmh": round(skm, 2)}
                )
                if bbox is not None:
                    coco_ann.append(
                        {
                            "id": ann_id,
                            "image_id": frame_num,
                            "category_id": cid,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            # CARLA actor id — matches ``actors[].id`` in speeds.jsonl for the same frame
                            "carla_actor_id": veh.id,
                        }
                    )
                    ann_id += 1

            fname = f"frame_{frame_num:05d}.png"
            cv2.imwrite(os.path.join(RGB_DIR, fname), rgb)
            cv2.imwrite(os.path.join(LABELED_DIR, fname), labeled)
            coco_images.append(
                {
                    "id": frame_num,
                    "file_name": fname,
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                }
            )
            speeds_file.write(
                json.dumps({"frame": frame_num, "actors": actors_payload}, separators=(",", ":"))
                + "\n"
            )

            display.fill((24, 24, 24))
            display.blit(font.render(f"Frame {frame_num}", True, (200, 200, 200)), (10, 12))
            display.blit(font.render(f"Vehicles: {len(vehicles)}", True, (160, 220, 160)), (10, 50))
            display.blit(font.render("ESC — stop", True, (150, 150, 150)), (10, 82))
            display.blit(
                font.render(
                    f"Cam Z {tf.location.z:.0f} m   patrol {math.hypot(CAMERA_VELOCITY_MPS[0], CAMERA_VELOCITY_MPS[1]):.1f} m/s",
                    True,
                    (130, 170, 200),
                ),
                (10, 112),
            )
            pygame.display.flip()
            frame_num += 1

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        pygame.quit()
        speeds_file.close()

        with open(os.path.join(LOG_DIR, "coco_annotations.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"images": coco_images, "annotations": coco_ann, "categories": categories},
                f,
                indent=2,
            )

        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for a in actor_list:
            if a.is_alive:
                a.destroy()

        print(f"Saved {frame_num} frames under {LOG_DIR}")


if __name__ == "__main__":
    main()
