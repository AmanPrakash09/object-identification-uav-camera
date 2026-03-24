import glob
import os
import sys
import math
import json

# Hunt down the pre-compiled CARLA library
try:
    sys.path.append(glob.glob('../../../../Downloads/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2
import queue
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_SPACE, K_ESCAPE

# --- Configuration ---
LOG_DIR = "_out_logs"
RGB_DIR = os.path.join(LOG_DIR, "rgb")
LABELED_DIR = os.path.join(LOG_DIR, "labeled")
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOV = 90.0

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_c = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    point_2d = np.dot(K, point_c)
    
    if point_2d[2] > 0.0:
        point_2d = np.array([point_2d[0] / point_2d[2], point_2d[1] / point_2d[2], point_2d[2]])
    return point_2d

def get_bbox_and_draw(image, vehicle, camera, K):
    bounding_box = vehicle.bounding_box
    transform = vehicle.get_transform()
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    
    verts = [loc for loc in bounding_box.get_world_vertices(transform)]
    points_2d = []
    for vert in verts:
        p2d = get_image_point(vert, K, w2c)
        if p2d[2] > 0: 
            points_2d.append((int(p2d[0]), int(p2d[1])))
    
    if len(points_2d) > 0:
        xs = [p[0] for p in points_2d]
        ys = [p[1] for p in points_2d]
        
        x_min = max(0, min(xs))
        x_max = min(IMAGE_WIDTH, max(xs))
        y_min = max(0, min(ys))
        y_max = min(IMAGE_HEIGHT, max(ys))
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width > 0 and height > 0:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, 'Motorcycle', (x_min, max(15, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return image, [int(x_min), int(y_min), int(width), int(height)]
            
    return image, None

def main():
    # Setup directories
    for directory in [LOG_DIR, RGB_DIR, LABELED_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((450, 200))
    pygame.display.set_caption("CARLA Control")
    font = pygame.font.SysFont('Arial', 24)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "motorcycle", "supercategory": "vehicle"}]
    }
    annotation_id = 1
    
    speed_history = []

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    custom_weather = world.get_weather()
    custom_weather.sun_altitude_angle = 83.0 
    custom_weather.cloudiness = 87.0       
    custom_weather.precipitation = 0.0   
    custom_weather.fog_density = 2.0      
    world.set_weather(custom_weather)
    
    actor_list = []
    speed_log_file = None

    try:
        speed_log_path = os.path.join(LOG_DIR, "speed_log.csv")
        speed_log_file = open(speed_log_path, "w")
        speed_log_file.write("frame,speed_kmh\n")

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # --- SPAWN MOTORCYCLE ---
        vehicle_bp = blueprint_library.filter('vehicle.yamaha.yzf')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Multiply by 2 because 'extent' is only half the size!
        bbox = vehicle.bounding_box
        car_length = bbox.extent.x * 2.0
        car_width = bbox.extent.y * 2.0
        car_height = bbox.extent.z * 2.0
        
        print(f"\n--- VEHICLE SPECS ---")
        print(f"Vehicle: Yamaha YZF Motorcycle")
        print(f"Length: {car_length:.2f} meters")
        print(f"Width:  {car_width:.2f} meters")
        print(f"Height: {car_height:.2f} meters")
        print(f"---------------------\n")
        
        spectator = world.get_spectator()

        # Spawn Recording Camera (Kept at 25m for better detail on the smaller bike)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        camera_bp.set_attribute('fov', str(FOV))

        camera_transform = carla.Transform(
            carla.Location(x=spawn_point.location.x, y=spawn_point.location.y, z=spawn_point.location.z + 25.0),
            carla.Rotation(pitch=-90.0) 
        )
        camera = world.spawn_actor(camera_bp, camera_transform)
        actor_list.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        K = build_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV)

        frame_num = 0
        running = True

        print("\nReady! CLICK THE SMALL PYGAME WINDOW to drive using WASD. Press ESC to stop.\n")
        
        while running:
            world.tick()
            
            # --- PYGAME KEYBOARD CONTROL ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    running = False
                    
            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()
            control.throttle = 1.0 if keys[K_w] else 0.0
            control.steer = -0.5 if keys[K_a] else (0.5 if keys[K_d] else 0.0)
            control.brake = 1.0 if keys[K_SPACE] else 0.0
            control.reverse = bool(keys[K_s])
            if keys[K_s]: control.throttle = 1.0 
                
            vehicle.apply_control(control)

            # --- SPECTATOR FOLLOWS MOTORCYCLE ---
            car_transform = vehicle.get_transform()
            # Moved closer: 7 meters back, 3 meters up
            spectator_offset = -7.0 * car_transform.rotation.get_forward_vector() + carla.Location(z=3.0)
            spectator_transform = carla.Transform(car_transform.location + spectator_offset, car_transform.rotation)
            spectator_transform.rotation.pitch -= 15.0 
            spectator.set_transform(spectator_transform)

            # --- IMAGE PROCESSING ---
            image = image_queue.get()
            img_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            img_bgr = img_array[:, :, :3]
            img_draw = img_bgr.copy()
            img_with_box, bbox = get_bbox_and_draw(img_draw, vehicle, camera, K)

            # --- LOGGING ---
            filename = f"frame_{frame_num:05d}.png"
            cv2.imwrite(os.path.join(RGB_DIR, filename), img_bgr)
            cv2.imwrite(os.path.join(LABELED_DIR, filename), img_with_box)
            
            coco_data["images"].append({"id": frame_num, "file_name": filename, "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT})
            if bbox is not None:
                coco_data["annotations"].append({"id": annotation_id, "image_id": frame_num, "category_id": 1, "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
                annotation_id += 1

            # --- SPEED MEASUREMENT ---
            velocity = vehicle.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_ms * 3.6
            speed_log_file.write(f"{frame_num},{speed_kmh:.2f}\n")
            
            speed_history.append(speed_kmh)
            
            # --- PYGAME DISPLAY ---
            display.fill((30, 30, 30))
            display.blit(font.render("🚨 CLICK THIS WINDOW TO DRIVE 🚨", True, (255, 80, 80)), (20, 20))
            display.blit(font.render("W: Gas | S: Reverse | A/D: Steer", True, (200, 200, 200)), (20, 70))
            display.blit(font.render(f"Speed: {speed_kmh:.2f} km/h", True, (100, 255, 100)), (20, 120))
            pygame.display.flip()
            
            frame_num += 1

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        print("\nCleaning up actors...")
        pygame.quit()
        
        with open(os.path.join(LOG_DIR, "coco_annotations.json"), "w") as json_file:
            json.dump(coco_data, json_file, indent=4)
            
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for actor in actor_list:
            actor.destroy()
            
        if speed_log_file:
            speed_log_file.close()

        # --- VIDEO GENERATION ---
        if frame_num > 0:
            print(f"Compiling {frame_num} frames into MP4 video. This might take a few seconds...")
            video_path = os.path.join(LOG_DIR, "telemetry_replay.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out_video = cv2.VideoWriter(video_path, fourcc, 20.0, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
            cv2_font = cv2.FONT_HERSHEY_SIMPLEX
            
            for i in range(frame_num):
                img_path = os.path.join(RGB_DIR, f"frame_{i:05d}.png")
                if os.path.exists(img_path):
                    frame = cv2.imread(img_path)
                    
                    text = f"{speed_history[i]:.1f} km/h"
                    text_size = cv2.getTextSize(text, cv2_font, 1, 2)[0]
                    text_x = IMAGE_WIDTH - text_size[0] - 20
                    text_y = IMAGE_HEIGHT - 20
                    
                    cv2.putText(frame, text, (text_x, text_y), cv2_font, 1, (0, 0, 0), 4)
                    cv2.putText(frame, text, (text_x, text_y), cv2_font, 1, (255, 255, 255), 2)
                    
                    out_video.write(frame)
                    
            out_video.release()
            print(f"Video saved successfully to {video_path}")
            
        print("Done. Dataset and video are ready!")

if __name__ == '__main__':
    main()