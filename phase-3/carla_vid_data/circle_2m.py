import os
import sys
import math
import random

from carla_api_paths import ensure_carla_on_path

ensure_carla_on_path()

import carla
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_SPACE, K_ESCAPE

NUM_VEHICLES = 20
DELTA_TIME = 0.05
MAX_DISTANCE = 90.0  # Distance before a car gets teleported back

def main():
    # Force Pygame window to the center of your screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = "center"
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("CARLA Hero Control (Top-Down)")
    font = pygame.font.SysFont('Arial', 24)

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # --- SETUP TRAFFIC MANAGER (For AI Cars) ---
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    # Slow the AI down by 80% so they crawl around the intersection
    traffic_manager.global_percentage_speed_difference(80.0)

    actor_list = []
    spawned_vehicles = []
    current_steer = 0.0  # Used to smooth out the steering

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DELTA_TIME
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        car_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        # --- SMART SPAWNING (Clustered) ---
        all_spawn_points = world.get_map().get_spawn_points()
        epicenter = all_spawn_points[0]
        
        all_spawn_points = sorted(all_spawn_points, key=lambda p: p.location.distance(epicenter.location))
        
        # Save a list of the closest spawn points for when we need to respawn cars
        nearby_spawn_points = all_spawn_points[:40] 

        print(f"\nSpawning {NUM_VEHICLES} cars near the epicenter...")
        for i, spawn_point in enumerate(all_spawn_points):
            if len(spawned_vehicles) >= NUM_VEHICLES:
                break
                
            vehicle = world.try_spawn_actor(car_bp, spawn_point)
            if vehicle is not None:
                spawned_vehicles.append(vehicle)
                actor_list.append(vehicle)
                
                # The first car is YOURS (Hero). The rest are AI.
                if i > 0:
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    traffic_manager.ignore_lights_percentage(vehicle, 100)
                
        print(f"Successfully spawned 1 Hero Car and {len(spawned_vehicles)-1} AI cars!\n")
        
        hero_car = spawned_vehicles[0]
        
        # --- FIXED BIRD'S EYE VIEW ---
        # Set the spectator directly above the epicenter, 50m up, looking straight down.
        spectator = world.get_spectator()
        spectator_transform = carla.Transform(
            carla.Location(x=epicenter.location.x, y=epicenter.location.y, z=epicenter.location.z + 50.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )
        spectator.set_transform(spectator_transform)

        running = True
        print("Drive the Hero Car using WASD in the Pygame window. Press ESC to stop.\n")
        
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
            # Cap the throttle at 0.3 (30%) so it isn't a rocket ship
            control.throttle = 0.3 if keys[K_w] else 0.0
            control.brake = 0.3 if keys[K_SPACE] else 0.0
            control.reverse = bool(keys[K_s])
            
            if keys[K_s]: 
                control.throttle = 0.3 
            
            # --- STEERING SMOOTHING ---
            target_steer = -0.55 if keys[K_a] else (0.55 if keys[K_d] else 0.0)
            
            # Gradually glide current_steer toward the target_steer
            steer_speed = 0.15  # How fast the "wheel" turns
            if current_steer < target_steer:
                current_steer = min(target_steer, current_steer + steer_speed)
            elif current_steer > target_steer:
                current_steer = max(target_steer, current_steer - steer_speed)
                
            control.steer = current_steer
            
            # Apply control to the Hero Car
            hero_car.apply_control(control)
            
            # --- DISTANCE LEASH (RESPAWNING) ---
            for car in spawned_vehicles:
                # If ANY car (even you) drives further than 200m from the center...
                if car.get_location().distance(epicenter.location) > MAX_DISTANCE:
                    # Pick a random point near the epicenter and teleport it there
                    new_spawn = random.choice(nearby_spawn_points)
                    car.set_transform(new_spawn)
                    # Kill its momentum so it doesn't go flying out of the new spawn
                    car.set_target_velocity(carla.Vector3D(0, 0, 0))
                    car.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

            # --- CALCULATE SPEED FOR DISPLAY ---
            velocity = hero_car.get_velocity()
            speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # --- PYGAME DISPLAY ---
            display.fill((30, 30, 30))
            display.blit(font.render("🚨 HERO CAR CONTROL 🚨", True, (255, 150, 0)), (20, 20))
            display.blit(font.render("W: Gas | S: Reverse | A/D: Steer", True, (200, 200, 200)), (20, 70))
            display.blit(font.render(f"Speed: {speed_kmh:.2f} km/h", True, (100, 255, 100)), (20, 120))
            pygame.display.flip()

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        print("\nCleaning up actors...")
        pygame.quit()
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    main()