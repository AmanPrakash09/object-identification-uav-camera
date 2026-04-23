import os
import sys
import math
import random

from carla_api_paths import ensure_carla_on_path

ensure_carla_on_path()

import carla
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_SPACE, K_ESCAPE

NUM_VEHICLES = 10
DELTA_TIME = 0.05
MAX_DISTANCE = 80.0  # Distance before a car gets teleported back
SPAWN_BUBBLE = 15.0    # Minimum distance (meters) between spawned cars

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
        
        # --- HERO CAR BLUEPRINT (Bright Red) ---
        hero_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        if hero_bp.has_attribute('color'):
            hero_bp.set_attribute('color', '255,0,0')

        # --- SMART SPAWNING (Clustered with 5m Bubble) ---
        all_spawn_points = world.get_map().get_spawn_points()
        epicenter = all_spawn_points[36]
        
        all_spawn_points = sorted(all_spawn_points, key=lambda p: p.location.distance(epicenter.location))
        nearby_spawn_points = all_spawn_points[:60] # Expand list to account for skipped spots

        print(f"\nSpawning {NUM_VEHICLES} cars near the epicenter...")
        
        # 1. Spawn the Hero Car FIRST
        hero_car = None
        for spawn_point in all_spawn_points:
            hero_car = world.try_spawn_actor(hero_bp, spawn_point)
            if hero_car:
                spawned_vehicles.append(hero_car)
                actor_list.append(hero_car)
                break # Hero spawned successfully!

        # 2. Spawn the AI Cars
        if hero_car:
            for spawn_point in all_spawn_points:
                if len(spawned_vehicles) >= NUM_VEHICLES:
                    break
                    
                # 5-METER DISTANCE CHECK: Ensure this spawn point isn't too close to an existing car
                too_close = False
                for existing_car in spawned_vehicles:
                    if spawn_point.location.distance(existing_car.get_location()) < SPAWN_BUBBLE:
                        too_close = True
                        break
                
                # If it's too close to another car, skip this spawn point!
                if too_close:
                    continue
                    
                ai_car = world.try_spawn_actor(car_bp, spawn_point)
                if ai_car is not None:
                    spawned_vehicles.append(ai_car)
                    actor_list.append(ai_car)
                    
                    # Turn on Autopilot and make them run red lights
                    ai_car.set_autopilot(True, traffic_manager.get_port())
                    traffic_manager.ignore_lights_percentage(ai_car, 100)
                    
                    # Tell this AI car to completely ignore the Hero car
                    traffic_manager.collision_detection(ai_car, hero_car, False)
                    
        print(f"Successfully spawned 1 Hero Car and {len(spawned_vehicles)-1} blind AI cars!\n")
        
        # --- FIXED BIRD'S EYE VIEW ---
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
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    running = False
                    
            keys = pygame.key.get_pressed()
            
            control = carla.VehicleControl()
            control.throttle = 0.3 if keys[K_w] else 0.0
            control.brake = 0.3 if keys[K_SPACE] else 0.0
            control.reverse = bool(keys[K_s])
            if keys[K_s]: control.throttle = 0.3 
            
            # --- STEERING SMOOTHING ---
            target_steer = -0.55 if keys[K_a] else (0.55 if keys[K_d] else 0.0)
            steer_speed = 0.15 
            if current_steer < target_steer:
                current_steer = min(target_steer, current_steer + steer_speed)
            elif current_steer > target_steer:
                current_steer = max(target_steer, current_steer - steer_speed)
            control.steer = current_steer
            
            # Apply control only if the hero car hasn't been glitched/destroyed
            if hero_car.is_alive:
                hero_car.apply_control(control)
            
            # --- DISTANCE LEASH (RESPAWNING) ---
            for car in spawned_vehicles:
                # CRASH FIX: Make sure the car still physically exists before getting its location
                if not car.is_alive:
                    continue
                    
                if car.get_location().distance(epicenter.location) > MAX_DISTANCE:
                    new_spawn = random.choice(nearby_spawn_points)
                    car.set_transform(new_spawn)
                    car.set_target_velocity(carla.Vector3D(0, 0, 0))
                    car.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

            # --- CALCULATE SPEED FOR DISPLAY ---
            speed_kmh = 0.0
            if hero_car.is_alive:
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
        
        # CRASH FIX: Check if actor is still alive before trying to destroy it
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()

if __name__ == '__main__':
    main()