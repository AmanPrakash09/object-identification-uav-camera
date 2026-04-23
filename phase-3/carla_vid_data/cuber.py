import os
import sys
import math
import random

from carla_api_paths import ensure_carla_on_path

ensure_carla_on_path()

import carla
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_SPACE, K_ESCAPE

# --- Driving & Traffic Configuration ---
SMOOTHING_WINDOW = 10  
MAX_THROTTLE = 0.3     
MAX_STEER = 0.4        
NUM_NPC_CARS = 30      # How many self-driving cars to spawn

def main():
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((450, 200))
    pygame.display.set_caption("CARLA Top-Down Control")
    font = pygame.font.SysFont('Arial', 24)

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # Clear, bright weather for driving
    custom_weather = world.get_weather()
    custom_weather.sun_altitude_angle = 83.0 
    custom_weather.cloudiness = 0.0       
    custom_weather.precipitation = 0.0   
    custom_weather.fog_density = 0.0      
    world.set_weather(custom_weather)
    
    actor_list = []
    
    # Input history for smoothed driving
    throttle_history = []
    steer_history = []
    brake_history = []

    try:
        # --- TRAFFIC MANAGER SETUP ---
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        # --- SPAWN YOUR MODEL 3 ---
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0] 
        player_spawn = spawn_points[3] # Grab the very first spawn point for you
        player_spawn.location.z += 1.0 
        
        vehicle = world.spawn_actor(vehicle_bp, player_spawn)
        actor_list.append(vehicle)

        # --- SPAWN AI TRAFFIC (ONLY MODEL 3s) ---
        print(f"\nSpawning {NUM_NPC_CARS} AI vehicles...")
        
        # CHANGED: Filter strictly for the Tesla Model 3 instead of all 4-wheeled vehicles
        car_blueprints = blueprint_library.filter('vehicle.tesla.model3')
        
        # Shuffle remaining spawn points so traffic is random
        random.shuffle(spawn_points)
        
        spawned_npcs = 0
        for spawn_point in spawn_points:
            if spawned_npcs >= NUM_NPC_CARS:
                break
                
            # Make sure we don't spawn an AI car directly on top of your Tesla
            if spawn_point.location.distance(player_spawn.location) < 8.0:
                continue

            npc_bp = random.choice(car_blueprints)
            
            # Try to spawn the car (sometimes spawn points collide, so we use try/except)
            try:
                npc_vehicle = world.spawn_actor(npc_bp, spawn_point)
                npc_vehicle.set_autopilot(True, traffic_manager.get_port())
                actor_list.append(npc_vehicle)
                spawned_npcs += 1
            except Exception:
                pass
                
        print(f"Successfully spawned {spawned_npcs} Tesla Model 3s!")
        
        # --- SET STATIC BIRD'S EYE VIEW (30m UP) ---
        spectator = world.get_spectator()
        static_transform = carla.Transform(
            player_spawn.location + carla.Location(z=30.0),
            carla.Rotation(pitch=-90.0, yaw=player_spawn.rotation.yaw, roll=0.0)
        )
        spectator.set_transform(static_transform)

        running = True
        print("\nReady! CLICK THE SMALL PYGAME WINDOW to drive using WASD.")
        print("The CARLA window is locked 30m above the spawn point. Press ESC to stop.\n")
        
        while running:
            world.tick()
            
            # --- PYGAME KEYBOARD CONTROL ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    running = False
                    
            keys = pygame.key.get_pressed()
            
            # 1. Get RAW inputs
            raw_throttle = MAX_THROTTLE if keys[K_w] else 0.0
            raw_steer = -MAX_STEER if keys[K_a] else (MAX_STEER if keys[K_d] else 0.0)
            raw_brake = 1.0 if keys[K_SPACE] else 0.0
            raw_reverse = bool(keys[K_s])
            
            if keys[K_s]: 
                raw_throttle = MAX_THROTTLE 
                
            # 2. Append to history lists
            throttle_history.append(raw_throttle)
            steer_history.append(raw_steer)
            brake_history.append(raw_brake)
            
            # 3. Trim lists to the maximum window size
            if len(throttle_history) > SMOOTHING_WINDOW: throttle_history.pop(0)
            if len(steer_history) > SMOOTHING_WINDOW: steer_history.pop(0)
            if len(brake_history) > SMOOTHING_WINDOW: brake_history.pop(0)
            
            # 4. Calculate the smoothed averages
            smooth_throttle = sum(throttle_history) / len(throttle_history)
            smooth_steer = sum(steer_history) / len(steer_history)
            smooth_brake = sum(brake_history) / len(brake_history)
            
            # 5. Apply the smoothed inputs to the vehicle
            control = carla.VehicleControl()
            control.throttle = smooth_throttle
            control.steer = smooth_steer
            control.brake = smooth_brake
            control.reverse = raw_reverse
            vehicle.apply_control(control)

            # --- SPEED MEASUREMENT (For UI Only) ---
            velocity = vehicle.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_ms * 3.6
            
            # --- PYGAME DISPLAY ---
            display.fill((30, 30, 30))
            display.blit(font.render("🚨 CLICK THIS WINDOW TO DRIVE 🚨", True, (255, 80, 80)), (20, 20))
            display.blit(font.render("W: Gas | S: Reverse | A/D: Steer", True, (200, 200, 200)), (20, 70))
            display.blit(font.render(f"Speed: {speed_kmh:.2f} km/h", True, (100, 255, 100)), (20, 120))
            pygame.display.flip()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("\nCleaning up actors...")
        pygame.quit()
        
        # Turn off synchronous modes before exiting
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
            
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    main()