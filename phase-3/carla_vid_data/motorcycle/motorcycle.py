import os
import sys
import math
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carla_api_paths import ensure_carla_on_path

ensure_carla_on_path()

import carla
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_SPACE, K_ESCAPE

# --- Smoothing Configuration ---
SMOOTHING_WINDOW = 10  # 10 frames = 0.5s of smoothed inputs

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
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # --- SPAWN MODEL 3 ---
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0] 
        spawn_point = world.get_map().get_spawn_points()[0]
        spawn_point.location.z += 1.0 
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        
        # Grab the main CARLA viewport
        spectator = world.get_spectator()

        running = True
        print("\nReady! CLICK THE SMALL PYGAME WINDOW to drive using WASD.")
        print("Watch the main CARLA window for your 30m top-down view. Press ESC to stop.\n")
        
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
            raw_throttle = 1.0 if keys[K_w] else 0.0
            raw_steer = -0.5 if keys[K_a] else (0.5 if keys[K_d] else 0.0)
            raw_brake = 1.0 if keys[K_SPACE] else 0.0
            raw_reverse = bool(keys[K_s])
            
            if keys[K_s]: 
                raw_throttle = 1.0 
                
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

            # --- SPECTATOR BIRD'S EYE VIEW (30m UP) ---
            car_transform = vehicle.get_transform()
            spectator_transform = carla.Transform(
                car_transform.location + carla.Location(z=30.0),
                carla.Rotation(pitch=-90.0, yaw=car_transform.rotation.yaw, roll=0.0)
            )
            spectator.set_transform(spectator_transform)

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
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    main()