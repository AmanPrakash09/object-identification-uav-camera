import airsim
import cv2
import numpy as np
import os
import time
import math

# --- 1. SETUP FOLDERS & LOG FILES ---
dataset_dir = "dataset"
ir_dir = os.path.join(dataset_dir, "ir")
label_dir = os.path.join(dataset_dir, "labels")
box_dir = os.path.join(dataset_dir, "boxes")
telemetry_file = os.path.join(dataset_dir, "telemetry_log.csv") # <-- NEW: CSV FILE

os.makedirs(ir_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(box_dir, exist_ok=True)

# Create the CSV file and write the column headers
with open(telemetry_file, "w") as f:
    f.write("frame_id,speed_m_s\n")

# --- 2. CONNECT TO AIRSIM (AS A CAR) ---
print("Connecting to AirSim...")
client = airsim.CarClient() 
client.confirmConnection()

# --- 3. CAMERA SETUP ---
camera_name = "0"
image_type = airsim.ImageType.Infrared 
altitude = -50 

# --- 4. SETUP DETECTIONS ---
client.simSetDetectionFilterRadius(camera_name, image_type, 20000) 

# Animals
client.simAddDetectionFilterMeshName(camera_name, image_type, "ANIM_Crocodile*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "ANIM_Rhinoceros*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "ANIM_Hippopotamus*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "African_Poacher*")

# Vehicles
client.simAddDetectionFilterMeshName(camera_name, image_type, "SUV*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "Car*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "PhysXCar*")

class_map = {
    "ANIM_Crocodile": 0,
    "ANIM_Rhinoceros": 1,
    "ANIM_Hippopotamus": 2,
    "African_Poacher": 3,
    "SUV": 4,
    "Car": 4,
    "PhysXCar": 4
}

display_names = {
    0: "Crocodile",
    1: "Rhinoceros",
    2: "Hippopotamus",
    3: "Poacher",
    4: "Car" 
}

# --- 5. THE "DROP ANCHOR" LOOP ---
print("\nReady! Drive the car around while I take pictures.")
print("Press Ctrl+C in this terminal when you are done collecting data.")
img_id = 0

# Variables for the Chase Camera
cam_x = -100.0 # Starts 100 meters behind you
cam_y = 0.0
catch_up_speed = 3.0 # Moves 3 meters closer every frame
locked_on = False

# Variables to store the GPS lock position
locked_world_x = 0.0
locked_world_y = 0.0
locked_world_z = 0.0

try:
    while True:
        # Get the car's current real-world GPS position and tilt
        car_pose = client.simGetVehiclePose()
        car_pitch, car_roll, car_yaw = airsim.utils.to_eularian_angles(car_pose.orientation)
        
        # --- CHASE & LOCK LOGIC ---
        if not locked_on:
            dist_from_center = math.hypot(cam_x, cam_y)
            
            if dist_from_center <= 4.0:
                # WE HIT THE ZONE. DROP THE ANCHOR IN GLOBAL SPACE.
                locked_on = True
                locked_world_x = car_pose.position.x_val
                locked_world_y = car_pose.position.y_val
                locked_world_z = car_pose.position.z_val + altitude 
                print("\n>>> CAMERA DROPPED ANCHOR IN THE SKY! DRIVE AWAY! <<<\n")
            else:
                # Still chasing: Move relative to the car
                angle = math.atan2(0 - cam_y, 0 - cam_x)
                cam_x += math.cos(angle) * catch_up_speed
                cam_y += math.sin(angle) * catch_up_speed

                sky_cam_pose = airsim.Pose(airsim.Vector3r(cam_x, cam_y, altitude), airsim.to_quaternion(-math.pi/2, 0, 0))
                client.simSetCameraPose(camera_name, sky_cam_pose)

        if locked_on:
            # Calculate the distance between the car and our permanent sky anchor
            dx = locked_world_x - car_pose.position.x_val
            dy = locked_world_y - car_pose.position.y_val
            dz = locked_world_z - car_pose.position.z_val

            # Rotate the world difference to counteract the direction the car is facing
            local_x = dx * math.cos(-car_yaw) - dy * math.sin(-car_yaw)
            local_y = dx * math.sin(-car_yaw) + dy * math.cos(-car_yaw)
            local_z = dz

            # Cancel out the car's pitch and roll so the camera doesn't spin
            cam_pitch = -math.pi/2 - car_pitch
            cam_roll = -car_roll
            cam_yaw = -car_yaw

            # Set the camera position so it perfectly counters your driving
            locked_pose = airsim.Pose(airsim.Vector3r(local_x, local_y, local_z), airsim.to_quaternion(cam_pitch, cam_roll, cam_yaw))
            client.simSetCameraPose(camera_name, locked_pose)

        # --- CAPTURE INFRARED ---
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Infrared, False, False)
        ])
        
        ir_response = responses[0]
        img1d_ir = np.frombuffer(ir_response.image_data_uint8, dtype=np.uint8)
        ir_img = img1d_ir.reshape(ir_response.height, ir_response.width, 3)

        box_img = ir_img.copy() 
        
        cv2.imwrite(os.path.join(ir_dir, f"{img_id:05d}.png"), ir_img)
        
        # --- GET BOUNDING BOXES ---
        detections = client.simGetDetections(camera_name, image_type)
        label_filepath = os.path.join(label_dir, f"{img_id:05d}.txt")
        
        with open(label_filepath, "w") as f:
            if detections:
                for d in detections:
                    class_id = -1
                    for key, val in class_map.items():
                        if key in d.name:
                            class_id = val
                            break
                    if class_id == -1: 
                        continue 
                    
                    x_min, y_min = d.box2D.min.x_val, d.box2D.min.y_val
                    x_max, y_max = d.box2D.max.x_val, d.box2D.max.y_val

                    cv2.rectangle(box_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    label_text = display_names[class_id]
                    cv2.putText(box_img, label_text, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    w, h = ir_response.width, ir_response.height
                    x_center = ((x_min + x_max) / 2.0) / w
                    y_center = ((y_min + y_max) / 2.0) / h
                    box_width = (x_max - x_min) / w
                    box_height = (y_max - y_min) / h
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        cv2.imwrite(os.path.join(box_dir, f"{img_id:05d}.png"), box_img)
                    
        # --- GET AND LOG TELEMETRY ---
        car_state = client.getCarState()
        speed_ms = car_state.speed
        
        # Save to the CSV file
        with open(telemetry_file, "a") as f:
            f.write(f"{img_id:05d},{speed_ms:.6f}\n")
            
        status = "ANCHORED" if locked_on else "CHASING"
        print(f"Frame {img_id:05d} | {status} | Speed: {speed_ms:.1f} m/s | Objects: {len(detections) if detections else 0}")
        
        img_id += 1
        time.sleep(0.05) 

except KeyboardInterrupt:
    print("\nCapture stopped. Safely disconnected!")