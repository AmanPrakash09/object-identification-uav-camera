import airsim
import cv2
import numpy as np
import os
import time
import math
import random

# --- 1. SETUP FOLDERS ---
dataset_dir = "dataset"
rgb_dir = os.path.join(dataset_dir, "rgb")
ir_dir = os.path.join(dataset_dir, "ir")
label_dir = os.path.join(dataset_dir, "labels")
box_dir = os.path.join(dataset_dir, "boxes")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(ir_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(box_dir, exist_ok=True)

# --- 2. CONNECT TO AIRSIM (AS A CAR) ---
print("Connecting to AirSim...")
client = airsim.CarClient() # <--- Changed to CarClient!
client.confirmConnection()

# --- 3. MOVE THE CAMERA TO THE SKY ---
camera_name = "0"
image_type = airsim.ImageType.Scene

print("Moving camera 40 meters above the car...")
# Vector3r(X, Y, Z) -> Z is negative to go UP relative to the car's roof
# to_quaternion(pitch, roll, yaw) -> Pitch -90 degrees (straight down)
sky_cam_pose = airsim.Pose(airsim.Vector3r(0, 0, -40), airsim.to_quaternion(-math.pi/2, 0, 0))
client.simSetCameraPose(camera_name, sky_cam_pose)

# --- 4. SETUP DETECTIONS ---
client.simSetDetectionFilterRadius(camera_name, image_type, 20000) 

# Animal meshes
client.simAddDetectionFilterMeshName(camera_name, image_type, "ANIM_Crocodile*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "ANIM_Rhinoceros*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "ANIM_Hippopotamus*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "African_Poacher*")

# --- NEW: Add the Car meshes ---
client.simAddDetectionFilterMeshName(camera_name, image_type, "SUV*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "Car*")
client.simAddDetectionFilterMeshName(camera_name, image_type, "PhysXCar*")

class_map = {
    "ANIM_Crocodile": 0,
    "ANIM_Rhinoceros": 1,
    "ANIM_Hippopotamus": 2,
    "African_Poacher": 3,
    # --- NEW: Map all car names to Class 4 ---
    "SUV": 4,
    "Car": 4,
    "PhysXCar": 4
}

display_names = {
    0: "Crocodile",
    1: "Rhinoceros",
    2: "Hippopotamus",
    3: "Poacher",
    4: "Car" # --- NEW: Display name for the box ---
}

# --- 5. CAPTURE LOOP WITH SWING PHYSICS ---
print("\nReady! Drive the car around while I take pictures.")
print("Press Ctrl+C in this terminal when you are done collecting data.")
img_id = 0

# Physics variables for the swingy camera
cam_x, cam_y = 0.0, 0.0
cam_vx, cam_vy = 0.0, 0.0
spring_constant = 0.15 # How strongly it pulls back to the car (higher = tighter tether)
damping = 0.85         # Friction so the camera doesn't swing out of control

try:
    while True:
        # 1. CALCULATE SWING PHYSICS
        # Random "wind" pushing the camera in random directions
        wind_x = random.uniform(-4.0, 4.0)
        wind_y = random.uniform(-4.0, 4.0)
        
        # Spring force pulling it back to center (0,0)
        force_x = -spring_constant * cam_x
        force_y = -spring_constant * cam_y
        
        # Apply forces to velocity, then apply friction (damping)
        cam_vx = (cam_vx + wind_x + force_x) * damping
        cam_vy = (cam_vy + wind_y + force_y) * damping
        
        # Update actual camera position
        cam_x += cam_vx
        cam_y += cam_vy
        
        # Add a tiny bit of random camera tilt (wobble)
        pitch = -math.pi/2 + random.uniform(-0.05, 0.05)
        roll = random.uniform(-0.05, 0.05)
        yaw = random.uniform(-0.1, 0.1)

        # Apply the new swinging pose!
        swing_pose = airsim.Pose(airsim.Vector3r(cam_x, cam_y, -40), airsim.to_quaternion(pitch, roll, yaw))
        client.simSetCameraPose(camera_name, swing_pose)
        
        # Give the game engine 0.1 seconds to register the new camera position
        time.sleep(0.1) 

        # 2. CAPTURE IMAGES
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera_name, airsim.ImageType.Infrared, False, False)
        ])
        
        rgb_response = responses[0]
        ir_response = responses[1]
        
        img1d_rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        rgb_img = img1d_rgb.reshape(rgb_response.height, rgb_response.width, 3)
        
        img1d_ir = np.frombuffer(ir_response.image_data_uint8, dtype=np.uint8)
        ir_img = img1d_ir.reshape(ir_response.height, ir_response.width, 3)

        box_img = rgb_img.copy()
        
        cv2.imwrite(os.path.join(rgb_dir, f"{img_id:05d}.png"), rgb_img)
        cv2.imwrite(os.path.join(ir_dir, f"{img_id:05d}.png"), ir_img)
        
        # 3. GET BOUNDING BOXES
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
                    
                    w, h = rgb_response.width, rgb_response.height
                    x_center = ((x_min + x_max) / 2.0) / w
                    y_center = ((y_min + y_max) / 2.0) / h
                    box_width = (x_max - x_min) / w
                    box_height = (y_max - y_min) / h
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        cv2.imwrite(os.path.join(box_dir, f"{img_id:05d}.png"), box_img)
                    
        print(f"Saved frame {img_id:05d} | Cam Offset: X:{cam_x:.1f} Y:{cam_y:.1f} | Objects: {len(detections) if detections else 0}")
        
        img_id += 1
        # Changed this to 0.9s so the total loop still takes exactly 1 second
        time.sleep(0.9) 

except KeyboardInterrupt:
    print("\nCapture stopped. Safely disconnected!")