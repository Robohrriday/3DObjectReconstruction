import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import time

def main():
    """
    Collects stereo image pairs (Left and Right) from an Intel RealSense camera.
    
    Note: Standard RealSense cameras (like D435, D455) have two Infrared sensors 
    that form the stereo pair. These produce grayscale images which are ideal for 
    stereo matching and 3D reconstruction due to perfect synchronization and 
    global shutter (on D435/D455).
    
    This script captures these Left (Infrared 1) and Right (Infrared 2) images.
    """
    
    # --- Configuration ---
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    OUTPUT_DIR = "data"
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable Left and Right Infrared streams
    # Infrared 1 = Left, Infrared 2 = Right
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)
    
    # Start streaming
    print(f"Starting RealSense pipeline with {WIDTH}x{HEIGHT} @ {FPS}fps...")
    try:
        profile = pipeline.start(config)
        
        # Disable the IR Emitter
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0) # 0 = Off
            
    except Exception as e:
        print(f"\nError starting pipeline: {e}")
        print("---------------------------------------------------------")
        print("Troubleshooting:")
        
        # Check for devices
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("  -> No RealSense devices detected! Check your USB cable.")
        else:
            for dev in devices:
                name = dev.get_info(rs.camera_info.name)
                usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
                print(f"  -> Found Device: {name} (USB {usb_type})")
                if "2.1" in usb_type or "2.0" in usb_type:
                    print("     WARNING: USB 2.0 detected. High resolutions/FPS may fail.")
                    print("     Try lowering FPS to 15 or 6 in the script.")
        
        print("---------------------------------------------------------")
        return

    # --- Calibration Info ---
    # We assume the left camera (Infrared 1) is the main reference
    stream_profile_left = profile.get_stream(rs.stream.infrared, 1)
    stream_profile_right = profile.get_stream(rs.stream.infrared, 2)
    
    intrinsics = stream_profile_left.as_video_stream_profile().get_intrinsics()
    extrinsics = stream_profile_left.get_extrinsics_to(stream_profile_right)
    baseline = abs(extrinsics.translation[0]) # usually in meters

    print("\n" + "="*60)
    print("CAMERA CALIBRATION PARAMETERS")
    print("Copy these values to your 'data/configs/base.yaml' file:")
    print("-" * 60)
    print(f"Resolution: {intrinsics.width}x{intrinsics.height}")
    print(f"Baseline:   {baseline:.5f} meters")
    print(f"Intrinsics: [{intrinsics.fx:.4f}, 0, {intrinsics.ppx:.4f}, 0, {intrinsics.fy:.4f}, {intrinsics.ppy:.4f}, 0, 0, 1]")
    print("="*60 + "\n")
    print("NOTE: RealSense Stereo is GRAYSCALE (Infrared).")
    print("="*60 + "\n")

    # --- Setup Directories ---
    left_dir = os.path.join(OUTPUT_DIR, "left")
    right_dir = os.path.join(OUTPUT_DIR, "right")
    
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    print(f"Saving images to:\n  - {left_dir}\n  - {right_dir}")
    print("\nINSTRUCTIONS:")
    print("1. Place your object in front of the camera.")
    print("2. Press 'Enter' to capture a stereo pair.")
    print("3. Rotate the object slightly.")
    print("4. Repeat until you have at least 20 pairs.")
    print("5. Type 'q' and press Enter to quit.")
    print("-" * 60)

    count = 0
    
    # Allow camera auto-exposure to settle
    print("Waiting for auto-exposure to settle...")
    for _ in range(30):
        pipeline.wait_for_frames()
    
    try:
        while True:
            user_input = input(f"\n[Pair #{count}] Press Enter to capture (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            
            ir1_frame = frames.get_infrared_frame(1) # Left
            ir2_frame = frames.get_infrared_frame(2) # Right
            
            if not ir1_frame or not ir2_frame:
                print("Error: Could not get stereo frames. Retrying...")
                continue

            # Convert images to numpy arrays
            # RealSense IR is 8-bit grayscale
            ir1_image = np.asanyarray(ir1_frame.get_data())
            ir2_image = np.asanyarray(ir2_frame.get_data())

            # Generate filenames: left000000.png, right000000.png
            filename_left = f"left{count:06d}.png"
            filename_right = f"right{count:06d}.png"
            
            path_left = os.path.join(left_dir, filename_left)
            path_right = os.path.join(right_dir, filename_right)
            
            # Save images
            cv2.imwrite(path_left, ir1_image)
            cv2.imwrite(path_right, ir2_image)
            
            print(f"  -> Captured!")
            print(f"     Saved {filename_left} and {filename_right}")
            
            count += 1
            
            if count < 20:
                print(f"  Progress: {count}/20 (Need {20 - count} more)")
            else:
                print(f"  Progress: {count}/20 (Minimum goal reached!)")

    finally:
        pipeline.stop()
        print("\nPipeline stopped.")
        print(f"Total pairs collected: {count}")

if __name__ == "__main__":
    main()
