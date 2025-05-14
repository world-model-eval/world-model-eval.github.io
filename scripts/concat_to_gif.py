#!/usr/bin/env python3
"""
Script to concatenate two MP4 videos side-by-side and create a GIF.
Requires: pip install opencv-python pillow numpy
"""

import cv2
import numpy as np
from PIL import Image
import argparse
import os

def create_side_by_side_gif(video1_path, video2_path, output_path, max_frames=30, fps=10, skip_frames1=0, skip_frames2=0, stride1=1, stride2=1):
    """
    Create a GIF by concatenating two videos side-by-side.
    
    Args:
        video1_path (str): Path to first MP4 video
        video2_path (str): Path to second MP4 video  
        output_path (str): Path for output GIF
        max_frames (int): Maximum number of frames to include
        fps (int): Frame rate for the output GIF
        skip_frames1 (int): Number of frames to skip at start of first video
        skip_frames2 (int): Number of frames to skip at start of second video
        stride1 (int): Number of frames to skip between captures for first video
        stride2 (int): Number of frames to skip between captures for second video
    """
    
    # Open video captures
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened():
        raise ValueError(f"Could not open video: {video1_path}")
    if not cap2.isOpened():
        raise ValueError(f"Could not open video: {video2_path}")
    
    # Skip initial frames
    for _ in range(skip_frames1):
        cap1.read()
    for _ in range(skip_frames2):
        cap2.read()
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use the minimum height to avoid distortion
    target_height = min(height1, height2)
    
    # Calculate proportional widths
    target_width1 = int(width1 * target_height / height1)
    target_width2 = int(width2 * target_height / height2)
    
    frames = []
    frame_count = 0
    
    print(f"Processing videos...")
    print(f"Video 1: {width1}x{height1} -> {target_width1}x{target_height}")
    print(f"Video 2: {width2}x{height2} -> {target_width2}x{target_height}")
    
    while frame_count < max_frames:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Break if either video ends
        if not ret1 or not ret2:
            break
            
        # Skip frames according to stride
        for _ in range(stride1 - 1):
            cap1.read()
        for _ in range(stride2 - 1):
            cap2.read()
            
        # Resize frames to target height while maintaining aspect ratio
        frame1_resized = cv2.resize(frame1, (target_width1, target_height))
        frame2_resized = cv2.resize(frame2, (target_width2, target_height))
        
        # Concatenate frames horizontally
        combined_frame = np.hstack([frame1_resized, frame2_resized])
        
        # Convert BGR to RGB for PIL
        combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(combined_frame_rgb)
        frames.append(pil_image)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release video captures
    cap1.release()
    cap2.release()
    
    if not frames:
        raise ValueError("No frames were processed. Check if videos are valid.")
    
    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)
    
    # Save as GIF
    print(f"Creating GIF with {len(frames)} frames...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    print(f"GIF saved to: {output_path}")
    print(f"Final dimensions: {frames[0].size[0]}x{frames[0].size[1]}")

def main():
    parser = argparse.ArgumentParser(description="Create side-by-side GIF from two MP4 videos")
    parser.add_argument("video1", help="Path to first MP4 video")
    parser.add_argument("video2", help="Path to second MP4 video")
    parser.add_argument("-o", "--output", default="output.gif", help="Output GIF path (default: output.gif)")
    parser.add_argument("-n", "--frames", type=int, default=30, help="Maximum number of frames (default: 30)")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate (default: 10)")
    parser.add_argument("--skip1", type=int, default=0, help="Number of frames to skip at start of first video (default: 0)")
    parser.add_argument("--skip2", type=int, default=0, help="Number of frames to skip at start of second video (default: 0)")
    parser.add_argument("--stride1", type=int, default=1, help="Number of frames to skip between captures for first video (default: 1)")
    parser.add_argument("--stride2", type=int, default=1, help="Number of frames to skip between captures for second video (default: 1)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.video1):
        print(f"Error: Video file not found: {args.video1}")
        return
    if not os.path.exists(args.video2):
        print(f"Error: Video file not found: {args.video2}")
        return
    
    try:
        create_side_by_side_gif(
            args.video1, 
            args.video2, 
            args.output, 
            args.frames,
            args.fps,
            args.skip1,
            args.skip2,
            args.stride1,
            args.stride2
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
