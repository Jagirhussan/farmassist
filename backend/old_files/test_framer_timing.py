import time
import os
from framer3 import framer

# Path to the videos folder
video_dir = "./videos"  # Updated to point to the videos folder
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# Get all video files in the directory
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(video_extensions)]

# Run framer on each video and measure time and frame count
for video_path in video_files:
    video_name = os.path.basename(video_path)
    print(f"Processing: {video_name}")

    start = time.time()
    frame_count = framer(video_path)
    end = time.time()

    duration = end - start
    print(f"{video_name} took {duration:.2f} seconds and extracted {frame_count} frames\n")
