import sys
import cv2
import os
import json
import subprocess
from datetime import datetime, timedelta
from extract_time import get_video_creation_time
import shutil

# === Main frame extraction ===
def framer(video, video_path):
    """
	Extract frames from a video file and save them as images in a folder named 'frames' along with the meta data .json files.
	Args:
		video (an mp4 file).
        video_path (str): path to the video file.
	Returns:
		None: A file will be created in the 'frames' directory for each frame extracted + their metadata.

	Author: Alex Foster
	Date: 2023-10-01
	"""
    if os.path.exists("frames"):
        # clear the directory if it exists
        shutil.rmtree("frames")
    os.makedirs("frames", exist_ok=True)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_index = 0

    # Get actual start time of the video
    start_time = get_video_creation_time(video_path)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % int(fps) == 0:  # Save 1 frame/sec
            frame_filename = f"frame_{saved_frame_index:04d}.jpg"
            frame_path = os.path.join("frames", frame_filename)
            cv2.imwrite(frame_path, frame)

            # Compute timestamp for this frame
            timestamp = start_time + timedelta(seconds=saved_frame_index)

            # Save metadata
            metadata = {
                "frame_index": saved_frame_index,
                "timestamp": timestamp.isoformat(),
                "source_video": os.path.basename(video_path)
            }

            metadata_path = os.path.join("frames", f"frame_{saved_frame_index:04d}.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved: {frame_filename} + metadata")

            saved_frame_index += 1

        frame_count += 1

    video.release()

# === Entry point ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 framer.py <video.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: could not open video file {video_path}")
        sys.exit(1)

    framer(video, video_path)

    print("Finished extracting frames and metadata.")

