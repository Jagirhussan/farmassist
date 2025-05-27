import sys
import cv2
import os
import json
import subprocess
from datetime import datetime, timedelta

# === Extract creation time using ffprobe ===
def get_video_creation_time(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        info = json.loads(result.stdout)
        creation_time = info["format"]["tags"]["creation_time"]
        return datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
    except Exception as e:
        print(f"Could not extract video creation time: {e}")
        sys.exit(1)

# === Main frame extraction ===
def framer(video, video_path):

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

