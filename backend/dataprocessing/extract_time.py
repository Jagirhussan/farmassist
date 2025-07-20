import subprocess
import json
from datetime import datetime
import os

def get_video_creation_time(video_path):
    # First try ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        info = json.loads(result.stdout)
        creation_time = (
            info.get("format", {}).get("tags", {}).get("creation_time")
            or next(
                (s.get("tags", {}).get("creation_time") for s in info.get("streams", []) if s.get("tags", {}).get("creation_time")),
                None
            )
        )
        if creation_time:
            return datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
    except Exception as e:
        print(f"ffprobe could not extract creation_time: {e}")

    # Fallback to filesystem creation time
    print("Falling back to filesystem creation time")
    stat = os.stat(video_path)
    return datetime.fromtimestamp(stat.st_ctime)


