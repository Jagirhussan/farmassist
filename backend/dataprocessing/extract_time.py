import sys
import cv2
import os
import json
import subprocess
from datetime import datetime, timedelta

def get_video_creation_time(video_path):
    """
	Finds the creation time of a video file using ffprobe.

    This function should be adjusted to find the actual timestamp of the video file. E.g. if the video was recorded on a camera, 
    it should return the timestamp of when the video was recorded, not when it was created on disk.
     
	Args:
        video_path (str): path to the video file.
	Returns:
		datetime: creation time of the video.

	Author: Alex Foster
	Date: 2023-10-01
	"""

    ##TODO: This function should be adjusted to find the actual timestamp of the video file.

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