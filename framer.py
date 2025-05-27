import sys
import cv2
import os



def framer(video):
	"""
	Extract frames from a video file and save them as images in a folder named 'frames'.
	Args:
		video (an mp4 file).
	Returns:
		None: A file will be created in the 'frames' directory for each frame extracted.

	Author: Alex Foster
	Date: 2023-10-01
	"""

	os.makedirs("frames", exist_ok=True)

	fps = video.get(cv2.CAP_PROP_FPS)

	frame_count = 0

	while video.isOpened():
		ret, frame = video.read()
		if not ret: 
			break

		if frame_count % int(fps) == 0:  # Save 1 frame/sec
			cv2.imwrite(f"frames/frame_{frame_count}.jpg", frame)

		frame_count +=1

	video.release()
	return

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python3 framer.py <video.mp4>")
		sys.exit(1)

	video = cv2.VideoCapture(sys.argv[1])
	if not video.isOpened():
		print(f"Error: could not open video file {sys.argv[1]}")
		sys.exit(1)

	framer(video)

	print(f"Finished Running.")


