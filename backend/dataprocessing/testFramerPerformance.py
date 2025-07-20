import os
import time
import psutil
import cv2
import threading
import unittest
from framer2 import framer

class TestFramerPerformance(unittest.TestCase):
    """
    Unit test for measuring peak RAM usage and runtime of framer() on a given video.
    """

    def test_ram_and_time(self):
        video_path = "videos/earth_spinning.mp4"
        self.assertTrue(os.path.exists(video_path), f"Video file '{video_path}' does not exist.")

        video = cv2.VideoCapture(video_path)
        process = psutil.Process(os.getpid())

        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        running = True

        def monitor():
            nonlocal peak_memory
            while running:
                mem = process.memory_info().rss / (1024 * 1024)
                if mem > peak_memory:
                    peak_memory = mem
                time.sleep(0.05)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        start_time = time.time()
        try:
            framer(video, video_path)
        finally:
            running = False
            monitor_thread.join()
        end_time = time.time()

        time_taken = end_time - start_time

        print(f"\nPeak memory used: {peak_memory:.2f} MB")
        print(f"Time taken: {time_taken:.2f} seconds")

        # Optional: assert performance constraints
        self.assertLess(peak_memory, 1024, "Peak memory used is unexpectedly high!")
        self.assertLess(time_taken, 120, "Processing took too long!")  # adjust as reasonable

if __name__ == "__main__":
    unittest.main()
