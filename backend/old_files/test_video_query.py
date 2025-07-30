import time
from video_query import VideoFrameRetriever

# Test query to search for
test_query = "person typing"

# Initialize the retriever
retriever = VideoFrameRetriever()

# Get all items from the collection
all_items = retriever.collection.get()

# Check if the database has any frames
if len(all_items["ids"]) == 0:
    print("[Test] No frames found in database.")
    exit()

# Organize frames by source video
video_sources = {}
for metadata in all_items["metadatas"]:
    video_name = metadata.get("source_video", "unknown")
    video_sources[video_name] = video_sources.get(video_name, 0) + 1

print("\n=== VIDEO QUERY TEST REPORT ===")
print(f"Total videos with frames: {len(video_sources)}\n")

# Run a test query for each video
for video_name, frame_count in video_sources.items():
    print(f"Running query on video: '{video_name}' ({frame_count} frames)")

    start = time.time()
    context = retriever.get_contextual_info(test_query, n_results=3, filter_video=video_name)
    end = time.time()

    print(f"Query time: {end - start:.2f} seconds")
    print("Returned Context:\n")
    print(context)
    print("\n" + "-"*60 + "\n")
