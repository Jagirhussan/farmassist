# debug_chromadb.py - Check what's in your ChromaDB collection
import chromadb
import json

def inspect_chromadb():
    """Inspect the contents of the ChromaDB collection"""
    try:
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path="video_db")
        collection = chroma_client.get_or_create_collection(
            name="video_frames",
            metadata={"hnsw:space": "cosine"}  # Use cosine distance
        )
        
        # Get all items in the collection
        results = collection.get()
        
        print(f"Total items in collection: {len(results['ids'])}")
        
        # Analyze source videos
        source_videos = {}
        for metadata in results['metadatas']:
            source_video = metadata.get('source_video', 'unknown')
            source_videos[source_video] = source_videos.get(source_video, 0) + 1
        
        print("\nVideos in database:")
        for video, count in source_videos.items():
            print(f"  - {video}: {count} frames")
        
        # Test the actual query that's problematic
        print("\n" + "="*50)
        print("TESTING QUERY: 'person typing'")
        print("="*50)
        
        # Import the actual video_query functionality
        import open_clip
        import torch
        import numpy as np
        
        # Load the same model as video_query.py
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k'
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Embed the problematic query
        query = "person typing"
        text_tokens = open_clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text(text_tokens)
        query_embedding = text_embedding.squeeze().cpu().numpy()
        
        # Search for similar frames
        search_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=10  # Get more results to see the pattern
        )
        
        print(f"\nTop 10 results for '{query}':")
        if search_results['ids'] and len(search_results['ids'][0]) > 0:
            for i in range(len(search_results['ids'][0])):
                frame_id = search_results['ids'][0][i]
                distance = search_results['distances'][0][i]
                metadata = search_results['metadatas'][0][i]
                similarity = 1 - distance
                
                print(f"  {i+1}. {frame_id} from '{metadata.get('source_video', 'unknown')}' "
                      f"at {metadata.get('timestamp', 'unknown')} "
                      f"(similarity: {similarity:.3f}, distance: {distance:.3f})")
        
        # Also test with filtering
        print(f"\n" + "-"*30)
        print("TESTING WITH FILTER: alex1min.mp4")
        print("-"*30)
        
        filtered_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
            where={"source_video": "alex1min.mp4"}
        )
        
        if filtered_results['ids'] and len(filtered_results['ids'][0]) > 0:
            for i in range(len(filtered_results['ids'][0])):
                frame_id = filtered_results['ids'][0][i]
                distance = filtered_results['distances'][0][i]
                metadata = filtered_results['metadatas'][0][i]
                similarity = 1 - distance
                
                print(f"  {i+1}. {frame_id} from '{metadata.get('source_video', 'unknown')}' "
                      f"(similarity: {similarity:.3f}, distance: {distance:.3f})")
        else:
            print("  No results found for alex1min.mp4")
        
        return source_videos
        
    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")
        return {}

if __name__ == "__main__":
    inspect_chromadb()
