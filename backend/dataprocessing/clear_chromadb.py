# clear_chromadb.py - Clear the ChromaDB collection to start fresh
import chromadb

def clear_chromadb():
    """Clear all data from ChromaDB collection"""
    try:
        chroma_client = chromadb.PersistentClient(path="video_db")
        
        # Delete the collection entirely
        try:
            chroma_client.delete_collection(name="video_frames")
            print("Successfully deleted 'video_frames' collection")
        except:
            print("Collection 'video_frames' doesn't exist or already empty")
        
        # Create a fresh collection with cosine distance
        collection = chroma_client.create_collection(
            name="video_frames"
        )
        print("Created fresh 'video_frames' collection with cosine distance")
        
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")

if __name__ == "__main__":
    clear_chromadb()
