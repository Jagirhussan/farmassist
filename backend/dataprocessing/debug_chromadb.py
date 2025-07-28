# debug_chromadb.py - Check what's in your ChromaDB collection
import chromadb
import json

def inspect_chromadb():
    """Inspect the contents of the ChromaDB collection"""
    try:
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path="video_db")
        collection = chroma_client.get_or_create_collection(
            name="video_frames"
        )
        
        # Get all items in the collection
        results = collection.get()
        
        print(f"Total items in collection: {len(results['ids'])}")
        
        # print the documents, embeddings, and metadata
        for i in range(min(5, len(results['ids']))):  # Print first 5 items
            print(f"Item {i+1}:")
            print(f"  ID: {results['ids'][i]}")
            print(f"  Embedding: {results['embeddings'][i]}...")
            print(f"  Documents: {results['documents'][i]}...")

        
    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")
        return {}

if __name__ == "__main__":
    inspect_chromadb()
