# simple_test.py - Simple test to see what's in ChromaDB
import chromadb

def simple_test():
    print("=== SIMPLE CHROMADB TEST ===")
    
    try:
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path="dataprocessing/video_db")
        print("Connected to ChromaDB")
        
        # List all collections
        collections = chroma_client.list_collections()
        print(f"Found {len(collections)} collections:")
        for col in collections:
            print(f"  - {col.name}: {col.metadata}")
        
        # Try to get the video_frames collection
        try:
            collection = chroma_client.get_collection(name="video_frames")
            print(f"Found 'video_frames' collection")
            print(f"Collection metadata: {collection.metadata}")
            
            # Get all items
            all_items = collection.get()
            print(f"Total items in collection: {len(all_items['ids'])}")
            
            if len(all_items['ids']) > 0:
                print("First 3 items:")
                for i in range(min(3, len(all_items['ids']))):
                    print(f"  {i+1}. ID: {all_items['ids'][i]}")
                    print(f"     Metadata: {all_items['metadatas'][i]}")
            else:
                print("Collection is empty!")
                
        except Exception as e:
            print(f"Error accessing 'video_frames' collection: {e}")
            
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")

if __name__ == "__main__":
    simple_test()
