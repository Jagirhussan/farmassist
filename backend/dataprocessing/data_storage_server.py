from xmlrpc import client
from fastapi import FastAPI, Request
import chromadb
from chromadb.config import Settings

app = FastAPI()
# Initialize ChromaDB with the new client configuration
client = chromadb.PersistentClient(path="video_db")

# get or create the collection
collection = client.get_or_create_collection(name="video_frames")


@app.post("/store_data")
async def store_data(request: Request):
    """
    Endpoint to receive and store processed data.
    """

    try:
        data = await request.json()

        print(f"[Storage] Received data: {data}")
        print(f"[Storage] Storing {len(data)} items to ChromaDB...")
        for item in data[0:1]:
            print(f"[Storage] Sample item: {item['ids']}, {item['documents']}, Embedding length: {len(item['embeddings'])}")

        # Add data to ChromaDB
        ids = [item["ids"] for item in data]
        embeddings = [item['embeddings'] for item in data]
        # metadatas = [{"caption": item["caption"]} for item in data]
        documents = [item["documents"] for item in data]

        print(f"[Storage] ids: {ids}")
        print(f"[Storage] embeddings: {embeddings[0:5]}")
        print(f"[Storage] documents: {documents}")

        collection.add(
            ids=ids,
            embeddings=embeddings,
            # metadatas=metadatas,
            documents=documents
        )
        
        client.persist()

        return {"message": "Data stored successfully"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5051)
