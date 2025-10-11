# backend/llm_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from numpy import dot
from numpy.linalg import norm

# Global variables to store the model (loaded once)
tokenizer = None
model = None
model_encoder = None


def load_models():
    """Load the LLM model once when the server starts"""
    global tokenizer, model, model_encoder

    if tokenizer is None or model is None or model_encoder is None:
        print("[LLM] Loading models...")
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ).to(device)
        print(
            "[LLM] Loaded tinyllama to ", next(model.parameters()).device
        )  # should show cuda:0
        model_encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
        print(
            "[LLM] Loaded sentence transformer to ",
            next(model_encoder.parameters()).device,
        )  # should show cuda:0

        # Set up pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("[LLM] Models loaded successfully!")


def retrieve_data():
    """Retrieve all data from ChromaDB collection."""
    print("[LLM] Retrieving data from ChromaDB...")
    # connect to the client
    client = chromadb.PersistentClient(path="dataprocessing/video_db")
    # get or create the collection
    collection = client.get_or_create_collection(name="video_frames")
    # return the data including documents and embeddings
    print(f"[LLM] Retrieved {len(collection.get()['ids'])} items from ChromaDB.")
    return collection.get(include=["documents", "embeddings"])


def retrieve_context(query, n=3, threshold=0.5):
    """Retrieve relevant context from ChromaDB based on the query."""
    print(f"[LLM] Retrieving context for query: {query}")
    # embed the query to the same format as the stored embeddings
    query_embedded = model_encoder.encode(query)
    # retrieve the video frame data
    data = retrieve_data()
    # calculate the similarities with the cosine similarity
    # similarities = model_encoder.similarity(np.array(data['embeddings'], dtype=np.float32), np.array(query_embedded, dtype=np.float32))

    similarities = np.array(
        [
            dot(e, query_embedded) / (norm(e) * norm(query_embedded))
            for e in data["embeddings"]
        ]
    )

    print(f"[LLM] Similarities: {similarities}")

    # check which similarities are higher than a threshold (e.g., 0.5)
    relevant_indices = np.where(similarities > threshold)

    # order the relevant indices by similarity score in descending order
    relevant_indices_sorted = np.array(np.sort(relevant_indices[0])[::-1])
    print(f"[LLM] Relevant indices (sorted): {relevant_indices_sorted}")

    # retrieve the most similar frames to be the context with a max of n items
    if relevant_indices_sorted.size > 0:
        # ensure if n is larger than available relevant indices, we don't exceed bounds
        if n > len(relevant_indices_sorted):
            n = len(relevant_indices_sorted)

        data["documents"] = np.array(data["documents"])
        data["ids"] = np.array(data["ids"])

        retrieved, timestamp = (
            data["documents"][relevant_indices_sorted[:n]],
            data["ids"][relevant_indices_sorted[:n]],
        )
        print(
            f"[LLM] Successfully retrieved context: {retrieved}, timestamp: {timestamp}"
        )
        return retrieved, timestamp
    else:
        print("[LLM] No relevant context found.")
        return None, None


def run_llm(prompt):
    """Process a prompt with the loaded LLM model, optionally with video context"""
    device = torch.device("cuda")

    print(f"[LLM] Processing prompt: {prompt}")

    maxtokens = None
    temp = None

    try:
        # get the most relevant observation from the data and it's timestamp.
        retrieved_texts, timestamp = retrieve_context(prompt, n=3, threshold=0.3)

        # formated as: "At time <timestamp>, <observation>"
        if retrieved_texts is not None and timestamp is not None:
            context = [
                f"At time {ts}, {text}" for ts, text in zip(timestamp, retrieved_texts)
            ]
            maxtokens = 100  # set a higher max token limit when context is available
            temp = 0.7  # set a higher temperature for more creative responses
        else:
            context = ["Provide a concise and friendly answer to the query."]
            maxtokens = 60  # enforce a concise answer when no context is available
            temp = 0.3  # lower temperature for more deterministic answers

        # Format as chat messages for TinyLlama with system message
        messages = [
            {
                "role": "system",
                "content": "You are an extremely kind and helpful AI assistant specalised in animal farming."
                "Provide one answer using the provided context shown to you in a video, provided in the context, if there is any available. "
                "Only answer the query asked by the user. "
                "Do NOT make up information. Never lie. If you don't know, do not attempt to answer. "
                "Provide a concise answer. "
                f"Your observed this context over video: {context}",
            },
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        # Set default values if not provided
        if maxtokens is None:
            maxtokens = 100  # default value if not set
        if temp is None:
            temp = 0.7  # default value if not set

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=maxtokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens (response)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()

    except Exception as e:
        print(f"[LLM] Error during inference: {e}")
        return f"Error processing request: {str(e)}"
