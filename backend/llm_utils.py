# backend/llm_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from numpy import dot
from numpy.linalg import norm
import time

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


def retrieve_context(query, n=3, threshold=0.3):
    """Retrieve relevant context from ChromaDB based on the query."""
    print(f"[LLM] Retrieving context for query: {query}")
    # embed the query to the same format as the stored embeddings
    query_embedded = model_encoder.encode(query)
    # retrieve the video frame data
    data = retrieve_data()
    # calculate the similarities with the cosine similarity
    # similarities = model_encoder.similarity(np.array(data['embeddings'], dtype=np.float32), np.array(query_embedded, dtype=np.float32))
    print(data)
    similarities = np.array(
        [
            dot(e, query_embedded) / (norm(e) * norm(query_embedded))
            for e in data["embeddings"]
        ]
    )

    print(f"[LLM] Similarities: {similarities}")

    sorted_indices = np.argsort(similarities)[::-1]

    # check which similarities are higher than a threshold (e.g., 0.5)
    filtered_indices = [i for i in sorted_indices if similarities[i] > threshold]

    top_n_indices = filtered_indices[:n]

    top_similarities = [similarities[i] for i in top_n_indices]
    print(f"[LLM] Relevant indices (sorted): {top_n_indices}")
    print(f"[LLM] Similarity scores of top relevant contexts: {top_similarities}")

    # order the relevant indices by similarity score in descending order
    print(f"[LLM] Relevant indices (sorted): {top_n_indices}")

    # retrieve the most similar frames to be the context with a max of n items
    if len(top_n_indices) > 0:
        # ensure if n is larger than available relevant indices, we don't exceed bounds
        if n > len(top_n_indices):
            n = len(top_n_indices)

        data["documents"] = np.array(data["documents"])
        data["ids"] = np.array(data["ids"])

        retrieved, timestamp = (
            data["documents"][top_n_indices[:n]],
            data["ids"][top_n_indices[:n]],
        )
        print(
            f"[LLM] Successfully retrieved context: {retrieved}, timestamp: {timestamp}"
        )
        return retrieved, timestamp
    else:
        print("[LLM] No relevant context found.")
        return None, None


def run_llm(prompt):
    device = torch.device("cuda")
    print(f"[LLM] Processing prompt: {prompt}")
    start_time = time.time()

    try:
        retrieved_texts, timestamp = retrieve_context(prompt, n=3, threshold=0.3)

        if retrieved_texts is not None:
            context = " ".join(
                [
                    f"At time {ts}, {text}."
                    for ts, text in zip(timestamp, retrieved_texts)
                ]
            )
            maxtokens, temp = 100, 0.7
        else:
            context = "Provide a concise and friendly answer to the query."
            maxtokens, temp = 60, 0.4

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant specialised in animal farming. "
                    "Use the provided context if available. "
                    "Do not make up information. "
                    f"Video context: {context}"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        # Generate once with scores
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=maxtokens,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_tokens = outputs.sequences[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Confidence estimation
        scores = torch.stack(outputs.scores)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        token_confidences = probs.max(dim=-1).values
        avg_confidence = token_confidences.mean().item()

        latency = round(time.time() - start_time, 2)
        num_tokens = len(generated_tokens)
        context_items = (
            len(retrieved_texts)
            if retrieved_texts is not None and len(retrieved_texts) > 0
            else 0
        )

        # print the system prompt for debugging
        print(f"[LLM] System prompt:\n{messages[0]['content']}")
        # print all metrics to server logs
        print(f"[LLM] Response: {response.strip()}")
        print(f"[LLM] Confidence: {avg_confidence:.3f}")
        print(f"[LLM] Tokens generated: {num_tokens}")
        print(f"[LLM] Latency: {latency}s")
        print(f"[LLM] Context items used: {context_items}")
        print(f"[LLM] Temperature: {temp}")

        return response.strip()

    except Exception as e:
        print(f"[LLM] Error during inference: {e}")
        return f"Error processing request: {str(e)}"
