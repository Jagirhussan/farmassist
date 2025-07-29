# backend/llm_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import chromadb


# Global variables to store the model (loaded once)
tokenizer = None
model = None
model_encoder = None

def load_models():
    """Load the LLM model once when the server starts"""
    global tokenizer, model, model_encoder

    if tokenizer is None or model is None or model_encoder is None:
        print("[LLM] Loading models...")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model_encoder = SentenceTransformer("all-MiniLM-L6-v2")

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
    return collection.get(include=['documents', 'embeddings'])

def retrieve_context(query):
    """Retrieve relevant context from ChromaDB based on the query."""
    print(f"[LLM] Retrieving context for query: {query}")
    # embed the query to the same format as the stored embeddings
    query_embedded = model_encoder.encode(query)
    # retrieve the video frame data
    data = retrieve_data()

    print(f"data['embeddings'] dtype: {type(data['embeddings'][0])} | {np.array(data['embeddings']).dtype}")
    print(f"query_embedded dtype: {type(query_embedded)} | {np.array(query_embedded).dtype}")


    # calculate the similarities with the cosine similarity
    similarities = model_encoder.similarity(data['embeddings'], query_embedded)
    # retrieve the most similar document for reference
    retrieved, timestamp = data['documents'][similarities.argmax().item()], data['ids'][similarities.argmax().item()]
    print(f"[LLM] Successfully retrieved context: {retrieved}, timestamp: {timestamp}")
    return retrieved, timestamp

def run_llm(prompt):
    """Process a prompt with the loaded LLM model, optionally with video context"""
    load_models()  # Ensure model is loaded
    
    print(f"[LLM] Processing prompt: {prompt}")
    
    try:
        # get the most relevant observation from the data and it's timestamp.
        retrieved_texts, timestamp = retrieve_context(prompt)

        # Format as chat messages for TinyLlama with system message
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."
             "Provide one Answer ONLY the following query based on the context provided below. "
             "Do not generate or answer any other questions. "
             "Do not make up or infer any information that is not directly stated in the context. "
             "Provide a concise answer."
             f"{retrieved_texts}"
             f"{timestamp}"},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,  # Use max_new_tokens instead of max_length
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (response)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    except Exception as e:
        print(f"[LLM] Error during inference: {e}")
        return f"Error processing request: {str(e)}"
