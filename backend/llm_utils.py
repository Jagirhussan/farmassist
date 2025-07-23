# backend/llm_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataprocessing.video_query import get_video_context

# Global variables to store the model (loaded once)
tokenizer = None
model = None

def load_model():
    """Load the LLM model once when the server starts"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        print("[LLM] Loading model...")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Set up pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("[LLM] Model loaded successfully!")

def run_llm(prompt: str, use_video_context: bool = True) -> str:
    """Process a prompt with the loaded LLM model, optionally with video context"""
    load_model()  # Ensure model is loaded
    
    print(f"[LLM] Processing prompt: {prompt}")
    
    try:
        # Get video context if requested
        user_message = prompt
        if use_video_context:
            print("[LLM] Retrieving video context...")
            video_context = get_video_context(prompt, n_results=3)
            if "No relevant video context found" not in video_context:
                user_message = f"{video_context}\n\nUser Question: {prompt}"
                print(f"[LLM] Added video context to prompt")
        
        # Format as chat messages for TinyLlama with system message
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on video content, specifically focused on animal welfare and health insights on a farm in New Zealand. Provide clear, concise, and helpful responses and don't provide any irrelevant information. If no information can be found please state this and don't make up any information. If you are unsure about the answer, please say so. "},
            {"role": "user", "content": user_message}
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
