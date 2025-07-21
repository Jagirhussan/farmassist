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
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")
        print("[LLM] Model loaded successfully!")

def run_llm(prompt: str, use_video_context: bool = True) -> str:
    """Process a prompt with the loaded LLM model, optionally with video context"""
    load_model()  # Ensure model is loaded
    
    print(f"[LLM] Processing prompt: {prompt}")
    
    try:
        # Get video context if requested
        full_prompt = prompt
        if use_video_context:
            print("[LLM] Retrieving video context...")
            video_context = get_video_context(prompt, n_results=3)
            if "No relevant video context found" not in video_context:
                full_prompt = f"{video_context}\n\nUser Question: {prompt}"
                print(f"[LLM] Added video context to prompt")
        
        # Tokenize the input
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 150,  # Longer for context
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        response = response[len(full_prompt):].strip()
        
        return response
    
    except Exception as e:
        print(f"[LLM] Error during inference: {e}")
        return f"Error processing request: {str(e)}"
