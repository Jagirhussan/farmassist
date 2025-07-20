import requests
from typing import Dict, Any

JETSON_IP = "172.23.98.136"

def call_llm(prompt: str) -> Dict[str, Any]:
    """
    Send a prompt to the Jetson server running the LLM.
    
    Args:
        prompt (str): The input prompt for the LLM
        
    Returns:
        Dict[str, Any]: Response from the LLM server
        
    Raises:
        requests.RequestException: If there's an error with the HTTP request
    """
    url = f"http://{JETSON_IP}:8000/run_llm"
    
    try:
        response = requests.post(
            url, 
            json={"prompt": prompt},
            timeout=30  # Add timeout to prevent hanging
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM server: {e}")
        return {"error": f"Failed to connect to LLM server: {str(e)}"}
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"} 

