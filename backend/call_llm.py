import requests

JETSON_IP = "172.17.0.1"

def call_llm(prompt):
	print("Opened call_llm")
	url = f"http://{JETSON_IP}:8000/run_llm"
	response = requests.post(url, json = {"prompt": prompt})
	print(response)
	return response.json()


# ether use ollama or huggingface to handle the LLM request
# probably go with huggingface 

