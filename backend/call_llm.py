import requests

JETSON_IP = "172.23.98.136"

def call_llm(prompt):
	url = f"http://{JETSON_IP}:8000/run_llm"
	response = requests.post(url, json = {"prompt": prompt})
	return response.json()
