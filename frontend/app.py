from flask import Flask, render_template, request
import requests

app = Flask(__name__)

BACKEND_URL = "http://localhost:5050/ask_llm"  # If backend is running on port 5050

@app.route('/', methods=['GET', 'POST'])
def home():
	user_input = ""
	response = ""

	if request.method == 'POST':
		user_input = request.form['input_text']
		print(f"[Frontend] User input: {user_input}")
		try:
			res = requests.post(BACKEND_URL, json={"prompt": user_input})
			print(f"[Frontend] Raw backend response: {res.text}")  # Debug print
			response_data = res.json()
			print(f"[Frontend] Parsed backend response: {response_data}")
			response = response_data.get("output", "No response from backend.")

		except Exception as e:
			response = f"Error: {e}"

	return render_template('index.html', input=user_input, output=response)

if __name__ == '__main__':
    print("[Frontend] Starting Flask app on http://localhost:3000 ...")
    app.run(host='0.0.0.0', port=3000, debug=True)


