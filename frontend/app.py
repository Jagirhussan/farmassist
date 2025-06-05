from flask import Flask, render_template, request
import requests

# Create a Flask app instance
app = Flask(__name__)

# URL of the backend service
BACKEND_URL = "http://localhost:5050/ask_llm"  # Backend runs on port 5050

# Helper function to handle errors and return readable messages
def handle_error(error_message):
    if "ConnectionError" in str(error_message):
        return "Unable to connect to the backend. Please try again later."
    elif "No response from backend" in str(error_message):
        return "The backend did not return a response. Please check the backend service."
    else:
        return "An unexpected error occurred. Please try again."

# Define the home route to handle GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def home():
    user_input = ""  # Stores user input from the form
    response = ""  # Stores the response from the backend

    if request.method == 'POST':  # Handle form submission
        user_input = request.form.get('input_text', "").strip()  # Get user input and remove extra spaces
        print(f"[Frontend] User input: {user_input}")

        if not user_input:  # Check if the input is empty
            response = "Please enter a prompt before clicking the send button."
        else:
            try:
                # Send the user input to the backend and get the response
                res = requests.post(BACKEND_URL, json={"prompt": user_input})
                response_data = res.json()  # Parse the backend's JSON response
                print(f"[Frontend] Parsed backend response: {response_data}")
                # Extract the "output" field from the backend response
                response = response_data.get("output", "No response from backend.")
            except Exception as e:
                # Handle errors during the request to the backend
                response = handle_error(e)

    # Render the HTML template with user input and backend response
    return render_template('index.html', input=user_input, output=response)

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    print("[Frontend] Starting Flask app on http://localhost:3000 ...")
    app.run(host='0.0.0.0', port=3000, debug=True)


