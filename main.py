import logging
import io
import sys
import contextlib
from flask import Flask, request, jsonify
from pydantic import BaseModel
from kubernetes import client, config
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

app = Flask(__name__)

class QueryResponse(BaseModel):
    query: str
    answer: str

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load kubeconfig
config.load_kube_config()

# Context manager to capture stdout
@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    sys.stdout = stdout or io.StringIO()
    yield sys.stdout
    sys.stdout = old

@app.route('/query', methods=['POST'])
def create_query():
    try:
        request_data = request.json
        query = request_data.get('query')
        logging.info(f"Received query: {query}")

        # System prompt to generate code
        system_prompt = (
            "As an AI assistant, translate the user's natural language query into Python code "
            "using the Kubernetes Python client library to retrieve the requested information. "
            "Output only the code necessary to fulfill the query, without any additional text or explanations. "
            "Do not include any import statements, as the `client` and `config` modules are already imported and available. "
            "Include `config.load_kube_config()` to initialize the client configuration. "
            "Assign `client.CoreV1Api()` to a variable before using it. "
            "Ensure all variables are properly defined before use. "
            "**Only print the required result without any additional text or messages in the print statements.**"
        )

        # Get the code from OpenAI
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}
            ],
            max_tokens=150,
            n=1,
            temperature=0,
        )

        # Extract the assistant's response
        response_text = response.choices[0].message.content.strip()
        logging.info(f"Assistant's raw response:\n{response_text}")

        # Remove code block markers if present
        response_text = response_text.strip('```python').strip('```').strip()

        # Define a whitelist of safe built-in functions
        safe_builtins = {
            'print': print,
            'len': len,
            'str': str,
            'list': list,
            'dict': dict,
            'enumerate': enumerate,
            'range': range,
            'min': min,
            'max': max,
            'abs': abs,
            'sum': sum,
            'sorted': sorted,
            '__import__': __import__,  # Add __import__ to the whitelist
        }

        # Prepare the execution environment
        exec_globals = {
            '__builtins__': safe_builtins,
            'client': client,
            'config': config
        }
        exec_locals = {}

        # Capture the output of the executed code
        with stdoutIO() as stdout:
            exec(response_text, exec_globals, exec_locals)

        output = stdout.getvalue().strip()
        logging.info(f"Execution output:\n{output}")

        # Create the response
        response_model = QueryResponse(query=query, answer=output)
        return jsonify(response_model.model_dump())

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
