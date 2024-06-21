import logging
import subprocess
from typing import Optional
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

app = Flask(__name__)

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.route('/query-agent', methods=['POST'])
def get_question_and_facts():
    try:
        # Extract the question from the request data
        request_data = request.json
        question = request_data.get('question')
        
        # Log the question
        logging.info(f"Received question: {question}")
        
        # Here, you can implement your logic to generate an answer for the given question.
        # For simplicity, we'll just echo the question back in the answer.
        answer = "14"
        
        # Log the answer
        logging.info(f"Generated answer: {answer}")
        
        # Create the response model
        response = AnswerResponse(question=question, answer=answer)
        
        return jsonify(response.dict())
    
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
