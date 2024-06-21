# Cleric Query Agent Assignment

## Introduction
This document outlines the requirements and guidelines for the Cleric Query Agent Assignment. Your task is to develop an AI agent capable of accurately answering questions about applications deployed on a Kubernetes cluster.

## Objective
Create an AI agent that interacts with a Kubernetes cluster to answer questions about its deployed applications. Your agent will be tested through an automated GitHub Actions pipeline using a Minikube cluster to simulate the Kubernetes environment.

## Assignment Details

### Technical Requirements
- Use Python 3.10
- The kubeconfig file will be located at `~/.kube/config`
- Utilize GPT-4 or a model with comparable performance for natural language processing

### API Specifications
Your agent should provide a POST endpoint for question submission:
- URL: `http://localhost:8000/query-agent`
- Port: 8000
- Payload format:
  ```json
  {
      "question": "How many pods are in the default namespace?"
  }
  ```
- Response format (using Pydantic):
  ```python
  from pydantic import BaseModel

  class SubmitQuestionResponse(BaseModel):
      question: str
      answer: str
  ```

### Question Scope
- Questions will require only read actions from your agent
- Topics may include status, information, or logs of resources deployed on Minikube
- Answers will not change dynamically
- Approximately 10 questions will be asked
- Questions are independent of each other
- Return only the answer, without identifiers (e.g., "mongodb" instead of "mongodb-56c598c8fc")

## Submission Guidelines
1. Submit your repository to [submission link]
   - The validator will return your score within a few minutes
   - If you encounter errors, wait a few minutes before retrying
   - Do not refresh the browser to avoid losing your session
2. Complete [this form] with the following:
   - Your GitHub repository URL
   - A 5-minute Loom video (use Loom.com only)
     - 2 minutes: Explain your approach to solving the problem
     - 3 minutes: Tell us about yourself and your interest in this field

### Submission Requirements
1. GitHub Repository
   - Include a `README.md` file describing your approach
   - Ensure your main script is named `agent.py`
2. Loom Video
   - Keep it informal and personal
   - Focus on your motivation and background

## Testing Your Agent
We recommend testing your agent locally before submission:
1. Install [Minikube](https://minikube.sigs.k8s.io/docs/start/)
2. Set up a local Kubernetes cluster
3. Deploy sample applications
4. Run your agent and test with sample questions

## Evaluation Criteria
- Accuracy of answers
- Code quality and organization
- Clarity of explanation in README and video

## Example Questions and Responses
1. Q: "Which pod is spawned by my-deployment?"
   A: "my-pod"
2. Q: "What is the status of the pod named 'example-pod'?"
   A: "Running"
3. Q: "How many nodes are there in the cluster?"
   A: "2"

## Deadline
Please submit your assignment by [insert deadline here].

For any questions or clarifications, please contact [insert contact information].
