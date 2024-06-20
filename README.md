# Query Agent Assignemnt @Cleric

This document outlines the requirements and guidlines about the Cleric Query Agent Assignment. We also included an example for handling question retrievals and answer submissions for your reference.

## Objective

Your task is to develop an AI agent that can accurately answer questions related to applications deployed on a Kubernetes cluster by interacting with it.
The testing of your AI agent will be automated through a GitHub Actions pipeline, where a Minikube cluster will be started to simulate the Kubernetes environment.

**Focus on creating your agent. Your agent will need to interact with the Kubernetes cluster in order to retrieve the correct answer, but the deployment details are handled for you.**

We will test your AI agent by:

1. Starting a Minikube cluster in a GitHub Actions pipeline.
2. Deploy a set of services to the Kubernetes cluster.
3. Running your agent with `python agent.py &`. (DO NOT modify the name of the script, or the evaluation pipeline will fail)
4. Triggering a set of questions, one by one, and expecting your agent to return the correct answers.
5. Evaluating the number of correct answers provided by your AI agent.

## Assignment Details

- Ensure your packages are compatible with **Python 3.10**
- The kubeconfig file will be located at `~/.kube/config`.
- Use `GPT-4o` or models with comparable performance.

### API

Your agent should offer a POST endpoint for question submission at `http://localhost:8000/query-agent`.
Please note that your agent should be running on port `8000`.

The POST endpoint should accept a JSON payload:

```json
{
    "question": "How many pods are there in the default namespace?"
}
```

Responses should be in JSON format, structured as follows (using pydantic):

```python
from pydantic import BaseModel

class SubmitQuestionResponse(BaseModel):
    question: str,
    answer: str
```

### Scope of Question

- We will ask questions requires only **read** actions from your agent.
- The set of questions might related to the status, information, or logs of the resources deployed on Minikube.
- The answer of the questions will not dynamically change.
- There will be around 10 questions, your agent will be evaluated with accuracy.
- The questions will be independent to each other.
- Return **only** the answer, and discard any identifier (e.g. return `mongodb` instead of `mongodb-56c598c8fc`).

## Submission

- Submit your repository to this [link]().
  - The validator should return your score in a few minutes.
  - If you encounter run errors, please wait a few minutes to try again.
  - DO NOT refresh the browser, your session will be lost.
- Use [this form]() to submit the following:
  - Your GitHub repo
    - Please include `README.md` describing your approach to solve the problem.
  - A short (5 minute) Loom video
    - Please only use Loom.com.
    - Please spend no more than 2 minutes talking through your submission.
    - Please spend 3 minutes telling us about yourself and why you are interested in working in this space.
    - We’d love to find out more about you as a person, so try not to be too formal or academic.

We suggest you to test your agent in your environment before submitting.
You can install [Minikue](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Fx86-64%2Fstable%2Fbinary+download) locally.

## Examples

Your response should contain **only** the answer. Here are some examples of the questions and expected responses:

- Question: `"Which pod is spawned by my-deployment"`
  - Expected Response: `"my-pod"`
- Question: `"What is the status of the pod named 'example-pod'?"`
  - Expected Response: `"Running"`
- Question: `"How many nodes are there in the cluster?"`
  - Expected Response: `"2"`
