import os
import logging

from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
import openai

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s - %(message)s',
    filename='agent.log',
    filemode='a'
)

# Load Kubernetes config (if available)
try:
    config.load_kube_config()
    v1_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()
    logging.info("Kubernetes config loaded successfully.")
except Exception as exc:
    logging.error(f"Could not load Kubernetes config: {exc}")
    v1_api = None
    apps_api = None

# Set your OpenAI API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# ------------------------------------------------------------------------------
# Pydantic Model for returning the query & answer
# ------------------------------------------------------------------------------
class QueryResponse(BaseModel):
    query: str
    answer: str

# ------------------------------------------------------------------------------
# A small helper class (optional) to retrieve minimal cluster data on demand
# ------------------------------------------------------------------------------
class K8sHelper:
    """Fetches smaller bits of K8s data as needed (pods, deployments, etc.)."""
    def __init__(self, core_v1: client.CoreV1Api, apps_v1: client.AppsV1Api):
        self.core_v1 = core_v1
        self.apps_v1 = apps_v1

    def get_pod_count_in_namespace(self, namespace="default") -> int:
        """Return how many pods are in a namespace."""
        try:
            pods = self.core_v1.list_namespaced_pod(namespace=namespace)
            return len(pods.items)
        except Exception as e:
            logging.error(f"Error fetching pods in namespace {namespace}: {e}")
            return 0

    def get_node_count(self) -> int:
        """Return how many nodes in the cluster."""
        try:
            nodes = self.core_v1.list_node()
            return len(nodes.items)
        except Exception as e:
            logging.error(f"Error fetching nodes: {e}")
            return 0

# ------------------------------------------------------------------------------
# QueryProcessor: Orchestrates the GPT call plus minimal K8s lookups
# ------------------------------------------------------------------------------
class QueryProcessor:
    def __init__(self, helper: K8sHelper):
        self.helper = helper

    def process_query(self, user_query: str) -> str:
        """
        1. Possibly collect minimal cluster data (based on user_query).
        2. Send everything (plus system instructions) to GPT.
        3. Return GPT's direct answer.
        """
        try:
            # Minimal example: if user query references “How many pods in default namespace?”
            # we can fetch that number from K8s. Otherwise, skip or do a different check.
            if "pods" in user_query.lower() and "default" in user_query.lower():
                pod_count = self.helper.get_pod_count_in_namespace("default")
                cluster_hint = f"[Pod count in default namespace: {pod_count}]"
            elif "nodes" in user_query.lower():
                node_count = self.helper.get_node_count()
                cluster_hint = f"[Node count in cluster: {node_count}]"
            else:
                # In a real app, you'd parse more conditions
                cluster_hint = "[No special cluster data gathered]"

            # The exact wording you asked for:
            system_message = (
                "You are a Kubernetes cluster assistant. "
                "You help users get information about their Kubernetes cluster. "
                "Provide direct, concise answers without any explanations, punctuations, or additional context. "
                "Return only the specific information requested, avoiding identifiers when possible."
                "Always return a straight answer."
                "When returning any resource name, remove any trailing hyphens and random strings."
            )

            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": (
                        f"Cluster info snippet: {cluster_hint}\n"
                        f"User query: {user_query}"
                    )
                }
            ]

            # Official openai library usage:
            response = openai.ChatCompletion.create(
                model="gpt-4o",  
                messages=messages,
                temperature=0.9,
                max_tokens=50
            )
            answer = response.choices[0].message.content.strip()
            logging.info(f"GPT Answer: {answer}")
            return answer

        except Exception as e:
            logging.error(f"Error during GPT query: {e}")
            return "Error processing your request."

# ------------------------------------------------------------------------------
# Setup the helper & processor objects if config loaded
# ------------------------------------------------------------------------------
helper_obj = K8sHelper(v1_api, apps_api) if v1_api and apps_api else None
processor = QueryProcessor(helper_obj) if helper_obj else None

# ------------------------------------------------------------------------------
# Flask Endpoint
# ------------------------------------------------------------------------------
@app.route("/query", methods=["POST"])
def create_query():
    try:
        data = request.get_json()
        query = data.get("query", "")
        logging.info(f"Received query: {query}")

        if processor is None:
            logging.error("Kubernetes helper or processor not initialized.")
            return jsonify({"error": "Kubernetes client not initialized"}), 500

        # Get the final answer from our QueryProcessor
        answer_text = processor.process_query(query)

        # Build Pydantic response
        response_data = QueryResponse(query=query, answer=answer_text)

        return jsonify(response_data.model_dump())
    except ValidationError as e:
        logging.error(f"Validation Error: {e.errors()}")
        return jsonify({"error": str(e)}), 400


# ------------------------------------------------------------------------------
# Main Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Flask will run on port 8000 by default if you specify
    app.run(host="0.0.0.0", port=8000)
