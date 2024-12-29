import json
import logging

import openai
from flask import Flask, request, jsonify
from kubernetes import config, client
from dotenv import load_dotenv
import os

from openai import OpenAI
from pydantic import BaseModel, ValidationError

# Load Openai API key from the .env and create open AI client.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai.api_key)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')
# Initialize the flask.
app = Flask(__name__)


# Define the query response format.
class QueryResponse(BaseModel):
    query: str
    answer: str


# Load the kube config and create kube client based on it.
def create_kube_client():
    try:
        config.load_kube_config(os.path.expanduser("~/.kube/config"))
        kube_client = client.CoreV1Api()
        logging.info("Kubernetes client initialized.")
        return kube_client
    except Exception as e:
        logging.error(f"Failed to initialize Kubernetes client: {str(e)}")
        return None


# Create kube client, so we could use it later in the query.
kube_client = create_kube_client()


# Create the kubernetes snapshot that contains all the kubernetes info in json format.
def create_kubernetes_snapshot():
    snapshot = {}
    if kube_client is None:
        logging.error(f"Kubernetes kube_client not initialized.")
        return snapshot
    # Nodes
    nodes = kube_client.list_node()
    snapshot['nodes'] = [
        {
            'name': node.metadata.name,
            'labels': node.metadata.labels,
            'status': node.status.conditions[-1].type if node.status.conditions else "Unknown"
        }
        for node in nodes.items
    ]

    # Namespaces
    namespaces = kube_client.list_namespace()
    snapshot['namespaces'] = [namespace.metadata.name for namespace in namespaces.items]

    # Pods
    all_pods = []
    for namespace in snapshot['namespaces']:
        pods = kube_client.list_namespaced_pod(namespace=namespace)
        for pod in pods.items:
            all_pods.append({
                'name': pod.metadata.name,
                'namespace': pod.metadata.namespace,
                'status': pod.status.phase,
                'node_name': pod.spec.node_name,
                'labels': pod.metadata.labels
            })
    snapshot['pods'] = all_pods

    # Services
    all_services = []
    for namespace in snapshot['namespaces']:
        services = kube_client.list_namespaced_service(namespace=namespace)
        for service in services.items:
            all_services.append({
                'name': service.metadata.name,
                'namespace': service.metadata.namespace,
                'type': service.spec.type,
                'cluster_ip': service.spec.cluster_ip,
                'ports': [{'port': port.port, 'target_port': port.target_port} for port in service.spec.ports]
            })
    snapshot['services'] = all_services

    # Persistent Volumes
    pvs = kube_client.list_persistent_volume()
    snapshot['persistent_volumes'] = [
        {
            'name': pv.metadata.name,
            'capacity': pv.spec.capacity,
            'access_modes': pv.spec.access_modes,
            'reclaim_policy': pv.spec.persistent_volume_reclaim_policy,
            'status': pv.status.phase
        }
        for pv in pvs.items
    ]

    # Persistent Volume Claims
    all_pvcs = []
    for namespace in snapshot['namespaces']:
        pvcs = kube_client.list_namespaced_persistent_volume_claim(namespace=namespace)
        for pvc in pvcs.items:
            all_pvcs.append({
                'name': pvc.metadata.name,
                'namespace': pvc.metadata.namespace,
                'volume': pvc.spec.volume_name,
                'access_modes': pvc.spec.access_modes,
                'resources': pvc.spec.resources.requests,
                'status': pvc.status.phase
            })
    snapshot['persistent_volume_claims'] = all_pvcs

    # Return the snapshot as a JSON object
    return json.dumps(snapshot, indent=4)


# Connect to the chatGpt 4 and ask them to return the precise answer.
# To implement that, we need to generate the kubernetes snapshot to make sure GPT has all the kubernetes info.
# Also, we need to give the system command to make sure GPT only returns precise answer instead of whole sentence.
def gpt_query(query):
    snapshot_json = create_kubernetes_snapshot()
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that provides information about Kubernetes clusters. Only output answer word. Don't give me the sentence."},
                {"role": "system", "content": f"Cluster Snapshot: {snapshot_json}"},
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )
        answer = completion.choices[0].message.content
        return answer
    except Exception as e:
        # Log any errors
        logging.error(f"Error processing query '{query}': {str(e)}")
        return "An error occurred while processing your query."

# Create a REST API  endpoint.
@app.route('/query', methods=['POST'])
def create_query():
    try:
        # Extract the question from the request data
        request_data = request.json
        query = request_data.get('query')

        # Log the question
        logging.info(f"Received query: {query}")
        answer = gpt_query(query)

        # Log the answer
        logging.info(f"Generated answer: {answer}")

        # Create the response model
        response = QueryResponse(query=query, answer=answer)

        return jsonify(response.dict())

    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
