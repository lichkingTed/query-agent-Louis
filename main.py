import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
from openai import OpenAI
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s %(levelname)s - %(message)s',
    filename='agent.log',
    filemode='a'
)

app = Flask(__name__)

class QueryResponse(BaseModel):
    query: str
    answer: str

# Try to load Kubernetes configuration
try:
    config.load_kube_config()
    logging.info("Successfully loaded Kubernetes configuration")
    k8s_client = client.CoreV1Api()
    k8s_deployment_client = client.AppsV1Api()
except Exception as e:
    logging.error(f"Failed to load Kubernetes configuration: {e}")
    k8s_client = None
    k8s_deployment_client = None

openai_client = OpenAI()

@app.route('/query', methods=['POST'])
def create_query():
    try:
        # Extract the question from the request data
        request_data = request.json
        query = request_data.get('query')
        
        # Log the question
        logging.info(f"Received query: {query}")

        # Validate the Kubernetes client initialization
        if k8s_client is None or k8s_deployment_client is None:
            logging.error("Kubernetes client not initialized")
            return jsonify({"error": "Kubernetes client not initialized"}), 500

        # Use OpenAI to analyze user's intent
        gpt_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Kubernetes administrator."},
                {"role": "system", "content": "Identify the query's intent then provide the user's intention and kubectl command."},
                {"role": "system", "content": "If the query's intent is vague, return: Could not understand your intent."},
                {"role": "user", "content": query}
            ],
            response_format={"type": "text"},
            max_tokens=100,
            temperature=0.1
        )

        logging.info(f"GPT response: {gpt_response}")
        intent = gpt_response.choices[0].message.content
        logging.info(f"User intent based on GPT analysis: {intent}")
        answer = (
            "Could not understand your intent.\n"
            + "I currently understand queries about the following Kubernetes operations:\n"
            + "- Pod count in the default namespace\n"
            + "- Pod count across all namespaces\n"
            + "- Pod status for a specific pod\n"
            + "- Logs for a specific pod\n"
            + "- Pods spawned by a specific deployment\n"
            + "- Running pods in the default namespace\n"
            + "- Pods with a specific label\n"
            + "- Service count in the default namespace\n"
            + "- Service count across all namespaces\n"
            + "- Deployment count in the default namespace\n"
            + "- Node count in the cluster\n"
            + "- Namespace count in the cluster\n"
            + "- Deployment details for a specific deployment\n"
            + "- Resource quota in the default namespace\n"
        )

        # Handle Kubernetes operations based on the user's intent
        try:
            if  "kubectl get pods" in intent and "default namespace" in intent:
                pods = k8s_client.list_namespaced_pod(namespace="default")
                answer = f"There are currently {len(pods.items)} pods in the default namespace."

            elif "kubectl get pods" in intent and "all" in intent:
                pods = k8s_client.list_pod_for_all_namespaces()
                answer = f"There are currently {len(pods.items)} pods for all namespaces."

            elif "kubectl get pod" in intent and "status" in intent:
                try:
                    pod_name = re.search(r'"(.*?)"', intent).group(1)
                    pod = k8s_client.read_namespaced_pod(name=pod_name, namespace="default")
                    answer = f"The status of the pod '{pod_name}' in the default namespace is {pod.status.phase}."
                except IndexError:
                    answer = "Pod name not provided in the query."
                except client.exceptions.ApiException:
                    answer = f"Pod '{pod_name}' not found in the default namespace."

            elif "kubectl logs" in intent and "logs" in intent:
                pod_name = re.search(r'"(.*?)"', intent).group(1)
                logs = k8s_client.read_namespaced_pod_log(name=pod_name, namespace="default")
                answer = f"Logs for pod '{pod_name}':\n{logs[:200]}..." 

            elif "kubectl get pods --selector=" in intent and "pod" in intent:
                deployment_name = re.search(r'"(.*?)"', intent).group(1)
                pods = k8s_client.list_namespaced_pod(namespace="default")
                deployment_pods = [
                    pod.metadata.name
                    for pod in pods.items
                    if any(
                        owner.kind == "ReplicaSet" and owner.name.startswith(deployment_name)
                        for owner in pod.metadata.owner_references
                    )
                ]
                if deployment_pods:
                    answer = f"The pod(s) spawned by deployment '{deployment_name}' are: {', '.join(deployment_pods)}"
                else:
                    answer = f"No pods found for deployment '{deployment_name}'"

            elif "kubectl get pods" in intent and "running" in intent:
                pods = k8s_client.list_namespaced_pod(namespace="default")
                running_pods = [pod for pod in pods.items if pod.status.phase == "Running"]
                answer = f"There are currently {len(running_pods)} running pods in the default namespace."

            elif "kubectl get pods" in intent and "label" in intent:
                try:
                    label = re.search(r'"(.*?)"', intent).group(1)
                    pods = k8s_client.list_namespaced_pod(namespace="default", label_selector=label)
                    answer = f"There are currently {len(pods.items)} pods with label '{label}' in the default namespace."
                except client.exceptions.ApiException:
                    answer = f"Could not find pods with label '{label}' in the default namespace."

            elif "kubectl get services" in intent and "default namespace" in intent:
                services = k8s_client.list_namespaced_service(namespace="default")
                answer = f"There are currently {len(services.items)} services in the default namespace."

            elif "kubectl get services" in intent:
                services = k8s_client.list_service_for_all_namespaces()
                answer = f"There are currently {len(services.items)} services in the namespace."

            elif "kubectl get deployments" in intent and "default namespace" in intent:
                deployments = k8s_deployment_client.list_namespaced_deployment(namespace="default")
                answer = f"There are currently {len(deployments.items)} deployments in the default namespace."

            elif "kubectl get nodes" in intent:
                nodes = k8s_client.list_node()
                logging.info(nodes.items)
                answer = f"There are currently {len(nodes.items)} nodes in the cluster."

            elif "kubectl get namespaces" in intent:
                namespaces = k8s_client.list_namespace()
                namespace_names = [ns.metadata.name for ns in namespaces.items]
                answer = f"There are {len(namespaces.items)} namespaces in the cluster. The current namespaces are {namespace_names}"

            elif "kubectl describe deployment" in intent:
                try:
                    deployment_name = re.findall(r'<(.*?)>', intent)
                    deployment = k8s_deployment_client.read_namespaced_deployment(name=deployment_name, namespace="default")
                    answer = f"Deployment: '{deployment_name}', Replicas: {deployment.spec.replicas}, Strategy: {deployment.spec.strategy.type}"
                except client.exceptions.ApiException:
                    answer = f"Could not describe deployment '{deployment_name}'."

            elif "kubectl get resourcequota" in intent:
                quotas = k8s_client.list_namespaced_resource_quota(namespace="default")
                answer = f"Resource quota for default namespace:\n{quotas.items[0].status.hard}" if quotas.items else "No resource quota set."

        except client.exceptions.ApiException as e:
            logging.error(f"Kubernetes API Exception: {e}")
            answer = "An error occurred while processing your Kubernetes query."
        
        # Log the answer
        logging.info(f"Generated answer: {answer}")
        
        # Create the response model
        response = QueryResponse(query=query, answer=answer)
        
        return jsonify(response.model_dump())
    
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
