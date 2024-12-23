import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
import os
from openai import OpenAI

config.load_kube_config()
v1 = client.CoreV1Api() # create an API client

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

# get basic info about the cluster
class KubernetesHelper:
    """
    A helper class that wraps common Kubernetes client actions
    """
    
    def __init__(self):
        """
        Initializes Kubernetes clients for CoreV1Api and AppsV1Api.
        """
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

    def get_pods(self, namespace="default"):
        return self.v1.list_namespaced_pod(namespace)

    def get_deployments(self, namespace="default"):
        return self.apps_v1.list_namespaced_deployment(namespace)

    def get_services(self, namespace="default"):
        return self.v1.list_namespaced_service(namespace)

    def get_nodes(self):
        return self.v1.list_node()

    def get_pod_logs(self, pod_name, namespace="default"):
        return self.v1.read_namespaced_pod_log(pod_name, namespace)

    def get_pod_status(self, pod_name, namespace="default"):
        return self.v1.read_namespaced_pod_status(pod_name, namespace)
    

# ----------------------------------------------------------------------------
# Query Processor: processing queries using Kubernetes information and GPT-based language understanding.
# ----------------------------------------------------------------------------
class Processer:
    def __init__(self):
        self.helper = KubernetesHelper()

    def process_query(self, query: str) -> str:
        '''
        1. retrieving the current cluster info
        2. Passing the info to GPT
        3. Retruning the answer
        '''
        try:
            system_message = (
                "You are a Kubernetes cluster assistant. "
                "You help users get information about their Kubernetes cluster. "
                "Provide direct, concise answers without any explanations, punctuation, or additional context. "
                "Return only the specific information requested, avoiding identifiers when possible."
                "When returning any resource name, remove any trailing hyphens and random strings."
                "Always return a straight answer."
            )

            cluster_info = self._get_cluster_info()

            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": (
                        f"Based on this Kubernetes cluster information:\n{cluster_info}"
                        f"\n\nQuery: {query}"
                    )
                },
            ]

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.9,
                max_tokens=3000
            )

            res = response.choices[0].message.content.strip()
            return res

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"Error: {e}"
        
        

    def _get_cluster_info(self) -> str:
        """
        Gathers relevant information about the cluster, such as:
          - Pods (names, statuses)
          - Deployments (names, replicas)
          - Services (names, types)
          - Nodes (names, statuses)

        Returns:
        str: A string representation of the collected cluster information.
        """
        try:
            pods = self.k8s_helper.get_pods()
            deployments = self.k8s_helper.get_deployments()
            services = self.k8s_helper.get_services()
            nodes = self.k8s_helper.get_nodes()

            # Build a dictionary with relevant data
            info_dict = {
                "pods": [
                    {"name": p.metadata.name, "status": p.status.phase}
                    for p in pods.items
                ],
                "deployments": [
                    {"name": d.metadata.name, "replicas": d.spec.replicas}
                    for d in deployments.items
                ],
                "services": [
                    {"name": s.metadata.name, "type": s.spec.type}
                    for s in services.items
                ],
                "nodes": [
                    {
                        "name": n.metadata.name,
                        "status": n.status.conditions[-1].type
                        if n.status.conditions
                        else "Unknown"
                    }
                    for n in nodes.items
                ],
            }

            return str(info_dict)

        except Exception as e:
            logging.error(f"Error gathering cluster info: {str(e)}")
            return str(e)
        


app = Flask(__name__)


class QueryResponse(BaseModel):
    query: str
    answer: str




@app.route('/query', methods=['POST'])
def create_query():
    try:
        # Extract the question from the request data
        request_data = request.json
        query = request_data.get('query')
        
        # Log the question
        logging.info(f"Received query: {query}")
        
        answer = Processer().process_query(query)

        # Log the answer
        logging.info(f"Generated answer: {answer}")
        
        # Create the response model
        response = QueryResponse(query=query, answer=answer)
        
        return jsonify(response.dict())
    
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
