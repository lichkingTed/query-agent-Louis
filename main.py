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
        
        
    def _get_cluster_info(self):
        '''
        Retrieves basic cluster information.
        '''
        try:
            pods = self.helper.get_pods()
            deployments = self.helper.get_deployments()
            services = self.helper.get_services()
            nodes = self.helper.get_nodes()
            
            # Build a dictionary with relevant data
            info_dict = {
                "pods": [
                    {
                        "name": p.metadata.name,
                        "status": p.status.phase,
                        "namespace": p.metadata.namespace,
                        "ip": p.status.pod_ip,
                        "node": p.spec.node_name,
                        "containers": [
                            {
                                "name": c.name,
                                "image": c.image,
                                "ready": c.ready,
                                "restart_count": c.restart_count if hasattr(c, 'restart_count') else 0
                            } for c in p.status.container_statuses
                        ] if p.status.container_statuses else [],
                        "labels": p.metadata.labels,
                        "creation_time": p.metadata.creation_timestamp.isoformat() if p.metadata.creation_timestamp else None,
                        "volumes": [
                            {
                                "name": v.name,
                                "type": next(iter(v.to_dict().keys() - {'name'}), None)
                            } for v in p.spec.volumes
                        ] if p.spec.volumes else []
                    }
                    for p in pods.items
                ],
                "deployments": [
                    {
                        "name": d.metadata.name,
                        "namespace": d.metadata.namespace,
                        "replicas": d.spec.replicas,
                        "available_replicas": d.status.available_replicas,
                        "ready_replicas": d.status.ready_replicas,
                        "strategy": d.spec.strategy.type if d.spec.strategy else None,
                        "labels": d.metadata.labels,
                        "selector": d.spec.selector.match_labels if d.spec.selector else None,
                        "creation_time": d.metadata.creation_timestamp.isoformat() if d.metadata.creation_timestamp else None
                    }
                    for d in deployments.items
                ],
                "services": [
                    {
                        "name": s.metadata.name,
                        "namespace": s.metadata.namespace,
                        "type": s.spec.type,
                        "cluster_ip": s.spec.cluster_ip,
                        "external_ip": s.spec.external_i_ps if hasattr(s.spec, 'external_i_ps') else None,
                        "ports": [
                            {
                                "port": p.port,
                                "target_port": p.target_port,
                                "protocol": p.protocol
                            } for p in s.spec.ports
                        ] if s.spec.ports else [],
                        "selector": s.spec.selector,
                        "creation_time": s.metadata.creation_timestamp.isoformat() if s.metadata.creation_timestamp else None
                    }
                    for s in services.items
                ],
                "nodes": [
                    {
                        "name": n.metadata.name,
                        "status": [
                            cond.type for cond in n.status.conditions
                            if cond.status == "True"
                        ] if n.status.conditions else [],
                        "capacity": n.status.capacity,
                        "allocatable": n.status.allocatable,
                        "architecture": n.status.node_info.architecture if n.status.node_info else None,
                        "container_runtime": n.status.node_info.container_runtime_version if n.status.node_info else None,
                        "kernel_version": n.status.node_info.kernel_version if n.status.node_info else None,
                        "os_image": n.status.node_info.os_image if n.status.node_info else None,
                        "addresses": [
                            {"type": addr.type, "address": addr.address}
                            for addr in n.status.addresses
                        ] if n.status.addresses else [],
                        "labels": n.metadata.labels
                    }
                    for n in nodes.items
                ]
            }
            logging.info(f"Cluster info: {info_dict}")

            return str(info_dict)

        except Exception as e:
            logging.error(f"Error retrieving cluster info: {e}")
            return f"Error: {e}"
        


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
