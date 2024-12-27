import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config


openai_client = OpenAI()


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s - %(message)s',
    filename='agent.log',
    filemode='a'
)

try:
    config.load_kube_config()
except Exception as e:
    logging.error(f"Error loading kube config: {e}")





# ------------------------------------------------------------------------------
# Pydantic Model for the Response
# ------------------------------------------------------------------------------
class QueryResponse(BaseModel):
    query: str
    answer: str

# ------------------------------------------------------------------------------
# Kubernetes Helper
# ------------------------------------------------------------------------------
class KubernetesHelper:
    """
    A helper class that wraps common Kubernetes client actions,
    simplifying access to pods, deployments, services, and nodes.
    """

    def __init__(self):
        """
        Initializes Kubernetes clients for CoreV1Api and AppsV1Api.
        """
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

    def get_pods(self, namespace="default"):
        """Returns a list of pods in the specified namespace."""
        return self.v1.list_namespaced_pod(namespace)

    def get_deployments(self, namespace="default"):
        """Returns a list of deployments in the specified namespace."""
        return self.apps_v1.list_namespaced_deployment(namespace)

    def get_services(self, namespace="default"):
        """Returns a list of services in the specified namespace."""
        return self.v1.list_namespaced_service(namespace)

    def get_nodes(self):
        """Returns a list of nodes in the cluster."""
        return self.v1.list_node()

    def get_pod_logs(self, pod_name, namespace="default"):
        """Returns the logs of a specific pod in the specified namespace."""
        return self.v1.read_namespaced_pod_log(pod_name, namespace)

# ------------------------------------------------------------------------------
# Query Processor
# ------------------------------------------------------------------------------
class QueryProcessor:
    """
    A class that:
    1. Collects extended Kubernetes info (across all namespaces).
    2. Sends that info + user query to GPT for direct, concise answers.
    3. Returns the result or an error if something goes wrong.
    """

    def __init__(self):
        """
        Initialize with a KubernetesHelper that fetches logs with a specified tail.
        """
        self.helper = KubernetesHelper()

    def process_query(self, query: str) -> str:
        try:
            system_message = (
                "You are a Kubernetes cluster assistant. "
                "You help users get information about their Kubernetes cluster. "
                "Provide direct, concise answers without any explanations, punctuations, or additional context. "
                "Return only the specific information requested, avoiding identifiers when possible."
                "Always return a straight answer."
                "When returning any resource name, remove any trailing hyphens and random strings."
            )

            # Gather cluster info from all namespaces
            cluster_info = self._get_cluster_info()

            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": (
                        f"Here is the current Kubernetes cluster data:\n{cluster_info}\n"
                        f"Query: {query}"
                    ),
                },
            ]

            response = openai_client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
                temperature=0.95,
                max_tokens=200
            )

            answer = response.choices[0].message.content.strip()
            logging.info(f"Generated answer: {answer}")
            return answer

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"


    def _get_cluster_info(self) -> str:
        """
        Gathers an extended set of data about the cluster:
          - Pods (with many attributes, volumes, logs, etc.)
          - Deployments
          - Services
          - Nodes

        Returns:
        str: A serialized dictionary (string) of that data.
        """
        try:
            pods = self.helper.get_pods()
            deployments = self.helper.get_deployments()
            services = self.helper.get_services()
            nodes = self.helper.get_nodes()

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

            # Add pod logs for each pod (be cautious: logs can be large)
            for pod in info_dict["pods"]:
                try:
                    pod["logs"] = self.helper.get_pod_logs(
                        pod["name"],
                        pod["namespace"]
                    )
                except Exception as e:
                    pod["logs"] = f"Error retrieving logs: {str(e)}"

            logging.info(f"Cluster info: {info_dict}")
            return str(info_dict)

        except Exception as e:
            logging.error(f"Error retrieving cluster info: {e}")
            return f"Error: {e}"




# ------------------------------------------------------------------------------
# Flask App
# ------------------------------------------------------------------------------
app = Flask(__name__)
processor = QueryProcessor()

@app.route("/query", methods=["POST"])
def create_query():
    try:
        data = request.get_json()
        query_text = data.get("query", "")
        logging.info(f"Received query: {query_text}")

        answer = processor.process_query(query_text)

        # Construct final response using Pydantic
        response = QueryResponse(query=query_text, answer=answer)
        return jsonify(response.model_dump())

    except ValidationError as e:
        logging.error(f"Validation Error: {e.errors()}")
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
