import os
import logging
import openai

from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s - %(message)s',
    filename='agent.log',
    filemode='a'
)

# ------------------------------------------------------------------------------
# Load Kube Config & OpenAI Key
# ------------------------------------------------------------------------------
try:
    # Load local kubeconfig (e.g. ~/.kube/config)
    config.load_kube_config()
except Exception as e:
    logging.error(f"Error loading kube config: {e}")

openai.api_key = os.getenv("OPENAI_API_KEY")

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
    allowing retrieval of pods, deployments, services, nodes, etc.
    from ALL namespaces to be more adaptive.
    """

    def __init__(self, tail_lines=100):
        """
        Initializes Kubernetes clients for CoreV1Api (pods, services, nodes)
        and AppsV1Api (deployments).
        
        :param tail_lines: The maximum number of log lines to fetch per pod.
        """
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.tail_lines = tail_lines

    def get_all_pods(self):
        """
        Returns a list of pods from all namespaces.
        """
        return self.v1.list_pod_for_all_namespaces()

    def get_all_deployments(self):
        """
        Returns a list of deployments from all namespaces.
        There's no single direct method, so we iterate each namespace.
        """
        all_deployments = []
        try:
            namespaces = self.v1.list_namespace()
            for ns in namespaces.items:
                ns_name = ns.metadata.name
                deps = self.apps_v1.list_namespaced_deployment(ns_name)
                all_deployments.extend(deps.items)
        except Exception as e:
            logging.error(f"Error fetching deployments: {str(e)}")
        return all_deployments

    def get_all_services(self):
        """
        Returns a list of services from all namespaces.
        Similarly, we iterate over each namespace.
        """
        all_services = []
        try:
            namespaces = self.v1.list_namespace()
            for ns in namespaces.items:
                ns_name = ns.metadata.name
                svcs = self.v1.list_namespaced_service(ns_name)
                all_services.extend(svcs.items)
        except Exception as e:
            logging.error(f"Error fetching services: {str(e)}")
        return all_services

    def get_nodes(self):
        """
        Returns a list of nodes in the cluster.
        """
        return self.v1.list_node()

    def get_pod_logs(self, pod_name, namespace):
        """
        Returns the tail of logs for a specific pod in the given namespace.
        """
        try:
            return self.v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=self.tail_lines
            )
        except Exception as e:
            return f"Error retrieving logs: {str(e)}"

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

    def __init__(self, tail_lines=100):
        """
        Initialize with a KubernetesHelper that fetches logs with a specified tail.
        """
        self.helper = KubernetesHelper(tail_lines=tail_lines)

    def process_query(self, query: str) -> str:
        try:
            system_message = (
                "You are a Kubernetes cluster assistant. "
                "You help users get information about their Kubernetes cluster. "
                "Provide direct, concise answers without any explanations, punctuation, or additional context. "
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

            response = openai.ChatCompletion.create(
                model="gpt-4", 
                messages=messages,
                temperature=0,
                max_tokens=300
            )

            answer = response.choices[0].message.content.strip()
            logging.info(f"Generated answer: {answer}")
            return answer

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def _get_cluster_info(self) -> str:
        """
        Gathers an extended set of cluster data from all namespaces:
          - Pods (plus limited logs, container details, etc.)
          - Deployments (all namespaces)
          - Services (all namespaces)
          - Nodes
        Returns a serialized dictionary (as string) of that data.
        """
        try:
            all_pods = self.helper.get_all_pods()
            all_deps = self.helper.get_all_deployments()
            all_svcs = self.helper.get_all_services()
            all_nodes = self.helper.get_nodes()

            info_dict = {
                "pods": [],
                "deployments": [],
                "services": [],
                "nodes": []
            }

            # Gather pods
            for p in all_pods.items:
                pod_name = p.metadata.name
                pod_ns = p.metadata.namespace
                pod_data = {
                    "name": pod_name,
                    "namespace": pod_ns,
                    "status": p.status.phase,
                    "ip": p.status.pod_ip,
                    "node": p.spec.node_name,
                    "creation_time": str(p.metadata.creation_timestamp) if p.metadata.creation_timestamp else None,
                    "containers": [],
                    "labels": p.metadata.labels,
                    "volumes": []
                }

                # Container details
                if p.status.container_statuses:
                    for c in p.status.container_statuses:
                        container_info = {
                            "name": c.name,
                            "image": c.image,
                            "ready": c.ready,
                            "restart_count": getattr(c, 'restart_count', 0)
                        }
                        pod_data["containers"].append(container_info)

                # Volumes
                if p.spec.volumes:
                    for v in p.spec.volumes:
                        vol_info = {
                            "name": v.name,
                            "type": next(iter(v.to_dict().keys() - {"name"}), None)
                        }
                        pod_data["volumes"].append(vol_info)

                # Pod logs (tail-limited)
                pod_data["logs"] = self.helper.get_pod_logs(pod_name, pod_ns)

                info_dict["pods"].append(pod_data)

            # Gather deployments
            for d in all_deps:
                dep_name = d.metadata.name
                dep_ns = d.metadata.namespace
                dep_data = {
                    "name": dep_name,
                    "namespace": dep_ns,
                    "replicas": d.spec.replicas,
                    "available_replicas": d.status.available_replicas,
                    "ready_replicas": d.status.ready_replicas,
                    "strategy": d.spec.strategy.type if d.spec.strategy else None,
                    "labels": d.metadata.labels,
                    "creation_time": str(d.metadata.creation_timestamp) if d.metadata.creation_timestamp else None
                }
                info_dict["deployments"].append(dep_data)

            # Gather services
            for s in all_svcs:
                svc_name = s.metadata.name
                svc_ns = s.metadata.namespace
                svc_data = {
                    "name": svc_name,
                    "namespace": svc_ns,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "external_ips": getattr(s.spec, 'external_i_ps', None),
                    "ports": [],
                    "selector": s.spec.selector,
                    "creation_time": str(s.metadata.creation_timestamp) if s.metadata.creation_timestamp else None
                }
                if s.spec.ports:
                    for p in s.spec.ports:
                        svc_data["ports"].append({
                            "port": p.port,
                            "target_port": p.target_port,
                            "protocol": p.protocol
                        })
                info_dict["services"].append(svc_data)

            # Gather nodes
            for n in all_nodes.items:
                node_name = n.metadata.name
                node_data = {
                    "name": node_name,
                    "status_conditions": [],
                    "capacity": n.status.capacity,
                    "allocatable": n.status.allocatable,
                    "addresses": [],
                    "labels": n.metadata.labels
                }

                if n.status.conditions:
                    for cond in n.status.conditions:
                        if cond.status == "True":
                            node_data["status_conditions"].append(cond.type)

                if n.status.addresses:
                    for addr in n.status.addresses:
                        node_data["addresses"].append({
                            "type": addr.type,
                            "address": addr.address
                        })

                # Node info (architecture, runtime, etc.)
                if n.status.node_info:
                    node_data["architecture"] = n.status.node_info.architecture
                    node_data["container_runtime"] = n.status.node_info.container_runtime_version
                    node_data["kernel_version"] = n.status.node_info.kernel_version
                    node_data["os_image"] = n.status.node_info.os_image

                info_dict["nodes"].append(node_data)

            logging.info(f"Cluster info: {info_dict}")
            return str(info_dict)

        except Exception as e:
            logging.error(f"Error retrieving cluster info: {e}")
            return f"Error retrieving cluster info: {e}"

# ------------------------------------------------------------------------------
# Flask App
# ------------------------------------------------------------------------------
app = Flask(__name__)
processor = QueryProcessor(tail_lines=100)

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

    except Exception as e:
        logging.error(f"Unexpected Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run Flask on port 8000
    app.run(host="0.0.0.0", port=8000)
