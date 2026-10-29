import numpy as np
from typing import List, Dict
from src.models.iot_classifier import IoTClassifier
from src.core.edge_node import EdgeAINode

class HeteroFederatedServer:
    """
    Layer 3: Hetero Federated Server (MEC-AI HetFL Server)
    Maintains the global model, broadcasts parameters to Edge AI nodes,
    and performs synchronous global aggregation across edge clusters:
    theta(t+1) = (1 / N_bc) * sum_c theta_c(t)
    """

    def __init__(self, edge_nodes: List[EdgeAINode], test_data_by_domain: Dict):
        self.edge_nodes = edge_nodes
        self.test_data_by_domain = test_data_by_domain

        # Global Model
        self.global_model = IoTClassifier()
        self.global_weights = self.global_model.get_weights()

        # Build combined global test dataset
        all_X_test = []
        all_y_test = []
        for domain, (X_t, y_t) in test_data_by_domain.items():
            all_X_test.append(X_t)
            all_y_test.append(y_t)
        self.X_global_test = np.vstack(all_X_test)
        self.y_global_test = np.concatenate(all_y_test)

    def global_aggregate(self, cluster_results: List[Dict]):
        """
        Formula from Slide 5 Step 3:
        theta(t+1) = sum_c theta_c(t) / N_bc
        """
        n_clusters = len(cluster_results)
        if n_clusters == 0:
            return

        new_weights = {}
        for key in self.global_weights.keys():
            new_weights[key] = np.zeros_like(self.global_weights[key])
            for res in cluster_results:
                # Direct cluster average as stated in formula (1 / N_bc)
                new_weights[key] += (1.0 / n_clusters) * res["weights"][key]

        self.global_weights = new_weights
        self.global_model.set_weights(self.global_weights)

    def evaluate_global(self) -> Dict:
        """Evaluates global model on combined multi-domain test set."""
        metrics = self.global_model.evaluate(self.X_global_test, self.y_global_test)

        # Domain-specific breakdowns
        domain_metrics = {}
        for domain, (X_t, y_t) in self.test_data_by_domain.items():
            domain_metrics[domain] = self.global_model.evaluate(X_t, y_t)

        metrics["domains"] = domain_metrics
        return metrics

    def run_round(
        self,
        round_idx: int,
        w1: float = 0.5,
        w2: float = 0.5,
        selection_ratio: float = 0.6,
        epochs: int = 3,
        lr: float = 0.05,
        algorithm: str = "hetfl"
    ) -> Dict:
        """
        Executes one complete hierarchical federated round:
        1. Broadcast global model to Edge AI Nodes
        2. Edge Nodes select candidates and perform intra-cluster aggregation
        3. Edge Nodes upload theta_c(t) to Global Server
        4. Global Server performs global aggregation theta(t+1)
        """
        cluster_results = []
        cluster_latencies = []
        total_round_energy = 0.0

        for edge_node in self.edge_nodes:
            c_res = edge_node.intra_cluster_train_and_aggregate(
                base_weights=self.global_weights,
                w1=w1,
                w2=w2,
                selection_ratio=selection_ratio,
                epochs=epochs,
                lr=lr,
                algorithm=algorithm
            )
            cluster_results.append(c_res)
            cluster_latencies.append(c_res["cluster_latency_sec"])
            total_round_energy += c_res["cluster_energy_joules"]

        # Global Aggregation
        self.global_aggregate(cluster_results)

        # In 3-layer architecture, edge-to-server backbone latency is ~0.15s
        backbone_delay = 0.15
        total_round_latency = max(cluster_latencies) + backbone_delay if cluster_latencies else 0.0

        eval_res = self.evaluate_global()

        return {
            "round": round_idx,
            "accuracy": eval_res["accuracy"],
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1_score": eval_res["f1_score"],
            "loss": eval_res["loss"],
            "round_latency_sec": float(total_round_latency),
            "round_energy_joules": float(total_round_energy),
            "cluster_results": cluster_results,
            "domain_eval": eval_res["domains"]
        }
