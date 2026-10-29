import numpy as np
from typing import List, Dict
from src.core.device import IoTDevice

class EdgeAINode:
    """
    Layer 2: MEC-in-Edge AI Node (Cluster Head)
    Orchestrates devices within an edge cluster, executes adaptive node selection
    based on Quality Score Q(e), and performs intra-cluster local aggregation.
    """

    def __init__(self, cluster_id: str, domain: str, devices: List[IoTDevice]):
        self.cluster_id = cluster_id
        self.domain = domain
        self.devices = devices
        self.cluster_model_weights: Dict = None

    def select_devices(self, w1: float = 0.5, w2: float = 0.5, selection_ratio: float = 0.6) -> List[IoTDevice]:
        """
        Step 1: Pick the best nodes based on Q(e) = w1*NetworkScore + w2*ExpertiseScore
        Sorts devices by Q(e) and selects top ratio.
        """
        scored_devices = []
        for dev in self.devices:
            # If battery is critical (<5%), device cannot participate
            if dev.battery_pct < 5.0:
                continue
            q = dev.calculate_quality_score(w1, w2)
            scored_devices.append((dev, q))

        # Sort descending by Q(e)
        scored_devices.sort(key=lambda item: item[1], reverse=True)

        k = max(1, int(len(scored_devices) * selection_ratio))
        selected = [item[0] for item in scored_devices[:k]]
        return selected

    def intra_cluster_train_and_aggregate(
        self,
        base_weights: dict,
        w1: float = 0.5,
        w2: float = 0.5,
        selection_ratio: float = 0.6,
        epochs: int = 3,
        lr: float = 0.05,
        algorithm: str = "hetfl"
    ):
        """
        Executes intra-cluster local training & aggregation:
        theta_c(t) = sum(n_i * theta_i) / sum(n_i)
        """
        if algorithm == "hetfl":
            # Adaptive Q-score selection
            active_devices = self.select_devices(w1=w1, w2=w2, selection_ratio=selection_ratio)
        elif algorithm == "edgefed":
            # Standard hierarchical FL (random device selection without Q-score)
            k = max(1, int(len(self.devices) * selection_ratio))
            active_devices = list(np.random.choice(self.devices, size=k, replace=False))
        elif algorithm == "fedavg":
            # Classical synchronous FedAvg (selects all or random including severe stragglers)
            active_devices = self.devices
        else:
            active_devices = self.devices

        updates = []
        latencies = []
        energies = []

        for dev in active_devices:
            result = dev.local_train(base_weights, epochs=epochs, lr=lr)
            updates.append(result)
            latencies.append(result["total_time_sec"])
            energies.append(result["energy_joules"])

        # Intra-cluster Federated Averaging
        total_samples = sum(u["n_samples"] for u in updates)
        agg_weights = {}

        for key in base_weights.keys():
            agg_weights[key] = np.zeros_like(base_weights[key])
            for u in updates:
                weight_factor = u["n_samples"] / total_samples
                agg_weights[key] += weight_factor * u["weights"][key]

        self.cluster_model_weights = agg_weights

        # In semi-asynchronous / clustered mode: cluster time is determined by
        # the slowest participant in the filtered set (since stragglers are filtered out by Q)
        cluster_latency = max(latencies) if latencies else 0.0
        cluster_energy = sum(energies)

        return {
            "cluster_id": self.cluster_id,
            "domain": self.domain,
            "weights": agg_weights,
            "n_participants": len(active_devices),
            "total_samples": total_samples,
            "cluster_latency_sec": float(cluster_latency),
            "cluster_energy_joules": float(cluster_energy),
            "selected_device_ids": [d.device_id for d in active_devices]
        }
