import copy
import numpy as np
from typing import Dict, List
from data.synthetic_generator import SensorDatasetGenerator
from src.core.device import IoTDevice
from src.core.edge_node import EdgeAINode
from src.core.server import HeteroFederatedServer

class SimulationManager:
    """
    Orchestrates the entire 3-Tier MEC-AI HetFL Simulation environment.
    Initializes clusters, partitions Non-IID data across domains,
    and runs multi-round federated training comparing different algorithms.
    """

    DOMAINS = ["Smart City", "Mobile Healthcare", "Agriculture"]

    def __init__(
        self,
        n_devices_per_cluster: int = 10,
        samples_per_device: int = 100,
        straggler_ratio: float = 0.25,
        random_seed: int = 42
    ):
        self.n_devices_per_cluster = n_devices_per_cluster
        self.samples_per_device = samples_per_device
        self.straggler_ratio = straggler_ratio
        self.random_seed = random_seed

        # Store test sets by domain
        self.test_data_by_domain = {}
        # Topology template: cluster_id -> list of raw device configs
        self.cluster_configs = {}

        self._generate_topology()

    def _generate_topology(self):
        """Generates domain datasets and initializes IoT device configurations."""
        rng = np.random.RandomState(self.random_seed)

        for c_idx, domain in enumerate(self.DOMAINS, start=1):
            cluster_id = f"Cluster_{c_idx}"
            device_data_list, (X_test, y_test) = SensorDatasetGenerator.partition_non_iid_to_devices(
                domain_name=domain,
                n_devices=self.n_devices_per_cluster,
                samples_per_device=self.samples_per_device,
                alpha=0.5
            )
            self.test_data_by_domain[domain] = (X_test, y_test)

            # Determine stragglers
            n_stragglers = int(round(self.n_devices_per_cluster * self.straggler_ratio))
            straggler_flags = [True] * n_stragglers + [False] * (self.n_devices_per_cluster - n_stragglers)
            rng.shuffle(straggler_flags)

            dev_specs = []
            for d_idx, (X_loc, y_loc) in enumerate(device_data_list, start=1):
                dev_id = f"Dev_{cluster_id}_{d_idx}"
                is_straggler = straggler_flags[d_idx - 1]
                dev_specs.append({
                    "device_id": dev_id,
                    "cluster_id": cluster_id,
                    "domain": domain,
                    "X_local": X_loc,
                    "y_local": y_loc,
                    "is_straggler": is_straggler,
                    "seed": self.random_seed + c_idx * 100 + d_idx
                })

            self.cluster_configs[cluster_id] = {
                "domain": domain,
                "devices": dev_specs
            }

    def instantiate_system(self):
        """Creates fresh instances of IoT Devices, Edge AI Nodes, and Global Server."""
        edge_nodes = []
        for cluster_id, cfg in self.cluster_configs.items():
            devices = []
            for spec in cfg["devices"]:
                dev = IoTDevice(
                    device_id=spec["device_id"],
                    cluster_id=spec["cluster_id"],
                    domain=spec["domain"],
                    X_local=np.copy(spec["X_local"]),
                    y_local=np.copy(spec["y_local"]),
                    is_straggler=spec["is_straggler"],
                    random_seed=spec["seed"]
                )
                devices.append(dev)

            edge_node = EdgeAINode(cluster_id=cluster_id, domain=cfg["domain"], devices=devices)
            edge_nodes.append(edge_node)

        server = HeteroFederatedServer(edge_nodes=edge_nodes, test_data_by_domain=self.test_data_by_domain)
        return server

    def run_simulation(
        self,
        algorithm: str = "hetfl",
        n_rounds: int = 15,
        w1: float = 0.5,
        w2: float = 0.5,
        selection_ratio: float = 0.6,
        epochs: int = 3,
        lr: float = 0.05
    ) -> List[Dict]:
        """
        Runs complete federated simulation for specified algorithm and returns round history.
        """
        server = self.instantiate_system()
        history = []

        for r in range(1, n_rounds + 1):
            round_res = server.run_round(
                round_idx=r,
                w1=w1,
                w2=w2,
                selection_ratio=selection_ratio,
                epochs=epochs,
                lr=lr,
                algorithm=algorithm
            )
            history.append(round_res)

        return history
