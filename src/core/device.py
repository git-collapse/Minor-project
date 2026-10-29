import numpy as np
from src.models.iot_classifier import IoTClassifier

class IoTDevice:
    """
    Layer 1: Resource-Constrained IoT Device
    Simulates edge heterogeneity: compute clock, network channel quality,
    battery drain, straggler behavior, and local non-IID data.
    """

    def __init__(
        self,
        device_id: str,
        cluster_id: str,
        domain: str,
        X_local: np.ndarray,
        y_local: np.ndarray,
        is_straggler: bool = False,
        random_seed: int = 42
    ):
        self.device_id = device_id
        self.cluster_id = cluster_id
        self.domain = domain
        self.X_local = X_local
        self.y_local = y_local
        self.is_straggler = is_straggler

        rng = np.random.RandomState(random_seed)

        # Hardware & Network Profiles
        if is_straggler:
            self.compute_speed_ghz = rng.uniform(0.3, 0.7)    # Slow throttled CPU
            self.bandwidth_mbps = rng.uniform(0.5, 3.0)        # Congested network
            self.latency_ms = rng.uniform(180.0, 450.0)        # High delay
            self.packet_loss = rng.uniform(0.10, 0.25)
            self.battery_pct = rng.uniform(15.0, 40.0)         # Low battery
        else:
            self.compute_speed_ghz = rng.uniform(1.2, 2.5)    # Capable edge CPU
            self.bandwidth_mbps = rng.uniform(15.0, 50.0)      # High-speed edge WiFi/5G
            self.latency_ms = rng.uniform(10.0, 60.0)          # Low delay
            self.packet_loss = rng.uniform(0.005, 0.03)
            self.battery_pct = rng.uniform(60.0, 100.0)

        self.model = IoTClassifier(random_seed=random_seed)

    def compute_network_score(self) -> float:
        """
        NetworkScore(e) in [0, 1]:
        Combines normalized Bandwidth (higher is better),
        Latency (lower is better), and Packet Loss (lower is better).
        """
        norm_bw = np.clip(self.bandwidth_mbps / 50.0, 0.0, 1.0)
        norm_lat = np.clip(1.0 - (self.latency_ms / 500.0), 0.0, 1.0)
        norm_ploss = np.clip(1.0 - (self.packet_loss / 0.30), 0.0, 1.0)

        score = 0.5 * norm_bw + 0.35 * norm_lat + 0.15 * norm_ploss
        return float(np.clip(score, 0.0, 1.0))

    def compute_expertise_score(self) -> float:
        """
        ExpertiseScore(e) in [0, 1]:
        Combines Compute Capability (CPU GHz), Battery Life, and Data Volume.
        """
        norm_cpu = np.clip(self.compute_speed_ghz / 2.5, 0.0, 1.0)
        norm_bat = np.clip(self.battery_pct / 100.0, 0.0, 1.0)
        norm_data = np.clip(len(self.X_local) / 150.0, 0.0, 1.0)

        score = 0.45 * norm_cpu + 0.35 * norm_bat + 0.20 * norm_data
        return float(np.clip(score, 0.0, 1.0))

    def calculate_quality_score(self, w1: float = 0.5, w2: float = 0.5) -> float:
        """
        Formula from Slide 5:
        Q(e) = w1 * NetworkScore(e) + w2 * ExpertiseScore(e)
        """
        net_score = self.compute_network_score()
        exp_score = self.compute_expertise_score()
        q_score = w1 * net_score + w2 * exp_score
        return float(q_score)

    def local_train(self, base_weights: dict, epochs: int = 3, lr: float = 0.05):
        """
        Trains model locally on device dataset:
        theta_c(t) = theta_c(t-1) - eta * grad_L(theta_c(t-1), D)
        """
        # Load weights
        self.model.set_weights(base_weights)

        # Compute simulated time & energy
        n_samples = len(self.X_local)
        # Compute delay (s): proportional to samples & inverse of CPU clock
        comp_time_sec = (n_samples * epochs * 0.005) / self.compute_speed_ghz
        # Network delay (s): model payload ~ 20 KB over bandwidth + latency
        model_size_mb = 0.02
        comm_time_sec = (model_size_mb * 8.0 / self.bandwidth_mbps) + (self.latency_ms / 1000.0)
        total_time_sec = comp_time_sec + comm_time_sec

        # Energy consumption (Joules): compute power + RF tx power
        energy_joules = (comp_time_sec * 0.8) + (comm_time_sec * 1.2)
        # Slight battery depletion
        self.battery_pct = max(1.0, self.battery_pct - (energy_joules * 0.01))

        # Perform actual gradient updates
        self.model.train_step(self.X_local, self.y_local, lr=lr, epochs=epochs)

        return {
            "device_id": self.device_id,
            "weights": self.model.get_weights(),
            "n_samples": n_samples,
            "total_time_sec": float(total_time_sec),
            "energy_joules": float(energy_joules),
            "q_score": self.calculate_quality_score()
        }
