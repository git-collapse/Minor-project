import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class SensorDatasetGenerator:
    """
    Generates non-IID sensor telemetry data mimicking the 3 IoT domains
    described in Mughal et al. (2024):
    1. Smart City (traffic, urban surveillance, air quality sensors)
    2. Mobile Healthcare (ECG, wearables, patient vital telemetry)
    3. Agriculture (soil moisture, pH, weather, crop status)
    """

    DOMAINS = ["Smart City", "Mobile Healthcare", "Agriculture"]

    @staticmethod
    def generate_domain_data(domain_name: str, n_samples: int = 1200, random_state: int = 42):
        """
        Creates synthetic multi-feature telemetry with domain-specific characteristics
        and label shifts representing domain-specific event/anomaly detection.
        """
        if domain_name == "Smart City":
            # 8 features: [traffic_density, noise_db, co2_ppm, temp, humidity, speed, light_lux, vibration]
            X, y = make_classification(
                n_samples=n_samples,
                n_features=8,
                n_informative=6,
                n_redundant=2,
                n_classes=2,
                weights=[0.6, 0.4],
                flip_y=0.03,
                random_state=random_state
            )
            # Apply positive shift & scaling to mimic physical sensor readings
            X[:, 0] = np.abs(X[:, 0] * 25 + 60)   # Traffic count
            X[:, 1] = np.abs(X[:, 1] * 15 + 65)   # Noise (dB)
            X[:, 2] = np.abs(X[:, 2] * 80 + 400)  # CO2 (ppm)

        elif domain_name == "Mobile Healthcare":
            # 8 features: [heart_rate, spo2, systolic_bp, diastolic_bp, body_temp, motion, ecg_delta, resp_rate]
            X, y = make_classification(
                n_samples=n_samples,
                n_features=8,
                n_informative=7,
                n_redundant=1,
                n_classes=2,
                weights=[0.7, 0.3],
                flip_y=0.02,
                random_state=random_state + 10
            )
            X[:, 0] = np.abs(X[:, 0] * 15 + 78)   # Heart rate
            X[:, 1] = np.clip(95 + X[:, 1] * 2, 80, 100) # SpO2 (%)
            X[:, 2] = np.abs(X[:, 2] * 20 + 120)  # Systolic BP

        elif domain_name == "Agriculture":
            # 8 features: [soil_moisture, soil_ph, ambient_temp, humidity, solar_rad, leaf_wetness, n_level, k_level]
            X, y = make_classification(
                n_samples=n_samples,
                n_features=8,
                n_informative=5,
                n_redundant=3,
                n_classes=2,
                weights=[0.55, 0.45],
                flip_y=0.04,
                random_state=random_state + 20
            )
            X[:, 0] = np.clip(X[:, 0] * 10 + 35, 5, 80) # Moisture %
            X[:, 1] = np.clip(X[:, 1] * 0.8 + 6.5, 4.0, 9.0) # pH
            X[:, 2] = np.abs(X[:, 2] * 8 + 26) # Ambient temp

        else:
            raise ValueError(f"Unknown domain: {domain_name}")

        # Normalize features
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-7
        X_norm = (X - mean) / std

        return X_norm, y

    @classmethod
    def partition_non_iid_to_devices(cls, domain_name: str, n_devices: int, samples_per_device: int = 100, alpha: float = 0.5):
        """
        Partitions domain data across IoT devices in a non-IID manner using Dirichlet distribution.
        """
        total_samples = n_devices * samples_per_device + 300
        X_all, y_all = cls.generate_domain_data(domain_name, n_samples=total_samples)

        # Split test set for edge/global evaluation
        X_pool, X_test, y_pool, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

        # Non-IID Dirichlet distribution over classes for each device
        n_classes = len(np.unique(y_pool))
        class_indices = [np.where(y_pool == c)[0] for c in range(n_classes)]

        device_data = []
        dirichlet_weights = np.random.dirichlet([alpha] * n_classes, size=n_devices)

        for dev_idx in range(n_devices):
            weights = dirichlet_weights[dev_idx]
            # determine how many samples from each class
            class_counts = np.round(weights * samples_per_device).astype(int)
            # adjust rounding differences
            diff = samples_per_device - np.sum(class_counts)
            class_counts[0] += diff

            dev_indices = []
            for c in range(n_classes):
                cnt = max(5, class_counts[c])
                chosen = np.random.choice(class_indices[c], size=min(cnt, len(class_indices[c])), replace=True)
                dev_indices.extend(chosen)

            dev_indices = np.array(dev_indices)[:samples_per_device]
            np.random.shuffle(dev_indices)
            device_data.append((X_pool[dev_indices], y_pool[dev_indices]))

        return device_data, (X_test, y_test)
