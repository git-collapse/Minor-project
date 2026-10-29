import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class IoTClassifier:
    """
    Lightweight 2-layer Multilayer Perceptron (MLP) for resource-constrained IoT devices.
    Implemented in pure NumPy for fast execution, deterministic gradient computation,
    and direct model parameter extraction for Federated Averaging.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 1, random_seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        rng = np.random.RandomState(random_seed)
        # Xavier/He initialization
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-z))

    def get_weights(self) -> dict:
        """Returns deep copy of current model weights (theta)."""
        return {
            "W1": np.copy(self.W1),
            "b1": np.copy(self.b1),
            "W2": np.copy(self.W2),
            "b2": np.copy(self.b2)
        }

    def set_weights(self, weights: dict):
        """Sets model weights from an aggregated or broadcast parameter dictionary."""
        self.W1 = np.copy(weights["W1"])
        self.b1 = np.copy(weights["b1"])
        self.W2 = np.copy(weights["W2"])
        self.b2 = np.copy(weights["b2"])

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)
        return z1, a1, z2, a2

    def train_step(self, X, y, lr: float = 0.05, epochs: int = 3, batch_size: int = 16):
        """
        Executes local SGD training on local dataset:
        theta_c(t) = theta_c(t-1) - eta * grad_L(theta_c(t-1), D)
        """
        n_samples = X.shape[0]
        y = y.reshape(-1, 1).astype(float)

        for _ in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]
                X_batch, y_batch = X[idx], y[idx]
                bs = len(idx)

                # Forward
                z1, a1, z2, a2 = self.forward(X_batch)

                # Backward (Binary Cross-Entropy Loss)
                dz2 = (a2 - y_batch) / bs
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self._relu_deriv(z1)
                dW1 = np.dot(X_batch.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # Update weights
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

    def compute_loss(self, X, y):
        y = y.reshape(-1, 1).astype(float)
        _, _, _, a2 = self.forward(X)
        eps = 1e-12
        loss = -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
        return float(loss)

    def evaluate(self, X, y):
        """Evaluates Accuracy, Precision, Recall, F1-Score, and Loss."""
        _, _, _, a2 = self.forward(X)
        preds = (a2 >= 0.5).astype(int).flatten()
        y_true = y.flatten()

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        loss = self.compute_loss(X, y)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "loss": float(loss)
        }
