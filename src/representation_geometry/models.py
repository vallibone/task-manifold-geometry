"""Small PyTorch MLP that exposes hidden-layer representations."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)


class _MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        self.output_layer = nn.Linear(previous_dim, output_dim)
        self.activation = _make_activation(activation)

    def forward(self, values, return_representations: bool = False):
        hidden = values
        representations = {}
        for index, layer in enumerate(self.hidden_layers, start=1):
            hidden = self.activation(layer(hidden))
            representations[f"hidden_{index}"] = hidden
        logits = self.output_layer(hidden)
        if return_representations:
            representations["logits"] = logits
            return logits, representations
        return logits


def _make_activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        options = ", ".join(sorted(activations))
        raise ValueError(f"Unknown activation {name!r}. Choose from: {options}") from exc


class MLPRepresentationModel:
    """Classification MLP with layer representation extraction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 100,
        weight_decay: float = 0.0,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [64, 64, 32]
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.verbose = verbose
        self.scaler_ = StandardScaler()
        self.label_encoder_ = LabelEncoder()
        self.network_ = None
        self.history_ = []
        self.is_fitted_ = False

    def fit(self, values, labels):
        self._set_seeds()
        values_scaled = self.scaler_.fit_transform(np.asarray(values)).astype(np.float32)
        labels_encoded = self.label_encoder_.fit_transform(np.asarray(labels)).astype(np.int64)

        if len(self.label_encoder_.classes_) > self.output_dim:
            raise ValueError(
                f"output_dim={self.output_dim} is smaller than the number of classes "
                f"({len(self.label_encoder_.classes_)})"
            )

        dataset = TensorDataset(torch.from_numpy(values_scaled), torch.from_numpy(labels_encoded))
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.network_ = _MLPNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            self.network_.train()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0

            for batch_values, batch_labels in loader:
                batch_values = batch_values.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                logits = self.network_(batch_values)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                batch_size = batch_labels.shape[0]
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                total_seen += batch_size

            record = {
                "epoch": epoch,
                "loss": total_loss / total_seen,
                "accuracy": total_correct / total_seen,
            }
            self.history_.append(record)
            if self.verbose:
                print(
                    f"epoch={record['epoch']} "
                    f"loss={record['loss']:.4f} "
                    f"accuracy={record['accuracy']:.4f}"
                )

        self.is_fitted_ = True
        return self

    def predict(self, values):
        logits = self._predict_logits(values)
        encoded = logits.argmax(axis=1)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, values):
        logits = self._predict_logits(values)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def representations(self, values):
        self._check_fitted()
        values_scaled = self.scaler_.transform(np.asarray(values)).astype(np.float32)
        values_tensor = torch.from_numpy(values_scaled).to(self.device)
        self.network_.eval()
        with torch.no_grad():
            _, representations = self.network_(values_tensor, return_representations=True)
        output = {"input": values_scaled}
        output.update({name: value.cpu().numpy() for name, value in representations.items()})
        return output

    def _predict_logits(self, values):
        self._check_fitted()
        values_scaled = self.scaler_.transform(np.asarray(values)).astype(np.float32)
        values_tensor = torch.from_numpy(values_scaled).to(self.device)
        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(values_tensor)
        return logits.cpu().numpy()

    def _check_fitted(self):
        if not self.is_fitted_ or self.network_ is None:
            raise RuntimeError("MLPRepresentationModel must be fitted before use.")

    def _set_seeds(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
