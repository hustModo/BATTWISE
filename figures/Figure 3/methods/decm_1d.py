"""One-dimensional DECM adaptation for the fair All_200 benchmark.

Paper source: Z. Chen et al., Applied Energy 388 (2025) 125677.

The implementation keeps the paper's two-stage optimization boundary:

* Section 3.1, Eqs. (3)-(4): pretrain an autoencoder with the
  positive/negative-pair mutual-information estimator, a standard-normal
  latent-prior KL term, and squared reconstruction error.
* Section 3.2, Eq. (10): restore the best pretrained model, discard the
  decoder as an optimization target, and alternate internal K-means with a
  greedy encoder update toward the projected cluster prototypes. The decoder is excluded from joint optimization
  and is used afterwards only to report a reconstruction diagnostic.

The paper uses a four-dimensional latent space and K=4. The benchmark's
explicit fair adaptation uses a continuous one-dimensional encoder output and
K=6 internally. In one dimension the orthogonal projection is a 1x1 sign
matrix, so Eq. (10) reduces exactly to the summed squared distance from each
latent value to its assigned internal prototype. Those assignments remain
training-only; final grouping is performed by the shared GA-KMeans pipeline.

The paper specifies beta1=1.0, beta2=0.01, phi=0.5, Adam, and lr=0.001, but
does not report epoch or early-stopping values. It also writes the prior term
as KL(p(z|u) || N(0,I)) while giving a deterministic one-output AE rather than
a variance head. The full-batch latent distribution is therefore represented
by its empirical Gaussian moments for this analytic KL evaluation. This is an
estimator for the paper's stated KL term, not an additional training loss.

Section 4.1.1 also says that the encoder's original weight matrix is based on
probabilistic K-nearest neighbors with three neighbors, but supplies neither a
construction equation nor code and does not define how that neighbor relation
initializes the fully connected layer matrices. To avoid inventing that missing
algorithm, this adaptation uses PyTorch's seed-controlled Linear initialization
and records the limitation in every result's diagnostics.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from ..contracts import All200Dataset, RepresentationResult


_INPUT_DIM = 400
_LATENT_DIM = 1
_PRIOR_VARIANCE_EPSILON = 0.000001
_KMEANS_N_INIT = 20


class _DECMNetwork(nn.Module):
    """Symmetric AE plus the Eq. (3) joint-vs-product discriminator."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, int],
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, _LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(_LATENT_DIM, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, input_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim + _LATENT_DIM, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, 1),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def discriminate(
        self,
        inputs: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.discriminator(torch.cat((inputs, latent), dim=1))

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(inputs)
        return self.decode(latent), latent


def _representation_loss_components(
    *,
    inputs: torch.Tensor,
    reconstruction: torch.Tensor,
    latent: torch.Tensor,
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    prior_variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unweighted Eq. (4) reconstruction and Eq. (3) terms."""

    reconstruction_loss = torch.sum((inputs - reconstruction).square())
    mutual_information_loss = (
        F.softplus(-positive_logits).mean()
        + F.softplus(negative_logits).mean()
    )

    latent_mean = latent.mean(dim=0)
    latent_variance = latent.var(dim=0, unbiased=False).clamp_min(
        prior_variance_epsilon
    )
    prior_kl_loss = 0.5 * torch.sum(
        latent_mean.square()
        + latent_variance
        - torch.log(latent_variance)
        - 1.0
    )
    return reconstruction_loss, mutual_information_loss, prior_kl_loss


def _greedy_clustering_loss(
    *,
    latent: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Implement Eq. (10), changing only the first projected dimension."""

    transformed = latent @ projection
    projected_centroids = centroids @ projection
    target = transformed.detach().clone()
    target[:, 0] = projected_centroids[assignments, 0]
    return torch.sum((transformed - target).square())


def _trace_ratio_projection(
    *,
    latent: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, float]:
    """Compute the ascending-eigenvalue projection from Eqs. (6)-(9)."""

    overall_mean = latent.mean(axis=0)
    within = np.zeros((_LATENT_DIM, _LATENT_DIM), dtype=float)
    between = np.zeros((_LATENT_DIM, _LATENT_DIM), dtype=float)
    for cluster_index in range(len(centroids)):
        members = latent[assignments == cluster_index]
        centered_members = members - centroids[cluster_index]
        within += centered_members.T @ centered_members
        center_delta = (centroids[cluster_index] - overall_mean).reshape(-1, 1)
        between += len(members) * (center_delta @ center_delta.T)

    regularized_within = within + regularization * np.eye(
        _LATENT_DIM,
        dtype=float,
    )
    denominator = float(np.trace(regularized_within))
    trace_ratio = float(np.trace(between) / denominator)
    difference = between - trace_ratio * regularized_within
    eigenvalues, eigenvectors = np.linalg.eigh(difference)
    order = np.argsort(eigenvalues, kind="stable")
    projection = np.asarray(eigenvectors[:, order], dtype=float)
    if projection.shape != (_LATENT_DIM, _LATENT_DIM):
        raise ValueError("DECM-1D produced an invalid projection matrix")
    if not np.isfinite(projection).all() or not np.isfinite(trace_ratio):
        raise ValueError("DECM-1D projection calculation became non-finite")
    return projection, trace_ratio


def _internal_partition(
    *,
    latent: np.ndarray,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=_KMEANS_N_INIT,
        algorithm="lloyd",
    )
    assignments = np.asarray(model.fit_predict(latent), dtype=int)
    centroids = np.asarray(model.cluster_centers_, dtype=float)
    if len(np.unique(assignments)) != n_clusters:
        raise ValueError(
            "DECM-1D internal K-means produced an empty cluster; "
            "the latent representation is degenerate"
        )
    if not np.isfinite(centroids).all():
        raise ValueError("DECM-1D internal prototypes became non-finite")
    return assignments, centroids


def _state_copy(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
    }


def _tensor_values(
    components: tuple[torch.Tensor, ...],
) -> tuple[float, ...]:
    values = tuple(
        float(component.detach().cpu().item())
        for component in components
    )
    if not np.isfinite(values).all():
        raise ValueError("DECM-1D produced non-finite loss diagnostics")
    return values


class DECM1DReducer:
    name = "DECM-1D"
    requires_targets = False

    def __init__(
        self,
        *,
        hidden_dims: Sequence[int] = (64, 16),
        pretrain_epochs: int = 1000,
        joint_epochs: int = 100,
        lr: float = 0.001,
        patience: int = 50,
        min_delta: float = 0.000001,
        beta1: float = 1.0,
        beta2: float = 0.01,
        phi: float = 0.5,
        internal_clusters: int = 6,
        scatter_regularization: float = 0.000001,
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(hidden_dims, (str, bytes)):
            raise ValueError("hidden_dims must contain exactly two positive integers")
        try:
            raw_dims = tuple(hidden_dims)
        except TypeError as exc:
            raise ValueError(
                "hidden_dims must contain exactly two positive integers"
            ) from exc
        if len(raw_dims) != 2 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in raw_dims
        ):
            raise ValueError("hidden_dims must contain exactly two positive integers")

        self.hidden_dims = tuple(int(value) for value in raw_dims)
        self.pretrain_epochs = self._positive_integer(
            pretrain_epochs,
            "pretrain_epochs",
        )
        self.joint_epochs = self._positive_integer(
            joint_epochs,
            "joint_epochs",
        )
        self.lr = self._finite_number(lr, "lr", strictly_positive=True)
        self.patience = self._positive_integer(patience, "patience")
        self.min_delta = self._finite_number(
            min_delta,
            "min_delta",
            non_negative=True,
        )
        self.beta1 = self._finite_number(
            beta1,
            "beta1",
            non_negative=True,
        )
        self.beta2 = self._finite_number(
            beta2,
            "beta2",
            non_negative=True,
        )
        self.phi = self._finite_number(phi, "phi", non_negative=True)
        self.internal_clusters = self._positive_integer(
            internal_clusters,
            "internal_clusters",
            minimum=2,
        )
        self.scatter_regularization = self._finite_number(
            scatter_regularization,
            "scatter_regularization",
            strictly_positive=True,
        )

        try:
            selected_device = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid torch device: {device}") from exc
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"Requested torch device is unavailable: {selected_device}"
            )
        self.device = selected_device

    @staticmethod
    def _positive_integer(
        value: object,
        name: str,
        *,
        minimum: int = 1,
    ) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
        ):
            qualifier = "positive" if minimum == 1 else f"at least {minimum}"
            raise ValueError(f"{name} must be a {qualifier} integer")
        return int(value)

    @staticmethod
    def _finite_number(
        value: object,
        name: str,
        *,
        non_negative: bool = False,
        strictly_positive: bool = False,
    ) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not np.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        if strictly_positive and converted <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        if non_negative and converted < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return converted

    @staticmethod
    def _seed_everything(seed: int) -> int:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed,
            (int, np.integer),
        ):
            raise ValueError("seed must be an integer")
        seed_value = int(seed)
        sklearn_seed = seed_value % (2**32)
        random.seed(seed_value)
        np.random.seed(sklearn_seed)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return sklearn_seed

    def _validate_dataset(
        self,
        dataset: All200Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        curves = np.asarray(dataset.curves, dtype=float)
        cell_ids = np.asarray(dataset.cell_ids)
        if cell_ids.ndim != 1:
            raise ValueError("dataset.cell_ids must be one-dimensional")
        if curves.ndim != 2:
            raise ValueError(
                "dataset.curves must have shape (cells, 400 features)"
            )
        if curves.shape[0] != len(cell_ids):
            raise ValueError(
                "dataset.curves row count must match the number of Cell_num values"
            )
        if curves.shape[1] != _INPUT_DIM:
            raise ValueError("dataset.curves must contain exactly 400 features")
        if curves.shape[0] < self.internal_clusters:
            raise ValueError(
                "dataset.curves must contain at least "
                f"{self.internal_clusters} cells for internal clustering"
            )
        if not np.isfinite(curves).all():
            raise ValueError("dataset.curves must contain only finite values")
        return curves, cell_ids

    def _pretraining_objective(
        self,
        components: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        reconstruction, mutual_information, prior_kl = components
        return (
            self.beta1 * mutual_information
            + self.beta2 * prior_kl
            + self.phi * reconstruction
        )

    def _representation_components(
        self,
        model: _DECMNetwork,
        inputs: torch.Tensor,
        *,
        negative_shift: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, latent = model(inputs)
        negative_latent = torch.roll(
            latent,
            shifts=negative_shift,
            dims=0,
        )
        return _representation_loss_components(
            inputs=inputs,
            reconstruction=reconstruction,
            latent=latent,
            positive_logits=model.discriminate(inputs, latent),
            negative_logits=model.discriminate(inputs, negative_latent),
            prior_variance_epsilon=_PRIOR_VARIANCE_EPSILON,
        )

    def _pretrain(
        self,
        *,
        model: _DECMNetwork,
        inputs: torch.Tensor,
    ) -> tuple[
        int,
        float,
        tuple[float, float, float],
        float,
        dict[str, torch.Tensor],
    ]:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_loss = float("inf")
        best_components: tuple[float, float, float] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        without_improvement = 0
        epochs_ran = 0

        for epoch in range(1, self.pretrain_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            random_shift = int(
                torch.randint(1, inputs.shape[0], (1,)).item()
            )
            training_components = self._representation_components(
                model,
                inputs,
                negative_shift=random_shift,
            )
            training_loss = self._pretraining_objective(training_components)
            if not bool(torch.isfinite(training_loss).item()):
                raise ValueError(
                    "DECM-1D pretraining produced a non-finite loss"
                )
            training_loss.backward()
            optimizer.step()
            epochs_ran = epoch

            model.eval()
            with torch.no_grad():
                monitor_components = self._representation_components(
                    model,
                    inputs,
                    negative_shift=1,
                )
                monitor_loss = self._pretraining_objective(
                    monitor_components
                )
            if not bool(torch.isfinite(monitor_loss).item()):
                raise ValueError(
                    "DECM-1D pretraining checkpoint became non-finite"
                )
            monitor_value = float(monitor_loss.detach().cpu().item())
            if monitor_value < best_loss - self.min_delta:
                best_loss = monitor_value
                best_components = _tensor_values(monitor_components)
                best_state = _state_copy(model)
                without_improvement = 0
            else:
                without_improvement += 1
                if without_improvement >= self.patience:
                    break

        if best_state is None or best_components is None:
            raise ValueError("DECM-1D pretraining produced no valid checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            restored_components = self._representation_components(
                model,
                inputs,
                negative_shift=1,
            )
            restored_total = self._pretraining_objective(
                restored_components
            )
        restored_total_value = float(restored_total.detach().cpu().item())
        if not np.isfinite(restored_total_value):
            raise ValueError("DECM-1D restored pretraining loss is non-finite")
        return (
            epochs_ran,
            best_loss,
            best_components,
            restored_total_value,
            _state_copy(model.encoder),
        )

    @staticmethod
    def _latent_numpy(
        model: _DECMNetwork,
        inputs: torch.Tensor,
    ) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            latent = model.encode(inputs)
        values = np.asarray(
            latent.detach().cpu().reshape(-1, _LATENT_DIM).tolist(),
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("DECM-1D encoder produced non-finite values")
        return values

    def _partition_and_projection(
        self,
        latent: np.ndarray,
        *,
        sklearn_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        assignments, centroids = _internal_partition(
            latent=latent,
            n_clusters=self.internal_clusters,
            seed=sklearn_seed,
        )
        projection, trace_ratio = _trace_ratio_projection(
            latent=latent,
            assignments=assignments,
            centroids=centroids,
            regularization=self.scatter_regularization,
        )
        return assignments, centroids, projection, trace_ratio

    def _joint_train(
        self,
        *,
        model: _DECMNetwork,
        inputs: torch.Tensor,
        sklearn_seed: int,
    ) -> tuple[int, float, float, float]:
        optimizer = torch.optim.Adam(model.encoder.parameters(), lr=self.lr)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        without_improvement = 0
        epochs_ran = 0

        for epoch in range(1, self.joint_epochs + 1):
            latent_before = self._latent_numpy(model, inputs)
            assignments, centroids, projection, _ = (
                self._partition_and_projection(
                    latent_before,
                    sklearn_seed=sklearn_seed,
                )
            )
            assignment_tensor = torch.tensor(
                assignments.tolist(),
                dtype=torch.long,
                device=self.device,
            )
            centroid_tensor = torch.tensor(
                centroids.tolist(),
                dtype=torch.float32,
                device=self.device,
            )
            projection_tensor = torch.tensor(
                projection.tolist(),
                dtype=torch.float32,
                device=self.device,
            )

            model.encoder.train()
            optimizer.zero_grad(set_to_none=True)
            latent = model.encode(inputs)
            training_loss = _greedy_clustering_loss(
                latent=latent,
                centroids=centroid_tensor,
                assignments=assignment_tensor,
                projection=projection_tensor,
            )
            if not bool(torch.isfinite(training_loss).item()):
                raise ValueError(
                    "DECM-1D joint training produced a non-finite loss"
                )
            training_loss.backward()
            optimizer.step()
            epochs_ran = epoch

            monitor_latent = self._latent_numpy(model, inputs)
            (
                monitor_assignments,
                monitor_centroids,
                monitor_projection,
                _,
            ) = self._partition_and_projection(
                monitor_latent,
                sklearn_seed=sklearn_seed,
            )
            with torch.no_grad():
                monitor_loss = _greedy_clustering_loss(
                    latent=model.encode(inputs),
                    centroids=torch.tensor(
                        monitor_centroids.tolist(),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    assignments=torch.tensor(
                        monitor_assignments.tolist(),
                        dtype=torch.long,
                        device=self.device,
                    ),
                    projection=torch.tensor(
                        monitor_projection.tolist(),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            monitor_value = float(monitor_loss.detach().cpu().item())
            if not np.isfinite(monitor_value):
                raise ValueError(
                    "DECM-1D joint checkpoint became non-finite"
                )
            if monitor_value < best_loss - self.min_delta:
                best_loss = monitor_value
                best_state = _state_copy(model.encoder)
                without_improvement = 0
            else:
                without_improvement += 1
                if without_improvement >= self.patience:
                    break

        if best_state is None:
            raise ValueError("DECM-1D joint training produced no valid checkpoint")
        model.encoder.load_state_dict(best_state)

        final_latent = self._latent_numpy(model, inputs)
        assignments, centroids, projection, trace_ratio = (
            self._partition_and_projection(
                final_latent,
                sklearn_seed=sklearn_seed,
            )
        )
        with torch.no_grad():
            final_loss = _greedy_clustering_loss(
                latent=model.encode(inputs),
                centroids=torch.tensor(
                    centroids.tolist(),
                    dtype=torch.float32,
                    device=self.device,
                ),
                assignments=torch.tensor(
                    assignments.tolist(),
                    dtype=torch.long,
                    device=self.device,
                ),
                projection=torch.tensor(
                    projection.tolist(),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        final_value = float(final_loss.detach().cpu().item())
        if not np.isfinite(final_value):
            raise ValueError("DECM-1D final clustering loss is non-finite")
        return epochs_ran, best_loss, final_value, trace_ratio

    @staticmethod
    def _encoder_update_l2(
        before: dict[str, torch.Tensor],
        after: nn.Module,
    ) -> float:
        squared = 0.0
        for key, current in after.state_dict().items():
            delta = current.detach().cpu() - before[key].detach().cpu()
            squared += float(torch.sum(delta.square()).item())
        return float(np.sqrt(squared))

    def fit_transform(
        self,
        dataset: All200Dataset,
        seed: int,
    ) -> RepresentationResult:
        started = time.perf_counter()
        curves, cell_ids = self._validate_dataset(dataset)
        sklearn_seed = self._seed_everything(seed)
        seed_value = int(seed)

        model = _DECMNetwork(
            input_dim=_INPUT_DIM,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
        inputs = torch.tensor(
            curves.tolist(),
            dtype=torch.float32,
            device=self.device,
        )

        (
            pretrain_epochs_ran,
            best_pretrain_loss,
            best_pretrain_components,
            final_pretrain_loss,
            pretrained_encoder_state,
        ) = self._pretrain(model=model, inputs=inputs)

        (
            joint_epochs_ran,
            best_cluster_loss,
            final_cluster_loss,
            final_trace_ratio,
        ) = self._joint_train(
            model=model,
            inputs=inputs,
            sklearn_seed=sklearn_seed,
        )
        encoder_update_l2 = self._encoder_update_l2(
            pretrained_encoder_state,
            model.encoder,
        )

        model.eval()
        with torch.no_grad():
            final_components_tensor = self._representation_components(
                model,
                inputs,
                negative_shift=1,
            )
        final_components = _tensor_values(final_components_tensor)
        z1 = self._latent_numpy(model, inputs).reshape(-1)
        if z1.shape != (curves.shape[0],) or not np.isfinite(z1).all():
            raise ValueError(
                "DECM-1D returned an invalid one-dimensional encoder output"
            )

        diagnostics = {
            "elapsed_s": time.perf_counter() - started,
            "pretrain_epochs_ran": pretrain_epochs_ran,
            "joint_epochs_ran": joint_epochs_ran,
            "best_pretrain_total_loss": best_pretrain_loss,
            "final_pretrain_total_loss": final_pretrain_loss,
            "best_reconstruction_loss": best_pretrain_components[0],
            "final_reconstruction_loss": final_components[0],
            "best_mutual_information_loss": best_pretrain_components[1],
            "final_mutual_information_loss": final_components[1],
            "best_prior_kl_loss": best_pretrain_components[2],
            "final_prior_kl_loss": final_components[2],
            "best_cluster_loss": best_cluster_loss,
            "final_cluster_loss": final_cluster_loss,
            "encoder_update_l2": encoder_update_l2,
            "final_trace_ratio": final_trace_ratio,
            "latent_dim": _LATENT_DIM,
            "input_dim": _INPUT_DIM,
            "seed": seed_value,
            "device": str(self.device),
            "internal_clusters": self.internal_clusters,
            "internal_only": True,
            "kmeans_n_init": _KMEANS_N_INIT,
            "optimizer": "Adam",
            "full_batch": True,
            "decoder_joint_optimized": False,
            "alternating_internal_kmeans": True,
            "representation": "final_encoder_continuous_z",
            "hidden_dims": list(self.hidden_dims),
            "pretrain_epochs": self.pretrain_epochs,
            "joint_epochs": self.joint_epochs,
            "lr": self.lr,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "phi": self.phi,
            "scatter_regularization": self.scatter_regularization,
            "paper_equation_mapping": {
                "pretraining": (
                    "Section 3.1 Eq. (3) uses beta1 for the positive/negative "
                    "JS logistic mutual-information estimator and beta2 for "
                    "the N(0,I) prior KL; Eq. (4) adds phi times squared "
                    "reconstruction error."
                ),
                "joint_training": (
                    "Section 3.2 discards the decoder and Eq. (4), alternates "
                    "internal K-means with Eqs. (6)-(9), and updates only the "
                    "encoder with the Eq. (10) projected prototype distance."
                ),
                "one_dimensional_adaptation": (
                    "The paper's 4D, K=4 latent clustering is adapted to a "
                    "continuous 1D latent and internal K=6. The 1x1 "
                    "orthogonal projection makes Eq. (10) the exact squared "
                    "distance to the assigned internal prototype."
                ),
                "prior_estimator": (
                    "The paper specifies a deterministic AE output but no "
                    "conditional variance head; full-batch latent mean and "
                    "population variance parameterize the stated Gaussian "
                    "KL without adding a new loss."
                ),
                "reported_hyperparameters": (
                    "Paper Section 4.1.1 reports Adam lr=0.001, beta1=1.0, "
                    "beta2=0.01, and phi=0.5."
                ),
                "initialization": (
                    "Paper Section 4.1.1 mentions probabilistic K-nearest "
                    "neighbors with k=3 for the original encoder weight "
                    "matrix, but gives no construction equation, layer "
                    "mapping, or code. This adaptation therefore uses "
                    "seed-controlled PyTorch Linear initialization rather "
                    "than inventing an unspecified algorithm."
                ),
                "implementation_details": (
                    "The paper specifies negative sampling through a "
                    "discriminator but not its network; this implementation "
                    "uses a one-hidden-layer logistic discriminator. The "
                    "deterministic AE also has no conditional variance head, "
                    "so the stated prior KL uses full-batch latent moments."
                ),
                "alternation": (
                    "Each joint epoch refreshes seeded internal K-means and "
                    "the Eqs. (6)-(9) projection, then performs one "
                    "full-batch Adam encoder step on Eq. (10). K-means uses "
                    "n_init=20; the paper does not report restart counts."
                ),
                "unreported_controls": (
                    "Epoch counts, patience, min_delta, and the positive "
                    "scatter regularizer value are reproducibility choices "
                    "because the paper does not report numeric values."
                ),
                "fairness_boundary": (
                    "Only dataset.curves trains this reducer. Internal K=6 "
                    "is representation-learning supervision only; final "
                    "[OCV, IR, z1] grouping remains in the shared GA-KMeans."
                ),
            },
        }
        return RepresentationResult(
            method=self.name,
            seed=seed_value,
            cell_ids=cell_ids.copy(),
            z1=z1,
            diagnostics=diagnostics,
        )
