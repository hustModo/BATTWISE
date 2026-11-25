from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import All200Dataset, RepresentationResult


_INPUT_DIM = 400
_TARGET_DIM = 17
_LATENT_DIM = 1


class _FCVAE(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        target_dim: int,
        hidden_dims: tuple[int, int],
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(second_hidden, _LATENT_DIM)
        self.logvar_head = nn.Linear(second_hidden, _LATENT_DIM)
        self.encoder_aux_head = nn.Linear(second_hidden, target_dim)

        self.decoder_input = nn.Linear(_LATENT_DIM, second_hidden)
        self.decoder_aux_head = nn.Linear(second_hidden, target_dim)
        self.decoder_output = nn.Sequential(
            nn.Linear(second_hidden, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, input_dim),
        )

    def encode(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs)
        return (
            self.mu_head(hidden),
            self.logvar_head(hidden),
            self.encoder_aux_head(hidden),
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.decoder_input(latent))
        return self.decoder_output(hidden), self.decoder_aux_head(hidden)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mu, logvar, encoder_aux = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        reconstruction, decoder_aux = self.decode(latent)
        return reconstruction, mu, logvar, encoder_aux, decoder_aux


def _loss_components(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    encoder_aux: torch.Tensor,
    decoder_aux: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, inputs)
    kl_loss = -0.5 * torch.mean(
        1.0 + logvar - mu.square() - torch.exp(logvar)
    )
    encoder_aux_loss = F.mse_loss(encoder_aux, targets)
    decoder_aux_loss = F.mse_loss(decoder_aux, targets)
    return (
        reconstruction_loss,
        kl_loss,
        encoder_aux_loss,
        decoder_aux_loss,
    )


class FCVAE1DReducer:
    name = "FC-VAE-1D"
    requires_targets = True

    def __init__(
        self,
        *,
        hidden_dims: Sequence[int] = (64, 16),
        beta_kl: float = 0.001,
        encoder_aux_weight: float = 1.0,
        decoder_aux_weight: float = 1.0,
        lr: float = 0.0001,
        max_epochs: int = 1000,
        patience: int = 50,
        min_delta: float = 0.000001,
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
        dims = tuple(int(value) for value in raw_dims)
        if not np.isfinite(beta_kl) or beta_kl < 0:
            raise ValueError("beta_kl must be finite and non-negative")
        if not np.isfinite(encoder_aux_weight) or encoder_aux_weight < 0:
            raise ValueError(
                "encoder_aux_weight must be finite and non-negative"
            )
        if not np.isfinite(decoder_aux_weight) or decoder_aux_weight < 0:
            raise ValueError(
                "decoder_aux_weight must be finite and non-negative"
            )
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if int(max_epochs) != max_epochs or int(max_epochs) <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if int(patience) != patience or int(patience) <= 0:
            raise ValueError("patience must be a positive integer")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")

        try:
            selected_device = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid torch device: {device}") from exc
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"Requested torch device is unavailable: {selected_device}"
            )

        self.hidden_dims = dims
        self.beta_kl = float(beta_kl)
        self.encoder_aux_weight = float(encoder_aux_weight)
        self.decoder_aux_weight = float(decoder_aux_weight)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.device = selected_device

    @staticmethod
    def _validate_dataset(
        dataset: All200Dataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        curves = np.asarray(dataset.curves, dtype=float)
        cell_ids = np.asarray(dataset.cell_ids)
        targets = np.asarray(dataset.train_targets, dtype=float)

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
        if curves.shape[0] == 0:
            raise ValueError("dataset.curves must contain at least one cell")
        if curves.shape[1] != _INPUT_DIM:
            raise ValueError("dataset.curves must contain exactly 400 features")
        if not np.isfinite(curves).all():
            raise ValueError("dataset.curves must contain only finite values")

        if targets.ndim != 2:
            raise ValueError(
                "dataset.train_targets must have shape (cells, 17 targets)"
            )
        if targets.shape[0] != curves.shape[0]:
            raise ValueError(
                "dataset.train_targets target row count must match "
                "dataset.curves"
            )
        if targets.shape[1] != _TARGET_DIM:
            raise ValueError(
                "dataset.train_targets must contain exactly 17 targets"
            )
        if not np.isfinite(targets).all():
            raise ValueError(
                "dataset.train_targets must contain only finite values"
            )
        return curves, targets, cell_ids

    @staticmethod
    def _seed_everything(seed: int) -> None:
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _weighted_loss(
        self,
        components: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        reconstruction, kl, encoder_aux, decoder_aux = components
        return (
            reconstruction
            + self.beta_kl * kl
            + self.encoder_aux_weight * encoder_aux
            + self.decoder_aux_weight * decoder_aux
        )

    def fit_transform(
        self,
        dataset: All200Dataset,
        seed: int,
    ) -> RepresentationResult:
        started = time.perf_counter()
        curves, targets, cell_ids = self._validate_dataset(dataset)
        seed_value = int(seed)
        self._seed_everything(seed_value)

        model = _FCVAE(
            input_dim=_INPUT_DIM,
            target_dim=_TARGET_DIM,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        inputs = torch.tensor(
            curves.tolist(),
            dtype=torch.float32,
            device=self.device,
        )
        target_tensor = torch.tensor(
            targets.tolist(),
            dtype=torch.float32,
            device=self.device,
        )

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        epochs_ran = 0

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar, encoder_aux, decoder_aux = model(inputs)
            components = _loss_components(
                inputs=inputs,
                targets=target_tensor,
                reconstruction=reconstruction,
                mu=mu,
                logvar=logvar,
                encoder_aux=encoder_aux,
                decoder_aux=decoder_aux,
            )
            training_loss = self._weighted_loss(components)
            if not bool(torch.isfinite(training_loss).item()):
                raise ValueError("FC-VAE-1D training produced a non-finite loss")
            training_loss.backward()
            optimizer.step()
            epochs_ran = epoch

            model.eval()
            with torch.no_grad():
                monitor_mu, monitor_logvar, monitor_encoder_aux = model.encode(
                    inputs
                )
                monitor_reconstruction, monitor_decoder_aux = model.decode(
                    monitor_mu
                )
                monitor_components = _loss_components(
                    inputs=inputs,
                    targets=target_tensor,
                    reconstruction=monitor_reconstruction,
                    mu=monitor_mu,
                    logvar=monitor_logvar,
                    encoder_aux=monitor_encoder_aux,
                    decoder_aux=monitor_decoder_aux,
                )
                monitor_loss = self._weighted_loss(monitor_components)
            if not bool(torch.isfinite(monitor_loss).item()):
                raise ValueError(
                    "FC-VAE-1D checkpoint evaluation produced a non-finite loss"
                )

            monitor_value = float(monitor_loss.detach().cpu().item())
            if monitor_value < best_loss - self.min_delta:
                best_loss = monitor_value
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is None:
            raise ValueError("FC-VAE-1D training did not produce a valid model")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            mu, logvar, encoder_aux = model.encode(inputs)
            reconstruction, decoder_aux = model.decode(mu)
            final_components = _loss_components(
                inputs=inputs,
                targets=target_tensor,
                reconstruction=reconstruction,
                mu=mu,
                logvar=logvar,
                encoder_aux=encoder_aux,
                decoder_aux=decoder_aux,
            )

        z1 = np.asarray(mu.detach().cpu().reshape(-1).tolist(), dtype=float)
        if z1.shape != (curves.shape[0],) or not np.isfinite(z1).all():
            raise ValueError(
                "FC-VAE-1D returned an invalid one-dimensional posterior mean"
            )
        final_values = [
            float(component.detach().cpu().item())
            for component in final_components
        ]
        if not np.isfinite(final_values).all():
            raise ValueError("FC-VAE-1D final diagnostics are non-finite")

        diagnostics = {
            "elapsed_s": time.perf_counter() - started,
            "epochs_ran": epochs_ran,
            "best_loss": best_loss,
            "final_reconstruction_loss": final_values[0],
            "final_kl_loss": final_values[1],
            "final_encoder_aux_loss": final_values[2],
            "final_decoder_aux_loss": final_values[3],
            "latent_dim": _LATENT_DIM,
            "input_dim": _INPUT_DIM,
            "target_dim": _TARGET_DIM,
            "seed": seed_value,
            "device": str(self.device),
            "representation": "posterior_mean_mu",
            "hidden_dims": list(self.hidden_dims),
            "beta_kl": self.beta_kl,
            "encoder_aux_weight": self.encoder_aux_weight,
            "decoder_aux_weight": self.decoder_aux_weight,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "paper_adaptation": (
                "Paper-inspired constrained VAE adapted to 400 standardized "
                "current/voltage residual features, paired 17-dimensional "
                "training-only auxiliary predictions, and a one-dimensional "
                "posterior-mean representation; downstream grouping remains "
                "outside this reducer."
            ),
        }
        return RepresentationResult(
            method=self.name,
            seed=seed_value,
            cell_ids=cell_ids.copy(),
            z1=z1,
            diagnostics=diagnostics,
        )
