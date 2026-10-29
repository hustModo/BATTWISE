from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from experiments.fair_1d_benchmark.contracts import All200Dataset
from experiments.fair_1d_benchmark.methods import fc_vae_1d as fc_vae_module
from experiments.fair_1d_benchmark.methods.fc_vae_1d import (
    FCVAE1DReducer,
    _FCVAE,
    _loss_components,
)


def _dataset(
    *,
    n_cells: int = 12,
    curves: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> All200Dataset:
    rng = np.random.default_rng(314159)
    cell_ids = np.arange(1001, 1001 + n_cells, dtype=int)
    rng.shuffle(cell_ids)
    health = np.linspace(-1.5, 1.5, n_cells, dtype=float)[:, None]

    if curves is None:
        curve_basis = rng.normal(size=(1, 400))
        curves = health @ curve_basis + 0.05 * rng.normal(size=(n_cells, 400))
    if targets is None:
        target_basis = np.linspace(0.4, 1.8, 17, dtype=float)[None, :]
        targets = health @ target_basis + 0.02 * rng.normal(size=(n_cells, 17))

    curves = np.asarray(curves, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return All200Dataset(
        cell_ids=cell_ids,
        curves=curves,
        cell_curves_raw=rng.normal(size=(n_cells, 200, 2)),
        model_curve_raw=rng.normal(size=(200, 2)),
        ocv_ir=rng.normal(size=(n_cells, 2)),
        train_targets=targets,
        capacity=rng.normal(size=n_cells),
        electrochemical=pd.DataFrame(
            {
                "state_a": rng.normal(size=n_cells),
                "state_b": rng.normal(size=n_cells),
            },
            index=cell_ids,
        ),
        curve_feature_names=tuple(f"curve_{index}" for index in range(17)),
        electrochemical_feature_names=("state_a", "state_b"),
        channel_mean=np.array([0.1, -0.2]),
        channel_std=np.array([0.9, 1.1]),
    )


def _small_reducer(**overrides: object) -> FCVAE1DReducer:
    kwargs: dict[str, object] = {
        "hidden_dims": (6, 3),
        "beta_kl": 0.001,
        "encoder_aux_weight": 1.0,
        "decoder_aux_weight": 1.0,
        "lr": 0.005,
        "max_epochs": 4,
        "patience": 4,
        "min_delta": 0.0,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return FCVAE1DReducer(**kwargs)


def test_fc_vae_1d_default_constructor_matches_benchmark_plan() -> None:
    reducer = FCVAE1DReducer()

    assert reducer.hidden_dims == (64, 16)
    assert reducer.beta_kl == 0.001
    assert reducer.encoder_aux_weight == 1.0
    assert reducer.decoder_aux_weight == 1.0
    assert reducer.lr == 0.0001
    assert reducer.max_epochs == 1000
    assert reducer.patience == 50
    assert reducer.min_delta == 0.000001
    assert str(reducer.device) == "cpu"


def test_fc_vae_1d_network_has_required_symmetric_topology() -> None:
    model = _FCVAE(
        input_dim=400,
        target_dim=17,
        hidden_dims=(64, 16),
    )

    assert tuple(model.encoder[0].weight.shape) == (64, 400)
    assert tuple(model.encoder[2].weight.shape) == (16, 64)
    assert tuple(model.mu_head.weight.shape) == (1, 16)
    assert tuple(model.logvar_head.weight.shape) == (1, 16)
    assert tuple(model.encoder_aux_head.weight.shape) == (17, 16)
    assert tuple(model.decoder_input.weight.shape) == (16, 1)
    assert tuple(model.decoder_aux_head.weight.shape) == (17, 16)
    assert tuple(model.decoder_output[0].weight.shape) == (64, 16)
    assert tuple(model.decoder_output[2].weight.shape) == (400, 64)

    inputs = torch.zeros((5, 400), dtype=torch.float32)
    mu, logvar, encoder_aux = model.encode(inputs)
    reconstruction, decoder_aux = model.decode(mu)
    assert mu.shape == (5, 1)
    assert logvar.shape == (5, 1)
    assert encoder_aux.shape == (5, 17)
    assert reconstruction.shape == (5, 400)
    assert decoder_aux.shape == (5, 17)


def test_fc_vae_1d_loss_matches_required_weighted_formula() -> None:
    components = _loss_components(
        inputs=torch.tensor([[1.0, -1.0]]),
        targets=torch.zeros((1, 17)),
        reconstruction=torch.zeros((1, 2)),
        mu=torch.ones((1, 1)),
        logvar=torch.zeros((1, 1)),
        encoder_aux=torch.ones((1, 17)),
        decoder_aux=torch.full((1, 17), 2.0),
    )
    component_values = [float(value.item()) for value in components]
    np.testing.assert_allclose(component_values, [1.0, 0.5, 1.0, 4.0])

    reducer = FCVAE1DReducer(
        beta_kl=0.2,
        encoder_aux_weight=0.3,
        decoder_aux_weight=0.4,
    )
    weighted = float(reducer._weighted_loss(components).item())
    assert weighted == pytest.approx(1.0 + 0.2 * 0.5 + 0.3 + 0.4 * 4.0)


def test_fc_vae_1d_returns_71_finite_values_in_cell_num_order() -> None:
    dataset = _dataset(n_cells=71)
    reducer = _small_reducer(max_epochs=2, patience=2)

    result = reducer.fit_transform(dataset, seed=7)

    assert reducer.name == "FC-VAE-1D"
    assert reducer.requires_targets is True
    assert result.method == "FC-VAE-1D"
    assert result.seed == 7
    np.testing.assert_array_equal(result.cell_ids, dataset.cell_ids)
    assert result.z1.shape == (71,)
    assert np.isfinite(result.z1).all()

    diagnostics = result.diagnostics
    assert diagnostics["elapsed_s"] >= 0.0
    assert 1 <= diagnostics["epochs_ran"] <= 2
    for key in (
        "best_loss",
        "final_reconstruction_loss",
        "final_kl_loss",
        "final_encoder_aux_loss",
        "final_decoder_aux_loss",
    ):
        assert np.isfinite(diagnostics[key]), key
    assert diagnostics["latent_dim"] == 1
    assert diagnostics["input_dim"] == 400
    assert diagnostics["target_dim"] == 17
    assert diagnostics["seed"] == 7
    assert diagnostics["device"] == "cpu"
    assert diagnostics["representation"] == "posterior_mean_mu"
    assert isinstance(diagnostics["paper_adaptation"], str)
    assert diagnostics["paper_adaptation"]


def test_fc_vae_1d_best_loss_describes_the_restored_checkpoint() -> None:
    reducer = _small_reducer(max_epochs=3, patience=3)

    diagnostics = reducer.fit_transform(_dataset(), seed=29).diagnostics

    restored_total = (
        diagnostics["final_reconstruction_loss"]
        + reducer.beta_kl * diagnostics["final_kl_loss"]
        + reducer.encoder_aux_weight
        * diagnostics["final_encoder_aux_loss"]
        + reducer.decoder_aux_weight
        * diagnostics["final_decoder_aux_loss"]
    )
    assert diagnostics["best_loss"] == pytest.approx(
        restored_total,
        rel=1e-7,
        abs=1e-7,
    )


def test_fc_vae_1d_is_exactly_deterministic_for_the_same_seed() -> None:
    dataset = _dataset()

    first = _small_reducer().fit_transform(dataset, seed=23)
    second = _small_reducer().fit_transform(dataset, seed=23)

    np.testing.assert_array_equal(first.z1, second.z1)
    assert first.diagnostics["epochs_ran"] == second.diagnostics["epochs_ran"]
    for key in (
        "best_loss",
        "final_reconstruction_loss",
        "final_kl_loss",
        "final_encoder_aux_loss",
        "final_decoder_aux_loss",
    ):
        assert first.diagnostics[key] == second.diagnostics[key]


def test_fc_vae_1d_ignores_non_training_dataset_fields() -> None:
    dataset = _dataset()
    changed = replace(
        dataset,
        cell_curves_raw=np.full_like(dataset.cell_curves_raw, -101.0),
        model_curve_raw=np.full_like(dataset.model_curve_raw, -103.0),
        ocv_ir=np.full_like(dataset.ocv_ir, -107.0),
        capacity=np.full_like(dataset.capacity, -109.0),
        electrochemical=pd.DataFrame(
            {
                "state_a": np.full(len(dataset.cell_ids), -113.0),
                "state_b": np.full(len(dataset.cell_ids), -127.0),
            },
            index=dataset.cell_ids,
        ),
        channel_mean=np.array([-131.0, -137.0]),
        channel_std=np.array([-139.0, -149.0]),
    )

    baseline = _small_reducer().fit_transform(dataset, seed=11)
    altered = _small_reducer().fit_transform(changed, seed=11)

    np.testing.assert_array_equal(baseline.z1, altered.z1)
    assert baseline.diagnostics["best_loss"] == altered.diagnostics["best_loss"]


def test_fc_vae_1d_training_targets_affect_supervised_learning() -> None:
    dataset = _dataset()
    changed_targets = dataset.train_targets[::-1].copy() + 2.5
    changed = replace(dataset, train_targets=changed_targets)

    baseline = _small_reducer(max_epochs=8, patience=8).fit_transform(
        dataset,
        seed=5,
    )
    altered = _small_reducer(max_epochs=8, patience=8).fit_transform(
        changed,
        seed=5,
    )

    representation_changed = not np.array_equal(baseline.z1, altered.z1)
    training_loss_changed = (
        baseline.diagnostics["best_loss"] != altered.diagnostics["best_loss"]
    )
    assert representation_changed
    assert training_loss_changed


def test_fc_vae_1d_uses_full_batch_adam_with_configured_lr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_batch_shapes: list[tuple[int, ...]] = []
    observed_lrs: list[float] = []
    original_forward = _FCVAE.forward
    original_adam = torch.optim.Adam

    def recording_forward(
        self: _FCVAE,
        inputs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        observed_batch_shapes.append(tuple(inputs.shape))
        return original_forward(self, inputs)

    def recording_adam(
        params: object,
        lr: float = 0.001,
        **kwargs: object,
    ) -> torch.optim.Optimizer:
        observed_lrs.append(float(lr))
        return original_adam(params, lr=lr, **kwargs)

    monkeypatch.setattr(_FCVAE, "forward", recording_forward)
    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    reducer = _small_reducer(lr=0.0123, max_epochs=2, patience=2)

    result = reducer.fit_transform(_dataset(), seed=31)

    assert observed_lrs == [0.0123]
    assert observed_batch_shapes == [(12, 400)] * result.diagnostics["epochs_ran"]


def test_fc_vae_1d_honors_patience_and_min_delta() -> None:
    result = _small_reducer(
        max_epochs=10,
        patience=2,
        min_delta=1e6,
    ).fit_transform(_dataset(), seed=37)

    assert result.diagnostics["epochs_ran"] == 3


def test_fc_vae_1d_reads_posterior_mean_without_eval_time_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _FCVAE.reparameterize

    def training_only_reparameterize(
        self: _FCVAE,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            raise AssertionError("evaluation representation sampled latent noise")
        return original(self, mu, logvar)

    monkeypatch.setattr(
        _FCVAE,
        "reparameterize",
        training_only_reparameterize,
    )

    result = _small_reducer().fit_transform(_dataset(), seed=17)

    assert result.diagnostics["representation"] == "posterior_mean_mu"
    assert np.isfinite(result.z1).all()


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        (
            replace(_dataset(), curves=np.zeros(400, dtype=float)),
            r"dataset\.curves must have shape",
        ),
        (
            replace(_dataset(), curves=np.zeros((12, 399), dtype=float)),
            r"exactly 400",
        ),
        (
            replace(
                _dataset(),
                curves=np.where(
                    np.arange(12 * 400).reshape(12, 400) == 0,
                    np.nan,
                    0.0,
                ),
            ),
            r"curves.*finite",
        ),
        (
            replace(_dataset(), train_targets=np.zeros((11, 17), dtype=float)),
            r"target row count",
        ),
        (
            replace(_dataset(), train_targets=np.zeros((12, 16), dtype=float)),
            r"exactly 17",
        ),
        (
            replace(
                _dataset(),
                train_targets=np.where(
                    np.arange(12 * 17).reshape(12, 17) == 0,
                    np.inf,
                    0.0,
                ),
            ),
            r"targets.*finite",
        ),
    ],
    ids=(
        "curves-not-matrix",
        "wrong-input-width",
        "non-finite-curves",
        "target-row-mismatch",
        "wrong-target-width",
        "non-finite-targets",
    ),
)
def test_fc_vae_1d_rejects_invalid_inputs_fast(
    dataset: All200Dataset,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _small_reducer(max_epochs=1, patience=1).fit_transform(dataset, seed=0)


def test_fc_vae_1d_has_no_final_grouping_interface_or_paper_clusterer() -> None:
    source = inspect.getsource(fc_vae_module).lower()
    reducer = _small_reducer(max_epochs=1, patience=1)

    assert "finch" not in source
    assert "cluster_representation" not in source
    assert not hasattr(reducer, "cluster")
    assert not hasattr(reducer, "fit_predict")


@pytest.mark.parametrize(
    "hidden_dims",
    [
        "64",
        (64.5, 16),
        (True, 16),
    ],
    ids=("string", "fractional", "boolean"),
)
def test_fc_vae_1d_rejects_non_integral_hidden_dimensions(
    hidden_dims: object,
) -> None:
    with pytest.raises(ValueError, match="hidden_dims"):
        FCVAE1DReducer(hidden_dims=hidden_dims)


@pytest.mark.parametrize("config_name", ["config.yaml", "config.smoke.yaml"])
def test_fc_vae_1d_is_registered_with_paper_adaptation_defaults(
    config_name: str,
) -> None:
    benchmark_dir = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (benchmark_dir / config_name).read_text(encoding="utf-8")
    )

    spec = config["methods"]["fc_vae_1d"]
    assert spec["class_path"] == (
        "experiments.fair_1d_benchmark.methods.fc_vae_1d:FCVAE1DReducer"
    )
    assert spec["kwargs"] == {
        "hidden_dims": [64, 16],
        "beta_kl": 0.001,
        "encoder_aux_weight": 1.0,
        "decoder_aux_weight": 1.0,
        "lr": 0.0001,
        "max_epochs": 1000,
        "patience": 50,
        "min_delta": 0.000001,
        "device": "cpu",
    }
