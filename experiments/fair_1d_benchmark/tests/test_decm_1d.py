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
from experiments.fair_1d_benchmark.methods import decm_1d as decm_module
from experiments.fair_1d_benchmark.methods.decm_1d import (
    DECM1DReducer,
    _DECMNetwork,
    _greedy_clustering_loss,
    _representation_loss_components,
)


def _dataset(
    *,
    n_cells: int = 12,
    curves: np.ndarray | None = None,
) -> All200Dataset:
    rng = np.random.default_rng(271828)
    cell_ids = np.arange(2001, 2001 + n_cells, dtype=int)
    rng.shuffle(cell_ids)
    health = np.linspace(-2.0, 2.0, n_cells, dtype=float)[:, None]
    if curves is None:
        current_basis = np.sin(np.linspace(0.0, 3.0, 200))[None, :]
        voltage_basis = np.cos(np.linspace(0.0, 4.0, 200))[None, :]
        basis = np.concatenate((current_basis, voltage_basis), axis=1)
        curves = health @ basis + 0.03 * rng.normal(size=(n_cells, 400))

    return All200Dataset(
        cell_ids=cell_ids,
        curves=np.asarray(curves, dtype=float),
        cell_curves_raw=rng.normal(size=(n_cells, 200, 2)),
        model_curve_raw=rng.normal(size=(200, 2)),
        ocv_ir=rng.normal(size=(n_cells, 2)),
        train_targets=rng.normal(size=(n_cells, 17)),
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


def _small_reducer(**overrides: object) -> DECM1DReducer:
    kwargs: dict[str, object] = {
        "hidden_dims": (8, 4),
        "pretrain_epochs": 2,
        "joint_epochs": 2,
        "lr": 0.003,
        "patience": 2,
        "min_delta": 0.0,
        "beta1": 1.0,
        "beta2": 0.01,
        "phi": 0.5,
        "internal_clusters": 6,
        "scatter_regularization": 0.000001,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return DECM1DReducer(**kwargs)


def test_decm_1d_default_constructor_matches_paper_and_adaptation() -> None:
    reducer = DECM1DReducer()

    assert reducer.hidden_dims == (64, 16)
    assert reducer.pretrain_epochs == 1000
    assert reducer.joint_epochs == 100
    assert reducer.lr == 0.001
    assert reducer.patience == 50
    assert reducer.min_delta == 0.000001
    assert reducer.beta1 == 1.0
    assert reducer.beta2 == 0.01
    assert reducer.phi == 0.5
    assert reducer.internal_clusters == 6
    assert reducer.scatter_regularization == 0.000001
    assert str(reducer.device) == "cpu"


def test_decm_1d_network_has_required_symmetric_topology() -> None:
    model = _DECMNetwork(input_dim=400, hidden_dims=(64, 16))

    assert tuple(model.encoder[0].weight.shape) == (64, 400)
    assert tuple(model.encoder[2].weight.shape) == (16, 64)
    assert tuple(model.encoder[4].weight.shape) == (1, 16)
    assert tuple(model.decoder[0].weight.shape) == (16, 1)
    assert tuple(model.decoder[2].weight.shape) == (64, 16)
    assert tuple(model.decoder[4].weight.shape) == (400, 64)

    inputs = torch.zeros((5, 400), dtype=torch.float32)
    latent = model.encode(inputs)
    reconstruction = model.decode(latent)
    assert latent.shape == (5, 1)
    assert reconstruction.shape == (5, 400)


def test_decm_1d_pretraining_loss_matches_equations_3_and_4() -> None:
    components = _representation_loss_components(
        inputs=torch.tensor([[1.0, -1.0], [2.0, 0.0]]),
        reconstruction=torch.zeros((2, 2)),
        latent=torch.tensor([[-1.0], [1.0]]),
        positive_logits=torch.zeros((2, 1)),
        negative_logits=torch.zeros((2, 1)),
        prior_variance_epsilon=0.000001,
    )
    reconstruction, mutual_information, prior_kl = components

    assert float(reconstruction.item()) == pytest.approx(6.0)
    assert float(mutual_information.item()) == pytest.approx(
        2.0 * np.log(2.0)
    )
    assert float(prior_kl.item()) == pytest.approx(0.0, abs=1e-7)

    reducer = DECM1DReducer(beta1=1.2, beta2=0.3, phi=0.4)
    objective = reducer._pretraining_objective(components)
    assert float(objective.item()) == pytest.approx(
        1.2 * 2.0 * np.log(2.0) + 0.3 * 0.0 + 0.4 * 6.0
    )


def test_decm_1d_prior_kl_matches_standard_normal_moment_formula() -> None:
    _, _, prior_kl = _representation_loss_components(
        inputs=torch.zeros((2, 1)),
        reconstruction=torch.zeros((2, 1)),
        latent=torch.tensor([[0.0], [2.0]]),
        positive_logits=torch.zeros((2, 1)),
        negative_logits=torch.zeros((2, 1)),
        prior_variance_epsilon=0.000001,
    )

    # Batch latent has mean=1 and population variance=1:
    # KL(N(1,1) || N(0,1)) = 0.5.
    assert float(prior_kl.item()) == pytest.approx(0.5)


def test_decm_1d_greedy_loss_is_exact_one_dimensional_equation_10() -> None:
    loss = _greedy_clustering_loss(
        latent=torch.tensor([[0.0], [2.0], [10.0], [12.0]]),
        centroids=torch.tensor([[1.0], [11.0]]),
        assignments=torch.tensor([0, 0, 1, 1]),
        projection=torch.ones((1, 1)),
    )

    assert float(loss.item()) == pytest.approx(4.0)


def test_decm_1d_returns_71_finite_values_in_cell_num_order() -> None:
    dataset = _dataset(n_cells=71)
    result = _small_reducer(
        pretrain_epochs=1,
        joint_epochs=1,
        patience=1,
    ).fit_transform(dataset, seed=7)

    assert result.method == "DECM-1D"
    assert result.seed == 7
    np.testing.assert_array_equal(result.cell_ids, dataset.cell_ids)
    assert result.z1.shape == (71,)
    assert np.isfinite(result.z1).all()

    diagnostics = result.diagnostics
    assert diagnostics["elapsed_s"] >= 0.0
    assert diagnostics["pretrain_epochs_ran"] == 1
    assert diagnostics["joint_epochs_ran"] == 1
    for key in (
        "best_pretrain_total_loss",
        "final_pretrain_total_loss",
        "best_reconstruction_loss",
        "final_reconstruction_loss",
        "best_mutual_information_loss",
        "final_mutual_information_loss",
        "best_prior_kl_loss",
        "final_prior_kl_loss",
        "best_cluster_loss",
        "final_cluster_loss",
        "encoder_update_l2",
    ):
        assert np.isfinite(diagnostics[key]), key
    assert diagnostics["latent_dim"] == 1
    assert diagnostics["input_dim"] == 400
    assert diagnostics["seed"] == 7
    assert diagnostics["device"] == "cpu"
    assert diagnostics["internal_clusters"] == 6
    assert diagnostics["internal_only"] is True
    assert diagnostics["optimizer"] == "Adam"
    assert diagnostics["full_batch"] is True
    assert diagnostics["decoder_joint_optimized"] is False
    assert diagnostics["alternating_internal_kmeans"] is True
    assert diagnostics["representation"] == "final_encoder_continuous_z"
    mapping = diagnostics["paper_equation_mapping"]
    assert "Eq. (4)" in mapping["pretraining"]
    assert "Eq. (10)" in mapping["joint_training"]
    assert "probabilistic K-nearest neighbors" in mapping["initialization"]
    assert "discriminator" in mapping["implementation_details"]
    assert diagnostics["kmeans_n_init"] == 20


def test_decm_1d_is_exactly_deterministic_for_the_same_seed() -> None:
    dataset = _dataset()

    first = _small_reducer().fit_transform(dataset, seed=23)
    second = _small_reducer().fit_transform(dataset, seed=23)

    np.testing.assert_array_equal(first.z1, second.z1)
    assert first.diagnostics["pretrain_epochs_ran"] == second.diagnostics[
        "pretrain_epochs_ran"
    ]
    assert first.diagnostics["joint_epochs_ran"] == second.diagnostics[
        "joint_epochs_ran"
    ]
    for key in (
        "best_pretrain_total_loss",
        "final_pretrain_total_loss",
        "final_reconstruction_loss",
        "final_mutual_information_loss",
        "final_prior_kl_loss",
        "best_cluster_loss",
        "final_cluster_loss",
        "encoder_update_l2",
    ):
        assert first.diagnostics[key] == second.diagnostics[key]


def test_decm_1d_ignores_every_dataset_field_except_curves_and_cell_ids() -> None:
    dataset = _dataset()
    changed = replace(
        dataset,
        cell_curves_raw=np.full_like(dataset.cell_curves_raw, -101.0),
        model_curve_raw=np.full_like(dataset.model_curve_raw, -103.0),
        ocv_ir=np.full_like(dataset.ocv_ir, -107.0),
        train_targets=np.full_like(dataset.train_targets, -109.0),
        capacity=np.full_like(dataset.capacity, -113.0),
        electrochemical=pd.DataFrame(
            {
                "state_a": np.full(len(dataset.cell_ids), -127.0),
                "state_b": np.full(len(dataset.cell_ids), -131.0),
            },
            index=dataset.cell_ids,
        ),
        channel_mean=np.array([-137.0, -139.0]),
        channel_std=np.array([-149.0, -151.0]),
    )

    baseline = _small_reducer().fit_transform(dataset, seed=11)
    altered = _small_reducer().fit_transform(changed, seed=11)

    np.testing.assert_array_equal(baseline.z1, altered.z1)
    assert baseline.diagnostics["final_cluster_loss"] == altered.diagnostics[
        "final_cluster_loss"
    ]


def test_decm_1d_curves_are_the_only_training_signal() -> None:
    dataset = _dataset()
    changed = replace(dataset, curves=dataset.curves[::-1].copy() + 0.75)

    baseline = _small_reducer().fit_transform(dataset, seed=5)
    altered = _small_reducer().fit_transform(changed, seed=5)

    assert not np.array_equal(baseline.z1, altered.z1)
    assert baseline.diagnostics["final_reconstruction_loss"] != altered.diagnostics[
        "final_reconstruction_loss"
    ]


def test_decm_1d_joint_stage_really_updates_the_encoder() -> None:
    result = _small_reducer(
        pretrain_epochs=1,
        joint_epochs=2,
        patience=2,
    ).fit_transform(_dataset(), seed=19)

    assert result.diagnostics["joint_epochs_ran"] >= 1
    assert result.diagnostics["encoder_update_l2"] > 0.0


def test_decm_1d_uses_full_batch_adam_and_seeded_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_shapes: list[tuple[int, ...]] = []
    observed_lrs: list[float] = []
    observed_parameter_tensor_counts: list[int] = []
    observed_kmeans: list[dict[str, object]] = []
    original_encode = _DECMNetwork.encode
    original_adam = torch.optim.Adam
    original_kmeans = decm_module.KMeans

    def recording_encode(
        self: _DECMNetwork,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        observed_shapes.append(tuple(inputs.shape))
        return original_encode(self, inputs)

    def recording_adam(
        params: object,
        lr: float = 0.001,
        **kwargs: object,
    ) -> torch.optim.Optimizer:
        parameter_list = list(params)
        observed_lrs.append(float(lr))
        observed_parameter_tensor_counts.append(len(parameter_list))
        return original_adam(parameter_list, lr=lr, **kwargs)

    def recording_kmeans(*args: object, **kwargs: object) -> object:
        observed_kmeans.append(dict(kwargs))
        return original_kmeans(*args, **kwargs)

    monkeypatch.setattr(_DECMNetwork, "encode", recording_encode)
    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    monkeypatch.setattr(decm_module, "KMeans", recording_kmeans)

    _small_reducer(lr=0.0123).fit_transform(_dataset(), seed=31)

    assert observed_lrs == [0.0123, 0.0123]
    # Pretraining sees encoder+decoder+discriminator; joint Eq. (10) sees
    # only the encoder's three Linear weight/bias pairs.
    assert observed_parameter_tensor_counts == [16, 6]
    assert observed_shapes
    assert set(observed_shapes) == {(12, 400)}
    assert observed_kmeans
    assert {entry["n_clusters"] for entry in observed_kmeans} == {6}
    assert {entry["random_state"] for entry in observed_kmeans} == {31}


def test_decm_1d_internal_k_is_training_only_and_result_has_no_labels() -> None:
    reducer = _small_reducer()
    result = reducer.fit_transform(_dataset(), seed=17)

    assert reducer.name == "DECM-1D"
    assert reducer.requires_targets is False
    assert result.diagnostics["internal_clusters"] == 6
    assert result.diagnostics["internal_only"] is True
    assert not hasattr(result, "labels")
    assert not hasattr(reducer, "cluster")
    assert not hasattr(reducer, "fit_predict")
    forbidden = {"labels", "cluster_labels", "assignments", "centroids"}
    assert forbidden.isdisjoint(result.diagnostics)


def test_decm_1d_output_is_continuous_final_encoder_latent() -> None:
    result = _small_reducer().fit_transform(_dataset(), seed=41)

    assert np.unique(result.z1).size > result.diagnostics["internal_clusters"]
    assert not np.allclose(result.z1, np.rint(result.z1))
    assert result.diagnostics["representation"] == "final_encoder_continuous_z"


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
            replace(_dataset(), curves=np.zeros((11, 400), dtype=float)),
            r"row count",
        ),
        (
            replace(_dataset(), cell_ids=np.arange(12).reshape(3, 4)),
            r"cell_ids.*one-dimensional",
        ),
        (
            _dataset(n_cells=5),
            r"at least 6 cells",
        ),
    ],
    ids=(
        "curves-not-matrix",
        "wrong-input-width",
        "non-finite-curves",
        "row-mismatch",
        "cell-ids-not-vector",
        "too-few-cells",
    ),
)
def test_decm_1d_rejects_invalid_inputs_fast(
    dataset: All200Dataset,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _small_reducer(
            pretrain_epochs=1,
            joint_epochs=1,
            patience=1,
        ).fit_transform(dataset, seed=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dims": "64"}, "hidden_dims"),
        ({"hidden_dims": (64.5, 16)}, "hidden_dims"),
        ({"hidden_dims": (True, 16)}, "hidden_dims"),
        ({"pretrain_epochs": 0}, "pretrain_epochs"),
        ({"joint_epochs": 0}, "joint_epochs"),
        ({"lr": 0.0}, "lr"),
        ({"patience": 0}, "patience"),
        ({"min_delta": -1.0}, "min_delta"),
        ({"beta1": -1.0}, "beta1"),
        ({"beta2": float("nan")}, "beta2"),
        ({"phi": -0.1}, "phi"),
        ({"internal_clusters": 1}, "internal_clusters"),
        ({"internal_clusters": True}, "internal_clusters"),
        ({"scatter_regularization": 0.0}, "scatter_regularization"),
        ({"device": "not-a-device"}, "device"),
    ],
)
def test_decm_1d_rejects_bad_hyperparameters_fast(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DECM1DReducer(**kwargs)


def test_decm_1d_module_documents_paper_specific_two_stage_mapping() -> None:
    source = inspect.getsource(decm_module)

    assert "Applied Energy 388 (2025) 125677" in source
    assert "Eq. (4)" in source
    assert "Eq. (10)" in source
    assert "decoder is excluded from joint optimization" in source


def test_decm_1d_configs_match_and_smoke_only_reduces_training_rounds() -> None:
    benchmark_dir = Path(__file__).resolve().parents[1]
    formal = yaml.safe_load(
        (benchmark_dir / "config.yaml").read_text(encoding="utf-8")
    )
    smoke = yaml.safe_load(
        (benchmark_dir / "config.smoke.yaml").read_text(encoding="utf-8")
    )

    formal_spec = formal["methods"]["decm_1d"]
    smoke_spec = smoke["methods"]["decm_1d"]
    assert formal_spec["class_path"] == (
        "experiments.fair_1d_benchmark.methods.decm_1d:DECM1DReducer"
    )
    assert smoke_spec["class_path"] == formal_spec["class_path"]
    expected = {
        "hidden_dims": [64, 16],
        "pretrain_epochs": 1000,
        "joint_epochs": 100,
        "lr": 0.001,
        "patience": 50,
        "min_delta": 0.000001,
        "beta1": 1.0,
        "beta2": 0.01,
        "phi": 0.5,
        "internal_clusters": 6,
        "scatter_regularization": 0.000001,
        "device": "cpu",
    }
    assert formal_spec["kwargs"] == expected

    formal_without_rounds = dict(formal_spec["kwargs"])
    smoke_without_rounds = dict(smoke_spec["kwargs"])
    formal_without_rounds.pop("pretrain_epochs")
    formal_without_rounds.pop("joint_epochs")
    smoke_without_rounds.pop("pretrain_epochs")
    smoke_without_rounds.pop("joint_epochs")
    assert smoke_without_rounds == formal_without_rounds
    assert smoke_spec["kwargs"]["pretrain_epochs"] < expected["pretrain_epochs"]
    assert smoke_spec["kwargs"]["joint_epochs"] < expected["joint_epochs"]
