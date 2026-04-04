"""
tests/dl_engine/test_dataset.py
dataset.py — subsample_by_unit, build_feature_matrix, fit/apply scaler,
NCMAPSSDataset windowing, DataLoader batch shapes.
No HDF5 file needed — all tests build synthetic data.

Run:  pytest tests/dl_engine/test_dataset.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from dl_engine.dataset import (
    subsample_by_unit,
    build_feature_matrix,
    fit_scaler,
    apply_scaler,
    NCMAPSSDataset,
    make_dataloaders,
    TRAIN_UNITS,
    TEST_UNITS,
)


def _synthetic_data(n_per_unit: int = 100, n_units: int = 2):
    N   = n_per_unit * n_units
    rng = np.random.default_rng(42)

    W        = rng.random((N, 4)).astype(np.float32)
    Xs       = rng.random((N, 14)).astype(np.float32)
    Y        = (rng.random((N, 1)).astype(np.float32) * 125.0)  # FIX: 2D (N,1) — dataset.py uses Y_u[i, 0]
    unit_ids = np.repeat(np.arange(1, n_units + 1), n_per_unit).astype(np.float32)
    A        = np.column_stack([unit_ids, rng.random((N, 4)).astype(np.float32)])

    return W, Xs, Y, A


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_train_units_count(self):
        assert len(TRAIN_UNITS) == 6

    def test_test_units_count(self):
        assert len(TEST_UNITS) == 3

    def test_no_overlap_between_splits(self):
        assert set(TRAIN_UNITS) & set(TEST_UNITS) == set()


# ── subsample_by_unit ─────────────────────────────────────────────────────────

class TestSubsampleByUnit:
    def test_stride1_is_noop(self):
        W, Xs, Y, A = _synthetic_data(n_per_unit=60, n_units=2)
        Wo, Xso, Yo, Ao = subsample_by_unit(W, Xs, Y, A, stride=1)
        assert np.array_equal(Wo, W)
        assert np.array_equal(Ao, A)

    def test_stride2_halves_rows_approximately(self):
        W, Xs, Y, A = _synthetic_data(n_per_unit=100, n_units=3)
        Wo, _, _, _ = subsample_by_unit(W, Xs, Y, A, stride=2)
        assert abs(len(Wo) - 150) <= 3

    def test_stride5_reduces_rows(self):
        W, Xs, Y, A = _synthetic_data(n_per_unit=100, n_units=2)
        Wo, _, _, _ = subsample_by_unit(W, Xs, Y, A, stride=5)
        assert len(Wo) < len(W)

    def test_all_outputs_have_same_row_count(self):
        W, Xs, Y, A = _synthetic_data(n_per_unit=80, n_units=2)
        Wo, Xso, Yo, Ao = subsample_by_unit(W, Xs, Y, A, stride=3)
        assert len(Wo) == len(Xso) == len(Yo) == len(Ao)

    def test_dtypes_preserved(self):
        W, Xs, Y, A = _synthetic_data()
        Wo, Xso, Yo, Ao = subsample_by_unit(W, Xs, Y, A, stride=2)
        assert Wo.dtype == np.float32
        assert Xso.dtype == np.float32

    def test_no_cross_unit_boundary_bleed(self):
        """After striding, unit IDs in output must still match original units."""
        W, Xs, Y, A = _synthetic_data(n_per_unit=50, n_units=3)
        _, _, _, Ao = subsample_by_unit(W, Xs, Y, A, stride=2)
        assert set(Ao[:, 0].astype(int)) == {1, 2, 3}


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_output_shape(self):
        W, Xs, _, _ = _synthetic_data(n_per_unit=100)  # n_units=2 → N=200
        X = build_feature_matrix(W, Xs)
        assert X.shape == (200, 18)

    def test_first_four_cols_are_W(self):
        W, Xs, _, _ = _synthetic_data(n_per_unit=50, n_units=2)
        X = build_feature_matrix(W, Xs)
        assert np.array_equal(X[:, :4], W)

    def test_last_14_cols_are_Xs(self):
        W, Xs, _, _ = _synthetic_data(n_per_unit=50, n_units=2)
        X = build_feature_matrix(W, Xs)
        assert np.array_equal(X[:, 4:], Xs)


# ── fit_scaler / apply_scaler ─────────────────────────────────────────────────

class TestScalerFunctions:
    @pytest.fixture
    def feature_matrix(self):
        W, Xs, _, _ = _synthetic_data(n_per_unit=200)
        return build_feature_matrix(W, Xs)

    def test_fit_scaler_returns_minmaxscaler(self, feature_matrix):
        s = fit_scaler(feature_matrix)
        assert isinstance(s, MinMaxScaler)

    def test_fit_scaler_covers_18_features(self, feature_matrix):
        s = fit_scaler(feature_matrix)
        assert s.n_features_in_ == 18

    def test_apply_scaler_range_within_0_1(self, feature_matrix):
        s      = fit_scaler(feature_matrix)
        scaled = apply_scaler(s, feature_matrix, clip=True)
        assert float(scaled.min()) >= 0.0
        assert float(scaled.max()) <= 1.0

    def test_apply_scaler_shape_preserved(self, feature_matrix):
        s      = fit_scaler(feature_matrix)
        scaled = apply_scaler(s, feature_matrix)
        assert scaled.shape == feature_matrix.shape

    def test_clip_false_may_exceed_bounds_on_test_data(self, feature_matrix):
        s      = fit_scaler(feature_matrix)
        test   = feature_matrix * 2.0
        scaled = apply_scaler(s, test, clip=False)
        assert scaled.max() > 1.0

    def test_clip_true_clamps_out_of_range_values(self, feature_matrix):
        s      = fit_scaler(feature_matrix)
        test   = feature_matrix * 2.0
        scaled = apply_scaler(s, test, clip=True)
        assert scaled.max() <= 1.0 and scaled.min() >= 0.0


# ── NCMAPSSDataset ────────────────────────────────────────────────────────────

class TestNCMAPSSDataset:
    @pytest.fixture
    def dataset(self):
        W, Xs, Y, A = _synthetic_data(n_per_unit=100, n_units=3)
        X         = build_feature_matrix(W, Xs)
        s         = fit_scaler(X)
        Xs_scaled = apply_scaler(s, X)
        return NCMAPSSDataset(Xs_scaled, Y, A, window=50, stride=1)

    def test_sample_count_is_positive(self, dataset):
        assert len(dataset) > 0

    def test_getitem_returns_tuple_of_two(self, dataset):
        item = dataset[0]
        assert isinstance(item, tuple) and len(item) == 2

    def test_sample_shape_is_window_x_features(self, dataset):
        x, _ = dataset[0]
        assert x.shape == (50, 18)

    def test_label_is_scalar_tensor(self, dataset):
        _, y = dataset[0]
        assert y.ndim == 0 or y.shape == torch.Size([])

    def test_sample_dtype_is_float32(self, dataset):
        x, _ = dataset[0]
        assert x.dtype == torch.float32

    def test_no_cross_unit_boundary_in_samples(self):
        """Windows must not span two different units."""
        W, Xs, Y, A = _synthetic_data(n_per_unit=60, n_units=2)
        X         = build_feature_matrix(W, Xs)
        s         = fit_scaler(X)
        Xs_scaled = apply_scaler(s, X)
        ds        = NCMAPSSDataset(Xs_scaled, Y, A, window=50, stride=1)
        # 60 rows per unit, window=50 → 11 windows per unit → 22 total
        assert len(ds) == 22

    def test_stride_reduces_sample_count(self):
        W, Xs, Y, A = _synthetic_data(n_per_unit=100, n_units=2)
        X          = build_feature_matrix(W, Xs)
        s          = fit_scaler(X)
        Xs_s       = apply_scaler(s, X)
        ds_stride1 = NCMAPSSDataset(Xs_s, Y, A, window=50, stride=1)
        ds_stride5 = NCMAPSSDataset(Xs_s, Y, A, window=50, stride=5)
        assert len(ds_stride5) < len(ds_stride1)

    def test_window_larger_than_unit_produces_zero_samples(self):
        W, Xs, Y, A = _synthetic_data(n_per_unit=30, n_units=2)
        X         = build_feature_matrix(W, Xs)
        s         = fit_scaler(X)
        Xs_s      = apply_scaler(s, X)
        ds        = NCMAPSSDataset(Xs_s, Y, A, window=50, stride=1)
        assert len(ds) == 0


# ── make_dataloaders ──────────────────────────────────────────────────────────

class TestMakeDataloaders:
    @pytest.fixture
    def train_and_test_ds(self):
        def _ds(n):
            W, Xs, Y, A = _synthetic_data(n_per_unit=n, n_units=2)
            X = build_feature_matrix(W, Xs)
            s = fit_scaler(X)
            return NCMAPSSDataset(apply_scaler(s, X), Y, A, window=50)
        return _ds(100), _ds(80)

    def test_returns_two_loaders(self, train_and_test_ds):
        train_ds, test_ds = train_and_test_ds
        tl, vl = make_dataloaders(train_ds, test_ds, batch_size=16, num_workers=0)
        assert tl is not None and vl is not None

    def test_train_batch_shape(self, train_and_test_ds):
        train_ds, test_ds = train_and_test_ds
        tl, _  = make_dataloaders(train_ds, test_ds, batch_size=8, num_workers=0)
        x, y   = next(iter(tl))
        assert x.shape[1:] == (50, 18)
        assert y.ndim == 1

    def test_test_batch_is_double_train_batch(self, train_and_test_ds):
        train_ds, test_ds  = train_and_test_ds
        tl, vl             = make_dataloaders(train_ds, test_ds, batch_size=8, num_workers=0)
        x_test, _          = next(iter(vl))
        x_train, _         = next(iter(tl))
        assert x_test.shape[0] <= x_train.shape[0] * 2
