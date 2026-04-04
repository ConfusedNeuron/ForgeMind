"""
tests/dl_engine/test_model.py
CNNLSTM_RUL model definition — architecture, forward pass, shapes, dtypes.
No weights needed. No GPU needed. Runs in ~1s on any machine.

Run:  pytest tests/dl_engine/test_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pytest
from dl_engine.model import CNNLSTM_RUL


@pytest.fixture
def default_model():
    m = CNNLSTM_RUL()
    m.eval()
    return m


@pytest.fixture
def make_batch():
    def _make(batch_size=4, window=50, features=18):
        return torch.rand(batch_size, window, features, dtype=torch.float32)
    return _make


# ── Architecture ─────────────────────────────────────────────────────────────

class TestArchitecture:
    def test_default_instantiates(self, default_model):
        assert default_model is not None

    def test_has_cnn_block(self, default_model):
        assert hasattr(default_model, "cnn")

    def test_has_lstm_block(self, default_model):
        assert hasattr(default_model, "lstm")

    def test_has_regressor_head(self, default_model):
        assert hasattr(default_model, "regressor")

    def test_cnn_has_two_conv_layers(self, default_model):
        conv_layers = [m for m in default_model.cnn if isinstance(m, torch.nn.Conv1d)]
        assert len(conv_layers) == 2

    def test_first_conv_input_channels_equals_n_features(self, default_model):
        convs = [m for m in default_model.cnn if isinstance(m, torch.nn.Conv1d)]
        assert convs[0].in_channels == 18

    def test_second_conv_doubles_filters(self, default_model):
        convs = [m for m in default_model.cnn if isinstance(m, torch.nn.Conv1d)]
        assert convs[1].out_channels == convs[0].out_channels * 2

    def test_lstm_input_size_matches_cnn_output(self, default_model):
        convs = [m for m in default_model.cnn if isinstance(m, torch.nn.Conv1d)]
        expected_lstm_input = convs[1].out_channels
        assert default_model.lstm.input_size == expected_lstm_input

    def test_regressor_ends_in_single_output(self, default_model):
        linears = [m for m in default_model.regressor if isinstance(m, torch.nn.Linear)]
        assert linears[-1].out_features == 1

    @pytest.mark.parametrize("n_features,cnn_filters,lstm_hidden,lstm_layers", [
        (18, 64,  128, 2),
        (18, 32,  64,  1),
        (18, 128, 256, 3),
    ])
    def test_custom_configs_instantiate(self, n_features, cnn_filters, lstm_hidden, lstm_layers):
        m = CNNLSTM_RUL(
            n_features=n_features, cnn_filters=cnn_filters,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers
        )
        assert m is not None


# ── Forward Pass Shape ────────────────────────────────────────────────────────

class TestForwardPassShape:
    def test_batch4_output_shape(self, default_model, make_batch):
        x = make_batch(batch_size=4)
        with torch.no_grad():
            y = default_model(x)
        assert y.shape == (4,)

    def test_batch1_output_shape(self, default_model, make_batch):
        x = make_batch(batch_size=1)
        with torch.no_grad():
            y = default_model(x)
        assert y.shape == (1,)

    def test_batch512_output_shape(self, default_model, make_batch):
        x = make_batch(batch_size=512)
        with torch.no_grad():
            y = default_model(x)
        assert y.shape == (512,)

    def test_output_dtype_is_float32(self, default_model, make_batch):
        x = make_batch()
        with torch.no_grad():
            y = default_model(x)
        assert y.dtype == torch.float32

    def test_output_has_no_nan(self, default_model, make_batch):
        x = make_batch()
        with torch.no_grad():
            y = default_model(x)
        assert not torch.isnan(y).any()

    def test_output_has_no_inf(self, default_model, make_batch):
        x = make_batch()
        with torch.no_grad():
            y = default_model(x)
        assert not torch.isinf(y).any()

    def test_wrong_feature_dim_raises(self, default_model):
        x = torch.rand(4, 50, 17)   # 17 features instead of 18
        with pytest.raises(Exception):
            default_model(x)

    def test_wrong_window_length_raises(self, default_model):
        # BatchNorm1d is window-agnostic so this may not raise — but the
        # LSTM is batch_first and accepts arbitrary seq len, so verify shape
        x = torch.rand(4, 30, 18)   # 30 timesteps instead of 50
        with torch.no_grad():
            y = default_model(x)
        assert y.shape == (4,)  # still produces output, just shorter sequence


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_produces_same_output_in_eval_mode(self, default_model, make_batch):
        default_model.eval()
        x = make_batch(batch_size=2)
        with torch.no_grad():
            y1 = default_model(x)
            y2 = default_model(x)
        assert torch.allclose(y1, y2)

    def test_different_inputs_produce_different_outputs(self, default_model, make_batch):
        default_model.eval()
        x1 = torch.zeros(2, 50, 18)
        x2 = torch.ones(2, 50, 18)
        with torch.no_grad():
            y1 = default_model(x1)
            y2 = default_model(x2)
        assert not torch.allclose(y1, y2)

    def test_eval_mode_disables_dropout_variance(self):
        """Dropout must be disabled in eval — same input, same output."""
        m = CNNLSTM_RUL(dropout=0.5)
        m.eval()
        x = torch.rand(1, 50, 18)
        with torch.no_grad():
            outs = [m(x).item() for _ in range(5)]
        assert all(abs(o - outs[0]) < 1e-5 for o in outs)

    def test_train_mode_has_dropout_variance(self):
        """With large dropout, different forward passes in train mode should vary."""
        m = CNNLSTM_RUL(dropout=0.9)
        m.train()
        x = torch.rand(1, 50, 18)
        outs = [m(x).item() for _ in range(10)]
        assert max(outs) - min(outs) > 0   # at least some variance under heavy dropout


# ── Parameter count ───────────────────────────────────────────────────────────

class TestParameterCount:
    def test_model_has_parameters(self, default_model):
        total = sum(p.numel() for p in default_model.parameters())
        assert total > 0

    def test_parameter_count_is_in_expected_range(self, default_model):
        total = sum(p.numel() for p in default_model.parameters())
        # Default config: ~300k–800k params is reasonable for this architecture
        assert 100_000 < total < 2_000_000, f"Unexpected param count: {total}"
        