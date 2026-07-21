"""파이프라인 핵심 동작 검증 (소형 표본으로 빠르게)."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from trip_duration import data_prep as dp
from trip_duration import eda
from trip_duration.ensemble import blend, learn_blend_weights
from trip_duration.features import CAT_COLS, NUM_COLS, FeatureBuilder
from trip_duration.models import build_model, metrics


@pytest.fixture(scope="module")
def sample():
    raw = dp.load_train().sample(8000, random_state=0)
    clean, _ = dp.clean_fold(raw)
    return clean.head(3000).reset_index(drop=True)


def test_data_columns(sample):
    for c in [*dp.RAW_FEATURES, dp.TARGET]:
        assert c in sample.columns
    assert len(sample) == 3000


def test_feature_builder_columns(sample):
    out = FeatureBuilder().fit_transform(sample[dp.RAW_FEATURES])
    assert list(out.columns) == NUM_COLS + CAT_COLS
    assert out["log_distance"].notna().all()


def test_model_fit_predict(sample):
    x, y = sample[dp.RAW_FEATURES], sample[dp.TARGET].to_numpy()
    model = build_model(Ridge())
    model.fit(x, y)
    pred = model.predict(x)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_metrics(sample):
    y = sample[dp.TARGET].to_numpy()
    m = metrics(y, y)
    assert set(m) == {"MAE", "RMSE", "R2"}
    assert m["MAE"] == 0.0 and m["R2"] == 1.0


def test_blend_weights_normalized():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    w = learn_blend_weights({"a": y, "b": y * 0.9}, y)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in w.values())
    out = blend({"a": y, "b": y}, w)
    assert np.allclose(out, y)


def test_hypotheses(sample):
    hyps = eda.run_hypotheses(sample)
    assert len(hyps) == 5
    for h in hyps:
        assert {"id", "question", "method", "result", "verdict"} <= set(h)
        assert h["verdict"] in {"채택", "기각"}
