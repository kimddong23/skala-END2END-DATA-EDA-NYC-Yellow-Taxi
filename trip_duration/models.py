"""모델 정의·평가 - 소요시간 회귀.

- 각 모델 = FeatureBuilder → 전처리(ColumnTransformer) → 추정기 를 한 Pipeline 으로
  묶고, 오른쪽으로 치우친 소요시간을 log1p 로 변환해 학습(예측은 expm1 로 복원).
- 평가는 원래 분(minute) 척도에서 MAE·RMSE·R2 로 계산.
- 후보 : 선형(Ridge, 기준선) · 랜덤포레스트 · HistGradientBoosting.

변경 이력
- 2026-07-21 최초 작성
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from .features import FeatureBuilder, build_preprocessor


def build_model(estimator) -> TransformedTargetRegressor:
    """파생피처+전처리+추정기 파이프라인을 log1p 타깃 래퍼로 감싼 회귀기 반환."""
    pipe = Pipeline(
        [
            ("features", FeatureBuilder()),
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipe, func=np.log1p, inverse_func=np.expm1
    )


def metrics(y_true, y_pred) -> dict:
    """분 척도 평가지표 : MAE·RMSE·R2 (표시는 4자리 반올림)."""
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }
