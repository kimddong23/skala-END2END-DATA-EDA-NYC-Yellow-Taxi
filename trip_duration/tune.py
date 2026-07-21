"""하이퍼파라미터 최적화 - HistGradientBoosting 을 무작위 탐색으로 튜닝.

- 후보 비교에서 부스팅이 가장 우수하므로 이 모델을 집중 튜닝.
- 3백만 건 전체로 탐색하면 느려, 부분표본+3겹 교차검증으로 후보를 고르고
  최종은 전체 train 으로 재학습(main).
- 평가는 원래 분 척도의 MAE(작을수록 우수).

변경 이력
- 2026-07-21 최초 작성
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

from .models import build_model

PARAM_DIST = {
    "regressor__model__learning_rate": [0.03, 0.05, 0.1, 0.15],
    "regressor__model__max_depth": [6, 8, 10, None],
    "regressor__model__max_iter": [300, 500, 800],
    "regressor__model__max_leaf_nodes": [31, 63, 127],
    "regressor__model__l2_regularization": [0.0, 0.1, 1.0],
    "regressor__model__min_samples_leaf": [20, 50, 100],
}


def tune_hgb(x: pd.DataFrame, y, n_iter: int = 20, sample: int = 400000) -> dict:
    """부분표본으로 HistGBM 무작위 탐색 후 최적 파라미터·MAE 반환.

    반환 : {best_params, best_mae(cv), n_iter, sample}
    """
    if len(x) > sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(x), sample, replace=False)
        x, y = x.iloc[idx], np.asarray(y)[idx]

    search = RandomizedSearchCV(
        build_model(HistGradientBoostingRegressor(random_state=42)),
        PARAM_DIST,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        refit=False,
    )
    search.fit(x, y)
    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    return {
        "best_params": best,
        "best_mae": round(-float(search.best_score_), 4),
        "n_iter": n_iter,
        "sample": min(sample, len(x)),
    }
