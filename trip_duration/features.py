"""피처 엔지니어링·전처리 - 소요시간 예측을 위한 설명변수 가공.

설계 근거
- 거리는 소요시간과 강한 양의 관계(r≈0.76)이며 로그를 취하면 더 선형(r≈0.78)이라
  log1p(거리) 파생.
- 시각(0~23)은 순환값이라 23시와 0시가 인접 → sin·cos 로 순환 인코딩.
- 러시아워(7~9·16~19)엔 같은 거리라도 소요시간이 길고, 그 격차가 거리에 비례해
  커짐(5~10mi +14%, 10mi+ +13%) → is_rush 와 log거리×러시 상호작용항 추가.
- 승하차 구역(약 260종)은 고카디널리티라 원-핫 대신 타깃 인코딩(내부 교차적합으로
  누수 차단)으로 구역별 평균 소요 경향을 수치화.

변경 이력
- 2026-07-21 최초 작성
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

RUSH_HOURS = set(range(7, 10)) | set(range(16, 20))
NIGHT_HOURS = set(range(0, 6))

NUM_COLS = [
    "trip_distance",
    "log_distance",
    "hour_sin",
    "hour_cos",
    "is_rush",
    "is_night",
    "is_weekend",
    "dist_x_rush",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
]
CAT_COLS = ["PULocationID", "DOLocationID"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """원본 컬럼에서 파생 피처를 생성하는 변환기 (fit 불필요, 무상태).

    입력 : [trip_distance, PULocationID, DOLocationID, pickup_hour,
            pickup_weekday, passenger_count]
    출력 : 위 원본 + log_distance·순환시각·러시/야간/주말 플래그·거리×러시
    """

    def fit(self, x: pd.DataFrame, y=None):  # noqa: ARG002
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        hour = df["pickup_hour"].astype(float)
        df["log_distance"] = np.log1p(df["trip_distance"])
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["is_rush"] = df["pickup_hour"].isin(RUSH_HOURS).astype(int)
        df["is_night"] = df["pickup_hour"].isin(NIGHT_HOURS).astype(int)
        df["is_weekend"] = (df["pickup_weekday"] >= 6).astype(int)
        df["dist_x_rush"] = df["log_distance"] * df["is_rush"]
        df[NUM_COLS] = df[NUM_COLS].astype("float64")
        return df[NUM_COLS + CAT_COLS]


def build_preprocessor() -> ColumnTransformer:
    """수치는 중앙값 대치+표준화, 고카디널리티 구역은 타깃 인코딩(cv=5)."""
    num = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cat = TargetEncoder(cv=cv)
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])
