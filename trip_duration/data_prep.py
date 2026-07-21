"""데이터 로딩·정제·검증 - NYC 택시 운행 소요시간 예측용 train/test.

- 원본 무정제 분할 parquet 을 Polars·Pandas 양쪽으로 로딩해 결과 일치 확인
- clean_fold 로 무효·이상치·중복 삭제 + 비정상 인원 결측화
  (인원 결측 대치는 파이프라인이 train 에만 fit → 누수 없음)
- 결측·중복 점검, 기본 기술통계 제공
- 예측 대상 : duration_min (하차−승차, 분) / 설명변수 : 거리·구역·시각·요일·인원

변경 이력
- 2026-07-21 최초 작성
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

BASE = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE / "data" / "duration_train.parquet"
TEST_PATH = BASE / "data" / "duration_test.parquet"

TARGET = "duration_min"
RAW_FEATURES = [
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_weekday",
    "passenger_count",
]

# 정제 임계값 (도메인 근거) — 소요시간 3시간·거리 100mi 초과는 미터 오류로 간주
DUR_MAX_MIN = 180.0
DIST_MAX_MI = 100.0


def clean_fold(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """원본 fold 정제 : 무효·이상치·중복 행 삭제 + 비정상 인원 결측화.

    - 행 삭제 규칙(무효 라벨·범위 이상치·오류 날짜·중복)은 fold 독립이라 누수 없음.
    - passenger_count 의 결측·비정상(≤0)은 결측으로 통일 → 대치는 파이프라인이
      train 에만 fit(SimpleImputer median)하므로 test 로의 누수 없음.
    - 시간 O(n), 공간 O(n).

    반환 : (정제된 DataFrame, funnel 통계 dict)
    """
    d = df
    funnel = [("원본 (분할된 fold)", 0, len(d))]

    def cut(keep: pd.Series, label: str) -> None:
        nonlocal d
        before = len(d)
        d = d[keep]
        funnel.append((label, before - len(d), len(d)))

    cut(d["duration_min"] > 0, "duration ≤ 0 (하차≤승차, 무효 라벨)")
    cut(d["duration_min"] <= DUR_MAX_MIN, "duration > 180분 (과대 이상치)")
    cut(d["trip_distance"] > 0, "distance ≤ 0 (무효)")
    cut(d["trip_distance"] < DIST_MAX_MI, "distance ≥ 100mi (과대 이상치)")
    dt = d["tpep_pickup_datetime"]
    cut((dt.dt.year == 2026) & (dt.dt.month == 5), "2026-05 이외 (오류 날짜)")

    before = len(d)
    d = d.drop_duplicates()
    funnel.append(("완전중복 제거", before - len(d), len(d)))

    d = d.copy()
    bad_pax = d["passenger_count"] <= 0
    n_bad_pax = int(bad_pax.sum())
    d.loc[bad_pax, "passenger_count"] = np.nan
    n_missing_pax = int(d["passenger_count"].isna().sum())
    d = d.reset_index(drop=True)

    stats = {
        "funnel": funnel,
        "raw": funnel[0][2],
        "clean": len(d),
        "removed": funnel[0][2] - len(d),
        "bad_pax_to_null": n_bad_pax,
        "missing_pax": n_missing_pax,
    }
    return d, stats


def load_train() -> pd.DataFrame:
    """train parquet 을 pandas DataFrame 으로 반환."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"train 파일 없음: {TRAIN_PATH}")
    return pd.read_parquet(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    """test parquet 을 pandas DataFrame 으로 반환."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"test 파일 없음: {TEST_PATH}")
    return pd.read_parquet(TEST_PATH)


def compare_engines() -> dict:
    """같은 집계(요일별 평균 소요시간)를 Polars·Pandas 로 각각 수행해 일치 검증.

    반환 : {polars_rows, pandas_rows, max_abs_diff, match}
    """
    pl_df = (
        pl.read_parquet(TRAIN_PATH)
        .group_by("pickup_weekday")
        .agg(pl.col(TARGET).mean().alias("mean_dur"))
        .sort("pickup_weekday")
    )
    pd_df = (
        pd.read_parquet(TRAIN_PATH)
        .groupby("pickup_weekday")[TARGET]
        .mean()
        .reset_index(name="mean_dur")
        .sort_values("pickup_weekday")
    )
    diff = pl_df["mean_dur"].to_numpy() - pd_df["mean_dur"].to_numpy()
    max_abs = float(abs(diff).max())
    return {
        "polars_rows": pl_df.height,
        "pandas_rows": len(pd_df),
        "max_abs_diff": max_abs,
        "match": max_abs < 1e-9,
    }


def quality_summary(df: pd.DataFrame) -> dict:
    """결측·중복·행수 요약."""
    miss = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    return {"rows": len(df), "duplicates": int(df.duplicated().sum()), "missing": miss}


def describe_stats(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 기술통계(평균·표준편차·분위수) 표 반환."""
    cols = [TARGET, "trip_distance", "pickup_hour", "passenger_count"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"기술통계 대상 컬럼 없음: {missing}")
    return df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).round(2)
