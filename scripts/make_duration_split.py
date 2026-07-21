"""팀 공통 정답셋 생성 - 운행 소요시간(duration) 예측용 train/test 고정 분할.

목적
- 팀원 전원이 '동일한 원본 분할(같은 행·같은 train/test)'로 경쟁하도록
  원본을 고정 규칙으로 정렬 → 고정 시드로 8:2 분할(train 80% · test 20%)
  → parquet 2개 저장. 같은 원본이면 어디서 돌려도 같은 분할이 나옴.

핵심 원칙 (원본 무정제 분할)
- 원본 parquet 의 '모든 행'을 정제 없이 그대로 고정 시드로 train/test 분할.
- 결측·이상치·비정상 행 정제는 이 단계에서 하지 않음 → 학습 파이프라인이
  train 에 fit 하는 방식으로 하류에서 수행(분석·리포트에 정제 과정 노출).
- target = duration_min = (하차시각 − 승차시각) 분. 승·하차 시각이 없거나
  음수·과대인 행도 여기서는 걸러내지 않고 그대로 남김(정제는 하류 판단).
- feature 는 '승차 시점에 알 수 있는 값'만 사용
  (거리 · 승차/하차 구역 · 시간대 · 요일 · 인원). 요금·팁·총액·속도·하차시각은
  target(시간) 누수라 feature 에서 제외(하차시각은 target 계산에만 사용).

시간·공간 복잡도
- 로딩·정렬 O(n log n) · 분할 O(n) · 공간 O(n). n = 원본 행수.

변경 이력
- 2026-07-21 최초 작성

실행 : python scripts/make_duration_split.py
출력 : data/duration_train.parquet, data/duration_test.parquet
"""

from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "yellow_tripdata_2026-05.parquet"
OUT_TRAIN = BASE / "data" / "duration_train.parquet"
OUT_TEST = BASE / "data" / "duration_test.parquet"
SEED = 42
TEST_SIZE = 0.2

# 분할에 담는 컬럼 : 누수 없는 설명변수 + 파생(시각·요일) + target(duration_min)
SORT_KEYS = ["tpep_pickup_datetime", "PULocationID", "trip_distance", "duration_min"]


def build() -> None:
    """원본 전체를 무정제로 고정 시드 분할 → parquet 2개 저장."""
    if not RAW.exists():
        raise FileNotFoundError(f"원본 없음: {RAW}")

    dur_sec = (
        pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
    ).dt.total_seconds()
    df = (
        pl.scan_parquet(RAW)
        .select(
            "tpep_pickup_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "passenger_count",
            (dur_sec / 60).alias("duration_min"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_weekday"),
        )
        .collect()
    )

    # 원본 무정제 : 어떤 행도 걸러내지 않고 정렬만으로 순서 고정 → 환경 무관 동일 분할
    df = df.sort(SORT_KEYS, nulls_last=True)
    idx = list(range(df.height))
    tr_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    train = df[sorted(tr_idx)]
    test = df[sorted(te_idx)]

    train.write_parquet(OUT_TRAIN)
    test.write_parquet(OUT_TEST)

    n_null_dur = int(df["duration_min"].null_count())
    n_bad_dur = df.filter(
        pl.col("duration_min").is_null() | (pl.col("duration_min") <= 0)
    ).height
    n_null_pax = int(df["passenger_count"].null_count())

    print(f"원본 총 {df.height:,}행 (무정제 분할)")
    print(f"train {train.height:,}행 → {OUT_TRAIN.name}")
    print(f"test  {test.height:,}행 → {OUT_TEST.name}")
    print(f"split seed={SEED}, test={TEST_SIZE:.0%} (전원 동일)")
    print("feature=거리·PU/DO구역·시각·요일·인원 · target=duration_min(분)")
    print(
        "참고(정제는 하류) · duration 결측 "
        f"{n_null_dur:,} · duration≤0 포함 비정상 {n_bad_dur:,} · "
        f"passenger 결측 {n_null_pax:,}"
    )


if __name__ == "__main__":
    build()
