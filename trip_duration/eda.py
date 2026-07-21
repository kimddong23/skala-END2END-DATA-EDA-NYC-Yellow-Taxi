"""탐색적 분석·통계 검정 - 소요시간을 무엇이 좌우하는가.

- 기술통계·상관계수로 관계 파악, Welch t-test 로 그룹 간 차이 검정.
- 거리를 통제한 상태에서도 러시아워 효과가 남는지 확인해 '같은 거리라도 러시엔
  더 오래' 라는 가설을 검증.

변경 이력
- 2026-07-21 최초 작성
"""

import numpy as np
import pandas as pd
from scipy import stats

from .data_prep import TARGET

RUSH = set(range(7, 10)) | set(range(16, 20))


def correlations(df: pd.DataFrame) -> pd.Series:
    """수치형 변수와 소요시간의 상관계수(내림차순)."""
    cols = [TARGET, "trip_distance", "pickup_hour", "pickup_weekday", "passenger_count"]
    return df[cols].corr()[TARGET].drop(TARGET).sort_values(ascending=False)


def _ttest(a, b) -> float:
    """Welch t-test p값."""
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def run_hypotheses(df: pd.DataFrame) -> list:
    """소요시간 관련 가설 검정 결과 목록."""
    d = df
    dur = d[TARGET]
    rush = d["pickup_hour"].isin(RUSH)
    weekend = d["pickup_weekday"] >= 6
    night = d["pickup_hour"].between(0, 5)
    band = d["trip_distance"].between(3, 5)  # 거리 통제 구간

    out = []
    p = _ttest(dur[rush], dur[~rush])
    out.append(
        {
            "id": "H1",
            "question": "러시아워엔 소요시간이 더 긴가",
            "method": "Welch t-test (러시 vs 그외)",
            "result": f"러시 {dur[rush].mean():.1f}분 vs 그외 "
            f"{dur[~rush].mean():.1f}분, p={p:.1e}",
            "verdict": "채택" if p < 0.05 else "기각",
            "note": "출퇴근 혼잡으로 통행 시간 증가",
        }
    )

    r = float(np.corrcoef(d["trip_distance"], dur)[0, 1])
    out.append(
        {
            "id": "H2",
            "question": "거리가 길수록 소요시간이 긴가",
            "method": "피어슨 상관",
            "result": f"r = {r:.3f}",
            "verdict": "채택" if r > 0.3 else "기각",
            "note": "거리는 소요시간의 가장 강한 예측변수",
        }
    )

    p = _ttest(dur[weekend], dur[~weekend])
    out.append(
        {
            "id": "H3",
            "question": "주말이 평일보다 소요시간이 다른가",
            "method": "Welch t-test (주말 vs 평일)",
            "result": f"주말 {dur[weekend].mean():.1f}분 vs 평일 "
            f"{dur[~weekend].mean():.1f}분, p={p:.1e}",
            "verdict": "채택" if p < 0.05 else "기각",
            "note": "주중 통근 수요가 통행 시간에 영향",
        }
    )

    p = _ttest(dur[night], dur[~night])
    out.append(
        {
            "id": "H4",
            "question": "심야(0~5시)엔 소요시간이 짧은가",
            "method": "Welch t-test (심야 vs 그외)",
            "result": f"심야 {dur[night].mean():.1f}분 vs 그외 "
            f"{dur[~night].mean():.1f}분, p={p:.1e}",
            "verdict": "채택" if p < 0.05 else "기각",
            "note": "도로 혼잡이 적어 같은 거리도 빠르게 통행",
        }
    )

    a, b = dur[band & rush], dur[band & ~rush]
    p = _ttest(a, b)
    out.append(
        {
            "id": "H5",
            "question": "거리(3~5mi)를 통제해도 러시가 더 긴가",
            "method": "Welch t-test (동일 거리대, 러시 vs 그외)",
            "result": f"러시 {a.mean():.1f}분 vs 그외 {b.mean():.1f}분, p={p:.1e}",
            "verdict": "채택" if p < 0.05 else "기각",
            "note": "거리를 맞춰도 러시 효과가 남아 시각(혼잡) 자체가 원인에 가까움",
        }
    )
    return out
