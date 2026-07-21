"""제출 파일·리더보드 생성 - 저장된 모델로 test 예측 후 submission 산출.

- 학습된 모델 묶음(joblib)을 불러와 test 를 예측, 모델별 성능을 리더보드로 정리.
- 최고 모델 예측을 submission.csv 로 저장하고, 표본 예측 미리보기를 만듦.

실행 : python -m trip_duration.submit (레포 루트에서)

변경 이력
- 2026-07-21 최초 작성
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data_prep as dp
from .ensemble import blend
from .models import metrics

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
WEEKDAY_KO = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}


def main() -> None:
    """저장된 모델로 test 예측 → 리더보드·submission.csv·미리보기 생성."""
    bundle = joblib.load(OUTPUT_DIR / "duration_model.joblib")
    models, weights = bundle["models"], bundle["weights"]
    test, _ = dp.clean_fold(dp.load_test())
    x_te, y_te = test[dp.RAW_FEATURES], test[dp.TARGET].to_numpy()

    preds = {n: m.predict(x_te) for n, m in models.items()}
    results = [{"name": n, **metrics(y_te, p)} for n, p in preds.items()]
    ens = blend(preds, weights)
    results.append({"name": "앙상블 (블렌딩)", **metrics(y_te, ens)})

    best = min(results, key=lambda r: r["MAE"])
    best_pred = ens if best["name"] == "앙상블 (블렌딩)" else preds[best["name"]]

    sub = pd.DataFrame(
        {"row_id": np.arange(len(y_te)), "duration_min_pred": np.round(best_pred, 2)}
    )
    sub.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(len(y_te), 10, replace=False))
    preview = pd.DataFrame(
        {
            "거리(mi)": np.round(test["trip_distance"].to_numpy()[idx], 1),
            "시각": test["pickup_hour"].to_numpy()[idx],
            "요일": [WEEKDAY_KO[w] for w in test["pickup_weekday"].to_numpy()[idx]],
            "실제(분)": np.round(y_te[idx], 1),
            "예측(분)": np.round(best_pred[idx], 1),
            "오차(분)": np.round(np.abs(best_pred[idx] - y_te[idx]), 1),
        }
    )

    board = pd.DataFrame(sorted(results, key=lambda r: r["MAE"]))
    board.insert(0, "순위", range(1, len(board) + 1))
    board.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)
    preview.to_csv(OUTPUT_DIR / "preview.csv", index=False)

    print("===== 제출 · 리더보드 =====")
    print(f"submission.csv : {len(sub):,}행 저장")
    for i, r in enumerate(sorted(results, key=lambda x: x["MAE"])):
        mark = " ★ 1위" if i == 0 else ""
        print(f"  {i + 1}. {r['name']:<20} MAE {r['MAE']:.3f} / R2 {r['R2']:.3f}{mark}")
    print("예측 미리보기 (표본):")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
