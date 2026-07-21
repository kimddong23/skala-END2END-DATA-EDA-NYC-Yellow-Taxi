"""앙상블 - 여러 모델 예측을 블렌딩.

- 검증셋(train 의 일부)에서 각 모델 예측을 모아 비음수 최소제곱으로 가중치를 학습해
  test 정보를 쓰지 않고 결합(누수 없음).
- 서로 다른 성격(선형·트리·부스팅)의 예측을 합쳐 개별 모델의 실수를 상쇄.

변경 이력
- 2026-07-21 최초 작성
"""

import numpy as np
from scipy.optimize import nnls


def learn_blend_weights(val_preds: dict, y_val) -> dict:
    """검증셋 예측으로 비음수 가중치 학습 후 합=1 로 정규화.

    입력 : val_preds - {모델명: 검증 예측 배열}, y_val - 검증 정답
    반환 : {모델명: 가중치}
    """
    names = list(val_preds)
    mat = np.column_stack([val_preds[n] for n in names])
    w, _ = nnls(mat, np.asarray(y_val, dtype=float))
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return {n: round(float(wi), 4) for n, wi in zip(names, w)}


def blend(preds: dict, weights: dict) -> np.ndarray:
    """예측들을 가중 합으로 결합."""
    if set(preds) != set(weights):
        raise ValueError("preds 와 weights 의 모델 구성이 다름")
    return sum(weights[n] * np.asarray(preds[n]) for n in weights)
