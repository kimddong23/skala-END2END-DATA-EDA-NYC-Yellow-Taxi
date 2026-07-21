"""운행 소요시간 예측 - 전체 파이프라인 오케스트레이션.

흐름 : 로딩·검증 → EDA·가설검정 → 학습/검증 분할 → HistGBM 튜닝 →
       모델 3종 학습·평가 → 검증셋 가중치로 앙상블 → 시각화 → report.md → 모델 저장.

실행 : python -m trip_duration.main (레포 루트에서)

변경 이력
- 2026-07-21 최초 작성
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from . import data_prep as dp
from . import eda
from . import visualize as viz
from .ensemble import blend, learn_blend_weights
from .models import build_model, metrics
from .report import write_report
from .tune import tune_hgb

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _importance(rf_model):
    """학습된 랜덤포레스트에서 (피처명, 중요도) 반환."""
    prep = rf_model.regressor_.named_steps["prep"]
    imp = rf_model.regressor_.named_steps["model"].feature_importances_
    names = [n.split("__")[-1] for n in prep.get_feature_names_out()]
    if len(names) != len(imp):
        raise ValueError("피처명과 중요도 길이가 다름")
    return names, imp


def main() -> None:
    """소요시간 예측 파이프라인 실행."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, test_raw = dp.load_train(), dp.load_test()
    train, tr_clean = dp.clean_fold(train_raw)
    test, te_clean = dp.clean_fold(test_raw)
    x_cols, target = dp.RAW_FEATURES, dp.TARGET

    # 1) 데이터 준비·검증 (원본 fold 상태 + 정제 funnel)
    compare = dp.compare_engines()
    quality = dp.quality_summary(train_raw)
    desc = dp.describe_stats(train)

    # 2) EDA·가설검정
    corr = eda.correlations(train)
    hyps = eda.run_hypotheses(train)

    # 3) 학습/검증 분할 (앙상블 가중치 학습용)
    x, y = train[x_cols], train[target].to_numpy()
    x_te, y_te = test[x_cols], test[target].to_numpy()
    x_fit, x_val, y_fit, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4) HistGBM 튜닝
    tuned = tune_hgb(x_fit, y_fit)
    hgb = HistGradientBoostingRegressor(random_state=42, **tuned["best_params"])

    # 5) 모델 3종 학습·평가
    defs = {
        "선형회귀 (Ridge)": build_model(Ridge(alpha=1.0)),
        "랜덤포레스트": build_model(
            RandomForestRegressor(
                n_estimators=40,
                max_depth=16,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "히스토그램 부스팅 (튜닝)": build_model(hgb),
    }
    results, val_pred, test_pred, fitted = [], {}, {}, {}
    for name, model in defs.items():
        model.fit(x_fit, y_fit)
        val_pred[name] = model.predict(x_val)
        test_pred[name] = model.predict(x_te)
        results.append({"name": name, **metrics(y_te, test_pred[name])})
        fitted[name] = model

    # 6) 앙상블 (검증셋 가중치 → test 블렌딩)
    weights = learn_blend_weights(val_pred, y_val)
    ens_pred = blend(test_pred, weights)
    ens_metrics = metrics(y_te, ens_pred)
    results.append({"name": "앙상블 (블렌딩)", **ens_metrics})
    best = min(results, key=lambda r: r["MAE"])["name"]

    # 7) 시각화 (표본으로 렌더)
    s = train.sample(n=min(300000, len(train)), random_state=42)
    names, imp = _importance(fitted["랜덤포레스트"])
    charts = [
        viz.target_distribution(s, OUTPUT_DIR),
        viz.hour_profile(s, OUTPUT_DIR),
        viz.distance_rush_interaction(s, OUTPUT_DIR),
        viz.corr_heatmap(s, OUTPUT_DIR),
        viz.model_comparison(results, OUTPUT_DIR),
        viz.feature_importance(names, imp, OUTPUT_DIR),
        viz.plotly_scatter(s, OUTPUT_DIR),
        viz.plotly_pred_actual(y_te, ens_pred, OUTPUT_DIR),
    ]

    # 8) 리포트·모델 저장
    clean = {
        "raw_total": tr_clean["raw"] + te_clean["raw"],
        "train": tr_clean,
        "test": te_clean,
    }
    report = write_report(
        {
            "quality": quality,
            "clean": clean,
            "compare": compare,
            "hypotheses": hyps,
            "results": results,
            "best": best,
            "tuned": tuned["best_params"],
            "weights": weights,
            "ensemble": ens_metrics,
        },
        OUTPUT_DIR,
    )
    model_path = OUTPUT_DIR / "duration_model.joblib"
    joblib.dump({"models": fitted, "weights": weights}, model_path)

    # 리포트 문서(PDF)용 수치 일괄 덤프 → 하드코딩 수치 드리프트 방지
    imp_top = sorted(zip(names, imp, strict=True), key=lambda t: -t[1])
    stats_out = {
        "raw_total": clean["raw_total"],
        "clean": clean,
        "compare": compare,
        "describe": desc.to_dict(),
        "corr": corr.round(4).to_dict(),
        "hypotheses": hyps,
        "tuned": {
            "best_params": tuned["best_params"],
            "sample": tuned["sample"],
            "best_mae": tuned["best_mae"],
        },
        "results": results,
        "weights": weights,
        "ensemble": ens_metrics,
        "best": best,
        "importance": [[n, round(float(v), 4)] for n, v in imp_top],
    }
    (OUTPUT_DIR / "report_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 9) 요약 출력
    print("===== NYC 택시 운행 소요시간 예측 =====")
    print("[1] 데이터 준비 (원본 무정제 분할 → 하류 정제)")
    print(f"    원본 총 {clean['raw_total']:,}행")
    print("    train 정제 funnel:")
    for label, removed, remaining in tr_clean["funnel"]:
        tag = f"-{removed:,}" if removed else "  "
        print(f"      {label:<28} {tag:>10} → {remaining:,}")
    print(
        f"    train {tr_clean['clean']:,} / test {te_clean['clean']:,} · "
        f"인원 결측 대치 {tr_clean['missing_pax']:,}(train)"
    )
    print(f"    Pandas·Polars 일치 : {compare['match']}")
    print("[2] 기술통계 (소요시간·거리)")
    print(desc[[target, "trip_distance"]].to_string())
    print("    상관(소요시간 기준):")
    print(corr.round(3).to_string())
    print("[3] 가설 검정")
    for h in hyps:
        print(f"    {h['id']} {h['question']} → {h['verdict']} ({h['result']})")
    print(
        f"[4] HistGBM 튜닝 (부분표본 {tuned['sample']:,}, cv-MAE {tuned['best_mae']}분)"
    )
    print(f"    {tuned['best_params']}")
    print("[5] 모델 비교 (test)")
    for r in results:
        mark = " *최고" if r["name"] == best else ""
        print(
            f"    - {r['name']:<20} MAE {r['MAE']:.3f} / "
            f"RMSE {r['RMSE']:.3f} / R2 {r['R2']:.3f}{mark}"
        )
    print(f"    앙상블 가중치 : {weights}")
    print(
        f"[6] 피처 중요도 상위 : "
        f"{sorted(zip(names, np.round(imp, 3)), key=lambda t: -t[1])[:5]}"
    )
    print("[7] 시각화 :", ", ".join(c.name for c in charts))
    print(f"[8] 리포트 : {report.name} · 모델 : {model_path.name}")
    print("완료")


if __name__ == "__main__":
    main()
