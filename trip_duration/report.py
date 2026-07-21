"""분석 결과를 report.md 로 자동 생성.

변경 이력
- 2026-07-21 최초 작성
"""

from pathlib import Path


def write_report(ctx: dict, out_dir: Path) -> Path:
    """분석·모델 결과를 담은 report.md 작성 후 경로 반환."""
    path = out_dir / "report.md"
    lines = ["# NYC 택시 운행 소요시간 예측 리포트", ""]

    lines += ["## 1. 데이터", ""]
    cl = ctx["clean"]
    tr, te = cl["train"], cl["test"]
    lines += [
        f"- 원본 무정제 분할 총 {cl['raw_total']:,}행 (정제는 하류·train fit)",
        "",
        "### 1-1. train 정제 funnel",
        "",
        "| 단계 | 제거 | 남은 행 |",
        "|---|---:|---:|",
    ]
    for label, removed, remaining in tr["funnel"]:
        lines.append(f"| {label} | {removed:,} | {remaining:,} |")
    lines += [
        f"| **정제 완료** | **{tr['removed']:,}** | **{tr['clean']:,}** |",
        "",
        f"- 비정상 인원(≤0) {tr['bad_pax_to_null']:,}건을 결측 처리 → "
        f"train 결측 {tr['missing_pax']:,}건 중앙값 대치(파이프라인)",
        f"- test 도 동일 규칙 : {te['raw']:,} → {te['clean']:,}행",
        f"- Pandas·Polars 집계 일치 : {ctx['compare']['match']} "
        f"(최대차 {ctx['compare']['max_abs_diff']:.2e})",
        "",
    ]

    lines += ["## 2. 가설 검정", ""]
    for h in ctx["hypotheses"]:
        lines += [
            f"### {h['id']}. {h['question']}",
            f"- 방법 : {h['method']}",
            f"- 결과 : {h['result']}",
            f"- 판정 : {h['verdict']} — {h['note']}",
            "",
        ]

    lines += [
        "## 3. 모델 비교 (test)",
        "",
        "| 모델 | MAE(분) | RMSE(분) | R2 |",
        "|---|---|---|---|",
    ]
    for r in ctx["results"]:
        mark = " ★" if r["name"] == ctx["best"] else ""
        lines.append(f"| {r['name']}{mark} | {r['MAE']} | {r['RMSE']} | {r['R2']} |")
    lines += [
        "",
        f"- 튜닝된 HistGBM 파라미터 : {ctx['tuned']}",
        f"- 앙상블 가중치 : {ctx['weights']}",
        f"- 앙상블 test : MAE {ctx['ensemble']['MAE']}분 · R2 {ctx['ensemble']['R2']}",
        "",
    ]

    lines += [
        "## 4. 산출물",
        "",
        "- 모델 : duration_model.joblib",
        "- 차트 : output/*.png, *.html",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
