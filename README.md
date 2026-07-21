# [Day 2] 종합 실습 · End2End 데이터 분석 프로젝트

- 작성자 : 광주 3반 4조
- 작성일 : 2026-07-21
- GitHub : https://github.com/kimddong23/skala-END2END-DATA-EDA-NYC-Yellow-Taxi
- 주제 : **NYC 택시 운행 소요시간 예측** (팀 target 예측)
- 데이터 : NYC Yellow Taxi 2026-05 (원본 약 409만 건 **무정제 8:2** train/test 분할, seed 42)
- 목표 : 승차 시점 정보(거리·구역·시각·요일·인원)로 **운행 소요시간(분)** 예측

## 팀원 (광주 3반 4조)
- 신주용 [@kimddong23](https://github.com/kimddong23)
- 문희주 [@creepygrape](https://github.com/creepygrape)
- 임승민 [@zmdals](https://github.com/zmdals)
- 성가연 [@gayeon9-9](https://github.com/gayeon9-9)
- 문성도 [@seogdo](https://github.com/seogdo)

## 접근
End2End 파이프라인 — 데이터 준비 → EDA·가설검정 → 피처 엔지니어링 →
여러 모델 비교 → 하이퍼파라미터 최적화 → 앙상블 → 자동 리포트.
- **원본 무정제 분할** : 정답셋은 원본 409만 행을 정제 없이 고정 시드로 **8:2**(train 80%
  · test 20%, seed 42) 분할 — 같은 원본이면 어디서 돌려도 동일하게 재현. 무효·이상치·중복
  삭제와 인원 결측 중앙값 대치는 **train 에 fit 하며 하류에서** 수행(누수 방지·funnel 노출).
- **누수 컬럼 제외** : 요금·팁·속도·하차시각처럼 시간을 이미 담은 값은 피처에서 제외.

## 코드 구조
```
Day2_종합실습(팀_target)/
├── trip_duration/           # 소스 패키지
│   ├── data_prep.py         # Polars·Pandas 로딩·일치 검증, 정제(clean_fold)·기술통계
│   ├── features.py          # 파생피처(로그·순환·상호작용)+전처리(타깃 인코딩)
│   ├── eda.py               # 상관·가설검정(Welch t-test)
│   ├── models.py            # 파이프라인+로그타깃 래퍼, 평가지표
│   ├── tune.py              # HistGradientBoosting 하이퍼파라미터 최적화
│   ├── ensemble.py          # 검증셋 가중치 블렌딩
│   ├── visualize.py         # Seaborn 6종 + Plotly 2종
│   ├── report.py            # report.md 자동 생성
│   ├── main.py              # 파이프라인 오케스트레이션
│   └── submit.py            # submission.csv · 리더보드 · 예측 미리보기 생성
├── scripts/                 # make_duration_split.py (원본 무정제 8:2 분할 생성)
├── tests/                   # pytest
├── data/                    # train/test parquet (대용량 → gitignore)
├── docs/                    # 제출 리포트 PDF · PDF 페이지 PNG · 실행 캡처
├── output/                  # 차트 · 모델 · report.md
├── requirements.txt · pyproject.toml · .gitignore · README.md
```

## 실행 방법
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_duration_split.py   # 원본 무정제 → 고정 8:2 분할
python -m trip_duration.main      # 전체 파이프라인 (EDA·모델·튜닝·앙상블·리포트)
python -m trip_duration.submit    # submission.csv · 리더보드 생성
ruff check . && pytest
```

## 평가지표 (test)
| 모델 | MAE(분) | RMSE(분) | R² | 상태 |
|---|---|---|---|---|
| 선형회귀 (Ridge) | 5.403 | 8.980 | 0.661 | 기준선 |
| 랜덤포레스트 | 4.205 | 7.180 | 0.783 | 통과 |
| 히스토그램 부스팅 (튜닝) | 4.048 | 7.005 | 0.793 | 통과 |
| 앙상블 (블렌딩) | **4.046** | 6.989 | **0.794** | 최고 |

- **MAE**(분) = 예측이 실제에서 평균 몇 분 빗나갔나 (**낮을수록 좋음**)
- **RMSE**(분) = 큰 오차에 더 벌점을 준 평균 오차, RMSE≥MAE (**낮을수록 좋음**)
- **R²** = 소요시간의 변동을 모델이 몇 % 설명하나, 1=완벽·0=평균 찍기 (**높을수록 좋음**)

## 분석 리포트
제출 리포트 [`docs/종합실습_리포트_광주_3반_신주용.pdf`](docs/종합실습_리포트_광주_3반_신주용.pdf) 전체 페이지 미리보기.

![리포트 p1](docs/pdf_pages/page-01.png)
![리포트 p2](docs/pdf_pages/page-02.png)
![리포트 p3](docs/pdf_pages/page-03.png)
![리포트 p4](docs/pdf_pages/page-04.png)
![리포트 p5](docs/pdf_pages/page-05.png)
![리포트 p6](docs/pdf_pages/page-06.png)
![리포트 p7](docs/pdf_pages/page-07.png)
![리포트 p8](docs/pdf_pages/page-08.png)
![리포트 p9](docs/pdf_pages/page-09.png)
![리포트 p10](docs/pdf_pages/page-10.png)
![리포트 p11](docs/pdf_pages/page-11.png)
![리포트 p12](docs/pdf_pages/page-12.png)
![리포트 p13](docs/pdf_pages/page-13.png)
![리포트 p14](docs/pdf_pages/page-14.png)
![리포트 p15](docs/pdf_pages/page-15.png)
![리포트 p16](docs/pdf_pages/page-16.png)
![리포트 p17](docs/pdf_pages/page-17.png)
![리포트 p18](docs/pdf_pages/page-18.png)
![리포트 p19](docs/pdf_pages/page-19.png)
![리포트 p20](docs/pdf_pages/page-20.png)
![리포트 p21](docs/pdf_pages/page-21.png)
![리포트 p22](docs/pdf_pages/page-22.png)
![리포트 p23](docs/pdf_pages/page-23.png)
![리포트 p24](docs/pdf_pages/page-24.png)

## 실행 로그
전체 파이프라인 실행 출력 — 데이터 준비 · 기술통계 · 상관 · 가설 5종 · 튜닝 · 모델 비교.

![실행 로그](docs/실행결과_0.png)

## 설계·간결화 방법
- 파생피처·전처리를 `FeatureBuilder`+`ColumnTransformer`로 묶어 sklearn Pipeline 단일화
- 로그 타깃은 `TransformedTargetRegressor`로 감싸 학습·역변환 자동화
- 고카디널리티 구역은 `TargetEncoder`(내부 교차적합)로 누수 없이 수치화
- 정제는 `clean_fold`로 일원화(무효·이상치·중복 삭제 + 인원 결측화) → 파이프라인이
  train 에 fit(누수 방지), 정제 funnel·수치는 `report_stats.json`으로 자동 기록
- 관심사 분리(준비/EDA/피처/모델/튜닝/앙상블/시각화/리포트 모듈)
