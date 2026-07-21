"""시각화 - 정적(Seaborn) + 인터랙티브(Plotly) 차트.

- 소요시간 분포·시간대 프로파일·거리×러시 상호작용·상관·모델비교·피처중요도.
- Plotly 로 인터랙티브 산점도와 예측-실제 대조도 생성(HTML+PNG).

변경 이력
- 2026-07-21 최초 작성
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sns.set_theme(context="talk", style="whitegrid")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

TARGET = "duration_min"
RUSH = set(range(7, 10)) | set(range(16, 20))
OK, WARN, BAD = "#2a9d8f", "#e9c46a", "#e76f51"


def target_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """소요시간 원본 vs log1p 분포 - 오른쪽 치우침과 로그변환 효과."""
    path = out_dir / "01_target_dist.png"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df[TARGET], bins=60, color=OK, ax=ax[0])
    ax[0].set_title("소요시간 분포 (원본, 오른쪽 치우침)")
    ax[0].set_xlabel("소요시간 (분)")
    sns.histplot(np.log1p(df[TARGET]), bins=60, color="#264653", ax=ax[1])
    ax[1].set_title("log1p(소요시간) 분포 (정규에 근접)")
    ax[1].set_xlabel("log1p(소요시간)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def hour_profile(df: pd.DataFrame, out_dir: Path) -> Path:
    """시간대별 평균 소요시간 - 러시아워 음영."""
    path = out_dir / "02_hour_profile.png"
    prof = df.groupby("pickup_hour")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x=prof.index, y=prof.values, marker="o", color="#264653", ax=ax)
    for h in RUSH:
        ax.axvspan(h - 0.5, h + 0.5, color=WARN, alpha=0.25)
    ax.set_title("시간대별 평균 소요시간 (노랑=러시아워)")
    ax.set_xlabel("승차 시각 (시)")
    ax.set_ylabel("평균 소요시간 (분)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def distance_rush_interaction(df: pd.DataFrame, out_dir: Path) -> Path:
    """거리 구간 × 러시 여부별 평균 소요시간 - 상호작용 확인."""
    path = out_dir / "03_dist_rush.png"
    d = df.copy()
    d["거리대"] = pd.cut(
        d["trip_distance"],
        [0, 1, 2, 3, 5, 10, 100],
        labels=["~1", "1-2", "2-3", "3-5", "5-10", "10+"],
    )
    d["구간"] = np.where(d["pickup_hour"].isin(RUSH), "러시아워", "그 외")
    g = d.groupby(["거리대", "구간"], observed=True)[TARGET].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=g,
        x="거리대",
        y=TARGET,
        hue="구간",
        palette={"러시아워": BAD, "그 외": OK},
        ax=ax,
    )
    ax.set_title("같은 거리라도 러시아워엔 소요시간이 길다")
    ax.set_xlabel("이동 거리 구간 (mi)")
    ax.set_ylabel("평균 소요시간 (분)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def corr_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """수치형 변수 상관 히트맵."""
    path = out_dir / "04_corr.png"
    cols = [TARGET, "trip_distance", "pickup_hour", "pickup_weekday", "passenger_count"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="mako", square=True, ax=ax)
    ax.set_title("변수 간 상관")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def model_comparison(results: list, out_dir: Path) -> Path:
    """모델별 test MAE 비교 막대."""
    path = out_dir / "05_model_mae.png"
    r = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=r, x="name", y="MAE", color=OK, ax=ax)
    for i, v in enumerate(r["MAE"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_title("모델별 test MAE (분, 낮을수록 우수)")
    ax.set_xlabel("")
    ax.set_ylabel("MAE (분)")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def feature_importance(names, values, out_dir: Path) -> Path:
    """피처 중요도 상위 막대."""
    path = out_dir / "06_importance.png"
    s = pd.Series(values, index=names).sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=s.values, y=s.index, color="#264653", ax=ax)
    ax.set_title("피처 중요도 (랜덤포레스트 기준)")
    ax.set_xlabel("중요도")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plotly_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    """거리 vs 소요시간 인터랙티브 산점도(러시 색상, 표본)."""
    html = out_dir / "07_scatter.html"
    s = df.sample(n=min(4000, len(df)), random_state=42).copy()
    s["구간"] = np.where(s["pickup_hour"].isin(RUSH), "러시아워", "그 외")
    fig = px.scatter(
        s,
        x="trip_distance",
        y=TARGET,
        color="구간",
        color_discrete_map={"러시아워": BAD, "그 외": OK},
        labels={"trip_distance": "이동 거리 (mi)", TARGET: "소요시간 (분)"},
        title="거리 vs 소요시간 (러시아워 색상)",
        opacity=0.5,
        template="plotly_white",
    )
    fig.write_html(html)
    fig.write_image(out_dir / "07_scatter.png", width=1000, height=600)
    return html


def plotly_pred_actual(y_true, y_pred, out_dir: Path) -> Path:
    """예측 vs 실제 소요시간 인터랙티브 대조(표본)."""
    html = out_dir / "08_pred_actual.html"
    n = min(4000, len(y_true))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_true), n, replace=False)
    s = pd.DataFrame({"실제": np.asarray(y_true)[idx], "예측": np.asarray(y_pred)[idx]})
    fig = px.scatter(
        s,
        x="실제",
        y="예측",
        opacity=0.4,
        labels={"실제": "실제 소요(분)", "예측": "예측 소요(분)"},
        title="예측 vs 실제 소요시간 (앙상블)",
        template="plotly_white",
    )
    lim = float(np.percentile(s["실제"], 99))
    fig.add_shape(
        type="line", x0=0, y0=0, x1=lim, y1=lim, line={"color": BAD, "dash": "dash"}
    )
    fig.write_html(html)
    fig.write_image(out_dir / "08_pred_actual.png", width=900, height=700)
    return html
