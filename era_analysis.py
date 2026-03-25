# %% [markdown]
#  # LifeCycle of a Creator: Era & Regime Analysis
#
#  Detects content eras and regime changes in HealthyGamerGG using transcript embeddings.
#
#
#
#  **Analyses:**
#
#  1. Embedding drift over time (rolling cosine similarity)
#
#  2. UMAP temporal visualization
#
#  3. Changepoint detection (PELT on PCA-reduced embeddings)
#
#  4. Topic tracking over time (KMeans + TF-IDF labels)
#
#  5. Novelty score (distance from historical content centroid)

# %% [markdown]
#  ## Imports & Setup

# %%
import sys
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sqlalchemy import text
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

import umap as umap_lib
import ruptures as rpt

sys.path.insert(0, ".")
from models import engine

plt.rcParams["figure.dpi"] = 120


# %% [markdown]
#  ## Config

# %%
# ===================== CONFIG =====================
CHANNEL_QUERY = "healthygamergg"  # ILIKE match on channel title
CONTENT_TYPE = "long"  # 'long' | 'short' | 'all'
SHORTS_MAX_SECS = 180  # YouTube Shorts threshold (seconds)
ROLLING_WINDOW = 10  # per-video rolling window for smoothing
N_TOPICS = 8  # number of KMeans topic clusters
CP_PENALTY = 10  # PELT penalty — lower = more changepoints
# ==================================================


# %% [markdown]
#  ## Load Data


# %%
def _duration_filter(content_type, max_secs):
    if content_type == "long":
        return f"AND v.duration_seconds > {max_secs}"
    if content_type == "short":
        return f"AND v.duration_seconds <= {max_secs}"
    return ""


_dur = _duration_filter(CONTENT_TYPE, SHORTS_MAX_SECS)

_sql = f"""
    SELECT
        v.video_id,
        v.title,
        v.published_at,
        v.duration_seconds,
        v.view_count,
        v.like_count,
        v.comment_count,
        v.engagement_rate,
        v.transcript_word_count,
        v.transcript_words_per_minute,
        v.days_since_channel_start,
        avg(tc.embedding) AS mean_embedding
    FROM videos v
    JOIN channels c ON v.channel_id = c.channel_id
    JOIN transcript_chunks tc ON v.video_id = tc.video_id
    WHERE LOWER(c.title) LIKE LOWER('%{CHANNEL_QUERY}%')
      AND tc.embedding IS NOT NULL
      {_dur}
    GROUP BY
        v.video_id, v.title, v.published_at, v.duration_seconds,
        v.view_count, v.like_count, v.comment_count, v.engagement_rate,
        v.transcript_word_count, v.transcript_words_per_minute, v.days_since_channel_start
    ORDER BY v.published_at
"""

with engine.connect() as conn:
    result = conn.execute(text(_sql))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
df["month"] = df["published_at"].dt.to_period("M")


def _parse_emb(v):
    if isinstance(v, np.ndarray):
        return v.astype(float)
    if isinstance(v, (list, tuple)):
        return np.array(v, dtype=float)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
    return np.array(nums, dtype=float)


df["embedding"] = df["mean_embedding"].apply(_parse_emb)
emb = np.stack(df["embedding"].values)
emb_norm = normalize(emb)

print(f"Videos loaded : {len(df)}")
print(
    f'Date range    : {df["published_at"].min().date()} -> {df["published_at"].max().date()}'
)
print(f"Embedding dim : {emb.shape[1]}")
print(f"Content type  : {CONTENT_TYPE!r}  (threshold = {SHORTS_MAX_SECS}s)")


# %% [markdown]
#  ## Monthly Aggregation

# %%
monthly_meta = (
    df.groupby("month")
    .agg(
        n_videos=("video_id", "count"),
        mean_views=("view_count", "mean"),
        median_views=("view_count", "median"),
        mean_engagement=("engagement_rate", "mean"),
        mean_duration_s=("duration_seconds", "mean"),
        mean_wpm=("transcript_words_per_minute", "mean"),
    )
    .reset_index()
)

monthly_emb = (
    df.groupby("month")["embedding"]
    .apply(lambda x: normalize(np.stack(x.values)).mean(axis=0))
    .reset_index()
)
monthly_emb.columns = ["month", "embedding"]

monthly = monthly_meta.merge(monthly_emb, on="month")
monthly["month_dt"] = monthly["month"].dt.to_timestamp()
monthly_emb_matrix = np.stack(monthly["embedding"].values)
monthly_emb_norm = normalize(monthly_emb_matrix)

print(
    monthly[["month", "n_videos", "mean_views", "mean_engagement"]].to_string(
        index=False
    )
)


# %% [markdown]
#  ## Analysis 1: Embedding Drift
#
#  Rolling cosine similarity between each video and the mean of the previous N videos.
#
#  A drop indicates a topic pivot.


# %%
def _rolling_drift(emb_norm, window):
    sims = np.full(len(emb_norm), np.nan)
    for i in range(window, len(emb_norm)):
        past = emb_norm[i - window : i].mean(axis=0)
        past = past / (np.linalg.norm(past) + 1e-10)
        sims[i] = float(emb_norm[i] @ past)
    return sims


df["drift_sim"] = _rolling_drift(emb_norm, ROLLING_WINDOW)
monthly["drift_sim"] = _rolling_drift(monthly_emb_norm, max(3, ROLLING_WINDOW // 3))

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

ax = axes[0]
ax.scatter(df["published_at"], df["drift_sim"], alpha=0.35, s=12, color="steelblue")
smooth = (
    pd.Series(df["drift_sim"].values)
    .rolling(ROLLING_WINDOW * 2, center=True, min_periods=1)
    .mean()
)
ax.plot(
    df["published_at"],
    smooth,
    color="crimson",
    lw=2,
    label=f"{ROLLING_WINDOW*2}-video rolling mean",
)
ax.set_title(
    "Embedding Drift — Per Video  (lower = more different from recent content)"
)
ax.set_ylabel("Cosine similarity to prior window")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[1]
ax.plot(monthly["month_dt"], monthly["drift_sim"], marker="o", color="darkorange", ms=5)
ax.fill_between(monthly["month_dt"], monthly["drift_sim"], alpha=0.2, color="orange")
ax.set_title("Embedding Drift — Monthly")
ax.set_ylabel("Cosine similarity")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("drift_analysis.png", bbox_inches="tight")
plt.show()


# %% [markdown]
#  ## Analysis 2: UMAP Temporal Visualization
#
#  Each dot is a video. Colour = publication date. Clusters that share colour = content era.

# %%
reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_2d = reducer.fit_transform(emb_norm)

df["umap_x"] = umap_2d[:, 0]
df["umap_y"] = umap_2d[:, 1]

dates_num = mdates.date2num(df["published_at"].dt.to_pydatetime())

fig, ax = plt.subplots(figsize=(12, 9))
sc = ax.scatter(df["umap_x"], df["umap_y"], c=dates_num, cmap="plasma", alpha=0.6, s=20)
cbar = plt.colorbar(sc, ax=ax)
tick_locs = np.linspace(dates_num.min(), dates_num.max(), 6)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([mdates.num2date(t).strftime("%Y-%m") for t in tick_locs])
ax.set_title("UMAP of Video Embeddings — Colored by Publication Date")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig("umap_temporal.png", bbox_inches="tight")
plt.show()


# %%
# Monthly trajectory — dots connected chronologically
monthly_2d = reducer.transform(monthly_emb_norm)
monthly["umap_x"] = monthly_2d[:, 0]
monthly["umap_y"] = monthly_2d[:, 1]

m_dates_num = mdates.date2num(monthly["month_dt"].dt.to_pydatetime())

fig, ax = plt.subplots(figsize=(12, 9))
sc = ax.scatter(
    monthly["umap_x"],
    monthly["umap_y"],
    c=m_dates_num,
    cmap="plasma",
    s=80,
    zorder=5,
    edgecolors="k",
    linewidths=0.4,
)
ax.plot(monthly["umap_x"].values, monthly["umap_y"].values, "k-", alpha=0.25, zorder=4)
for _, row in monthly.iterrows():
    ax.annotate(
        str(row["month"]),
        (row["umap_x"], row["umap_y"]),
        fontsize=6.5,
        alpha=0.8,
        xytext=(3, 3),
        textcoords="offset points",
    )
cbar = plt.colorbar(sc, ax=ax)
tick_locs = np.linspace(m_dates_num.min(), m_dates_num.max(), 6)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([mdates.num2date(t).strftime("%Y-%m") for t in tick_locs])
ax.set_title("Monthly Content Trajectory in Embedding Space")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig("umap_monthly_trajectory.png", bbox_inches="tight")
plt.show()


# %% [markdown]
#  ## Analysis 3: Changepoint Detection
#
#  PELT algorithm on PCA-reduced embeddings. Red lines = detected regime breaks.
#
#  Tune `CP_PENALTY` in config — lower values give more breakpoints.

# %%
pca = PCA(n_components=min(20, emb_norm.shape[1]), random_state=42)
emb_pca = pca.fit_transform(emb_norm)
monthly_pca = pca.transform(monthly_emb_norm)


def _changepoints(signal, penalty):
    algo = rpt.Pelt(model="rbf").fit(signal)
    bkps = algo.predict(pen=penalty)
    return bkps[:-1]  # last entry is always len(signal)


video_bkps = _changepoints(emb_pca, CP_PENALTY)
monthly_bkps = _changepoints(monthly_pca, CP_PENALTY)

print(f"Video-level changepoints ({len(video_bkps)}):")
for idx in video_bkps:
    print(f'  {df["published_at"].iloc[idx].date()}  —  "{df["title"].iloc[idx]}"')

print(f"\nMonthly changepoints ({len(monthly_bkps)}):")
for idx in monthly_bkps:
    print(f'  {monthly["month"].iloc[idx]}')

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

ax = axes[0]
ax.plot(df["published_at"], emb_pca[:, 0], alpha=0.6, lw=0.8, label="PC1")
ax.plot(df["published_at"], emb_pca[:, 1], alpha=0.6, lw=0.8, label="PC2")
for idx in video_bkps:
    ax.axvline(df["published_at"].iloc[idx], color="red", lw=1.5, ls="--", alpha=0.8)
ax.set_title("Changepoint Detection — Per Video  (red = regime break)")
ax.set_ylabel("PCA value")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[1]
ax.plot(monthly["month_dt"], monthly_pca[:, 0], marker="o", ms=4, lw=1.2, label="PC1")
ax.plot(monthly["month_dt"], monthly_pca[:, 1], marker="s", ms=4, lw=1.2, label="PC2")
for idx in monthly_bkps:
    ax.axvline(monthly["month_dt"].iloc[idx], color="red", lw=1.5, ls="--", alpha=0.8)
ax.set_title("Changepoint Detection — Monthly")
ax.set_ylabel("PCA value")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("changepoints.png", bbox_inches="tight")
plt.show()


# %% [markdown]
#  ## Analysis 4: Topic Tracking Over Time
#
#  KMeans clusters on embeddings. Topics labeled via TF-IDF on video titles.
#
#  Stacked area chart shows when each topic dominates.

# %%
# Elbow plot — use to validate N_TOPICS
inertias = []
_ks = range(3, 16)
for k in _ks:
    inertias.append(
        KMeans(n_clusters=k, random_state=42, n_init=5).fit(emb_norm).inertia_
    )

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(_ks), inertias, marker="o")
ax.axvline(N_TOPICS, color="red", ls="--", label=f"N_TOPICS = {N_TOPICS}")
ax.set_title("KMeans Elbow Plot — adjust N_TOPICS in config if needed")
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
ax.legend()
plt.tight_layout()
plt.show()


# %%
km = KMeans(n_clusters=N_TOPICS, random_state=42, n_init=10)
df["topic_id"] = km.fit_predict(emb_norm)

tfidf = TfidfVectorizer(max_features=300, stop_words="english")
topic_labels = {}
for t in range(N_TOPICS):
    titles = df.loc[df["topic_id"] == t, "title"].tolist()
    if not titles:
        topic_labels[t] = f"Topic {t}"
        continue
    try:
        mat = tfidf.fit_transform(titles)
        terms = tfidf.get_feature_names_out()
        top3 = [terms[i] for i in mat.mean(axis=0).A1.argsort()[::-1][:3]]
        topic_labels[t] = f'T{t}: {", ".join(top3)}'
    except Exception:
        topic_labels[t] = f"Topic {t}"

df["topic_label"] = df["topic_id"].map(topic_labels)

print("Topic clusters:")
for tid, label in topic_labels.items():
    n = (df["topic_id"] == tid).sum()
    print(f"  {label}  ({n} videos)")


# %%
monthly_topic_counts = df.groupby(["month", "topic_label"]).size().unstack(fill_value=0)
monthly_topic_pct = monthly_topic_counts.div(monthly_topic_counts.sum(axis=1), axis=0)
monthly_topic_pct.index = monthly_topic_pct.index.to_timestamp()

fig, ax = plt.subplots(figsize=(16, 6))
monthly_topic_pct.plot(kind="area", stacked=True, ax=ax, colormap="tab10", alpha=0.82)
ax.set_title("Topic Proportions Over Time")
ax.set_ylabel("Proportion of videos")
ax.set_xlabel("")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig("topic_tracking.png", bbox_inches="tight")
plt.show()


# %%
# UMAP re-colored by topic
palette = plt.cm.tab10(np.linspace(0, 1, N_TOPICS))

fig, ax = plt.subplots(figsize=(12, 9))
for tid, label in topic_labels.items():
    grp = df[df["topic_id"] == tid]
    ax.scatter(
        grp["umap_x"], grp["umap_y"], label=label, alpha=0.6, s=20, color=palette[tid]
    )
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.set_title("UMAP — Colored by Topic Cluster")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig("umap_topics.png", bbox_inches="tight")
plt.show()


# %% [markdown]
#  ## Analysis 5: Novelty Score
#
#  Cosine distance from each video to the centroid of **all prior** videos.
#
#  High novelty = creator is venturing into new territory.
#
#  Sustained high novelty = exploration era. Low novelty = refinement/exploitation era.


# %%
def _novelty_score(emb_norm):
    scores = np.full(len(emb_norm), np.nan)
    for i in range(1, len(emb_norm)):
        centroid = emb_norm[:i].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        scores[i] = 1.0 - float(emb_norm[i] @ centroid)
    return scores


df["novelty"] = _novelty_score(emb_norm)
monthly["novelty"] = df.groupby("month")["novelty"].mean().values

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

ax = axes[0]
ax.scatter(df["published_at"], df["novelty"], alpha=0.3, s=12, color="mediumpurple")
smooth_nov = (
    pd.Series(df["novelty"].values)
    .rolling(ROLLING_WINDOW * 2, center=True, min_periods=1)
    .mean()
)
ax.plot(
    df["published_at"],
    smooth_nov,
    color="darkviolet",
    lw=2,
    label=f"{ROLLING_WINDOW*2}-video rolling mean",
)
ax.set_title("Novelty Score — Per Video  (higher = further from all prior content)")
ax.set_ylabel("Cosine distance to historical centroid")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[1]
ax.plot(monthly["month_dt"], monthly["novelty"], marker="o", color="purple", ms=5)
ax.fill_between(monthly["month_dt"], monthly["novelty"], alpha=0.2, color="purple")
ax.set_title("Novelty Score — Monthly")
ax.set_ylabel("Avg cosine distance to historical centroid")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("novelty_score.png", bbox_inches="tight")
plt.show()


# %% [markdown]
#  ## Era Summary
#
#  Use detected changepoints to define eras, then compare engagement metrics across them.

# %%
if video_bkps:
    era_boundaries = [0] + video_bkps + [len(df)]
    df["era"] = 0
    for era_num, (start, end) in enumerate(
        zip(era_boundaries[:-1], era_boundaries[1:])
    ):
        df.iloc[start:end, df.columns.get_loc("era")] = era_num

    era_start_strs = [
        df["published_at"].iloc[i].strftime("%Y-%m") for i in era_boundaries[:-1]
    ]
    df["era_label"] = df["era"].map(
        {i: f"Era {i+1} ({era_start_strs[i]})" for i in range(len(era_boundaries) - 1)}
    )

    era_summary = df.groupby("era_label").agg(
        n_videos=("video_id", "count"),
        mean_views=("view_count", "mean"),
        median_views=("view_count", "median"),
        mean_engagement=("engagement_rate", "mean"),
        mean_duration_s=("duration_seconds", "mean"),
        mean_wpm=("transcript_words_per_minute", "mean"),
        mean_novelty=("novelty", "mean"),
    )
    print(era_summary.to_string())

    metrics = [
        "view_count",
        "engagement_rate",
        "transcript_words_per_minute",
        "novelty",
    ]
    labels = ["View Count", "Engagement Rate", "Words / Min", "Novelty Score"]
    era_order = sorted(df["era_label"].unique(), key=lambda x: int(x.split()[1]))

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        data = [
            df.loc[df["era_label"] == era, metric].dropna().values for era in era_order
        ]
        ax.boxplot(
            data,
            labels=[e.replace(" ", "\n") for e in era_order],
            patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue", color="navy"),
            medianprops=dict(color="crimson", lw=2),
        )
        ax.set_title(label)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    plt.suptitle("Engagement & Content Metrics by Detected Era", y=1.02)
    plt.tight_layout()
    plt.savefig("era_engagement.png", bbox_inches="tight")
    plt.show()
else:
    print("No changepoints detected — try lowering CP_PENALTY in config")
