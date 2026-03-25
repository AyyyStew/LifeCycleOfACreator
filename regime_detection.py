# %% [markdown]
# # Regime Detection — Alternative Methods
# Five approaches to finding eras in HealthyGamerGG's content history.
#
# 1. Hidden Markov Model (HMM)
# 2. Bayesian Online Changepoint Detection (BOCPD)
# 3. Matrix Profile / STUMPY
# 4. Rolling Cluster Dominance
# 5. Temporal Graph Segmentation

# %%
# !pip install hmmlearn stumpy bayesian-changepoint-detection networkx -q

# %% [markdown]
# ## Imports & Setup

# %%
import sys
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sqlalchemy import text
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, '.')
from models import engine

plt.rcParams['figure.dpi'] = 120

# %% [markdown]
# ## Config

# %%
CHANNEL_QUERY   = 'healthygamergg'
CONTENT_TYPE    = 'long'
SHORTS_MAX_SECS = 180
N_REGIMES       = 5    # number of regimes/states for HMM and graph segmentation
N_TOPICS        = 8    # KMeans clusters for rolling dominance
PCA_DIMS        = 10   # dimensionality before feeding into models
ROLL_WIN        = 20   # rolling window (videos) for dominance + smoothing

# %% [markdown]
# ## Load Data

# %%
_dur = {
    'long':  f'AND v.duration_seconds > {SHORTS_MAX_SECS}',
    'short': f'AND v.duration_seconds <= {SHORTS_MAX_SECS}',
    'all':   '',
}[CONTENT_TYPE]

_sql = f"""
    SELECT
        v.video_id, v.title, v.published_at, v.duration_seconds,
        v.view_count, v.engagement_rate,
        v.transcript_words_per_minute,
        avg(tc.embedding) AS mean_embedding
    FROM videos v
    JOIN channels c ON v.channel_id = c.channel_id
    JOIN transcript_chunks tc ON v.video_id = tc.video_id
    WHERE LOWER(c.title) LIKE LOWER('%{CHANNEL_QUERY}%')
      AND tc.embedding IS NOT NULL
      {_dur}
    GROUP BY
        v.video_id, v.title, v.published_at, v.duration_seconds,
        v.view_count, v.engagement_rate, v.transcript_words_per_minute
    ORDER BY v.published_at
"""

with engine.connect() as conn:
    result = conn.execute(text(_sql))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
df['month'] = df['published_at'].dt.to_period('M')


def _parse_emb(v):
    if isinstance(v, np.ndarray):
        return v.astype(float)
    if isinstance(v, (list, tuple)):
        return np.array(v, dtype=float)
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(v))
    return np.array(nums, dtype=float)


df['embedding'] = df['mean_embedding'].apply(_parse_emb)
emb      = np.stack(df['embedding'].values)
emb_norm = normalize(emb)

# Shared PCA reduction used by most methods
pca     = PCA(n_components=PCA_DIMS, random_state=42)
emb_pca = pca.fit_transform(emb_norm)

print(f'Loaded {len(df)} videos  |  {df["published_at"].min().date()} -> {df["published_at"].max().date()}')
print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}')


# Shared helper — label topic clusters by TF-IDF on titles
def label_clusters(df, id_col, n_clusters):
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    labels = {}
    for t in range(n_clusters):
        titles = df.loc[df[id_col] == t, 'title'].tolist()
        if not titles:
            labels[t] = f'Regime {t}'
            continue
        try:
            mat   = tfidf.fit_transform(titles)
            terms = tfidf.get_feature_names_out()
            top3  = [terms[i] for i in mat.mean(axis=0).A1.argsort()[::-1][:3]]
            labels[t] = f'R{t}: {", ".join(top3)}'
        except Exception:
            labels[t] = f'Regime {t}'
    return labels


def regime_timeline(df, label_col, date_col='published_at'):
    """Print a compact timeline of regime transitions."""
    prev = None
    for _, row in df.iterrows():
        if row[label_col] != prev:
            print(f'  {row[date_col].date()}  ->  {row[label_col]}')
            prev = row[label_col]


def add_changepoint_vlines(ax, df, bkp_indices, date_col='published_at'):
    for idx in bkp_indices:
        ax.axvline(df[date_col].iloc[idx], color='red', lw=1.5, ls='--', alpha=0.7)


# %% [markdown]
# ## Method 1: Hidden Markov Model
# Fits a Gaussian HMM on PCA-reduced embeddings.
# Hidden states = regimes. The model learns which videos belong together
# and where the transitions are — no penalty to tune.

# %%
from hmmlearn import hmm

hmm_model = hmm.GaussianHMM(
    n_components=N_REGIMES,
    covariance_type='diag',
    n_iter=200,
    random_state=42,
)
hmm_model.fit(emb_pca)
df['hmm_state'] = hmm_model.predict(emb_pca)

# Stabilise labels: re-number states by order of first appearance
_state_order = list(dict.fromkeys(df['hmm_state'].tolist()))
_remap = {s: i for i, s in enumerate(_state_order)}
df['hmm_state'] = df['hmm_state'].map(_remap)

hmm_labels = label_clusters(df, 'hmm_state', N_REGIMES)
df['hmm_label'] = df['hmm_state'].map(hmm_labels)

print('=== HMM Regime Timeline ===')
regime_timeline(df, 'hmm_label')

# Posterior state probabilities (soft assignments)
posteriors = hmm_model.predict_proba(emb_pca)
# remap columns to match stabilised order
posteriors = posteriors[:, _state_order]

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

ax = axes[0]
palette = plt.cm.tab10(np.linspace(0, 1, N_REGIMES))
for i, label in hmm_labels.items():
    ax.fill_between(df['published_at'], posteriors[:, i], alpha=0.7,
                    color=palette[i], label=label)
ax.set_title('HMM — Posterior Regime Probabilities Over Time')
ax.set_ylabel('P(regime)')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax = axes[1]
colors = [palette[s] for s in df['hmm_state']]
ax.scatter(df['published_at'], df['hmm_state'], c=colors, s=15, alpha=0.7)
# mark transitions
transitions = df['hmm_state'].ne(df['hmm_state'].shift()).to_numpy().copy()
transitions[0] = False
for idx in np.where(transitions)[0]:
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1, ls='--', alpha=0.6)
ax.set_title('HMM — Viterbi State Sequence  (red = transition)')
ax.set_ylabel('Regime')
ax.set_yticks(range(N_REGIMES))
ax.set_yticklabels([hmm_labels[i] for i in range(N_REGIMES)], fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('regime_hmm.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Method 2: Bayesian Online Changepoint Detection (BOCPD)
# Probabilistic — gives P(changepoint at t) at every timestep.
# No hard threshold needed; you can see "soft" pivots the channel drifts through.
# We run it on PC1 (the dominant content axis).

# %%
# Minimal BOCPD implementation (Gaussian with unknown mean, known variance)
# Based on Adams & MacKay 2007. No extra library required.

def bocpd(data, hazard=1/200, obs_noise=1.0):
    """
    Returns run-length posterior matrix R where R[t, l] = P(run length = l at time t).
    Changepoint probability at t = R[t, 0].
    """
    T    = len(data)
    R    = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Sufficient statistics for each run length
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, obs_noise

    muT    = np.array([mu0])
    kappaT = np.array([kappa0])
    alphaT = np.array([alpha0])
    betaT  = np.array([beta0])

    cp_prob = np.zeros(T)

    for t in range(T):
        x = data[t]

        # Predictive probability under Student-t
        nu     = 2 * alphaT
        sig2   = betaT * (kappaT + 1) / (alphaT * kappaT)
        # log p(x | run length) via Student-t
        log_pred = (
            np.log(nu / (nu + ((x - muT) ** 2) / sig2)) * ((nu + 1) / 2)
            - 0.5 * np.log(np.pi * nu * sig2)
            + np.log(1 + ((x - muT) ** 2) / (nu * sig2)) * (-(nu + 1) / 2)
        )
        pred = np.exp(log_pred - log_pred.max())  # numerical stability

        # Growth (no changepoint)
        R[t + 1, 1:t + 2] = R[t, :t + 1] * pred * (1 - hazard)
        # Reset (changepoint)
        R[t + 1, 0]        = np.sum(R[t, :t + 1] * pred) * hazard

        # Normalise
        R[t + 1, :] /= R[t + 1, :].sum() + 1e-300

        cp_prob[t] = R[t + 1, 0]

        # Update sufficient statistics
        muT    = np.append(mu0,    (kappaT * muT + x) / (kappaT + 1))
        kappaT = np.append(kappa0, kappaT + 1)
        alphaT = np.append(alpha0, alphaT + 0.5)
        betaT  = np.append(beta0,  betaT + (kappaT[:-1] * (x - muT[:-1]) ** 2) / (2 * (kappaT[:-1] + 1)))

    return R, cp_prob


# Run BOCPD on the top 3 PCs independently, then combine
all_cp = np.zeros(len(df))
for dim in range(3):
    signal = (emb_pca[:, dim] - emb_pca[:, dim].mean()) / (emb_pca[:, dim].std() + 1e-10)
    _, cp  = bocpd(signal, hazard=1/150)
    all_cp += cp

all_cp /= 3  # average across dims

# Smooth for display
cp_smooth = pd.Series(all_cp).rolling(5, center=True, min_periods=1).mean().values

# Find peaks above threshold
from scipy.signal import find_peaks
cp_peaks, _ = find_peaks(cp_smooth, height=np.percentile(cp_smooth, 90), distance=20)

print(f'=== BOCPD: {len(cp_peaks)} high-probability changepoints ===')
for idx in cp_peaks:
    print(f'  {df["published_at"].iloc[idx].date()}  —  "{df["title"].iloc[idx]}"')

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax = axes[0]
ax.plot(df['published_at'], emb_pca[:, 0], lw=0.7, alpha=0.7, label='PC1')
ax.plot(df['published_at'], emb_pca[:, 1], lw=0.7, alpha=0.7, label='PC2')
for idx in cp_peaks:
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1.5, ls='--', alpha=0.7)
ax.set_title('BOCPD — Embedding Signal with Detected Changepoints')
ax.set_ylabel('PCA value')
ax.legend()

ax = axes[1]
ax.plot(df['published_at'], cp_smooth, color='darkorange', lw=1.2)
ax.fill_between(df['published_at'], cp_smooth, alpha=0.3, color='orange')
for idx in cp_peaks:
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1.5, ls='--', alpha=0.7)
threshold = np.percentile(cp_smooth, 90)
ax.axhline(threshold, color='grey', ls=':', lw=1, label=f'90th pct threshold')
ax.set_title('BOCPD — Changepoint Probability Over Time')
ax.set_ylabel('P(changepoint)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('regime_bocpd.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Method 3: Matrix Profile (STUMPY)
# Finds the most anomalous subsequences (discords) in the embedding time series.
# Discords = videos maximally unlike any other window = regime pivots.
# Also finds motifs = windows that recur = eras the channel returns to.

# %%
import stumpy

# Matrix profile on PC1 (the dominant content axis)
# window = how many videos define a "regime window"
MP_WINDOW = max(10, len(df) // 20)

mp = stumpy.stump(emb_pca[:, 0].astype(np.float64), m=MP_WINDOW)
mp_distances = mp[:, 0].astype(float)   # nearest-neighbor distances

# Discords: high distance = most anomalous window = regime break candidates
discord_indices = np.argsort(mp_distances)[::-1]

# Filter to keep well-separated discords
min_separation = MP_WINDOW
discords = []
for idx in discord_indices:
    if all(abs(idx - d) >= min_separation for d in discords):
        discords.append(idx)
    if len(discords) >= N_REGIMES:
        break
discords = sorted(discords)

# Motifs: low distance = most recurring pattern
motif_idx = int(np.argmin(mp_distances))

print(f'=== Matrix Profile: top {len(discords)} discords (regime breaks) ===')
print(f'Window size: {MP_WINDOW} videos')
for idx in discords:
    print(f'  [{idx}] {df["published_at"].iloc[idx].date()}  —  "{df["title"].iloc[idx]}"')
print(f'\nTop motif (most recurring pattern) at index {motif_idx}:')
print(f'  {df["published_at"].iloc[motif_idx].date()}  —  "{df["title"].iloc[motif_idx]}"')

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax = axes[0]
ax.plot(df['published_at'], emb_pca[:, 0], lw=0.8, alpha=0.7, color='steelblue')
for idx in discords:
    ax.axvspan(df['published_at'].iloc[idx],
               df['published_at'].iloc[min(idx + MP_WINDOW, len(df) - 1)],
               alpha=0.25, color='red', label='_nolegend_')
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1.5, ls='--', alpha=0.8)
ax.axvspan(df['published_at'].iloc[motif_idx],
           df['published_at'].iloc[min(motif_idx + MP_WINDOW, len(df) - 1)],
           alpha=0.25, color='green')
ax.set_title(f'Matrix Profile — PC1 Signal  (red = discord/regime break, green = top motif)')
ax.set_ylabel('PC1')

ax = axes[1]
ax.plot(df['published_at'].iloc[:len(mp_distances)], mp_distances,
        color='darkorange', lw=0.9)
for idx in discords:
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1.5, ls='--', alpha=0.8)
ax.set_title('Matrix Profile Distance  (peaks = most anomalous windows)')
ax.set_ylabel('Nearest-neighbour distance')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('regime_matrixprofile.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Method 4: Rolling Cluster Dominance
# Uses KMeans topic clusters from era_analysis.py logic.
# For each rolling window of videos, the dominant topic = the regime.
# When the dominant topic switches and stays switched = regime change.
# Most interpretable method — regimes have human-readable topic labels.

# %%
km = KMeans(n_clusters=N_TOPICS, random_state=42, n_init=10)
df['topic_id'] = km.fit_predict(emb_norm)

topic_labels = label_clusters(df, 'topic_id', N_TOPICS)
df['topic_label'] = df['topic_id'].map(topic_labels)

# Rolling dominant topic
rolling_dominant = (
    df['topic_id']
    .rolling(ROLL_WIN, center=True, min_periods=1)
    .apply(lambda x: int(pd.Series(x.astype(int)).mode().iloc[0]))
    .astype(int)
)
df['dominant_topic']       = rolling_dominant
df['dominant_topic_label'] = df['dominant_topic'].map(topic_labels)

# Smooth: only register a regime switch if the new topic holds for > ROLL_WIN//2 consecutive videos
def smooth_regimes(labels, min_run=10):
    out = labels.copy()
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_run and i > 0:
            out[i:j] = [out[i - 1]] * run_len
        i = j
    return out


df['regime_smooth'] = smooth_regimes(df['dominant_topic_label'].tolist(), min_run=ROLL_WIN // 2)

print('=== Rolling Cluster Dominance — Regime Timeline ===')
regime_timeline(df, 'regime_smooth')

# Stacked rolling topic proportions
topic_one_hot = pd.get_dummies(df['topic_label'])
rolling_props = topic_one_hot.rolling(ROLL_WIN, center=True, min_periods=1).mean()
rolling_props.index = df['published_at']

fig, axes = plt.subplots(2, 1, figsize=(16, 9))

ax = axes[0]
palette = plt.cm.tab10(np.linspace(0, 1, N_TOPICS))
rolling_props.plot(kind='area', stacked=True, ax=ax, colormap='tab10', alpha=0.82, legend=False)

# Mark regime transitions from smoothed signal
transitions = pd.Series(df['regime_smooth'].values).ne(pd.Series(df['regime_smooth'].values).shift())
transitions.iloc[0] = False
for idx in transitions[transitions].index:
    ax.axvline(df['published_at'].iloc[idx], color='black', lw=1, alpha=0.5)

ax.set_title(f'Rolling Topic Dominance  (window={ROLL_WIN} videos, black = regime switch)')
ax.set_ylabel('Proportion of window')
ax.legend(topic_labels.values(), bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax = axes[1]
# Regime band plot
prev_label = None
prev_date  = df['published_at'].iloc[0]
color_map  = {label: palette[i % len(palette)] for i, label in enumerate(set(df['regime_smooth']))}
for _, row in df.iterrows():
    if row['regime_smooth'] != prev_label and prev_label is not None:
        ax.axvspan(prev_date, row['published_at'],
                   alpha=0.5, color=color_map[prev_label], label=prev_label)
        prev_date = row['published_at']
    prev_label = row['regime_smooth']
ax.axvspan(prev_date, df['published_at'].iloc[-1],
           alpha=0.5, color=color_map[prev_label], label=prev_label)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[l]) for l in color_map]
ax.legend(handles, list(color_map.keys()), bbox_to_anchor=(1.01, 1),
          loc='upper left', fontsize=7)
ax.set_title('Smoothed Regime Bands')
ax.set_yticks([])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('regime_rolling_dominance.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Method 5: Temporal Graph Segmentation
# Build a similarity graph where videos are nodes, edges weighted by cosine similarity.
# Apply spectral clustering with a temporal smoothness prior — videos that are
# both semantically similar AND close in time cluster together.
# Produces contiguous regime blocks.

# %%
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans as _KMeans

n = len(df)

# --- Semantic similarity graph (k-nearest neighbours) ---
K_NEIGHBOURS = 15
similarity    = emb_norm @ emb_norm.T   # (n, n) cosine similarity

# Keep only top-K neighbours per row
knn_mask = np.zeros_like(similarity, dtype=bool)
for i in range(n):
    top_k = np.argpartition(similarity[i], -(K_NEIGHBOURS + 1))[-(K_NEIGHBOURS + 1):]
    top_k = top_k[top_k != i][:K_NEIGHBOURS]
    knn_mask[i, top_k] = True

W_semantic = np.where(knn_mask | knn_mask.T, similarity, 0.0)
W_semantic = np.clip(W_semantic, 0, None)

# --- Temporal proximity graph ---
TEMPORAL_SIGMA = 30  # videos; controls how strongly adjacent videos are pulled together
idx_diff       = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
W_temporal     = np.exp(-idx_diff ** 2 / (2 * TEMPORAL_SIGMA ** 2))

# --- Combined graph ---
ALPHA     = 0.6   # weight of semantic vs temporal (0=pure temporal, 1=pure semantic)
W         = ALPHA * W_semantic + (1 - ALPHA) * W_temporal
np.fill_diagonal(W, 0)

# Normalised spectral clustering
D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))
L_sym      = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

# Eigenvectors corresponding to smallest N_REGIMES eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
spectral_emb = eigenvectors[:, :N_REGIMES]
spectral_emb = normalize(spectral_emb)

km_spectral      = _KMeans(n_clusters=N_REGIMES, random_state=42, n_init=20)
df['graph_state'] = km_spectral.fit_predict(spectral_emb)

# Relabel by temporal order of first appearance
_order = list(dict.fromkeys(df['graph_state'].tolist()))
df['graph_state'] = df['graph_state'].map({s: i for i, s in enumerate(_order)})

graph_labels    = label_clusters(df, 'graph_state', N_REGIMES)
df['graph_label'] = df['graph_state'].map(graph_labels)

print('=== Temporal Graph Segmentation — Regime Timeline ===')
regime_timeline(df, 'graph_label')

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

palette = plt.cm.tab10(np.linspace(0, 1, N_REGIMES))

ax = axes[0]
colors = [palette[s] for s in df['graph_state']]
ax.scatter(df['published_at'], df['graph_state'], c=colors, s=15, alpha=0.7)
transitions_g = pd.Series(df['graph_state'].values).ne(pd.Series(df['graph_state'].values).shift()).copy()
transitions_g.iloc[0] = False
for idx in transitions_g[transitions_g].index:
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1, ls='--', alpha=0.5)
ax.set_title(f'Temporal Graph Segmentation — Regime per Video  (alpha={ALPHA})')
ax.set_ylabel('Regime')
ax.set_yticks(range(N_REGIMES))
ax.set_yticklabels([graph_labels[i] for i in range(N_REGIMES)], fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax = axes[1]
# Eigenvalue spectrum — elbow shows natural number of regimes
ax.plot(range(1, min(20, n)), eigenvalues[1:20], marker='o', ms=5)
ax.axvline(N_REGIMES, color='red', ls='--', label=f'N_REGIMES = {N_REGIMES}')
ax.set_title('Spectral Gap — use elbow to validate N_REGIMES')
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Eigenvalue')
ax.legend()

plt.tight_layout()
plt.savefig('regime_graph.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cross-Method Comparison
# Stack all five methods to see where they agree — consensus = high-confidence regime boundary.

# %%
fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)
titles = [
    'HMM States',
    'BOCPD Changepoint Probability',
    'Matrix Profile Distance',
    'Rolling Cluster Dominance',
    'Graph Segmentation States',
]

palette5 = plt.cm.tab10(np.linspace(0, 1, max(N_REGIMES, N_TOPICS)))

# 1. HMM
ax = axes[0]
ax.scatter(df['published_at'], df['hmm_state'],
           c=[palette5[s] for s in df['hmm_state']], s=10, alpha=0.7)
ax.set_ylabel('State', fontsize=8)

# 2. BOCPD
ax = axes[1]
ax.fill_between(df['published_at'], cp_smooth, alpha=0.5, color='darkorange')
ax.plot(df['published_at'], cp_smooth, lw=0.8, color='darkorange')
for idx in cp_peaks:
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1, ls='--', alpha=0.6)
ax.set_ylabel('P(cp)', fontsize=8)

# 3. Matrix profile
ax = axes[2]
ax.plot(df['published_at'].iloc[:len(mp_distances)], mp_distances,
        lw=0.8, color='steelblue')
for idx in discords:
    ax.axvline(df['published_at'].iloc[idx], color='red', lw=1, ls='--', alpha=0.6)
ax.set_ylabel('MP dist', fontsize=8)

# 4. Rolling dominance
ax = axes[3]
rolling_props.plot(kind='area', stacked=True, ax=ax, colormap='tab10', alpha=0.82, legend=False)
ax.set_ylabel('Topic mix', fontsize=8)

# 5. Graph
ax = axes[4]
ax.scatter(df['published_at'], df['graph_state'],
           c=[palette5[s] for s in df['graph_state']], s=10, alpha=0.7)
ax.set_ylabel('State', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=9, loc='left', pad=2)

plt.suptitle('Cross-Method Regime Comparison — HealthyGamerGG', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('regime_comparison.png', bbox_inches='tight')
plt.show()
