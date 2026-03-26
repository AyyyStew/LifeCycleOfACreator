# %% [markdown]
# # What Changed Between Eras? — HealthyGamerGG
#
# Seven years of content from a single channel. Did it actually change — or just feel
# like it did?
#
# This notebook answers that question empirically. Starting from nothing but transcript
# embeddings and upload dates, it automatically segments the channel's history into
# content eras, then characterises what made each one distinct.
#
# ---
#
# ## How It Works
#
# ### Step 1 — Era Detection via Matrix Profile + CUSUM
#
# Each video is represented as a high-dimensional embedding — a vector that captures
# the semantic content of its transcript. We compress these into a lower-dimensional
# space with PCA, then use the **matrix profile** to measure, for every window of
# consecutive videos, how anomalous that window looks compared to the rest of the
# channel's history.
#
# This anomaly signal is smoothed and fed into three independent change-detection
# algorithms (EMA crossover, Bollinger Band breakout, and CUSUM). Where all three
# agree a shift occurred, we have high confidence. **CUSUM** drives the final
# segmentation — it accumulates evidence for a sustained shift before firing, which
# makes it robust to noise.
#
# ### Step 2 — What Changed? Log-Odds Ratio
#
# For each era, we find the nouns that are statistically *over-represented* compared
# to all other eras combined. This is explicitly contrastive — a word only ranks high
# if it appears *unusually often* in that era relative to the channel's overall history.
#
# Transcripts are pre-processed with **spaCy**: we keep only nouns (the content-bearing
# words), strip named entities (people and organisations that appear everywhere and
# add no signal), and lemmatise so "relationships" and "relationship" count as one token.
#
# ### Step 3 — Where Did the Channel Move? Embedding Dimension Attribution
#
# The channel's content lives in a high-dimensional embedding space. PCA finds the
# independent axes of variation — the directions along which videos differ most from
# each other. By tracking how each era's centroid moved through this space, we can
# identify which axes captured the real shifts.
#
# Three visualisations make these abstract axes interpretable:
# - **Radar chart** — each era as a polygon; spokes where polygons diverge = axes
#   that distinguish eras
# - **Annotated 2D scatter** — videos on the two most-changed axes, extreme videos
#   labelled so you can read off what each axis means
# - **Word clouds at each pole** — vocabulary of the high vs low end of each axis,
#   showing the semantic content that defines each direction

# %%
# !pip install spacy wordcloud -q
# !python -m spacy download en_core_web_sm -q

# %% [markdown]
# ## Setup
# *Run this cell first. Install deps with the commented pip commands if needed.*

# %%
import sys
import re
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter

import spacy
from wordcloud import WordCloud

from sqlalchemy import text
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

sys.path.insert(0, '.')
from models import engine

plt.rcParams['figure.dpi'] = 120

# %% [markdown]
# ## Config
#
# All tuneable parameters live here. The channel and content type are the first
# things to change if you want to run this on a different creator.
#
# **Channel / content:**
# - `CHANNEL_QUERY` — partial match against the channel title in the database.
# - `CONTENT_TYPE` — `'long'` (videos > 3 min), `'short'` (≤ 3 min), or `'all'`.
#
# **Matrix profile:**
# - `MP_WINDOW` — sliding window size in videos. At ~2.7 uploads/week, 25 ≈ 9 weeks.
#   Larger = smoother signal but can blur short eras. Try 15–35.
#
# **Signal smoothing:**
# - `SMOOTH_WINDOW` — rolling mean window applied before boundary detection.
#   This is the primary tuning lever: wider = fewer, broader eras. At 2.7 vids/week,
#   35 ≈ 3 months. Try 20–50.
# - `MIN_DISCORD_SEP` — minimum gap (videos) between any two boundaries.
#   Prevents two detections firing on the same event. 40 ≈ 3.5 months.
#
# **Boundary detection (all three run for the convergence plot; CUSUM drives the result):**
# - `EMA_FAST` / `EMA_SLOW` — fast and slow EMA spans. Wider ratio = less sensitive.
# - `BB_WINDOW` / `BB_N_STD` / `BB_CONFIRM` — Bollinger window, band width, and
#   the number of videos the signal must stay outside the band before a boundary fires.
# - `CUSUM_THRESHOLD` — how much evidence must accumulate before CUSUM fires.
#   The main CUSUM dial. Higher = fewer boundaries. Try 4–10.
# - `CUSUM_DRIFT` — per-step tolerance subtracted from the accumulator.
#   Higher = only large, sustained shifts trigger. Try 0.3–0.8.
#
# **NLP:**
# - `NLP_CHAR_LIMIT` — transcript characters fed to spaCy per video. 8000 is a good
#   balance between accuracy and runtime.
# - `LOG_ODDS_MIN_COUNT` — minimum era appearances for a noun to enter the log-odds
#   ranking. Filters out one-off words that score high by chance.
#
# **Visualisation:**
# - `PCA_DIMS` — dimensions for the radar chart and scatter plot.
#   Does not affect era detection.

# %%
CHANNEL_QUERY    = 'healthygamergg'
CONTENT_TYPE     = 'long'
SHORTS_MAX_SECS  = 180
PCA_DIMS         = 10   # dims for scatter/radar — does not affect era detection

MP_WINDOW        = 25   # matrix profile window (~9 weeks at 2.7 vids/week)
SMOOTH_WINDOW    = 35   # rolling mean window (~3 months) — primary era-count dial
MIN_DISCORD_SEP  = 40   # min videos between boundaries (~3.5 months)

# Boundary detection — all three run for the convergence plot
EMA_FAST         = 15
EMA_SLOW         = 75
EMA_MIN_SEP      = MIN_DISCORD_SEP

BB_WINDOW        = 30
BB_N_STD         = 0.75
BB_CONFIRM       = 20
BB_MIN_SEP       = MIN_DISCORD_SEP

CUSUM_THRESHOLD  = 6.0  # main dial — higher = fewer eras
CUSUM_DRIFT      = 0.5
CUSUM_MIN_SEP    = MIN_DISCORD_SEP

NLP_CHAR_LIMIT     = 8000
LOG_ODDS_MIN_COUNT = 5

# %% [markdown]
# ## Load Data
#
# Two queries from the database:
#
# 1. **Embeddings** — one row per video, where the embedding is the mean of all
#    transcript chunk embeddings. This is the input to the matrix profile and PCA.
# 2. **Transcript text** — all chunks concatenated in order per video. This is the
#    input to the spaCy NLP pipeline for log-odds analysis.

# %%
_dur = {
    'long':  f'AND v.duration_seconds > {SHORTS_MAX_SECS}',
    'short': f'AND v.duration_seconds <= {SHORTS_MAX_SECS}',
    'all':   '',
}[CONTENT_TYPE]

_sql_videos = f"""
    SELECT
        v.video_id, v.title, v.published_at, v.duration_seconds,
        v.view_count, v.engagement_rate, v.transcript_words_per_minute,
        avg(tc.embedding) AS mean_embedding
    FROM videos v
    JOIN channels c ON v.channel_id = c.channel_id
    JOIN transcript_chunks tc ON v.video_id = tc.video_id
    WHERE LOWER(c.title) LIKE LOWER('%{CHANNEL_QUERY}%')
      AND tc.embedding IS NOT NULL
      {_dur}
    GROUP BY v.video_id, v.title, v.published_at, v.duration_seconds,
             v.view_count, v.engagement_rate, v.transcript_words_per_minute
    ORDER BY v.published_at
"""

_sql_text = f"""
    SELECT v.video_id,
           string_agg(tc.text, ' ' ORDER BY tc.chunk_index) AS full_text
    FROM videos v
    JOIN channels c ON v.channel_id = c.channel_id
    JOIN transcript_chunks tc ON v.video_id = tc.video_id
    WHERE LOWER(c.title) LIKE LOWER('%{CHANNEL_QUERY}%')
      {_dur}
    GROUP BY v.video_id
"""

with engine.connect() as conn:
    r = conn.execute(text(_sql_videos))
    df = pd.DataFrame(r.fetchall(), columns=r.keys())
    r2 = conn.execute(text(_sql_text))
    df_text = pd.DataFrame(r2.fetchall(), columns=r2.keys())

df = df.merge(df_text, on='video_id', how='left')
df['published_at'] = pd.to_datetime(df['published_at'], utc=True)


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

pca     = PCA(n_components=PCA_DIMS, random_state=42)
emb_pca = pca.fit_transform(emb_norm)

print(f'Loaded {len(df)} videos')
print(f'PCA explained variance ({PCA_DIMS} dims): {pca.explained_variance_ratio_.sum():.1%}')

# %% [markdown]
# ## Era Detection
#
# ### Part 1 — Matrix Profile
#
# The **matrix profile** measures, for every sliding window of `MP_WINDOW` consecutive
# videos, how similar that window is to the most similar other window in the channel's
# history. A high distance means this stretch of content looks unlike anything else —
# that's a signal of a regime shift.
#
# We compute the profile separately for each PCA dimension and combine them into a
# single anomaly score, weighting each dimension by its explained variance. The result
# is a time series where spikes correspond to windows of anomalous content.
#
# The raw signal is then smoothed with a rolling mean (`SMOOTH_WINDOW` videos) to
# suppress high-frequency noise. A genuine era transition shows up as a *sustained*
# elevation, not a single spike — smoothing makes those sustained periods stand out.

# %%
import stumpy

print('Computing matrix profiles...')
weights     = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
n_profile   = len(df) - MP_WINDOW + 1
mp_combined = np.zeros(n_profile)

for dim in range(PCA_DIMS):
    mp_dim       = stumpy.stump(emb_pca[:, dim].astype(np.float64), m=MP_WINDOW)[:, 0].astype(float)
    mp_combined += weights[dim] * mp_dim

mp_z     = (mp_combined - mp_combined.mean()) / (mp_combined.std() + 1e-10)
mp_dates = df['published_at'].values[:n_profile]

# Rolling smooth
mp_smooth = pd.Series(mp_z).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().values
mp_smooth = (mp_smooth - mp_smooth.mean()) / (mp_smooth.std() + 1e-10)

fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
axes[0].plot(mp_dates, mp_z, lw=0.7, color='steelblue', alpha=0.6, label='Raw MP (z-score)')
axes[0].set_title('Matrix Profile Anomaly Signal — Raw')
axes[0].set_ylabel('z-score')
axes[0].legend(fontsize=8)

axes[1].plot(mp_dates, mp_smooth, lw=1.3, color='seagreen', label=f'Smoothed (window={SMOOTH_WINDOW})')
axes[1].axhline(0, color='k', lw=0.5, alpha=0.4)
axes[1].set_title(f'Matrix Profile Anomaly Signal — Smoothed  (window={SMOOTH_WINDOW} videos ≈ 3 months)')
axes[1].set_ylabel('z-score')
axes[1].legend(fontsize=8)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Part 2 — Boundary Detection: Three-Method Convergence
#
# A single detection algorithm can produce false positives. Instead, we run three
# independent methods on the smoothed signal and compare. Where all three agree a
# shift occurred, we have strong evidence it's real.
#
# - **EMA Crossover** — two exponential moving averages (fast and slow) run in
#   parallel. A boundary fires when the short-term trend crosses the long-term
#   baseline and stays crossed. Because the slow EMA requires sustained movement to
#   shift, individual spikes don't trigger it.
#
# - **Bollinger Band Breakout** — computes a rolling mean ± N standard deviations.
#   A boundary fires when the signal escapes this band *and stays outside* for a
#   confirmation window. Requires both magnitude and duration.
#
# - **CUSUM (Cumulative Sum)** — accumulates evidence for an upward or downward
#   shift separately, and only fires when it exceeds a threshold, then resets.
#   Originally developed for industrial quality control, it's designed to answer
#   exactly our question: *has the mean of this process sustainably shifted?*
#
# The convergence panel below shows all three. **CUSUM drives the final segmentation.**

# %%
# --- EMA Crossover ---
_s       = pd.Series(mp_smooth)
ema_fast = _s.ewm(span=EMA_FAST, adjust=False).mean().values
ema_slow = _s.ewm(span=EMA_SLOW, adjust=False).mean().values
diff     = ema_fast - ema_slow
raw_cross = np.where(np.diff(np.sign(diff)) != 0)[0]
ema_boundaries = []
for idx in raw_cross:
    if not ema_boundaries or (idx - ema_boundaries[-1]) >= EMA_MIN_SEP:
        ema_boundaries.append(int(idx))

# --- Bollinger Band Breakout ---
bb_mean  = _s.rolling(BB_WINDOW, center=True, min_periods=1).mean().values
bb_std   = _s.rolling(BB_WINDOW, center=True, min_periods=1).std().fillna(0).values
outside  = (mp_smooth > bb_mean + BB_N_STD * bb_std) | (mp_smooth < bb_mean - BB_N_STD * bb_std)
bb_boundaries = []
i = 0
while i < len(outside):
    if outside[i]:
        end = min(i + BB_CONFIRM, len(outside))
        if outside[i:end].sum() >= BB_CONFIRM * 0.7:
            if not bb_boundaries or (i - bb_boundaries[-1]) >= BB_MIN_SEP:
                bb_boundaries.append(i)
            i += BB_MIN_SEP
            continue
    i += 1

# --- CUSUM ---
S_pos, S_neg   = 0.0, 0.0
cusum_boundaries = []
last_detection = -CUSUM_MIN_SEP
for i, x in enumerate(mp_smooth):
    S_pos = max(0, S_pos + x - CUSUM_DRIFT)
    S_neg = max(0, S_neg - x - CUSUM_DRIFT)
    if (S_pos > CUSUM_THRESHOLD or S_neg > CUSUM_THRESHOLD):
        if (i - last_detection) >= CUSUM_MIN_SEP:
            cusum_boundaries.append(i)
            last_detection = i
        S_pos, S_neg = 0.0, 0.0

print(f'EMA boundaries     : {len(ema_boundaries)}')
print(f'Bollinger boundaries: {len(bb_boundaries)}')
print(f'CUSUM boundaries   : {len(cusum_boundaries)}  ← used for era segmentation')

# %% [markdown]
# #### Convergence Panel
#
# Each row shows one detection method on the same smoothed signal. Where dashed lines
# stack up vertically across all three rows, multiple independent algorithms agree — those
# are the highest-confidence era boundaries.

# %%
all_methods = {
    f'EMA ({EMA_FAST}/{EMA_SLOW})  —  {len(ema_boundaries)} boundaries':
        (ema_boundaries,   'darkorange'),
    f'Bollinger (±{BB_N_STD}σ)  —  {len(bb_boundaries)} boundaries':
        (bb_boundaries,    'steelblue'),
    f'CUSUM (threshold={CUSUM_THRESHOLD})  —  {len(cusum_boundaries)} boundaries  ★ used':
        (cusum_boundaries, 'seagreen'),
}

fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
for ax, (label, (boundaries, color)) in zip(axes, all_methods.items()):
    ax.plot(mp_dates, mp_smooth, lw=1.0, color='grey', alpha=0.55)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    for b in boundaries:
        ax.axvline(mp_dates[min(b, len(mp_dates)-1)], color=color, lw=2,
                   ls='--', alpha=0.85)
    ax.set_title(label)
    ax.set_ylabel('z-score')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.suptitle('Era Boundary Method Comparison — Convergence = Confidence', fontsize=12)
plt.tight_layout()
plt.savefig('boundary_comparison.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Part 3 — Era Segmentation
#
# CUSUM boundaries become the era assignments used in all downstream analysis.
# Each video gets labelled with the era it falls in, and a summary is printed below.

# %%
boundary_indices = sorted([min(b, len(df) - 1) for b in cusum_boundaries])
era_slices = list(zip([0] + boundary_indices, boundary_indices + [len(df)]))

df['era'] = -1
for era_num, (s, e) in enumerate(era_slices):
    df.iloc[s:e, df.columns.get_loc('era')] = era_num

n_eras  = len(era_slices)
palette = plt.cm.tab10(np.linspace(0, 1, n_eras))

print(f'{n_eras} eras detected via CUSUM '
      f'(threshold={CUSUM_THRESHOLD}, drift={CUSUM_DRIFT}, min_sep={CUSUM_MIN_SEP}):\n')
for i, (s, e) in enumerate(era_slices):
    dominant = 'high-anomaly' if mp_smooth[min(s, len(mp_smooth)-1):min(e, len(mp_smooth))].mean() > 0 \
               else 'consistent'
    print(f'  Era {i+1}: {df.iloc[s]["published_at"].date()} '
          f'-> {df.iloc[e-1]["published_at"].date()}  ({e-s} videos)  [{dominant}]')

# %%
fig, ax = plt.subplots(figsize=(16, 3.5))
ax.plot(mp_dates, mp_smooth, lw=1.3, color='seagreen')
ax.axhline(0, color='k', lw=0.5, alpha=0.4)
for era_num, (s, e) in enumerate(era_slices):
    span_start = mp_dates[min(s, len(mp_dates) - 1)]
    span_end   = mp_dates[min(e - 1, len(mp_dates) - 1)]
    ax.axvspan(span_start, span_end, alpha=0.10, color=palette[era_num])
for rank, b in enumerate(cusum_boundaries, 1):
    ax.axvline(mp_dates[min(b, len(mp_dates)-1)], color='crimson', lw=2, ls='--', alpha=0.85)
    ax.text(mp_dates[min(b, len(mp_dates)-1)], mp_smooth.max() * 0.88, f'#{rank}',
            color='crimson', fontsize=8, ha='center')
ax.set_title(f'CUSUM Era Boundaries  ({n_eras} eras)')
ax.set_ylabel('z-score')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('hmm_states.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Method 1 — Log-Odds Ratio
#
# ### Text Preprocessing: NER + Noun Extraction
#
# Raw spoken transcripts are noisy. To compare vocabulary across eras meaningfully,
# we first need to strip out words that carry no topical signal.
#
# We use **spaCy** to do three things in one pass over each transcript:
#
# 1. **Part-of-speech tagging** — we keep only **nouns**. Verbs ("talk", "feel",
#    "make") and filler words appear constantly regardless of topic and add noise.
#    Nouns are the content-bearing words.
#
# 2. **Named entity recognition (NER)** — we strip anything tagged as a **PERSON**
#    or **ORG**. Names like "Dr K", "Twitch", and "Harvard" appear across all eras
#    and tell us nothing about *what topics* changed.
#
# 3. **Lemmatisation** — words are reduced to their base form so "relationships",
#    "relationship", and "relating" all count as the same token.
#
# Each transcript is capped at `NLP_CHAR_LIMIT` characters and processed in batches.

# %%
print('Loading spaCy model...')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'senter'])
nlp.max_length = NLP_CHAR_LIMIT + 100

EXTRA_STOPWORDS = {
    'thing', 'stuff', 'lot', 'bit', 'way', 'time', 'day', 'year',
    'video', 'channel', 'stream', 'chat', 'comment', 'question', 'answer',
    'guy', 'man', 'woman', 'people', 'person', 'everyone', 'someone',
    'kind', 'sort', 'type', 'part', 'point', 'place', 'end', 'world',
    'sense', 'fact', 'case', 'reason', 'problem', 'issue', 'example',
    'number', 'level', 'moment', 'mind', 'word', 'life',
}


def extract_nouns(text, char_limit=NLP_CHAR_LIMIT):
    """
    Returns a list of lemmatised noun tokens with person/org entities removed.
    """
    if not isinstance(text, str):
        return []
    text = text[:char_limit]
    doc  = nlp(text)

    # Collect spans that are PERSON or ORG entities — we'll skip these
    ent_spans = set()
    for ent in doc.ents:
        if ent.label_ in ('PERSON', 'ORG'):
            for tok in ent:
                ent_spans.add(tok.i)

    nouns = []
    for tok in doc:
        if tok.i in ent_spans:
            continue
        if tok.pos_ != 'NOUN':
            continue
        lemma = tok.lemma_.lower()
        if len(lemma) < 3:
            continue
        if lemma in EXTRA_STOPWORDS:
            continue
        nouns.append(lemma)

    return nouns


# Process all transcripts in batches
texts  = df['full_text'].fillna('').tolist()
n_docs = len(texts)
BATCH  = 50

print(f'Extracting nouns from {n_docs} transcripts (batch size {BATCH})...')
df['nouns'] = None
noun_lists  = []

for i in range(0, n_docs, BATCH):
    batch_texts = texts[i:i + BATCH]
    batch_nouns = [extract_nouns(t) for t in batch_texts]
    noun_lists.extend(batch_nouns)
    print(f'  {min(i + BATCH, n_docs)}/{n_docs}')

df['nouns'] = noun_lists
print('Done.')

# %% [markdown]
# ### Log-Odds Ratio
#
# With clean noun lists per video, we can now measure how distinctive each word is
# to each era. The **log-odds ratio** (Monroe et al. 2008) asks:
#
# > How much more likely is this word in *this era* vs. all other eras combined?
#
# Concretely: `score = log( P(word | this era) / P(word | all other eras) )`
#
# - **High positive** → word appears far more often in this era than the rest of the
#   channel's history — this is what the era is "about"
# - **Near zero** → word is equally common across all eras — no signal
# - **Negative** → word is *less* common in this era than usual
#
# We apply Laplace smoothing so rare words don't get artificially extreme scores.
#
# The key advantage over TF-IDF or plain frequency: scores are explicitly *contrastive*
# against the rest of the channel's history, not just against other words in the era.

# %%
def log_odds(era_counts, bg_counts, alpha=0.05):
    vocab    = set(era_counts) | set(bg_counts)
    total_e  = sum(era_counts.values()) + alpha * len(vocab)
    total_bg = sum(bg_counts.values())  + alpha * len(vocab)
    return {
        term: math.log(
            (era_counts.get(term, 0) + alpha) / total_e /
            ((bg_counts.get(term, 0)  + alpha) / total_bg)
        )
        for term in vocab
    }


era_counts = []
all_counts = Counter()

for era_num in range(n_eras):
    nouns  = []
    for token_list in df.loc[df['era'] == era_num, 'nouns']:
        if isinstance(token_list, list):
            nouns.extend(token_list)
    c = Counter(nouns)
    era_counts.append(c)
    all_counts.update(c)

era_lo = []
for era_num in range(n_eras):
    bg     = all_counts - era_counts[era_num]
    scores = log_odds(era_counts[era_num], bg)
    # Only keep nouns with sufficient presence in this era
    scores = {k: v for k, v in scores.items()
              if era_counts[era_num].get(k, 0) >= LOG_ODDS_MIN_COUNT}
    era_lo.append(scores)

# --- Bar charts ---
TOP_N = 15
fig, axes = plt.subplots(1, n_eras, figsize=(4 * n_eras, 6))
if n_eras == 1:
    axes = [axes]

for era_num, (ax, scores) in enumerate(zip(axes, era_lo)):
    s, e   = era_slices[era_num]
    top    = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
    vals   = [scores[t] for t in top]
    ax.barh(top[::-1], vals[::-1], color=palette[era_num], edgecolor='k', linewidth=0.4)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title(
        f'Era {era_num+1}\n'
        f'{df.iloc[s]["published_at"].strftime("%Y-%m")} – '
        f'{df.iloc[e-1]["published_at"].strftime("%Y-%m")}',
        fontsize=9
    )
    ax.set_xlabel('Log-odds vs all other eras', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

plt.suptitle('Log-Odds Ratio: Nouns Most Distinctive to Each Era\n'
             '(person names and organisations removed)', fontsize=12)
plt.tight_layout()
plt.savefig('era_logodds.png', bbox_inches='tight')
plt.show()

print('=== TOP NOUNS PER ERA ===\n')
for era_num, scores in enumerate(era_lo):
    s, e  = era_slices[era_num]
    top10 = sorted(scores, key=scores.get, reverse=True)[:10]
    print(f'Era {era_num+1}  ({df.iloc[s]["published_at"].date()} -> {df.iloc[e-1]["published_at"].date()})')
    print(f'  {", ".join(top10)}\n')

# %% [markdown]
# ## Method 2 — Embedding Dimension Attribution
#
# Log-odds tells us *which words* changed. This method tells us *which directions in
# content space* the channel moved along — and by how much.
#
# ### Background: what is a PCA dimension?
#
# Each video's transcript embedding is a point in 384-dimensional space. PCA finds
# the independent axes of greatest variation — the directions along which videos
# differ from each other most. PC1 explains the most variance, PC2 the next most
# independent amount, and so on.
#
# By computing the centroid (average position) of each era in this space, we can
# measure which axes the channel moved along over time. A large centroid shift along
# PC3 means the channel moved in whatever direction PC3 represents — we can figure
# out what that is by reading the videos at each extreme.
#
# ### Three visualisations
#
# **A. Era Fingerprint Radar Chart**
# Each spoke is a PCA dimension; each coloured polygon is an era. Spokes where all
# polygons cluster together = dimensions that were consistent across history. Spokes
# where they diverge = the directions the channel actually moved.
#
# **B. Annotated 2D Scatter**
# Every video projected onto the two dimensions that changed most across all era
# transitions. Extreme videos are labelled — reading those titles is how you interpret
# what each axis represents in human terms.
#
# **C. Word Clouds at Each Pole**
# For each key dimension, we split all videos into a "high" and "low" half and run
# log-odds between them. The resulting word clouds show the vocabulary that defines
# each end of the axis — together with the labelled titles, this gives you a complete
# semantic picture of what each dimension captures.

# %%
era_centroids = np.array([
    emb_pca[s:e].mean(axis=0) for s, e in era_slices
])

# Normalise centroids to [-1, 1] per dim for radar display
dim_max = np.abs(era_centroids).max(axis=0) + 1e-10
era_centroids_norm = era_centroids / dim_max

# %% [markdown]
# ### A. Era Fingerprint Radar Chart
#
# Each era is a polygon across all PCA dimensions. The further a spoke extends from
# the centre, the more that era's content leaned toward the positive end of that axis.
#
# - **Spokes where polygons cluster** → that dimension was consistent across history
# - **Spokes where polygons spread** → that dimension captures something that genuinely shifted
# - **Two eras with similar shapes** → those eras are close to each other in content space

# %%
angles = np.linspace(0, 2 * np.pi, PCA_DIMS, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

labels = [f'PC{i+1}' for i in range(PCA_DIMS)]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for era_num in range(n_eras):
    values = era_centroids_norm[era_num].tolist()
    values += values[:1]
    s, e = era_slices[era_num]
    label = f'Era {era_num+1} ({df.iloc[s]["published_at"].strftime("%Y-%m")})'
    ax.plot(angles, values, 'o-', lw=1.8, color=palette[era_num], label=label)
    ax.fill(angles, values, alpha=0.08, color=palette[era_num])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(-1, 1)
ax.set_yticks([-0.5, 0, 0.5])
ax.set_yticklabels(['-0.5', '0', '0.5'], fontsize=7)
ax.set_title('Era Content Fingerprints\nEach spoke = PCA dimension, each polygon = era centroid',
             pad=20, fontsize=11)
ax.legend(bbox_to_anchor=(1.35, 1.1), loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('era_pca_radar.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### B. Annotated 2D Scatter
#
# Every video projected onto the two PCA dimensions that changed most across all
# era transitions. Colour = era. Stars = era centroids. Arrows connect centroids
# chronologically, showing the channel's trajectory through content space.
#
# Videos at the extremes of each axis are labelled — reading those titles is how you
# figure out what each axis actually represents. For example, if the high end of PC2
# is all mental health videos and the low end is all gaming content, PC2 is roughly
# a "mental health vs gaming" axis.
#
# Label colour key:
# - **Dark red** = high end of the horizontal axis
# - **Dark blue** = low end of the horizontal axis
# - **Dark green** = high end of the vertical axis
# - **Purple** = low end of the vertical axis

# %%
n_transitions = n_eras - 1

if n_transitions > 0:
    total_change = np.sum([
        np.abs(era_centroids[i+1] - era_centroids[i]) for i in range(n_transitions)
    ], axis=0)
else:
    total_change = np.ones(PCA_DIMS)  # no transitions — treat all dims equally

dim_x, dim_y = total_change.argsort()[::-1][:2]

N_LABELS = 5   # videos to label at each extreme

fig, ax = plt.subplots(figsize=(13, 10))

for era_num, (s, e) in enumerate(era_slices):
    ax.scatter(emb_pca[s:e, dim_x], emb_pca[s:e, dim_y],
               color=palette[era_num], alpha=0.4, s=18,
               label=f'Era {era_num+1} ({df.iloc[s]["published_at"].strftime("%Y-%m")})')
    ax.scatter(era_centroids[era_num, dim_x], era_centroids[era_num, dim_y],
               color=palette[era_num], s=160, marker='*',
               edgecolors='k', linewidths=0.8, zorder=5)

for i in range(n_transitions):
    ax.annotate('',
        xy     =(era_centroids[i+1, dim_x], era_centroids[i+1, dim_y]),
        xytext =(era_centroids[i,   dim_x], era_centroids[i,   dim_y]),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Label extreme videos along dim_x
x_scores   = emb_pca[:, dim_x]
x_max_idx  = x_scores.argsort()[::-1][:N_LABELS]
x_min_idx  = x_scores.argsort()[:N_LABELS]

# Label extreme videos along dim_y
y_scores   = emb_pca[:, dim_y]
y_max_idx  = y_scores.argsort()[::-1][:N_LABELS]
y_min_idx  = y_scores.argsort()[:N_LABELS]

def _annotate(ax, idx, x, y, color='#333333'):
    title = df['title'].iloc[idx]
    title = title[:32] + '…' if len(title) > 32 else title
    ax.annotate(
        title,
        xy=(x, y),
        xytext=(x + (x_scores.max() - x_scores.min()) * 0.02, y),
        fontsize=6, color=color,
        arrowprops=dict(arrowstyle='-', color=color, lw=0.5),
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, lw=0),
    )

for idx in x_max_idx:
    _annotate(ax, idx, emb_pca[idx, dim_x], emb_pca[idx, dim_y], color='darkred')
for idx in x_min_idx:
    _annotate(ax, idx, emb_pca[idx, dim_x], emb_pca[idx, dim_y], color='navy')
for idx in y_max_idx:
    _annotate(ax, idx, emb_pca[idx, dim_x], emb_pca[idx, dim_y], color='darkgreen')
for idx in y_min_idx:
    _annotate(ax, idx, emb_pca[idx, dim_x], emb_pca[idx, dim_y], color='purple')

ax.set_xlabel(
    f'PC{dim_x+1}  ({pca.explained_variance_ratio_[dim_x]:.1%} variance)\n'
    f'← dark blue titles (low)          dark red titles (high) →',
    fontsize=10
)
ax.set_ylabel(
    f'PC{dim_y+1}  ({pca.explained_variance_ratio_[dim_y]:.1%} variance)\n'
    f'↑ dark green (high)   purple (low) ↓',
    fontsize=10
)
ax.set_title(
    f'Videos Projected onto Most-Changed Dimensions (PC{dim_x+1} vs PC{dim_y+1})\n'
    'Stars = era centroids  |  Arrows = chronological direction  |  '
    'Labelled = most extreme videos',
    fontsize=10
)
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig('era_pca_scatter.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### C. Word Clouds at Each Pole
#
# The scatter plot tells you *which videos* sit at the extremes of each axis. The word
# clouds tell you *what those videos are talking about*.
#
# For each of the two most-changed PCA dimensions, we split all videos into a "high"
# half (above median on that axis) and a "low" half. We run log-odds between the two
# halves — the same method as Method 1, but now comparing the two ends of an axis
# rather than comparing eras.
#
# - **Blue cloud (left)** = vocabulary of the LOW end of the axis
# - **Red cloud (right)** = vocabulary of the HIGH end of the axis
#
# The titles show *what the videos are*; the word clouds show *what they say*. Together
# they give you a complete semantic interpretation of each axis.

# %%
TOP_DIMS_FOR_WC = PCA_DIMS  # word clouds for all dims, ordered by most-changed first

for dim in total_change.argsort()[::-1][:TOP_DIMS_FOR_WC]:
    scores_dim = emb_pca[:, dim]
    median_val = np.median(scores_dim)

    high_mask = scores_dim >= median_val
    low_mask  = ~high_mask

    high_counts = Counter()
    low_counts  = Counter()
    all_wc      = Counter()

    for i, tokens in enumerate(df['nouns']):
        if not isinstance(tokens, list):
            continue
        c = Counter(tokens)
        all_wc.update(c)
        if high_mask[i]:
            high_counts.update(c)
        else:
            low_counts.update(c)

    # Log-odds: high vs low
    lo_high = log_odds(high_counts, low_counts)
    lo_low  = log_odds(low_counts,  high_counts)

    # Filter to terms with minimum presence
    lo_high = {k: v for k, v in lo_high.items() if high_counts.get(k, 0) >= 5}
    lo_low  = {k: v for k, v in lo_low.items()  if low_counts.get(k, 0) >= 5}

    # Convert to positive weights for word cloud (scores are already positive log-odds)
    wc_high = {k: max(v, 0) for k, v in lo_high.items() if v > 0}
    wc_low  = {k: max(v, 0) for k, v in lo_low.items()  if v > 0}

    if not wc_high or not wc_low:
        print(f'PC{dim+1}: not enough data for word clouds, skipping.')
        continue

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    wc_obj_low = WordCloud(
        width=700, height=400, background_color='white',
        colormap='Blues_r', max_words=60,
    ).generate_from_frequencies(wc_low)

    wc_obj_high = WordCloud(
        width=700, height=400, background_color='white',
        colormap='Reds_r', max_words=60,
    ).generate_from_frequencies(wc_high)

    axes[0].imshow(wc_obj_low, interpolation='bilinear')
    axes[0].set_title(f'← LOW end of PC{dim+1}', fontsize=11, color='navy')
    axes[0].axis('off')

    axes[1].imshow(wc_obj_high, interpolation='bilinear')
    axes[1].set_title(f'HIGH end of PC{dim+1} →', fontsize=11, color='darkred')
    axes[1].axis('off')

    # Print top 10 per side
    print(f'\n=== PC{dim+1} Poles  (explains {pca.explained_variance_ratio_[dim]:.1%} of variance) ===')
    print(f'  LOW end  → {", ".join(sorted(wc_low, key=wc_low.get, reverse=True)[:10])}')
    print(f'  HIGH end → {", ".join(sorted(wc_high, key=wc_high.get, reverse=True)[:10])}')

    plt.suptitle(
        f'PC{dim+1} Word Clouds  ({pca.explained_variance_ratio_[dim]:.1%} variance)\n'
        'Blue = low end of axis, Red = high end',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'era_wc_pc{dim+1}.png', bbox_inches='tight')
    plt.show()

# %% [markdown]
# ### Supplementary — Dimension Change at Each Era Transition
#
# For each pair of adjacent eras, this bar chart shows how much the centroid moved
# along each PCA dimension. Red bars = the two dimensions that changed most at that
# specific transition.
#
# Use this to identify *which transition* drove the biggest shift and *on which axis*.
# Cross-reference with the word clouds to understand what that shift meant in content terms.

# %%
if n_transitions > 0:
    fig, axes = plt.subplots(1, n_transitions, figsize=(5 * n_transitions, 4))
    if n_transitions == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        diff       = np.abs(era_centroids[i+1] - era_centroids[i])
        colors_bar = ['crimson' if d >= sorted(diff)[-2] else 'steelblue' for d in diff]
        ax.bar(range(PCA_DIMS), diff, color=colors_bar, edgecolor='k', linewidth=0.4)
        ax.set_xticks(range(PCA_DIMS))
        ax.set_xticklabels([f'PC{j+1}' for j in range(PCA_DIMS)], fontsize=8)
        ax.set_title(f'Era {i+1} → Era {i+2}  (red = top 2 dims)', fontsize=9)
        ax.set_ylabel('|Centroid diff|')

    plt.suptitle('PCA Dimension Change at Each Era Transition', fontsize=12)
    plt.tight_layout()
    plt.savefig('era_pca_transition.png', bbox_inches='tight')
    plt.show()

# %% [markdown]
# ### Supplementary — Extreme Videos Per Dimension
#
# A plain-text printout of the videos at the high and low ends of the three
# most-changed PCA dimensions. Each entry shows the era, publish date, and title.
#
# This is the fastest way to answer "what does this dimension represent?" — if the
# high-end videos are all about ADHD and the low-end videos are all about relationship
# dynamics, you know exactly what the axis is capturing.

# %%
N_EXTREME = 5

print('=== WHAT DO THE KEY DIMENSIONS MEAN? ===\n')
for dim in total_change.argsort()[::-1][:3]:
    scores_dim = emb_pca[:, dim]
    print(f'PC{dim+1}  ({pca.explained_variance_ratio_[dim]:.1%} of variance  |  '
          f'total era change: {total_change[dim]:.3f})')

    top_idx = scores_dim.argsort()[::-1][:N_EXTREME]
    print(f'  HIGH end:')
    for idx in top_idx:
        print(f'    [Era {int(df["era"].iloc[idx])+1}]  {df["published_at"].iloc[idx].date()}'
              f'  {df["title"].iloc[idx][:65]}')

    bot_idx = scores_dim.argsort()[:N_EXTREME]
    print(f'  LOW end:')
    for idx in bot_idx:
        print(f'    [Era {int(df["era"].iloc[idx])+1}]  {df["published_at"].iloc[idx].date()}'
              f'  {df["title"].iloc[idx][:65]}')
    print()
