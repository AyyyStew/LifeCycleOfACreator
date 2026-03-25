# %% [markdown]
# # What Changed Between Eras? — HealthyGamerGG
#
# This notebook detects the channel's content eras from the data and then characterises
# what was topically different about each one.
#
# It is self-contained — era detection and topic analysis all happen here.
#
# ---
#
# ### Era Detection — Topic Dominance Shift
# We separate the problem into two steps:
# 1. **KMeans clustering** finds the channel's distinct content topics (ignoring time).
#    The number of topics is chosen automatically via the silhouette score.
# 2. A **rolling dominant topic** window shows when each topic held sway.
#    Era boundaries are where the dominant topic shifted and stayed shifted.
#
# This produces clean, contiguous eras by construction — no flickering or
# state reuse across non-adjacent periods.
#
# ### Method 1 — Log-Odds Ratio
# Finds the words that are statistically *over-represented* in each era compared to all
# other eras combined. Unlike a plain word count, this is explicitly contrastive — a word
# only scores high if it appears *unusually often* in that era relative to the rest of the
# channel's history.
#
# To cut through the noise of conversational speech, we first run the transcripts through
# a natural language processing pipeline (spaCy) that:
# - Keeps only **nouns** — the content-bearing words
# - Strips out **named entities** (people, organisations) that appear everywhere and add no signal
#
# Results are shown as bar charts per era, and as word clouds at the poles of each key
# content axis (see Method 2).
#
# ### Method 2 — Embedding Dimension Attribution
# The transcript embeddings live in a 384-dimensional space. We compress that down to
# `PCA_DIMS` dimensions — each one an independent axis of variation in the content.
# If an era's centroid moved a lot along one of these axes, that axis captures whatever changed.
#
# We make these abstract axes interpretable through three visualisations:
# - **Radar / fingerprint chart** — each era as a polygon, instantly shows which
#   dimensions distinguish eras from each other
# - **Annotated 2D scatter** — videos plotted on the two most-changed axes, with the
#   most extreme videos labelled so you can read off what each axis means
# - **Word clouds at each pole** — log-odds nouns for the high vs low end of each axis,
#   showing the semantic content that defines each direction

# %%
# !pip install spacy wordcloud -q
# !python -m spacy download en_core_web_sm -q

# %% [markdown]
# ## Setup

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
# All tuneable parameters are here.
#
# **Era detection:**
# - `ROLL_WIN_ERAS` — rolling window size (videos) for computing dominant topic.
#   Larger = smoother transitions, less sensitive to one-off videos. Try 15–40.
# - `MIN_ERA_VIDEOS` — minimum era length. Runs shorter than this get merged into
#   the adjacent era. Increase if the timeline still looks fragmented.
# - `K_MIN_TOPICS` / `K_MAX_TOPICS` — range of topic counts to evaluate.
#   The silhouette score picks the best k automatically within this range.
#
# **NLP:**
# - `NLP_CHAR_LIMIT` — characters of each transcript fed to spaCy.
#   Higher = more accurate noun extraction, slower runtime. 8000 is a good balance.
# - `LOG_ODDS_MIN_COUNT` — a noun must appear at least this many times in an era
#   to be included in the log-odds ranking. Filters out rare one-off words.
#
# **Visualisation:**
# - `PCA_DIMS` — dimensions used for the radar chart and scatter plot.
#   Does not affect era detection.

# %%
CHANNEL_QUERY    = 'healthygamergg'
CONTENT_TYPE     = 'long'
SHORTS_MAX_SECS  = 180
PCA_DIMS         = 10   # PCA dims for the scatter/radar visualisations later
ROLL_WIN_ERAS    = 20   # rolling window (videos) for computing dominant topic
MIN_ERA_VIDEOS   = 15   # minimum run length — shorter runs get merged into neighbour
K_MIN_TOPICS     = 3    # smallest k to try for topic clustering
K_MAX_TOPICS     = 12   # largest k to try

NLP_CHAR_LIMIT   = 8000   # chars per transcript fed to spaCy (for speed)
LOG_ODDS_MIN_COUNT = 5    # minimum era term count to include in log-odds

# %% [markdown]
# ## Load Data
# We pull two things from the database:
# 1. **Video metadata + mean-pooled embeddings** — one row per video, the embedding is the
#    average of all its transcript chunk embeddings
# 2. **Full transcript text** — all chunks concatenated in order, used for the NLP analysis

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
# ## Era Detection: Topic Dominance Shift
#
# Rather than fitting a single model to find eras directly, we separate the problem
# into two cleaner questions:
#
# **Question 1: What are the distinct topics this channel makes content about?**
# We answer this with **KMeans clustering** on the video embeddings, ignoring time
# entirely. KMeans finds groups of videos that are similar to each other in embedding
# space — each group is a topic the channel covers.
#
# We automatically select the number of topics using the **silhouette score** — a
# measure of how well-separated the clusters are. Higher = more distinct clusters.
# We try k = `K_MIN_TOPICS` to `K_MAX_TOPICS` and pick the k with the highest score.
#
# **Question 2: When did the dominant topic shift and stay shifted?**
# Once every video has a topic label, we compute a **rolling dominant topic** over a
# window of `ROLL_WIN_ERAS` videos — the topic that appears most in that window.
# We then apply a minimum run-length filter: any stretch of fewer than `MIN_ERA_VIDEOS`
# videos in the same dominant topic gets absorbed into its neighbour.
# What remains are the clean, contiguous era boundaries.
#
# ### Why this is better than HMM for this data
# The HMM was non-stationary — it had no concept of time direction, so it would assign
# the same state to similar content in 2019 and 2024, producing a flickering timeline.
# This approach separates "what are the topics" (KMeans, time-agnostic) from "when did
# they dominate" (rolling window, time-aware), so the output is guaranteed to be
# contiguous and chronologically sensible.

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _enforce_min_run(states, min_run):
    """Merge runs shorter than min_run into the adjacent state."""
    out = list(states)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            j = i
            while j < len(out) and out[j] == out[i]:
                j += 1
            if (j - i) < min_run and len(set(out)) > 1:
                neighbour = out[i - 1] if i > 0 else out[j] if j < len(out) else out[i]
                out[i:j] = [neighbour] * (j - i)
                changed = True
            i = j
    return out


# --- Stage 1: Silhouette-based k selection ---
print(f'Selecting number of topics (k={K_MIN_TOPICS}..{K_MAX_TOPICS})...\n')

sil_scores, km_models = [], []
sample_n = min(1000, len(emb_norm))  # silhouette is O(n²), sample for speed

for k in range(K_MIN_TOPICS, K_MAX_TOPICS + 1):
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_norm)
    sil    = silhouette_score(emb_norm, labels, sample_size=sample_n, random_state=42)
    sil_scores.append(sil)
    km_models.append(km)
    print(f'  k={k:>2}  silhouette={sil:.4f}')

best_k_idx    = int(np.argmax(sil_scores))
best_k_topics = K_MIN_TOPICS + best_k_idx
best_km       = km_models[best_k_idx]
df['topic']   = best_km.labels_
print(f'\n→ Best k = {best_k_topics}  (silhouette = {sil_scores[best_k_idx]:.4f})')

# %% [markdown]
# ### Silhouette Score Curve
# Higher silhouette = more distinct, better-separated topic clusters.
# The peak is the data's natural number of topics.

# %%
ks_plot = list(range(K_MIN_TOPICS, K_MAX_TOPICS + 1))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ks_plot, sil_scores, marker='o', color='steelblue', lw=2)
ax.axvline(best_k_topics, color='crimson', ls='--', lw=1.5, label=f'Best k = {best_k_topics}')
ax.set_title('Silhouette Score vs Number of Topic Clusters\nHigher = more distinct clusters', fontsize=10)
ax.set_xlabel('k (number of topic clusters)')
ax.set_ylabel('Silhouette score')
ax.legend()
plt.tight_layout()
plt.savefig('hmm_bic.png', bbox_inches='tight')   # reuse filename for compatibility
plt.show()

# %% [markdown]
# ### Stage 2: Rolling Dominant Topic → Era Boundaries
#
# For each video we compute the most common topic in the surrounding window of
# `ROLL_WIN_ERAS` videos. When the dominant topic shifts and holds for at least
# `MIN_ERA_VIDEOS` videos, that's an era boundary.
#
# The result is a clean, contiguous timeline — no flickering, no state reuse across
# different periods. If the channel returns to a previous topic later, that counts as
# a new era (same topic, different period).

# %%
# Rolling dominant topic
rolling_dom = (
    df['topic']
    .rolling(ROLL_WIN_ERAS, center=True, min_periods=1)
    .apply(lambda x: int(pd.Series(x.astype(int)).mode().iloc[0]))
    .astype(int)
    .tolist()
)

# Enforce minimum run length
smoothed_eras = _enforce_min_run(rolling_dom, MIN_ERA_VIDEOS)

# Build contiguous era slices (sequential runs, even if same topic reappears)
era_slices  = []
era_topics  = []   # which topic cluster each era corresponds to
i = 0
while i < len(smoothed_eras):
    j = i
    while j < len(smoothed_eras) and smoothed_eras[j] == smoothed_eras[i]:
        j += 1
    era_slices.append((i, j))
    era_topics.append(smoothed_eras[i])
    i = j

# Assign sequential era index to each video
df['era'] = -1
for era_idx, (s, e) in enumerate(era_slices):
    df.iloc[s:e, df.columns.get_loc('era')] = era_idx

n_eras  = len(era_slices)
palette = plt.cm.Set2(np.linspace(0, 1, best_k_topics))   # colour by topic, not era index

print(f'Detected {n_eras} eras  ({best_k_topics} underlying topics):\n')
for era_idx, (s, e) in enumerate(era_slices):
    topic = era_topics[era_idx]
    print(f'  Era {era_idx+1:>2}  [topic {topic}]  '
          f'{df.iloc[s]["published_at"].date()} -> {df.iloc[e-1]["published_at"].date()}'
          f'  ({e-s} videos)')

# Visual timeline — colour = topic so returning topics are visually obvious
fig, ax = plt.subplots(figsize=(16, 2.5))
for era_idx, (s, e) in enumerate(era_slices):
    topic = era_topics[era_idx]
    ax.axvspan(df['published_at'].iloc[s], df['published_at'].iloc[e - 1],
               alpha=0.5, color=palette[topic])
    mid = df['published_at'].iloc[s] + (df['published_at'].iloc[e - 1] - df['published_at'].iloc[s]) / 2
    ax.text(mid, 0.5, f'E{era_idx+1}\nT{topic}', ha='center', va='center', fontsize=7)

ax.set_xlim(df['published_at'].min(), df['published_at'].max())
ax.set_yticks([])
ax.set_title(
    f'Era Timeline — {n_eras} eras from {best_k_topics} topics  '
    f'(window={ROLL_WIN_ERAS}, min run={MIN_ERA_VIDEOS})\n'
    'Colour = topic cluster. Same colour reappearing = channel returned to that topic.'
)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('hmm_states.png', bbox_inches='tight')   # reuse filename for compatibility
plt.show()

# %% [markdown]
# ## Text Preprocessing: NER + Noun Extraction
#
# Before we can compare vocabulary across eras, we need to clean the transcripts.
# Raw spoken transcripts are messy — they're full of filler words, repeated names,
# and organisation references that appear in every era and drown out the actual signal.
#
# We use **spaCy** (a standard NLP library) to do three things:
#
# **1. Part-of-speech tagging** — every word gets tagged as a noun, verb, adjective, etc.
# We keep only **nouns**, since those are the content-bearing words. Verbs like
# "talk", "feel", "make" appear constantly regardless of topic.
#
# **2. Named entity recognition (NER)** — spaCy identifies spans of text that are
# proper names. We remove anything tagged as a **PERSON** or **ORG** — names like
# "Dr K", "Twitch", "Harvard" show up across all eras and tell us nothing about what
# *topics* changed.
#
# **3. Lemmatisation** — words are reduced to their base form so "relationships",
# "relationship", and "relating" all count as the same token.
#
# Each transcript is capped at `NLP_CHAR_LIMIT` characters and processed in batches
# to keep the runtime manageable.

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
# ## Method 1: Log-Odds Ratio
#
# Now that we have clean noun lists, we can measure how distinctive each word is to each era.
#
# The **log-odds ratio** (Monroe et al. 2008) compares how often a word appears in one era
# versus all other eras combined:
#
# > score = log( P(word | *this era*) / P(word | *all other eras*) )
#
# - **High positive score** → word appears *much more often* in this era than the rest of the channel's history — strongly era-defining
# - **Near zero** → word appears at roughly the same rate everywhere — not useful
# - **Negative score** → word is *less common* in this era than elsewhere
#
# We use Laplace smoothing (a small constant added to all counts) so that rare words
# don't get artificially extreme scores.
#
# This is the key advantage over plain word frequency or TF-IDF: the score is
# explicitly *relative to the rest of the channel*, not just to other words within the era.

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
    ax.barh(top[::-1], vals[::-1], color=palette[era_topics[era_num]], edgecolor='k', linewidth=0.4)
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
# ## Method 2: Embedding Dimension Attribution
#
# The log-odds analysis tells us *which words* changed. This section tells us *which
# directions in content space* the channel moved along — and by how much.
#
# ### Background: what is a PCA dimension?
# Each video's embedding is a point in 384-dimensional space. PCA finds the directions
# of greatest variance in that space — the axes along which videos differ from each other
# the most. PC1 is the single biggest axis of variation, PC2 the next biggest independent
# one, and so on.
#
# By computing the average (centroid) position of each era in this space, we can see
# which axes the channel moved along over time. A large shift along PC3 means the channel
# moved in whatever direction PC3 represents — and we can figure out what that is by
# looking at which videos sit at each extreme.
#
# ### Three visualisations
#
# **A. Era fingerprint radar chart**
# Each spoke = one PCA dimension. Each era = one filled polygon.
# Spokes where polygons overlap = dimensions that didn't distinguish eras.
# Spokes where they diverge = the directions the channel actually moved.
#
# **B. Annotated 2D scatter**
# Videos projected onto the two dimensions that changed most across all era transitions.
# Extreme videos are labelled directly on the plot — reading those titles is how you
# interpret what each axis represents.
#
# **C. Word clouds at each pole**
# For each key dimension, we split all videos into a "high" and "low" half, run log-odds
# between them, and render word clouds. The words on each side tell you what the axis
# semantically captures from the content itself.

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
# Each era is represented as a polygon across all 10 PCA dimensions.
# The further a spoke extends from the centre, the more that era's content
# leaned toward the high end of that dimension.
#
# **How to read it:**
# - Spokes where all polygons are close together → that dimension is consistent across the channel's history
# - Spokes where polygons spread apart → that dimension captures something that genuinely shifted
# - Two eras with similar polygon shapes → those eras are close to each other in content space

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
    ax.plot(angles, values, 'o-', lw=1.8, color=palette[era_topics[era_num]], label=label)
    ax.fill(angles, values, alpha=0.08, color=palette[era_topics[era_num]])

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
# Every video is plotted as a dot on the two PCA dimensions that changed most across
# all era transitions. Colour = era. Stars = era centroids. Arrows connect centroids
# in chronological order, showing the direction of travel through content space.
#
# The videos at the extremes of each axis are labelled directly on the chart.
# **Reading those titles is how you interpret what the axis means** — e.g. if the
# high end of PC2 is all "How to Deal With Anxiety" videos and the low end is all
# gaming strategy content, you know PC2 is roughly a "mental health vs gaming" axis.
#
# Colour key for labels:
# - 🔴 Dark red = high end of the horizontal axis
# - 🔵 Dark blue = low end of the horizontal axis
# - 🟢 Dark green = high end of the vertical axis
# - 🟣 Purple = low end of the vertical axis

# %%
n_transitions = n_eras - 1

if n_transitions > 0:
    total_change = np.sum([
        np.abs(era_centroids[i+1] - era_centroids[i]) for i in range(n_transitions)
    ], axis=0)
    dim_x, dim_y = total_change.argsort()[::-1][:2]
else:
    dim_x, dim_y = 0, 1

N_LABELS = 5   # videos to label at each extreme

fig, ax = plt.subplots(figsize=(13, 10))

for era_num, (s, e) in enumerate(era_slices):
    ax.scatter(emb_pca[s:e, dim_x], emb_pca[s:e, dim_y],
               color=palette[era_topics[era_num]], alpha=0.4, s=18,
               label=f'Era {era_num+1} ({df.iloc[s]["published_at"].strftime("%Y-%m")})')
    ax.scatter(era_centroids[era_num, dim_x], era_centroids[era_num, dim_y],
               color=palette[era_topics[era_num]], s=160, marker='*',
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
# The scatter plot tells you *which videos* sit at the extremes of each axis.
# The word clouds tell you *what those videos are talking about*.
#
# For each of the top 2 most-changed PCA dimensions, we split all videos into a
# "high" half (above median on that dimension) and a "low" half (below median).
# We then run the same log-odds analysis from Method 1 — but now comparing the
# two halves against each other rather than comparing eras.
#
# Words that score high in log-odds for one half but not the other are the ones
# that *define* that end of the axis in content terms.
#
# - **🔵 Blue word cloud (left)** = vocabulary of the LOW end of the axis
# - **🔴 Red word cloud (right)** = vocabulary of the HIGH end of the axis
#
# Together with the labelled titles from the scatter, this gives you a complete
# picture: the titles show *what the videos are*, the word clouds show *what they say*.

# %%
TOP_DIMS_FOR_WC = 2   # word clouds for this many most-changed dims

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
# ### Supplementary: Dimension Change at Each Era Transition
#
# For each pair of adjacent eras, this shows how much the centroid moved along
# each PCA dimension. Red bars = the two dimensions that changed most at that transition.
#
# Use this to identify *which transition* drove the biggest shift, and *on which axis*.
# Those are the transitions and dimensions worth investigating further in the word clouds.

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
# ### Supplementary: Extreme Videos Per Dimension
#
# A plain-text printout of the videos that sit at the high and low ends of the three
# most-changed PCA dimensions. Each entry shows the era it belongs to, its publish date,
# and its title.
#
# This is the most direct way to answer "what does this dimension represent?" —
# if the high-end videos are all about ADHD and the low-end videos are all about
# relationship dynamics, you know exactly what axis you're looking at.

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
