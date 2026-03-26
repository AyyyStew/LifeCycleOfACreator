# %% [markdown]
# # BERTTopic on Video Transcripts
#
# Aggregates transcript chunks to video level, filters to nouns via spaCy, then fits BERTTopic using pre-computed mean-pooled embeddings.

# %% Imports
# !pip install bertopic spacy umap-learn hdbscan tqdm -q
# !python -m spacy download en_core_web_sm -q

import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from umap import UMAP

sys.path.insert(0, ".")
from models import Session, TranscriptChunk

# %% Load chunks and aggregate to video level
session = Session()

rows = (
    session.query(
        TranscriptChunk.video_id,
        TranscriptChunk.chunk_index,
        TranscriptChunk.text,
        TranscriptChunk.embedding,
    )
    .filter(TranscriptChunk.embedding.isnot(None))
    .order_by(TranscriptChunk.video_id, TranscriptChunk.chunk_index)
    .all()
)

session.close()
print(f"Loaded {len(rows)} chunks — aggregating to video level...")

video_texts = defaultdict(list)
video_embeddings = defaultdict(list)

for r in rows:
    video_texts[r.video_id].append(r.text or "")
    video_embeddings[r.video_id].append(r.embedding)

video_ids = list(video_texts.keys())
raw_texts = [" ".join(video_texts[vid]) for vid in video_ids]
embeddings = normalize(
    np.array(
        [np.mean(video_embeddings[vid], axis=0) for vid in video_ids],
        dtype=np.float32,
    )
)

print(f"{len(video_ids)} videos")

# %% Extract nouns via spaCy
EXTRA_STOPWORDS = {
    "thing",
    "stuff",
    "lot",
    "bit",
    "way",
    "time",
    "day",
    "year",
    "video",
    "channel",
    "stream",
    "chat",
    "comment",
    "question",
    "answer",
    "guy",
    "man",
    "woman",
    "people",
    "person",
    "everyone",
    "someone",
    "kind",
    "sort",
    "type",
    "part",
    "point",
    "place",
    "end",
    "world",
    "sense",
    "fact",
    "case",
    "reason",
    "problem",
    "issue",
    "example",
    "number",
    "level",
    "moment",
    "mind",
    "word",
    "life",
}

NOUN_CACHE = "noun_cache.csv"

if os.path.exists(NOUN_CACHE):
    print(f"Loading noun cache from {NOUN_CACHE}...")
    cache_df  = pd.read_csv(NOUN_CACHE).set_index("video_id")
    noun_docs = [cache_df.loc[vid, "nouns"] if vid in cache_df.index else "" for vid in video_ids]
    print("Done.")
else:
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "senter"])
    nlp.max_length = max(len(t) for t in raw_texts) + 100

    def _doc_to_nouns(doc):
        ent_spans = set()
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG"):
                for tok in ent:
                    ent_spans.add(tok.i)
        nouns = []
        for tok in doc:
            if tok.i in ent_spans:
                continue
            if tok.pos_ != "NOUN":
                continue
            lemma = tok.lemma_.lower()
            if len(lemma) < 3 or lemma in EXTRA_STOPWORDS:
                continue
            nouns.append(lemma)
        return " ".join(nouns)

    n = len(raw_texts)
    print(f"Extracting nouns from {n} videos...")
    noun_docs = [
        _doc_to_nouns(doc)
        for doc in tqdm(nlp.pipe(raw_texts, batch_size=64, n_process=6), total=n, unit="video")
    ]

    pd.DataFrame({"video_id": video_ids, "nouns": noun_docs}).to_csv(NOUN_CACHE, index=False)
    print(f"Saved cache to {NOUN_CACHE}.")

# %% Drop videos with no nouns
mask = np.array([bool(d.strip()) for d in noun_docs])
video_ids = [vid for vid, m in zip(video_ids, mask) if m]
noun_docs = [d for d, m in zip(noun_docs, mask) if m]
embeddings = embeddings[mask]

print(f"{len(video_ids)} videos with nouns / {n} total")

# %% Fit BERTTopic
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

topic_model = BERTopic(
    embedding_model=None,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,
    verbose=True,
)

topics, _ = topic_model.fit_transform(noun_docs, embeddings=embeddings)

# %% Reduce to 30 topics by merging most similar
print(f"Topics before reduction: {len(topic_model.get_topic_info()) - 1}")  # -1 excludes outlier
topics = topic_model.reduce_topics(noun_docs, nr_topics=30)
print(f"Topics after reduction:  {len(topic_model.get_topic_info()) - 1}")

# %% Topic summary
topic_info = topic_model.get_topic_info()
print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

# %% Top words per topic
print("\n=== TOP WORDS PER TOPIC ===\n")
for topic_id in sorted(topic_model.get_topics()):
    if topic_id == -1:
        continue
    words = [w for w, _ in topic_model.get_topic(topic_id)]
    print(f"Topic {topic_id:3d}: {', '.join(words)}")

# %% Save video → topic mapping with labels
topic_info = topic_model.get_topic_info().set_index("Topic")

def _make_label(topic_id):
    if topic_id == -1:
        return "Outlier"
    name = topic_info.loc[topic_id, "Name"]          # e.g. "0_dopamine_game_pleasure"
    words = str(name).split("_")[1:]                  # drop leading id
    return " / ".join(words[:3])

df_topics = pd.DataFrame({"video_id": video_ids, "topic": topic_model.topics_})
df_topics["topic_label"] = df_topics["topic"].apply(_make_label)
df_topics.to_csv("video_topics_labeled.csv", index=False)
print(f"\nSaved video_topics_labeled.csv ({len(df_topics)} rows)")
print(df_topics["topic_label"].value_counts().head(10))
