"""
Joins video_topics.csv with topic labels parsed from topics_per_video.txt.
Outputs video_topics_labeled.csv with columns: video_id, topic, topic_label.
"""
import pandas as pd

# Parse labels from the summary table at the top of topics_per_video.txt
# Format: "    0    107    0_dopamine_game_pleasure_gaming"
topic_labels = {-1: "Outlier"}
with open("topics_per_video.txt") as f:
    for line in f:
        parts = line.strip().split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue
        try:
            topic_id = int(parts[0])
            words = parts[-1].split("_")[1:]  # drop leading id number
            topic_labels[topic_id] = " / ".join(words[:3])
        except (ValueError, IndexError):
            continue

labels_df = pd.DataFrame(
    [{"topic": k, "topic_label": v} for k, v in topic_labels.items()]
)

topics_df = pd.read_csv("video_topics.csv")
out = topics_df.merge(labels_df, on="topic", how="left")
out.to_csv("video_topics_labeled.csv", index=False)

print(f"Saved {len(out)} rows to video_topics_labeled.csv")
print(out.head())
