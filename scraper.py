import uuid
from datetime import timezone
from dateutil import parser as dateparser

import yaml
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
from sqlalchemy.dialects.postgresql import insert

from models import Session, Channel, Video, TranscriptChunk

with open("config.yml") as f:
    _config = yaml.safe_load(f)

YOUTUBE_API_KEY = _config["youtube"]["api_key"]
CHUNK_SIZE = _config["scraper"]["chunk_size"]
_proxy_cfg = _config["scraper"]["proxy"]

_proxy_config = None
if _proxy_cfg.get("http") or _proxy_cfg.get("https"):
    _proxy_config = GenericProxyConfig(
        http_url=_proxy_cfg.get("http", ""),
        https_url=_proxy_cfg.get("https", ""),
    )

_transcript_api = YouTubeTranscriptApi(proxy_config=_proxy_config)


def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def scrape_channel(channel_id: str, limit: int | None = None):
    """
    Scrape a channel's videos and transcripts.
    limit: number of most recent videos to scrape. None = all.
    """
    youtube = get_youtube_client()
    session = Session()

    try:
        print("Fetching channel info...")
        channel_title = _upsert_channel(youtube, session, channel_id)
        session.commit()
        print(f"  Channel: {channel_title}")

        print("Fetching video list...")
        ranked_videos = _get_video_ids_ranked(youtube, channel_id, limit)
        total = len(ranked_videos)
        print(f"Found {total} videos to scrape\n")

        for i, (video_id, rank) in enumerate(ranked_videos):
            print(f"[{i+1}/{total}] Fetching video details...")
            title = _upsert_video(youtube, session, video_id, channel_id, rank)
            print(f"  {title} ({video_id})")
            print(f"  Fetching transcript...")
            _upsert_transcript_chunks(session, video_id)
            session.commit()
            print(f"  Saved.\n")

        print("Done.")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def scrape_channels(channel_ids: list[str], limit: int = None):
    for channel_id in channel_ids:
        print(f"\n=== Scraping channel {channel_id} ===")
        scrape_channel(channel_id, limit)


# --- Channel ---

def _upsert_channel(youtube, session, channel_id: str):
    resp = youtube.channels().list(
        part="snippet,statistics",
        id=channel_id
    ).execute()

    item = resp["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    stmt = insert(Channel).values(
        channel_id=channel_id,
        title=snippet.get("title"),
        channel_created_at=dateparser.parse(snippet["publishedAt"]),
        subscriber_count=int(stats.get("subscriberCount", 0)),
        total_video_count=int(stats.get("videoCount", 0)),
        total_channel_views=int(stats.get("viewCount", 0)),
    ).on_conflict_do_update(
        index_elements=["channel_id"],
        set_={
            "title": snippet.get("title"),
            "subscriber_count": int(stats.get("subscriberCount", 0)),
            "total_video_count": int(stats.get("videoCount", 0)),
            "total_channel_views": int(stats.get("viewCount", 0)),
            "updated_at": _now(),
        }
    )
    session.execute(stmt)
    return snippet.get("title")


# --- Videos ---

def _get_video_ids_ranked(youtube, channel_id: str, limit: int = None) -> list[tuple[str, int]]:
    """
    Returns list of (video_id, rank) tuples.
    Rank 1 = first video ever uploaded. Playlist returns newest first,
    so we fetch all, reverse, then assign ranks. limit takes the most recent N.
    """
    resp = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    uploads_playlist = resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    all_ids = []
    next_page_token = None

    while True:
        pl_resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in pl_resp["items"]:
            all_ids.append(item["contentDetails"]["videoId"])

        next_page_token = pl_resp.get("nextPageToken")
        if not next_page_token:
            break

    # Reverse to chronological order and assign ranks
    all_ids.reverse()
    ranked = [(video_id, rank + 1) for rank, video_id in enumerate(all_ids)]

    if limit:
        # Take the most recent N (end of the chronological list)
        ranked = ranked[-limit:]

    return ranked


def _upsert_video(youtube, session, video_id: str, channel_id: str, video_rank: int = None):
    resp = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    ).execute()

    if not resp["items"]:
        return None

    item = resp["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    published_at = dateparser.parse(snippet["publishedAt"])

    # Get channel creation date to compute days_since_channel_start
    channel = session.get(Channel, channel_id)
    days_since_start = None
    if channel and channel.channel_created_at:
        delta = published_at - channel.channel_created_at
        days_since_start = delta.days

    view_count = int(stats.get("viewCount", 0))
    like_count = int(stats.get("likeCount", 0))
    comment_count = int(stats.get("commentCount", 0))
    engagement_rate = (like_count + comment_count) / view_count if view_count else None

    duration_seconds = _parse_duration(item["contentDetails"]["duration"])

    stmt = insert(Video).values(
        video_id=video_id,
        channel_id=channel_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=snippet.get("title"),
        description=snippet.get("description"),
        tags=snippet.get("tags"),
        category_id=snippet.get("categoryId"),
        thumbnail_url=snippet.get("thumbnails", {}).get("high", {}).get("url"),
        has_captions=snippet.get("caption") == "true",
        published_at=published_at,
        duration_seconds=duration_seconds,
        days_since_channel_start=days_since_start,
        video_rank=video_rank,
        view_count=view_count,
        like_count=like_count,
        comment_count=comment_count,
        engagement_rate=engagement_rate,
    ).on_conflict_do_update(
        index_elements=["video_id"],
        set_={
            "view_count": view_count,
            "like_count": like_count,
            "comment_count": comment_count,
            "engagement_rate": engagement_rate,
            "updated_at": _now(),
        }
    )
    session.execute(stmt)
    return snippet.get("title")


# --- Transcripts ---

def _upsert_transcript_chunks(session, video_id: str):
    import time

    existing = session.query(TranscriptChunk).filter_by(video_id=video_id).first()
    if existing:
        print(f"  Transcript already in DB, skipping.")
        return

    max_retries = 4
    for attempt in range(max_retries):
        try:
            transcript = _transcript_api.fetch(video_id)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"  Transcript fetch failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Transcript unavailable after {max_retries} attempts, skipping.")
                return

    full_text = " ".join(entry.text for entry in transcript)
    words = full_text.split()
    chunks = [words[i:i+CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]

    # Delete existing chunks for this video before reinserting
    session.query(TranscriptChunk).filter_by(video_id=video_id).delete()

    for i, chunk_words in enumerate(chunks):
        chunk = TranscriptChunk(
            id=str(uuid.uuid4()),
            video_id=video_id,
            chunk_index=i,
            text=" ".join(chunk_words),
        )
        session.add(chunk)

    # Update transcript stats on video
    word_count = len(words)
    video = session.get(Video, video_id)
    if video and video.duration_seconds:
        video.transcript_words_per_minute = word_count / (video.duration_seconds / 60)
    if video:
        video.transcript_word_count = word_count


# --- Helpers ---

def _now():
    from datetime import datetime
    return datetime.now(timezone.utc)


def _parse_duration(duration: str) -> int:
    """Parse ISO 8601 duration (PT1H2M3S) to seconds."""
    import re
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _resolve_channel_id(youtube, value: str) -> str:
    """
    Accepts a channel ID, @handle, or YouTube channel URL.
    Returns the resolved channel_id or exits if invalid.
    Supported formats:
      - UCxxxxxx
      - https://www.youtube.com/channel/UCxxxxxx
      - https://www.youtube.com/@handle
    """
    import re
    import sys

    # Strip whitespace
    value = value.strip()

    # Extract from URL
    channel_match = re.search(r"youtube\.com/channel/([A-Za-z0-9_-]+)", value)
    handle_match = re.search(r"youtube\.com/@([A-Za-z0-9_.-]+)", value)
    custom_match = re.search(r"youtube\.com/(?:c|user)/([A-Za-z0-9_.-]+)", value)

    if channel_match:
        channel_id = channel_match.group(1)
    elif handle_match:
        handle = handle_match.group(1)
        resp = youtube.channels().list(part="id", forHandle=handle).execute()
        if not resp.get("items"):
            print(f"Error: could not find a channel for handle @{handle}")
            sys.exit(1)
        channel_id = resp["items"][0]["id"]
    elif custom_match:
        name = custom_match.group(1)
        # Try forUsername first (legacy /user/ URLs), then fall back to search
        resp = youtube.channels().list(part="id", forUsername=name).execute()
        if resp.get("items"):
            channel_id = resp["items"][0]["id"]
        else:
            resp = youtube.search().list(part="snippet", q=name, type="channel", maxResults=1).execute()
            if not resp.get("items"):
                print(f"Error: could not find a channel for '{name}'")
                sys.exit(1)
            channel_id = resp["items"][0]["snippet"]["channelId"]
    elif re.fullmatch(r"UC[A-Za-z0-9_-]{22}", value):
        channel_id = value
    else:
        print(f"Error: '{value}' is not a valid YouTube channel ID or URL")
        sys.exit(1)

    # Confirm the channel exists
    resp = youtube.channels().list(part="id", id=channel_id).execute()
    if not resp.get("items"):
        print(f"Error: channel '{channel_id}' not found on YouTube")
        sys.exit(1)

    return channel_id


if __name__ == "__main__":
    import argparse
    import sys

    if not _proxy_config:
        print("Error: no proxy configured. Set proxy.http and proxy.https in config.yml before running.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Scrape a YouTube channel into the database.")
    parser.add_argument("channel", help="Channel ID, @handle, or YouTube channel URL")
    args = parser.parse_args()

    youtube = get_youtube_client()
    channel_id = _resolve_channel_id(youtube, args.channel)

    limit_input = input("How many recent videos to scrape? (press Enter for all): ").strip()
    limit = int(limit_input) if limit_input else None

    scrape_channel(channel_id, limit=limit)
