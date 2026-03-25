import yaml
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Float, Text, JSON
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

with open("config.yml") as f:
    _config = yaml.safe_load(f)

DATABASE_URL = _config["database"]["url"]

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Channel(Base):
    __tablename__ = "channels"

    channel_id = Column(String, primary_key=True)
    title = Column(String)
    channel_created_at = Column(TIMESTAMP(timezone=True))
    subscriber_count = Column(Integer)
    total_video_count = Column(Integer)
    total_channel_views = Column(Integer)
    inserted_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))
    last_scraped_at = Column(TIMESTAMP(timezone=True))


class Video(Base):
    __tablename__ = "videos"

    video_id = Column(String, primary_key=True)
    channel_id = Column(String)
    url = Column(String)
    title = Column(String)
    description = Column(Text)
    tags = Column(JSON)
    category_id = Column(String)
    thumbnail_url = Column(String)
    has_captions = Column(Boolean)
    published_at = Column(TIMESTAMP(timezone=True))
    duration_seconds = Column(Integer)
    days_since_channel_start = Column(Integer)
    video_rank = Column(Integer)
    view_count = Column(Integer)
    like_count = Column(Integer)
    comment_count = Column(Integer)
    engagement_rate = Column(Float)
    transcript_word_count = Column(Integer)
    transcript_words_per_minute = Column(Float)
    inserted_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))


class TranscriptChunk(Base):
    __tablename__ = "transcript_chunks"

    id = Column(String, primary_key=True)
    video_id = Column(String)
    chunk_index = Column(Integer)
    text = Column(Text)
    embedding = Column(Vector(384))
    inserted_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))
