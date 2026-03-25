import torch
from sentence_transformers import SentenceTransformer

from models import Session, TranscriptChunk

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1024  # increase if GPU util is still low, decrease if OOM


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    session = Session()

    try:
        chunks = (
            session.query(TranscriptChunk)
            .filter(TranscriptChunk.embedding == None)
            .all()
        )

        if not chunks:
            print("No chunks missing embeddings.")
            return

        total = len(chunks)
        print(f"Embedding {total} chunks in batches of {BATCH_SIZE}...\n")

        for batch_start in range(0, total, BATCH_SIZE):
            batch = chunks[batch_start:batch_start + BATCH_SIZE]
            texts = [c.text for c in batch]

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding.tolist()

            session.commit()
            done = min(batch_start + BATCH_SIZE, total)
            print(f"  [{done}/{total}] embedded and saved")

        print("\nDone.")

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    run()
