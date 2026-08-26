from pathlib import Path


RUNBOOK_PATH = Path("knowledge_base/payment_failure_runbook.txt")


def load_runbook():
    return RUNBOOK_PATH.read_text()


def split_into_chunks(text: str):
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


if __name__ == "__main__":
    document = load_runbook()
    chunks = split_into_chunks(document)

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)