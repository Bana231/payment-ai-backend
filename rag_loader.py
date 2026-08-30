from supabase_store import list_knowledge_documents


def load_runbook():
    documents = list_knowledge_documents()
    if not documents:
        raise RuntimeError("Supabase knowledge_documents is empty.")
    return "\n\n".join(document["content"] for document in documents)


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
