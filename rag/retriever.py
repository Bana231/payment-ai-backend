# Turns a knowledge-base query into the most relevant chunks of approved
# knowledge-document text. Chunk embeddings are stored in Postgres (the
# knowledge_chunks table added by
# supabase/migrations/20260918_add_pgvector_chunks.sql) and searched with
# pgvector's cosine-distance operator via a single RPC call — this is
# what makes retrieval a real vector-database lookup instead of the
# previous in-process, recompute-on-every-request brute-force scan.
#
# Both public functions below (refresh_index, retrieve_relevant_chunks)
# keep the exact same names/signatures/return shape they had before this
# change, so nothing that calls them (agent_graph.py, api.py) had to
# change.
from sentence_transformers import SentenceTransformer

from rag_loader import split_into_chunks
from supabase_store import get_supabase, list_knowledge_documents


model = SentenceTransformer("all-MiniLM-L6-v2")


def refresh_index() -> None:
    """Re-embed every knowledge_documents row and replace knowledge_chunks.

    Called once at import time (below), and again whenever a document is
    created/deleted via the knowledge-base API, so newly added/removed
    content is searchable right away instead of waiting for a restart.
    Does a full delete-and-reinsert every time — simplest way to
    guarantee knowledge_chunks never drifts out of sync with a renamed or
    edited document (no leftover stale chunks to reconcile).
    """
    documents = list_knowledge_documents()

    # Each document is chunked and embedded on its own (rather than
    # joining every document into one big string first, as the old
    # in-memory version did) so every chunk row can record which
    # document it actually came from.
    rows = []

    for document in documents:

        chunks = split_into_chunks(
            document["content"]
        )

        if not chunks:
            continue

        embeddings = model.encode(
            chunks,
            convert_to_tensor=False,
        )

        for chunk_index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            rows.append(
                {
                    "document_slug":
                        document["slug"],

                    "chunk_index":
                        chunk_index,

                    "content":
                        chunk,

                    # pgvector accepts a plain list of floats here and
                    # casts it to its `vector` column type.
                    "embedding":
                        embedding.tolist(),
                }
            )

    supabase = get_supabase()

    # id is a bigint identity column starting at 1, so this always
    # matches (and deletes) every existing row.
    supabase.table("knowledge_chunks").delete().gte("id", 0).execute()

    if rows:
        supabase.table("knowledge_chunks").insert(rows).execute()


refresh_index()


def retrieve_relevant_chunks(question: str, top_k: int = 2):
    question_embedding = model.encode(
        question,
        convert_to_tensor=False,
    ).tolist()

    response = (
        get_supabase()
        .rpc(
            "match_knowledge_chunks",
            {
                "query_embedding": question_embedding,
                "match_count": top_k,
            },
        )
        .execute()
    )

    return [
        {
            "content": row["content"],
            "score": round(row["score"], 4),
        }
        for row in response.data
    ]
