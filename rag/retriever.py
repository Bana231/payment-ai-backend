from sentence_transformers import SentenceTransformer, util

from rag_loader import load_runbook, split_into_chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

document = load_runbook()
chunks = split_into_chunks(document)

chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True,
)


def retrieve_relevant_chunks(question: str, top_k: int = 2):
    question_embedding = model.encode(
        question,
        convert_to_tensor=True,
    )

    similarities = util.cos_sim(
        question_embedding,
        chunk_embeddings,
    )[0]

    top_results = similarities.topk(
        k=min(top_k, len(chunks))
    )

    retrieved = []

    for score, index in zip(top_results.values, top_results.indices):
        retrieved.append(
            {
                "content": chunks[index.item()],
                "score": round(score.item(), 4),
            }
        )

    return retrieved