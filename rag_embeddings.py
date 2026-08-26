from sentence_transformers import SentenceTransformer, util

from rag_loader import load_runbook, split_into_chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

document = load_runbook()
chunks = split_into_chunks(document)

chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True,
)

question = "What should I check if the issuer or switch is unavailable?"

question_embedding = model.encode(
    question,
    convert_to_tensor=True,
)

similarities = util.cos_sim(
    question_embedding,
    chunk_embeddings,
)[0]

best_match_index = similarities.argmax().item()
best_chunk = chunks[best_match_index]
best_score = similarities[best_match_index].item()

print("Question:")
print(question)

print("\nBest matching chunk:")
print(best_chunk)

print("\nSimilarity score:")
print(round(best_score, 4))