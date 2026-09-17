-- Adds pgvector-backed storage for knowledge-base chunk embeddings, so
-- rag/retriever.py can do real vector similarity search in Postgres
-- instead of recomputing an in-memory brute-force scan on every request.
--
-- ADDITIVE ONLY: unlike 20260913_schema.sql, this does not drop or modify
-- knowledge_documents (or anything else) — it only adds a new extension,
-- a new table, and a new function. Every statement below is guarded (if
-- not exists / or replace), so this file is safe to re-run.
--
-- Run this in the Supabase SQL Editor after 20260913_schema.sql.

-- pgvector: adds the `vector` column type and the <=> distance operator
-- used below. This is a Postgres extension, not a separate hosted
-- service — no new credentials are needed beyond the existing
-- SUPABASE_SERVICE_ROLE_KEY.
create extension if not exists vector;

-- One row per chunk of one knowledge_documents row's content (the same
-- chunks rag_loader.split_into_chunks already produces today). Deleting
-- a knowledge document cascades to delete its chunks, so a document
-- removed via the Knowledge Base page's delete button never leaves
-- orphaned embeddings behind.
create table if not exists public.knowledge_chunks (
  id bigint generated always as identity primary key,
  document_slug text not null references public.knowledge_documents(slug) on delete cascade,
  chunk_index int not null,
  content text not null,
  -- all-MiniLM-L6-v2 (the embedding model rag/retriever.py already uses
  -- today) produces 384-dimensional embeddings.
  embedding vector(384) not null,
  created_at timestamptz not null default now(),
  unique (document_slug, chunk_index)
);

create index if not exists knowledge_chunks_document_slug_idx
  on public.knowledge_chunks (document_slug);

-- No ANN index (ivfflat/hnsw) on the embedding column: at a few dozen
-- chunks total, an exact nearest-neighbor scan is simpler, always exact,
-- and plenty fast. Worth adding only if the knowledge base grows much
-- larger than it is today.

-- Similarity search, exposed as a Postgres function so it can be called
-- as a plain RPC from the existing supabase-py client
-- (supabase.rpc("match_knowledge_chunks", {...})) — no raw SQL access
-- needed from the app, and no new credentials.
--
-- Score is `1 - cosine_distance`, matching the scale
-- sentence_transformers.util.cos_sim already returns today (1.0 =
-- identical), so nothing downstream that reads a chunk's "score" has to
-- change.
create or replace function public.match_knowledge_chunks(
  query_embedding vector(384),
  match_count int default 8
)
returns table (
  content text,
  score float
)
language sql stable
as $$
  select
    knowledge_chunks.content,
    1 - (knowledge_chunks.embedding <=> query_embedding) as score
  from public.knowledge_chunks
  order by knowledge_chunks.embedding <=> query_embedding
  limit match_count;
$$;
