-- Run after 20260826_initial_schema.sql.
-- Stores synthetic ML training examples and RAG source documents outside the repository.

create table if not exists public.knowledge_documents (
  slug text primary key,
  title text not null,
  category text not null,
  tags jsonb not null default '[]'::jsonb,
  content text not null,
  updated_at timestamptz not null default now()
);

create table if not exists public.ml_training_examples (
  id uuid primary key default gen_random_uuid(),
  label text not null,
  features jsonb not null,
  created_at timestamptz not null default now()
);

alter table public.knowledge_documents enable row level security;
alter table public.ml_training_examples enable row level security;
