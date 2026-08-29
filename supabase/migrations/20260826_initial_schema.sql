-- Run this in the Supabase SQL Editor before starting the backend.
-- All records are synthetic demonstration data. Do not store cardholder data.

create table if not exists public.transactions (
  id uuid primary key default gen_random_uuid(),
  transaction_id text not null unique,
  amount numeric(12, 2) not null check (amount >= 0),
  currency text not null check (char_length(currency) = 3),
  merchant text not null,
  status text not null,
  response_code text not null,
  reason_code text,
  service text not null,
  network text,
  environment text not null default 'sandbox',
  created_at timestamptz not null default now()
);

create index if not exists transactions_created_at_idx on public.transactions (created_at desc);

create table if not exists public.investigations (
  investigation_id text primary key,
  created_at timestamptz not null,
  question text not null,
  status text not null,
  scenario_id text,
  predicted_cause text,
  assessment text,
  selected_paths jsonb not null default '[]'::jsonb,
  top_probability double precision not null default 0,
  probability_gap double precision not null default 0,
  validation_passed boolean not null default false,
  human_escalation_required boolean not null default true,
  failure_count integer not null default 0
);

create index if not exists investigations_created_at_idx on public.investigations (created_at desc);

-- This server uses the service-role key. Keep RLS enabled for browser clients.
alter table public.transactions enable row level security;
alter table public.investigations enable row level security;
