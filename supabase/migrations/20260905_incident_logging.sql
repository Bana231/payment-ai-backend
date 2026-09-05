-- Run this migration in the Supabase SQL Editor after the existing schema migrations.
-- Incidents are operational records generated from failed synthetic authorizations.

create table if not exists public.incidents (
  incident_id text primary key,
  transaction_id text not null references public.transactions(transaction_id) on delete restrict,
  title text not null,
  status text not null default 'OPEN' check (status in ('OPEN', 'INVESTIGATING', 'RESOLVED')),
  severity text not null default 'SEV2' check (severity in ('SEV1', 'SEV2', 'SEV3', 'SEV4')),
  failure_domain_code text not null references public.payment_failure_domains(domain_code),
  failure_reason_code text references public.payment_failure_reasons(reason_code),
  summary text not null,
  source text not null default 'payment-simulator',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (
    failure_reason_code is null
    or failure_domain_code = left(failure_reason_code, 3)
  )
);

create index if not exists incidents_created_at_idx on public.incidents (created_at desc);
create index if not exists incidents_status_created_at_idx on public.incidents (status, created_at desc);

alter table public.incidents enable row level security;
