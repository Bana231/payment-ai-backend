-- Shared synthetic payment-failure catalog for the simulator and investigation UI.
-- Run in the Supabase SQL Editor after the initial schema migration.

create table if not exists public.payment_failure_catalog (
  reason_code text primary key,
  response_code text not null,
  display_name text not null,
  category text not null,
  retriable boolean not null default false
);

insert into public.payment_failure_catalog (
  reason_code, response_code, display_name, category, retriable
) values
  ('issuer_decline', '05', 'Issuer decline', 'issuer', false),
  ('issuer_down', '91', 'Issuer unavailable', 'issuer', true),
  ('merchant_offline', '58', 'Merchant unavailable', 'merchant', false),
  ('pos_net_down', '68', 'Local POS network down', 'network', true),
  ('merchant_net_down', '96', 'Merchant network down', 'merchant', true),
  ('mcc_blocked', '57', 'Merchant category not allowed', 'merchant', false),
  ('fraud_decline', '59', 'Suspected fraud', 'issuer', false),
  ('invalid_txn', '12', 'Invalid transaction', 'payment_service', false)
on conflict (reason_code) do update set
  response_code = excluded.response_code,
  display_name = excluded.display_name,
  category = excluded.category,
  retriable = excluded.retriable;

alter table public.payment_failure_catalog enable row level security;
