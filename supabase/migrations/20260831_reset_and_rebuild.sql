-- DESTRUCTIVE: resets only PayOps Sentinel tables in the public schema.
-- It does not delete Supabase Auth users, Storage buckets, or unrelated tables.
-- Run this script once in the Supabase SQL Editor after reviewing it.

drop table if exists public.ml_training_examples;
drop table if exists public.knowledge_documents;
drop table if exists public.investigations;
drop table if exists public.transactions;
drop table if exists public.payment_failure_catalog;
drop table if exists public.payment_failure_reasons;
drop table if exists public.payment_failure_domains;

create table public.payment_failure_domains (
  domain_code text primary key,
  display_name text not null,
  description text not null,
  sort_order smallint not null unique check (sort_order between 1 and 99)
);

create table public.payment_failure_reasons (
  reason_code text primary key,
  domain_code text not null references public.payment_failure_domains(domain_code),
  display_name text not null,
  description text not null,
  expected_response_code text,
  retriable boolean not null default false,
  sort_order smallint not null check (sort_order between 1 and 99),
  unique (domain_code, sort_order)
);

create table public.transactions (
  id uuid primary key default gen_random_uuid(),
  transaction_id text not null unique,
  amount numeric(12, 2) not null check (amount >= 0),
  currency text not null check (char_length(currency) = 3),
  merchant text not null,
  network text not null,
  status text not null check (status in ('SUCCESS', 'FAILED')),
  response_code text not null,
  -- Retained temporarily for compatibility with the current backend/API.
  reason_code text,
  failure_domain_code text references public.payment_failure_domains(domain_code),
  failure_reason_code text references public.payment_failure_reasons(reason_code),
  decision_source text not null default 'issuer' check (
    decision_source in ('issuer', 'network_stip', 'merchant', 'acquirer', 'gateway', 'pos', 'risk_engine')
  ),
  risk_source text check (risk_source in ('issuer', 'merchant', 'acquirer', 'network')),
  fallback_used boolean not null default false,
  service text not null,
  environment text not null default 'sandbox' check (environment in ('sandbox', 'production')),
  created_at timestamptz not null default now(),
  check (
    failure_reason_code is null
    or failure_domain_code = left(failure_reason_code, 3)
  )
);

create index transactions_created_at_idx on public.transactions (created_at desc);
create index transactions_failure_reason_idx on public.transactions (failure_reason_code);
create index transactions_network_created_at_idx on public.transactions (network, created_at desc);

create table public.investigations (
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

create index investigations_created_at_idx on public.investigations (created_at desc);

create table public.knowledge_documents (
  slug text primary key,
  title text not null,
  category text not null,
  tags jsonb not null default '[]'::jsonb,
  content text not null,
  updated_at timestamptz not null default now()
);

create table public.ml_training_examples (
  id uuid primary key default gen_random_uuid(),
  label text not null,
  features jsonb not null,
  created_at timestamptz not null default now()
);

insert into public.payment_failure_domains (
  domain_code, display_name, description, sort_order
) values
  ('F01', 'Issuer decision', 'Issuer authorization or account decision.', 1),
  ('F02', 'Issuer availability', 'Issuer host, processor, or response availability.', 2),
  ('F03', 'Card-network availability', 'Card scheme or network-route availability.', 3),
  ('F04', 'Merchant acceptance', 'Merchant acceptance, category, or terminal configuration.', 4),
  ('F05', 'Merchant / acquirer connectivity', 'Merchant or acquirer connectivity failures.', 5),
  ('F06', 'POS / terminal', 'Local point-of-sale or terminal failures.', 6),
  ('F07', 'Gateway / processor', 'Payment-gateway or processor failures.', 7),
  ('F08', 'Authentication / risk gateway', 'Pre-authorization authentication or risk blocks.', 8),
  ('F09', 'Invalid request', 'Malformed or unsupported authorization request.', 9);

insert into public.payment_failure_reasons (
  reason_code, domain_code, display_name, description, expected_response_code, retriable, sort_order
) values
  ('F01.01', 'F01', 'Insufficient funds', 'Issuer declined because available funds were insufficient.', '51', false, 1),
  ('F01.02', 'F01', 'Account restricted', 'Issuer account restriction or card status block.', '62', false, 2),
  ('F01.03', 'F01', 'Limit exceeded', 'Issuer approval amount or velocity limit exceeded.', '61', false, 3),
  ('F01.04', 'F01', 'Card expired', 'Issuer card expiry validation failed.', '54', false, 4),
  ('F01.05', 'F01', 'Suspected fraud', 'Issuer risk controls declined a suspected fraudulent transaction.', '59', false, 5),
  ('F01.06', 'F01', 'Issuer decline', 'Generic issuer authorization decline.', '05', false, 6),
  ('F02.01', 'F02', 'Issuer host down', 'Issuer authorization host is unavailable.', '91', true, 1),
  ('F02.02', 'F02', 'Issuer timeout', 'Issuer did not respond within the authorization timeout.', '91', true, 2),
  ('F02.03', 'F02', 'Issuer maintenance', 'Issuer authorization service is unavailable for maintenance.', '91', true, 3),
  ('F03.01', 'F03', 'Visa unavailable', 'Visa scheme route is unavailable.', '91', true, 1),
  ('F03.02', 'F03', 'Mastercard unavailable', 'Mastercard scheme route is unavailable.', '91', true, 2),
  ('F03.03', 'F03', 'RuPay unavailable', 'RuPay scheme route is unavailable.', '91', true, 3),
  ('F03.04', 'F03', 'Scheme route timeout', 'Network route response timed out.', '68', true, 4),
  ('F04.01', 'F04', 'Merchant inactive', 'Merchant is inactive or disabled for authorization.', '58', false, 1),
  ('F04.02', 'F04', 'MCC blocked', 'Merchant category is not permitted for the card or product.', '57', false, 2),
  ('F04.03', 'F04', 'Terminal not enabled', 'Terminal is not enabled for the authorization type.', '58', false, 3),
  ('F05.01', 'F05', 'Merchant network down', 'Merchant internet or local network connection is unavailable.', '96', true, 1),
  ('F05.02', 'F05', 'Acquirer link down', 'Acquirer connectivity path is unavailable.', '91', true, 2),
  ('F06.01', 'F06', 'POS offline', 'Local point-of-sale terminal is offline.', '68', true, 1),
  ('F06.02', 'F06', 'POS timeout', 'POS authorization request timed out.', '68', true, 2),
  ('F06.03', 'F06', 'Terminal message invalid', 'Terminal submitted malformed authorization data.', '12', false, 3),
  ('F07.01', 'F07', 'Gateway down', 'Payment gateway service is unavailable.', '96', true, 1),
  ('F07.02', 'F07', 'Processor timeout', 'Payment processor did not respond in time.', '68', true, 2),
  ('F07.03', 'F07', 'Processing error', 'Internal payment-processing error occurred.', '96', true, 3),
  ('F08.01', 'F08', '3DS challenge failed', 'Strong customer authentication challenge failed.', null, false, 1),
  ('F08.02', 'F08', 'Merchant risk block', 'Merchant risk rules blocked the transaction.', null, false, 2),
  ('F08.03', 'F08', 'Acquirer risk block', 'Acquirer risk rules blocked the transaction.', null, false, 3),
  ('F09.01', 'F09', 'Malformed request', 'Authorization request structure is invalid.', '12', false, 1),
  ('F09.02', 'F09', 'Missing required field', 'Authorization request is missing required information.', '12', false, 2),
  ('F09.03', 'F09', 'Unsupported currency', 'Authorization currency is not supported.', '12', false, 3);

-- Minimal synthetic knowledge base required by the RAG loader.
insert into public.knowledge_documents (slug, title, category, tags, content) values
  ('issuer-decision', 'Issuer decision runbook', 'Runbook', '["issuer", "decline"]', 'Analyze issuer declines by response code, account pattern, card network, and merchant spread. Do not retry hard issuer declines automatically.'),
  ('issuer-availability', 'Issuer and network availability runbook', 'Runbook', '["issuer", "network", "stip"]', 'For issuer or scheme outages, segment failures by network and issuer. Record whether stand-in processing approved transactions on the issuer''s behalf.'),
  ('merchant-pos', 'Merchant and POS failure runbook', 'Runbook', '["merchant", "pos", "mcc"]', 'Check merchant status, MCC restrictions, terminal enablement, local POS connectivity, and acquirer links.'),
  ('gateway-requests', 'Gateway and request failure runbook', 'Runbook', '["gateway", "processor", "request"]', 'Check gateway availability, processor latency, malformed messages, and missing request fields.'),
  ('risk-authentication', 'Risk and authentication runbook', 'Runbook', '["risk", "3ds", "fraud"]', 'Separate issuer fraud decisions from merchant or acquirer risk blocks and authentication failures.');

-- Minimal synthetic examples keep the current Random Forest bootable until
-- the expanded feature extractor and training set are updated in the next phase.
insert into public.ml_training_examples (label, features) values
  ('issuer_issue', '{"failure_count":20,"code_05_count":18,"code_91_count":0,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":18,"payment_gateway_failures":2,"unique_affected_merchants":8,"issuer_decline_ratio":0.9,"network_failure_ratio":0,"account_issue_ratio":0,"merchant_issue_ratio":0}'::jsonb),
  ('issuer_issue', '{"failure_count":16,"code_05_count":14,"code_91_count":0,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":15,"payment_gateway_failures":1,"unique_affected_merchants":7,"issuer_decline_ratio":0.875,"network_failure_ratio":0,"account_issue_ratio":0,"merchant_issue_ratio":0}'::jsonb),
  ('network_switch_issue', '{"failure_count":20,"code_05_count":0,"code_91_count":19,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":1,"payment_gateway_failures":19,"unique_affected_merchants":10,"issuer_decline_ratio":0,"network_failure_ratio":0.95,"account_issue_ratio":0,"merchant_issue_ratio":0}'::jsonb),
  ('network_switch_issue', '{"failure_count":16,"code_05_count":0,"code_91_count":14,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":2,"payment_gateway_failures":14,"unique_affected_merchants":8,"issuer_decline_ratio":0,"network_failure_ratio":0.875,"account_issue_ratio":0,"merchant_issue_ratio":0}'::jsonb),
  ('merchant_issue', '{"failure_count":18,"code_05_count":0,"code_91_count":0,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":1,"payment_gateway_failures":17,"unique_affected_merchants":1,"issuer_decline_ratio":0,"network_failure_ratio":0,"account_issue_ratio":0,"merchant_issue_ratio":0.94}'::jsonb),
  ('merchant_issue', '{"failure_count":15,"code_05_count":0,"code_91_count":1,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":1,"payment_gateway_failures":14,"unique_affected_merchants":1,"issuer_decline_ratio":0,"network_failure_ratio":0.067,"account_issue_ratio":0,"merchant_issue_ratio":0.87}'::jsonb),
  ('payment_service_issue', '{"failure_count":18,"code_05_count":2,"code_91_count":3,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":8,"payment_gateway_failures":10,"unique_affected_merchants":9,"issuer_decline_ratio":0.11,"network_failure_ratio":0.17,"account_issue_ratio":0,"merchant_issue_ratio":0}'::jsonb),
  ('payment_service_issue', '{"failure_count":14,"code_05_count":1,"code_91_count":2,"reason_001_count":0,"reason_002_count":0,"reason_004_count":0,"reason_005_count":0,"authorization_service_failures":7,"payment_gateway_failures":7,"unique_affected_merchants":8,"issuer_decline_ratio":0.07,"network_failure_ratio":0.14,"account_issue_ratio":0,"merchant_issue_ratio":0}'::jsonb);

alter table public.payment_failure_domains enable row level security;
alter table public.payment_failure_reasons enable row level security;
alter table public.transactions enable row level security;
alter table public.investigations enable row level security;
alter table public.knowledge_documents enable row level security;
alter table public.ml_training_examples enable row level security;
