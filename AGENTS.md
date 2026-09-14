# AGENTS.md — PayOps Sentinel Backend

This file orients any AI coding assistant (or a human skimming fast) working in
this repo. Read this before touching `agent_graph.py`, `investigation_service.py`,
or the ML/RAG pipeline — several behaviors here are non-obvious and have already
caused real bugs once (see "Gotchas" at the bottom).

## What this is

FastAPI + LangGraph backend for **PayOps Sentinel**, an evidence-grounded agentic
system that investigates *why synthetic payment transactions failed*. It classifies
a question, extracts diagnostic features from transaction evidence, ranks four
candidate root causes with a Random Forest, decides whether that verdict is
confident enough to act on alone, retrieves grounding evidence from a runbook
knowledge base, drafts a report, validates the report against the evidence, and
either returns an autonomous recommendation or forces human escalation. All data —
transactions, ML training examples, runbook documents — is synthetic. No real
cardholder data.

## Related repos (siblings under `/Users/anirban/AI_Projects/`)

| Repo | Role |
|---|---|
| `payment-ai-backend` (this repo) | FastAPI + LangGraph + ML + RAG backend, Supabase persistence |
| `transaction-guardian` | React/TanStack Start frontend ("PayOps Sentinel" UI) — Dashboard, Investigate, Transactions, Knowledge Base, Agent Activity, Audit Logs |
| `payment-simulator` | Standalone FastAPI app that generates synthetic authorizations (single, live-stream, or batch) into the shared Supabase `transactions` table |

All three share **one Supabase project** (Postgres + PostgREST) as the source of truth.
There is no `.env.example` in this repo — required vars are listed below.

## Repo map

```
api.py                    # LIVE entrypoint (uvicorn api:app). All current routes live here.
main.py                   # LEGACY, unused second FastAPI app — see Gotchas. Do not extend this.
investigation_service.py  # Question routing: single-tx lookup / factual answers / root-cause graph
agent_graph.py            # The LangGraph pipeline itself (node wiring, thresholds, evidence grading)
diagnosis_map.py          # Static map: 4 diagnosis keys -> failure domains + services
failure_taxonomy.py       # F01-F09 domain/reason-code taxonomy + lookup helpers
feature_extractor.py      # Transaction list -> 14-key diagnostic feature dict
ml_diagnosis.py           # RandomForestClassifier: trains at import time from Supabase, then predicts
llm_service.py            # All OpenAI calls (gpt-4.1-mini): planning, summary, validation, recommendations
rag_loader.py             # Loads knowledge_documents from Supabase, naive blank-line chunking
rag/retriever.py          # sentence-transformers embeddings + cosine similarity (NOT FAISS — see below)
supabase_store.py         # All Supabase table reads/writes, including pagination past PostgREST's cap
evaluate_ml_model.py      # Offline script: accuracy / confusion matrix on synthetic_ml_dataset.json
evaluate_confidence_coverage.py  # Offline script: coverage-vs-accuracy trade-off across thresholds
synthetic_ml_dataset.json # 400 labeled training incidents (14 features + label), seeds ml_training_examples
supabase/migrations/      # ONE consolidated schema file (20260913_schema.sql) — destructive, drop+recreate
```

## Request flow: `investigation_service.run_investigation(question, transactions?, timezone_offset_minutes)`

Called from `POST /api/investigate`. Routes a question down one of **three** paths,
checked in this order:

1. **Single-transaction lookup** — if the question contains a transaction ID
   matching `SIM-[0-9A-F]{12}` (case-insensitive, then upper-cased), it is looked
   up directly via `supabase_store.get_transaction()`. If found and `status ==
   "FAILED"`, its domain (`failure_domain_code()`) is mapped straight through
   `DIAGNOSIS_MAP` to one cause with probability 1.0 — **no aggregate batch, no
   LangGraph, no LLM call**. If not found, or found-but-not-failed, a short factual
   answer is returned instead. This exists because the aggregate root-cause path
   (below) has no ID-awareness at all and used to dilute a single known failure
   into a 500-1000-row batch, producing an ambiguous verdict for a question that
   had a deterministic answer. **IDs must be exactly `SIM-` + 12 uppercase hex
   chars** (e.g. `SIM-05BCD473606F`) — no hyphen, wrong length, or lowercase input
   will silently fall through to path 2 or 3 instead of matching.
2. **Factual/observational questions** — `classify_question_intent()` keyword-matches
   things like "which merchant," "how many transactions failed," "what response
   code" and answers directly from `build_factual_answer()` over the relevant
   transaction window — no ML, no LLM.
3. **Root-cause investigation** — everything else. Pulls transactions (explicit
   caller-supplied list, or up to `list_transactions(10_000)` from Supabase),
   applies any explicit reason/domain/date scope from the question text
   (`select_root_cause_evidence()`), then invokes `agent_graph.investigation_graph`.

A guardrail (`is_payment_domain_question()`) rejects anything not payment-related
before any of this runs, raising `ValueError("OUT_OF_SCOPE: ...")` which `api.py`
turns into a 422.

Every branch ends by calling `add_history_record()`, which persists a summarized
row to Supabase's `investigations` table — **this happens on every real call**,
including ad-hoc verification calls made from a Python shell. See Gotchas.

## The LangGraph pipeline (`agent_graph.py`)

Real execution order, from the actual `add_node`/`add_edge` wiring — **not** the
order function definitions appear in the file:

```
transaction_analysis (analyze_transactions)
        |
feature_extraction (extract_features)
        |
ml_diagnosis (run_ml_diagnosis)                      <- Random Forest scores 4 causes
        |
diagnosis_assessment (assess_ml_diagnosis)            <- clear/ambiguous DECIDED HERE, early
        |                                                 (top_prob >= 0.70 AND gap >= 0.30 = clear)
response_code_analysis (analyze_response_codes)
        |
investigation_intelligence (investigation_intelligence_agent)  <- consults diagnosis_assessment
        |                                                          + diagnosis_map, does NOT recompute it
rag_retrieval (retrieve_rag_evidence)                 <- path-aware: scopes retrieval to selected causes
        |
   +----+-------------------------+
   |                              |
insufficient_evidence          llm_reasoning (generate_llm_summary)
   |                              |
  END (forced human review)   validation (validate_investigation)
                                  |
                        +---------+---------+
                        |                   |
                  recommendation      validation_failed
                  (recommendation_agent)  (forced human review,
                        |                  canned message)
                       END                  |
                                            END
```

Key facts that are easy to get backwards:

- **`diagnosis_assessment` is the 4th node, not a final step.** It runs right after
  `ml_diagnosis` and *before* the Investigation Intelligence Agent, RAG, reporting,
  or validation. Nothing downstream recomputes it — they all read the same verdict.
- **Clear ≠ "needs human review."** `human_escalation_required = (assessment ==
  "ambiguous")`, set inside `recommendation_agent`. A **clear** verdict is the
  autonomous, no-human-needed path by design (see thresholds below). **Ambiguous**
  is what routes to human review — plus `insufficient_evidence` and
  `validation_failed`, which force escalation via separate early-exit branches.
- **`diagnosis_map.py` is not a graph node.** It's a static lookup table consulted
  from inside `investigation_intelligence_agent` and again inside
  `retrieve_rag_evidence` — never registered via `add_node`.
- Thresholds live in `agent_graph.py`: `MINIMUM_CLEAR_PROBABILITY = 0.70`,
  `MINIMUM_CLEAR_GAP = 0.30`. `grade_evidence_strength()` grades every candidate
  cause (not just the winner) as `strong`/`moderate`/`weak`/`negligible` against the
  uninformative baseline `1 / num_causes`, so an "ambiguous" verdict still
  distinguishes a real contender from noise in the `evidence_map` it produces.

## Diagnosis map (`diagnosis_map.py`)

Four diagnosis keys (a 5th, `unknown`, exists as a fallback with no ML class):

| Key | Failure domains | Services |
|---|---|---|
| `issuer_issue` | F01 | authorization-service |
| `network_switch_issue` | F02, F03 | payment-gateway |
| `merchant_issue` | F04, F05, F06 | (none) |
| `payment_service_issue` | F07, F08, F09 | (none) |

F05/F06 belong **only** to `merchant_issue` — they used to also appear under
`network_switch_issue`, diluting evidence for both hypotheses at once (confirmed
fix: probability gap on a real network-only batch went from 9.5%/ambiguous to
65%/clear). Don't re-add them there.

## Failure taxonomy (`failure_taxonomy.py`)

9 domains, 29 reason codes:

| Domain | Name | Reason codes |
|---|---|---|
| F01 | Issuer decision | F01.01 Insufficient funds · F01.02 Account restricted · F01.03 Limit exceeded · F01.04 Card expired · F01.05 Suspected fraud · F01.06 Issuer decline |
| F02 | Issuer availability | F02.01 Issuer host down · F02.02 Issuer timeout · F02.03 Issuer maintenance |
| F03 | Card-network availability | F03.01 Visa unavailable · F03.02 Mastercard unavailable · F03.03 RuPay unavailable · F03.04 Scheme route timeout |
| F04 | Merchant acceptance | F04.01 Merchant inactive · F04.02 MCC blocked · F04.03 Terminal not enabled |
| F05 | Merchant/acquirer connectivity | F05.01 Merchant network down · F05.02 Acquirer link down |
| F06 | POS/terminal | F06.01 POS offline · F06.02 POS timeout · F06.03 Terminal message invalid |
| F07 | Gateway/processor | F07.01 Gateway down · F07.02 Processor timeout · F07.03 Processing error |
| F08 | Authentication/risk gateway | F08.01 3DS challenge failed · F08.02 Merchant risk block · F08.03 Acquirer risk block |
| F09 | Invalid request | F09.01 Malformed request · F09.02 Missing required field · F09.03 Unsupported currency |

Helpers: `failure_reason_code(txn)`, `failure_domain_code(txn)` (first 3 chars,
`None` if not a recognized domain), `failure_reason_details(code)` (safe defaults,
never raises).

## ML diagnosis (`feature_extractor.py` + `ml_diagnosis.py`)

`extract_diagnostic_features(transactions)` filters to `status == "FAILED"` and
computes exactly 14 keys (order matters — it's a fixed feature vector):
`failure_count`, `code_05_count`, `code_91_count`, `reason_001_count`,
`reason_002_count`, `reason_004_count`, `reason_005_count`,
`authorization_service_failures`, `payment_gateway_failures`,
`unique_affected_merchants`, `issuer_decline_ratio`, `network_failure_ratio`,
`account_issue_ratio`, `merchant_issue_ratio`. Despite the legacy `code_05`/`code_91`
names (holdovers from literal ISO codes), every count is now taxonomy-derived via
`failure_domain_code()`.

`ml_diagnosis.py` trains a `RandomForestClassifier(n_estimators=200,
random_state=42)` **at module import time**, from `list_training_examples()`
(Supabase `ml_training_examples`, seeded from `synthetic_ml_dataset.json`'s 400
records). Class labels: `issuer_issue`, `network_switch_issue`, `merchant_issue`,
`payment_service_issue`. **Importing this module makes a live Supabase call and
retrains from scratch every time** — no persisted model artifact, no caching.

Held-out eval (`evaluate_ml_model.py` / `evaluate_confidence_coverage.py`, 80/20
stratified split, same 200-tree RF): 93.75% overall accuracy; at the 0.70/0.30
threshold, 92.5% coverage at 94.59% accuracy on accepted cases.

**Known ML blind spots:** F07/F09 have no dedicated feature — the model can't
distinguish gateway/processor or invalid-request failures from account issues.
A perfectly pure single-domain batch (ratio = 1.0) is out-of-distribution versus
training data (which tops out around 0.75) and can get a *less* confident verdict
than realistic noisy data — expected behavior, not a bug.

## RAG retrieval (`rag/retriever.py`, `rag_loader.py`, `agent_graph.py`)

**Current mechanism is not a vector database.** `rag_loader.load_runbook()` pulls
every row from Supabase's `knowledge_documents` table and joins their `content`;
`split_into_chunks()` just splits on blank lines. `rag/retriever.py` embeds all
chunks once at import time with `sentence-transformers` (`all-MiniLM-L6-v2`) into
an in-memory tensor, and `retrieve_relevant_chunks(question, top_k)` does a plain
`sentence_transformers.util.cos_sim` + `.topk()` over that tensor — no FAISS, no
scikit-learn nearest-neighbors, nothing persisted past process lifetime.

Path-aware scoping happens **in `agent_graph.py`, not in the retriever**. Node
`retrieve_rag_evidence()` builds a query string from the selected diagnosis paths'
names/descriptions/domains/services, calls `retrieve_relevant_chunks(query,
top_k=8)`, then post-filters with `filter_rag_evidence()` — a keyword-weight
lookup (`PATH_RELEVANCE_TERMS`) that scores each chunk against the selected paths
and keeps only `path_score >= 2`, capped to the top 4.

*(If migrating this to FAISS: the swap point is `rag/retriever.py`'s
`chunk_embeddings`/`cos_sim`/`.topk()` — keep the same `retrieve_relevant_chunks(question, top_k)`
signature so `agent_graph.py`'s path-aware filtering layer needs no changes.)*

## API surface (`api.py` — the live entrypoint; run via `uvicorn api:app`)

CORS: explicit allowlist of 8 localhost/127.0.0.1 origins (ports 8080/8081/3000/5173).
**No auth of any kind** — no API keys, no JWT, nothing gating any route.

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /api/investigate` | Body: `{question, transactions?, timezone_offset_minutes?}`. Runs `investigation_service.run_investigation()`. Out-of-scope questions → 422. |
| `GET /api/investigations?limit=20` | Recent investigation history from Supabase |
| `GET /api/transactions?limit=100` | Persisted synthetic transactions (clamped to 5000; paginated internally past PostgREST's 1000-row default — see Gotchas) |
| `GET /api/knowledge-base` | Runbook documents, from Supabase (not local files — see Gotchas) |

## Environment variables

No `.env.example` exists. Required (set in `.env` and/or `.env.supabase`, both
gitignored — `supabase_store.py` loads both):

- `OPENAI_API_KEY` — used by `llm_service.py` (`gpt-4.1-mini` via `client.responses.create`)
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` — used by `supabase_store.py`; service-role key is server-only, never expose to the frontend

## Gotchas (things that already bit us once)

- **PostgREST silently truncates at 1000 rows** regardless of what `.limit()` asks
  for, unless you paginate with `.range()`. `list_transactions()` and
  `list_transactions_in_window()` in `supabase_store.py` already do this correctly
  — don't revert them to a single `.limit()` call, and don't add a new Supabase
  query elsewhere without the same `.range()` loop.
- **`response_code_analysis` has a specific shape**: `{reason_code: {"count": int,
  "meaning": str, "category": str}}` — keyed by code, not a flat object. The
  frontend iterates it with `Object.entries(...).map(([code, info]) => ...
  formatPath(info.category))`, which throws `Cannot read properties of undefined
  (reading 'replaceAll')` on any other shape. This exact mismatch crashed the
  Investigate page once already — check this shape first if that page white-screens
  after a backend change.
- **Any direct call to `investigation_service.run_investigation()`** (e.g. from a
  Python shell while testing) **writes a real row to the live `investigations`
  table**, indistinguishable from a real user question in the "recent questions"
  UI. If you test this way, capture the returned `investigation_id` and delete it
  from Supabase afterward — don't leave fabricated history behind.
- **`main.py` is a second, separate FastAPI app** with wide-open CORS that calls
  `agent_graph.investigation_graph.invoke()` directly, bypassing
  `investigation_service.py` (no single-tx lookup, no factual shortcuts, no
  history persistence in the same shape). It is not what `uvicorn api:app` serves
  and the README doesn't mention it. Treat it as legacy; don't extend it, and
  don't confuse its output shape with `api.py`'s.
- **`api.py` has a dead local-file knowledge-base loader** (`KNOWLEDGE_BASE_DIR`,
  `load_knowledge_documents()`, ~250 lines) that nothing calls — the live
  `/api/knowledge-base` route uses `supabase_store.list_knowledge_documents()`
  instead. Don't assume the local-file loader is in the request path.
