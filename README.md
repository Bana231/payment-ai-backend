# PayOps Sentinel Backend

FastAPI and LangGraph backend for evidence-grounded payment incident investigation. The prototype separates observed transaction facts, Random Forest diagnostic signals, selected investigation hypotheses, retrieved runbook guidance and LLM-generated reporting.

## Pipeline

1. Classify the question as a factual observation or root-cause investigation.
2. Calculate diagnostic features from synthetic transaction evidence.
3. Rank issuer, merchant, network/switch and payment-service classes with a Random Forest.
4. Apply the confidence policy: top probability at least 0.70 and probability gap at least 0.30.
5. Select one clear path or preserve competing ambiguous paths.
6. Retrieve relevant passages from the approved local knowledge base.
7. Generate an evidence-bounded report and validate it with a critic stage.
8. Persist the investigation record and route ambiguous cases to human review.

## Decision policy: coverage vs. confidence, not speed

Step 4 above is a **selective classifier with a reject option**, not a system optimized for a fast single verdict. Rather than always emitting one root-cause answer, it only commits to a cause when the ML evidence clears an explicit bar (`MINIMUM_CLEAR_PROBABILITY = 0.70`, `MINIMUM_CLEAR_GAP = 0.30` in `agent_graph.py`), and routes every other case to human review. Moving that bar trades **coverage** (the share of investigations the system resolves on its own) against **confidence** (the accuracy of the ones it does resolve): a looser bar raises coverage but accepts more wrong autonomous calls; a tighter bar does the opposite. `evaluate_confidence_coverage.py` measures where the current bar sits empirically: 92.5% coverage at 94.59% accuracy on accepted cases, against 93.75% overall accuracy on the full held-out set.

**Why this use case sits on the confidence side of that trade-off.** An incorrect payment root-cause handed to an on-call engineer as settled fact can send remediation in the wrong direction, prolonging the real incident — a cost that compounds over time. A short human-review delay on a genuinely ambiguous case does not compound the same way; it costs a fixed, bounded amount of engineer time. This system is therefore tuned to tolerate delay over error: the thresholds are set high enough that a "clear" verdict should be safe to act on without a human in the loop, at the cost of routing more borderline cases to review than a looser policy would. A different domain where a wrong-but-fast answer is cheaper and a delay is expensive (e.g. a low-stakes content-tagging pipeline) would reasonably sit at a different point on the same curve — the point chosen here is a deliberate property of this domain, not a universal default.

**"Good vs. weak evidence" is graded, not binary, even though the accept/reject decision is.** The accept/reject split at step 4 answers only one question — is the top cause safe to act on alone? It does not describe how strong the *other* candidate causes' evidence is, and collapsing every rejected case into one flat "ambiguous" label was itself a bug: two domains sharing overlapping evidence sources could report as equally "ambiguous" even when one is a real contender and the other is noise. `grade_evidence_strength()` in `agent_graph.py` addresses this by scoring every candidate cause's probability against the uninformative baseline (1 / number of causes), producing a `strong` / `moderate` / `weak` / `negligible` grade per cause (`strong` is exactly the step-4 accept region; the rest is graded by lift over baseline). This full per-cause `evidence_map` is threaded through the LLM prompts, the critic's validation rules, and the API response, so an "ambiguous" verdict still distinguishes a genuinely competing alternative from noise instead of reporting both as equally uncertain.

## Local setup

Create a virtual environment, install the project dependencies, and set `OPENAI_API_KEY` in a local `.env` file. Never commit that file.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

The offline flags avoid optional Hugging Face metadata requests when the sentence-transformer model is already cached.

## Supabase database

Run `supabase/migrations/20260826_initial_schema.sql` in the Supabase SQL Editor, then add these server-only values to `.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

Investigation history is stored in Supabase, and saved synthetic authorization records are available through the transactions endpoint. Never expose the service-role key to the frontend or store real cardholder data.

Run `supabase/migrations/20260831_payment_failure_catalog.sql` to add the shared synthetic failure-code catalog used by the simulator and UI.

For a fresh, destructive rebuild of only the PayOps tables, use `supabase/migrations/20260831_reset_and_rebuild.sql`. It includes the ordered failure taxonomy and the minimum synthetic RAG/ML seed data required for the current backend to start.

Synthetic transactions, runbook documents, and ML training examples are stored in Supabase. Do not commit local copies of those datasets. `ml_training_examples` is seeded from `synthetic_ml_dataset.json` (see "Evaluation and regression checks" below) rather than the minimal placeholder rows in the reset script — reseed it after a fresh rebuild if you want the model to train on the full 400-example set instead of the reset script's bootstrap examples.

## API

- `GET /health`
- `POST /api/investigate`
- `GET /api/investigations?limit=20`
- `GET /api/transactions?limit=100`
- `GET /api/knowledge-base`

## Evaluation and regression checks

```sh
.venv/bin/python evaluate_ml_model.py
.venv/bin/python evaluate_confidence_coverage.py
```

Both scripts are self-contained: they load `synthetic_ml_dataset.json` (400 labelled synthetic incidents, roughly balanced across the 4 classes) directly, split it 80/20 train/test (stratified, `random_state=42`), and report held-out metrics independently of whatever is currently seeded in Supabase. `evaluate_ml_model.py` reports overall accuracy, a per-class classification report, and a confusion matrix. `evaluate_confidence_coverage.py` reports the coverage/accuracy trade-off (see "Decision policy" above) across several candidate thresholds, not just the one currently configured — this is the direct empirical evidence behind that section's claims.

Current held-out evaluation: 93.75% overall accuracy on 80 test incidents. With the configured 0.70 minimum top probability and 0.30 minimum probability gap, coverage is 92.5% and accepted-case accuracy is 94.59%.

`synthetic_ml_dataset.json` is also the source for the live model: it replaces an earlier 8-example placeholder set that only existed to keep the Random Forest bootable, seeded into Supabase's `ml_training_examples` table so the deployed model actually trains on the same 400 examples these scripts evaluate against, not a token bootstrap set.

## Data boundary and limitations

The prototype uses synthetic transactions, generated labelled incidents and a local runbook. It is not validated on production payment data. Model outputs are hypotheses, not confirmed causes; ambiguous or validation-failed cases require human review.

**Known ML limitations:**

- `synthetic_ml_dataset.json`'s incidents are deliberately noisy and realistic — even a `merchant_issue`-labelled incident typically carries some background issuer/network signal (`merchant_issue_ratio` tops out at 0.75 across the whole dataset; median 0.46). A perfectly pure, single-domain batch (`merchant_issue_ratio = 1.0`) is a valid transaction pattern, but it falls outside anything the model was trained on — Random Forests do not extrapolate reliably past the range of their training data, so a "clean" synthetic test batch can get a less confident verdict than a messier, more realistic one would. That is expected behavior on an out-of-distribution input, not a bug, and worth keeping in mind when hand-crafting test batches via the simulator.
- F07 (gateway/processor) and F09 (invalid request) have no dedicated feature in `feature_extractor.py`; only F08 is partially represented, folded into `account_issue_ratio` alongside F01 — a combination that does not correspond to any single diagnosis class's domain list. The model is effectively blind to F07/F09-dominated evidence. Not yet fixed.
- Two related issues were found and fixed during this evaluation pass, both now resolved: `feature_extractor.py`/`diagnosis_map.py` previously double-counted F05/F06 (merchant/acquirer connectivity, POS/terminal) into both `network_switch_issue`'s and `merchant_issue`'s evidence at once, diluting both — confirmed on a real network-only batch: probability gap went from 9.5% (ambiguous) to 65% (clear, correct). Separately, the simulator (`payment-simulator/app.py`) used to write the same `service` value for every transaction, which the model's `authorization_service_failures`/`payment_gateway_failures` features never saw vary across any of the 400 training examples — a genuine train/serve skew, not just a labelling inconsistency. The simulator now assigns a domain-weighted `service` value matching the class-level distribution measured in the training set.
