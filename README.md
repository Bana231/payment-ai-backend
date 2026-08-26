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

## API

- `GET /health`
- `POST /api/investigate`
- `GET /api/investigations?limit=20`
- `GET /api/knowledge-base`

## Evaluation and regression checks

```sh
.venv/bin/python test_ml_diagnosis.py
.venv/bin/python test_ml_class_coverage.py
.venv/bin/python test_clear_case.py
.venv/bin/python evaluate_ml_model.py
.venv/bin/python evaluate_confidence_coverage.py
```

Current held-out evaluation: 93.75% overall accuracy on 80 test incidents. With the configured 0.70 minimum top probability and 0.30 minimum probability gap, coverage is 92.5% and accepted-case accuracy is 94.59%.

## Data boundary and limitations

The prototype uses synthetic transactions, generated labelled incidents and a local runbook. It is not validated on production payment data. Model outputs are hypotheses, not confirmed causes; ambiguous or validation-failed cases require human review.
