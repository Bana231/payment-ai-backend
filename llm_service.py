import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Environment / OpenAI Client
# =========================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================================================
# Investigation Summary Agent
# =========================================================

def generate_investigation_summary(
    question: str,
    observed_transaction_evidence: dict,
    response_code_analysis: dict,
    ml_diagnosis: dict,
    investigation_plan: dict,
    rag_evidence: list,
):
    assessment = investigation_plan.get(
        "assessment",
        "ambiguous",
    )

    selected_paths = investigation_plan.get(
        "selected_paths",
        [],
    )

    prompt = f"""
You are a payment incident investigation reporting agent.

Original question:
{question}

OBSERVED TRANSACTION EVIDENCE:
{observed_transaction_evidence}

RESPONSE-CODE ANALYSIS:
{response_code_analysis}

ML DIAGNOSTIC SIGNAL:
{ml_diagnosis}

INVESTIGATION PLAN:
{investigation_plan}

SELECTED DIAGNOSIS PATHS:
{selected_paths}

ASSESSMENT:
{assessment}

RETRIEVED RUNBOOK KNOWLEDGE:
{rag_evidence}


=========================================================
SOURCE BOUNDARIES
=========================================================

Observed transaction evidence:
- Directly calculated facts from current transactions.

Response-code analysis:
- Observed response-code counts and approved meanings.

ML diagnostic signal:
- Model estimates only.
- NOT confirmed root cause evidence.

Investigation plan / Diagnosis Map:
- Defines investigation scope and hypotheses.
- NOT evidence that a condition occurred.

RAG knowledge:
- Approved domain knowledge and investigation guidance.
- NOT evidence that the current incident exhibits a condition.


=========================================================
STRICT RULES
=========================================================

- Do not invent payment-domain explanations.

- Do not introduce causes such as credit limits, fraud rules,
  issuer policies, risk controls, authentication problems,
  cardholder behavior, or similar explanations unless they
  explicitly exist in supplied evidence.

- Never convert an ML probability into a confirmed root cause.

- Never treat Diagnosis Map definitions as observed evidence.

- Never treat investigation guidance as proof.

- Never claim merchant concentration unless observed
  transaction evidence supports it.

- Never claim that an alternative diagnosis has been ruled out
  merely because its ML probability is lower.

- A "clear" ML assessment means the leading diagnostic class
  passed the prototype routing thresholds.
  It does NOT mean certainty.

- Use wording such as:
  "leading hypothesis"
  "current evidence is most consistent with"
  "lower model support"
  "not currently selected as a primary path"

- Do NOT use wording such as:
  "confirmed"
  "proven"
  "definitely"
  "rules out"
  "cannot be"
  unless direct evidence genuinely establishes it.

- Do not claim that the system fixed the payment issue.


=========================================================
CLEAR CASE
=========================================================

If assessment == "clear":

- Discuss only the selected diagnosis path as the
  leading root-cause hypothesis.

- Do NOT manufacture competing hypotheses from every
  non-zero ML probability.

- You may mention that other classes received lower model
  probabilities, but do not claim they have been excluded.

- The selected cause remains a hypothesis requiring
  evidence-based confirmation.


=========================================================
AMBIGUOUS CASE
=========================================================

If assessment == "ambiguous":

- Present only selected paths as competing hypotheses.

- Preserve uncertainty.

- Do not combine the competing hypotheses into one cause.


=========================================================
OUTPUT
=========================================================

For a CLEAR case:

1. Investigation Summary
2. Confirmed / Observed Evidence
3. Leading Root Cause Hypothesis
4. Current Assessment


For an AMBIGUOUS case:

1. Investigation Summary
2. Confirmed / Observed Evidence
3. Competing Root Cause Hypotheses
4. Current Assessment
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


# =========================================================
# Validation / Critic Agent
# =========================================================

def validate_investigation_summary(
    question: str,
    llm_summary: str,
    observed_transaction_evidence: dict,
    response_code_analysis: dict,
    ml_diagnosis: dict,
    investigation_plan: dict,
    rag_evidence: list,
):
    assessment = investigation_plan.get(
        "assessment",
        "ambiguous",
    )

    selected_paths = investigation_plan.get(
        "selected_paths",
        [],
    )

    prompt = f"""
You are a strict evidence-grounding critic for a payment
incident investigation system.

Original question:
{question}

REPORT TO VALIDATE:
{llm_summary}

OBSERVED TRANSACTION EVIDENCE:
{observed_transaction_evidence}

RESPONSE-CODE ANALYSIS:
{response_code_analysis}

ML DIAGNOSTIC SIGNAL:
{ml_diagnosis}

INVESTIGATION PLAN:
{investigation_plan}

SELECTED PATHS:
{selected_paths}

ASSESSMENT:
{assessment}

RETRIEVED RAG KNOWLEDGE:
{rag_evidence}


=========================================================
VALIDATION PRINCIPLE
=========================================================

Every factual or causal statement must be traceable to the
supplied evidence.

Absence of supporting evidence for an alternative cause is
NOT proof that the alternative cause is impossible.


=========================================================
MARK VALIDATION FAIL IF THE REPORT
=========================================================

- Invents transaction facts.

- Invents domain explanations.

- Adds unsupported causes such as credit limits,
  fraud controls, risk rules, issuer policies,
  authentication issues or cardholder behavior.

- Treats ML probability as confirmed truth.

- Says a lower-probability class has been ruled out
  without direct supporting evidence.

- Uses statements such as:
  "rather than merchant/network/service problems"
  when those alternatives have not actually been excluded.

- Treats Diagnosis Map definitions as evidence.

- Treats RAG guidance as evidence that a condition occurred.

- Claims merchant concentration without quantitative support.

- Presents an ambiguous case as confirmed.

- Creates unnecessary competing hypotheses in a clear case.

- Uses causal certainty stronger than the evidence allows.


=========================================================
CLEAR CASE
=========================================================

A clear case should:

- Have one selected leading hypothesis.
- Describe it as leading or best-supported.
- NOT describe it as confirmed.
- NOT claim other causes are impossible.


=========================================================
AMBIGUOUS CASE
=========================================================

An ambiguous case must:

- Preserve all selected paths as competing hypotheses.
- Explicitly state that current evidence cannot
  conclusively distinguish them.


Return exactly:

Validation Status: PASS or FAIL
Unsupported Claims: <list unsupported claims, or None>
Validation Reason: <short explanation>
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


# =========================================================
# Recommendation Agent
# =========================================================

def generate_recommendations(
    question: str,
    observed_transaction_evidence: dict,
    response_code_analysis: dict,
    rag_evidence: list,
    llm_summary: str,
    investigation_plan: dict,
):
    selected_paths = investigation_plan.get(
        "selected_paths",
        [],
    )

    assessment = investigation_plan.get(
        "assessment",
        "ambiguous",
    )

    prompt = f"""
You are the Recommendation Agent for a payment incident
investigation and reporting system.

Question:
{question}

OBSERVED TRANSACTION EVIDENCE:
{observed_transaction_evidence}

SELECTED DIAGNOSIS PATHS:
{selected_paths}

ASSESSMENT:
{assessment}

RESPONSE-CODE ANALYSIS:
{response_code_analysis}

RETRIEVED APPROVED RUNBOOK EVIDENCE:
{rag_evidence}

INVESTIGATION REPORT:
{llm_summary}


=========================================================
STRICT RULES
=========================================================

- Recommend only investigation actions explicitly supported
  by retrieved runbook evidence.

- Prioritize selected diagnosis paths.

- Do not invent procedures.

- Do not introduce a new primary diagnosis path.

- Do not recommend contacting issuers, customers, merchants,
  vendors, cardholders or external support unless retrieved
  evidence explicitly instructs that action.

- Do not recommend checking credit limits, fraud controls,
  risk rules or other conditions unless retrieved evidence
  explicitly supports them.

- Do not state that a condition exists merely because the
  runbook recommends checking it.

- Phrase checks as:
  "Review..."
  "Check whether..."
  "Compare..."

- Do not perform autonomous remediation.

- Do not restart services.

- Do not modify production configuration.

- Do not retry transactions automatically.


=========================================================
CLEAR CASE
=========================================================

If assessment == "clear":

- Focus recommendations on the selected path.
- Do not create unrelated investigation paths.
- Do not require human escalation solely because other ML
  classes have non-zero probabilities.


=========================================================
AMBIGUOUS CASE
=========================================================

If assessment == "ambiguous":

- Investigate all selected paths.
- Recommend human review if ambiguity persists.


Return ONLY:

Recommended Actions:
1. ...
2. ...
3. ...
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text.strip()


# =========================================================
# Investigation Intelligence Agent
# =========================================================

def plan_investigation(
    question: str,
    observed_transaction_evidence: dict,
    ml_diagnosis: dict,
    response_code_analysis: dict,
    diagnosis_map: dict,
):
    prompt = f"""
You are the Investigation Intelligence Agent for a payment
incident investigation system.

Question:
{question}

OBSERVED TRANSACTION EVIDENCE:
{observed_transaction_evidence}

ML DIAGNOSTIC OUTPUT:
{ml_diagnosis}

RESPONSE-CODE ANALYSIS:
{response_code_analysis}

AVAILABLE DIAGNOSIS MAP:
{diagnosis_map}


Your role is to explain why the supplied candidate paths
should be investigated.

The clear-versus-ambiguous decision has already been made
quantitatively outside this agent.

Do NOT override it.


=========================================================
RULES
=========================================================

- Diagnosis Map definitions are not observed evidence.

- ML probabilities are diagnostic signals, not proof.

- Do not use words such as:
  confirmed,
  proven,
  definitely,
  certain
  when describing a root cause.

- A high or separated probability means a path is suitable
  for prioritized investigation, not confirmed causality.

- Do not infer a condition merely because it appears in
  a Diagnosis Map definition.

- Do not invent generic payment-domain causes.

- Do not perform remediation.

- Do not declare final root cause.

- Explain the investigation rationale using actual observed
  transaction facts, response-code evidence, and ML signals.


Return ONLY valid JSON:

{{
  "assessment": "clear" or "ambiguous",
  "selected_paths": ["diagnosis_key"],
  "reason": "short investigation rationale"
}}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    raw_output = response.output_text.strip()

    try:
        result = json.loads(
            raw_output
        )

        assessment = result.get(
            "assessment",
            "ambiguous",
        )

        selected_paths = result.get(
            "selected_paths",
            [],
        )

        reason = result.get(
            "reason",
            "",
        )

        if assessment not in [
            "clear",
            "ambiguous",
        ]:
            assessment = "ambiguous"

        if not isinstance(
            selected_paths,
            list,
        ):
            selected_paths = []

        valid_paths = [
            path
            for path in selected_paths
            if path in diagnosis_map
        ]

        return {
            "assessment": assessment,
            "selected_paths": valid_paths,
            "reason": reason,
        }

    except json.JSONDecodeError:
        return {
            "assessment": "ambiguous",
            "selected_paths": [],
            "reason": (
                "The Investigation Intelligence Agent "
                "returned invalid structured output."
            ),
        }