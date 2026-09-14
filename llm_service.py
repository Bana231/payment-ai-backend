# This file is the ONLY place in the whole backend that actually talks to
# the AI model (OpenAI). Every function here builds a big block of plain-
# English instructions (a "prompt"), sends it to the model, and returns
# whatever text comes back. Four different jobs, four different prompts:
# 1. generate_investigation_summary — writes the final report a user reads.
# 2. validate_investigation_summary — a second AI call that double-checks
#    job 1's report against the real evidence, catching invented claims.
# 3. generate_recommendations — writes the "what to do next" checklist.
# 4. plan_investigation — decides which root-cause path(s) to focus on.
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Environment / OpenAI Client
# =========================================================

# Load the OPENAI_API_KEY secret from the .env file, and build one shared
# client object every function below re-uses to actually call the model.
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================================================
# Investigation Summary Agent
# =========================================================

# Writes the human-readable investigation report — the actual paragraphs
# of text a user sees explaining what happened and why. It's given every
# piece of evidence the pipeline has gathered so far and told, in detail,
# what it's allowed and not allowed to claim (see STRICT RULES below).
def generate_investigation_summary(
    question: str,
    observed_transaction_evidence: dict,
    response_code_analysis: dict,
    ml_diagnosis: dict,
    investigation_plan: dict,
    rag_evidence: list,
    # A single pre-computed, guaranteed-correct sentence naming which
    # failure domain (e.g. "F06") actually has the most failures, and its
    # exact share. This is computed once in Python by adding up real
    # counts (see analyze_response_codes in agent_graph.py) — it is not
    # something the LLM has to work out for itself from the scattered
    # per-exact-code FAILURE-REASON ANALYSIS dict below. Optional/None
    # when there is no failure data to summarize.
    dominant_domain_summary: str | None = None,
    # One level MORE specific than dominant_domain_summary above — the
    # exact reason code (e.g. "F06.01", not just "F06") when it alone
    # clearly covers most failures (see analyze_response_codes' Step 3 in
    # agent_graph.py). Also pre-computed by counting, not by the LLM.
    # Optional/None when no single reason code is that dominant.
    sub_root_cause_summary: str | None = None,
    # The TAXONOMY's own definition of the root cause category (e.g.
    # "F08" belongs to "payment_service_issue" per diagnosis_map.py),
    # computed whenever one domain clearly dominates (see
    # analyze_response_codes' Step 4 in agent_graph.py). This is a fixed
    # DEFINITION, not a statistical guess — when present, it is THE
    # answer, even if ml_diagnosis's probabilities point somewhere else.
    # Optional/None when no domain is dominant enough to confirm.
    confirmed_root_cause_summary: str | None = None,
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

FAILURE-REASON ANALYSIS:
{response_code_analysis}

DOMINANT FAILURE DOMAIN (pre-computed, deterministic — do not recompute
or contradict this; it is the ground truth for which domain has the
most failures, already summed from the exact reason codes above):
{dominant_domain_summary or "Not available."}

SUB ROOT CAUSE (pre-computed, deterministic — the exact reason code
within the dominant domain, only given here when it alone clearly
accounts for a majority of failures; treat this as confirmed fact,
not a hypothesis, when present):
{sub_root_cause_summary or "Not available — no single reason code was dominant enough to confirm."}

CONFIRMED ROOT CAUSE PER TAXONOMY (pre-computed from the fixed
diagnosis_map.py DEFINITION of which category a domain belongs to —
NOT a statistical guess. When present, this OVERRIDES the ML
diagnostic signal below: report this category as the confirmed root
cause, even if ML DIAGNOSTIC SIGNAL assigns a higher probability to a
different category. The taxonomy's own definition is not something an
ML probability can outvote):
{confirmed_root_cause_summary or "Not available — no domain was dominant enough to confirm a category via the taxonomy; rely on the ML diagnostic signal below instead."}

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

Failure-reason analysis:
- Observed taxonomy reason counts and approved meanings.

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

- When you describe which failure reason/domain has the largest
  share of failures, your statement MUST match the DOMINANT FAILURE
  DOMAIN line above exactly. Do not name a different domain as the
  biggest one, and do not add up the FAILURE-REASON ANALYSIS numbers
  yourself — the dominant domain has already been calculated for you.

- If SUB ROOT CAUSE above is "Not available...", do not name any
  specific reason code as confirmed or dominant — only the domain-level
  fact is established in that case. If SUB ROOT CAUSE above gives an
  actual code, you MAY call that exact code the confirmed sub root
  cause (this is allowed because it is a counted fact, not an ML
  guess) — but do not name a different code.

- The sub root cause being confirmed does NOT, by itself, confirm the
  overall root CAUSE CATEGORY (issuer/merchant/network/payment_service)
  — that is a separate fact, controlled ONLY by CONFIRMED ROOT CAUSE
  PER TAXONOMY above, not by SUB ROOT CAUSE.

- If CONFIRMED ROOT CAUSE PER TAXONOMY above gives an actual category,
  you MUST report that category as the confirmed root cause — this is
  the one case where the overall category itself (not just a specific
  reason code) may be called "confirmed", because it comes from the
  taxonomy's own fixed definition, not a probability. State it plainly
  even if ML DIAGNOSTIC SIGNAL assigns a higher probability to a
  different category — when they disagree, the taxonomy fact wins and
  the ML signal is mentioned only as a secondary, weaker data point,
  never as a competing "leading hypothesis". If CONFIRMED ROOT CAUSE
  PER TAXONOMY above is "Not available...", do not call any category
  confirmed — fall back to describing the ML signal as a hypothesis,
  same as before.

- Never claim that an alternative diagnosis has been ruled out
  merely because its ML probability is lower.

- A "clear" ML assessment means the leading diagnostic class
  passed the prototype routing thresholds.
  It does NOT mean certainty.

- The investigation plan's evidence_map grades every candidate cause's
  evidence as "strong", "moderate", "weak" or "negligible" (a formal
  probability-based grading against the uninformative baseline, not a
  judgment call). Use these exact grades when characterizing how much
  support a cause has. Do not invent your own qualitative language
  ("some evidence", "fairly likely") for this — use the grade that is
  actually present in evidence_map for that cause.

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
CONFIRMED ROOT CAUSE OVERRIDE (check this BEFORE clear/ambiguous below)
=========================================================

If CONFIRMED ROOT CAUSE PER TAXONOMY above gives an actual category,
that OVERRIDES the clear/ambiguous framing in the two sections below —
follow this section's instructions instead, even if ASSESSMENT says
"ambiguous". This can genuinely happen: the ML model can be unsure
between two categories (e.g. 55% vs 33%, not clearing the routing
threshold) while the taxonomy is still completely clear about which
domain dominates and what category it belongs to. In that situation:

- State the taxonomy-confirmed category as THE root cause, not as one
  of several competing hypotheses.

- You may still mention the ML model's probabilities as additional
  context, but frame them as secondary/supporting detail, not as
  genuine alternative hypotheses still requiring investigation.

- Use the OUTPUT structure below for "a case with a confirmed root
  cause", not the CLEAR CASE or AMBIGUOUS CASE structure.

If CONFIRMED ROOT CAUSE PER TAXONOMY above is "Not available...",
ignore this section entirely and follow CLEAR CASE / AMBIGUOUS CASE
below as normal.


=========================================================
CLEAR CASE
=========================================================

If assessment == "clear" AND no CONFIRMED ROOT CAUSE PER TAXONOMY
override applies:

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

If assessment == "ambiguous" AND no CONFIRMED ROOT CAUSE PER TAXONOMY
override applies:

- Present only selected paths as competing hypotheses.

- Preserve uncertainty.

- Do not combine the competing hypotheses into one cause.

- State each competing hypothesis's evidence_map grade explicitly (e.g.
  "moderate evidence" vs "weak evidence") rather than presenting them as
  equally uncertain — "ambiguous" describes the accept/reject decision, not
  the relative strength of the alternatives, and those are not the same
  thing.

- If SUB ROOT CAUSE above gives an actual code, state it in the
  "Confirmed / Observed Evidence" section as a confirmed fact even in an
  otherwise ambiguous case — the exact reason code being confirmed is a
  separate, counted fact from whether the broader root-cause category
  (issuer/merchant/network/payment_service) is confirmed.


=========================================================
OUTPUT
=========================================================

For a case where CONFIRMED ROOT CAUSE PER TAXONOMY above gives an
actual category (regardless of what ASSESSMENT says):

1. Investigation Summary
2. Confirmed / Observed Evidence
3. Confirmed Root Cause (per taxonomy)
4. Current Assessment


For a CLEAR case (and no taxonomy override applies):

1. Investigation Summary
2. Confirmed / Observed Evidence
3. Leading Root Cause Hypothesis
4. Current Assessment


For an AMBIGUOUS case (and no taxonomy override applies):

1. Investigation Summary
2. Confirmed / Observed Evidence
3. Competing Root Cause Hypotheses
4. Current Assessment
"""

    # Send the whole prompt above to the model and get its written report
    # back as plain text — that's all this line does.
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


# =========================================================
# Validation / Critic Agent
# =========================================================

# A SECOND, separate AI call whose only job is to re-read the report that
# generate_investigation_summary just wrote and check it against the same
# evidence — like a strict editor fact-checking an article before it's
# published. If it finds something unsupported, the pipeline throws the
# report away and escalates to a human instead of showing it to the user.
def validate_investigation_summary(
    question: str,
    llm_summary: str,
    observed_transaction_evidence: dict,
    response_code_analysis: dict,
    ml_diagnosis: dict,
    investigation_plan: dict,
    rag_evidence: list,
    # Same pre-computed "which domain actually has the most failures"
    # fact given to the report-writer above. Handing it to the critic too
    # means it can directly compare the report's own claim against this
    # ground truth, instead of only checking looser things like "is this
    # grounded in some evidence".
    dominant_domain_summary: str | None = None,
    # Same sub-root-cause fact given to the report-writer — lets the
    # critic check the report named the right exact reason code (or
    # correctly said nothing, when none was confirmed) instead of only
    # checking the broader domain-level claim.
    sub_root_cause_summary: str | None = None,
    # Same taxonomy-confirmed-category fact given to the report-writer —
    # lets the critic check the report correctly reported it as
    # confirmed (or correctly said nothing, when no domain was dominant
    # enough) instead of letting a wrong or missing category claim slip
    # through.
    confirmed_root_cause_summary: str | None = None,
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

FAILURE-REASON ANALYSIS:
{response_code_analysis}

DOMINANT FAILURE DOMAIN (pre-computed, deterministic ground truth —
already summed from the exact reason codes above; use this to check
the report's own dominant-domain claim, do not recompute it yourself):
{dominant_domain_summary or "Not available."}

SUB ROOT CAUSE (pre-computed, deterministic — the exact reason code
within the dominant domain, only given here when it alone clearly
accounts for a majority of failures):
{sub_root_cause_summary or "Not available — no single reason code was dominant enough to confirm."}

CONFIRMED ROOT CAUSE PER TAXONOMY (pre-computed from the fixed
diagnosis_map.py DEFINITION of which category a domain belongs to —
NOT a statistical guess. When present, the report SHOULD have reported
this category as confirmed, overriding the ML signal, even if ML
DIAGNOSTIC SIGNAL below assigns a higher probability elsewhere):
{confirmed_root_cause_summary or "Not available — no domain was dominant enough to confirm a category via the taxonomy."}

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

- Presents an ambiguous case as confirmed. NOTE: two exceptions to this
  rule, both deterministic counted facts rather than ML guesses, so
  neither should fail this check:
  (1) the SUB ROOT CAUSE fact — the report may state that exact reason
      code as confirmed even while the broader root-cause category
      stays ambiguous;
  (2) the CONFIRMED ROOT CAUSE PER TAXONOMY fact — when it gives an
      actual category, the report is REQUIRED to state that category
      itself as the confirmed root cause, even overriding a different,
      higher-probability ML category. Only fail this rule if the report
      calls something confirmed that ISN'T backed by either of these
      two facts.

- Names a different reason code as the confirmed sub root cause than
  the SUB ROOT CAUSE fact above, or states one as confirmed when SUB
  ROOT CAUSE above says "Not available".

- Names a different category as the confirmed root cause than the
  CONFIRMED ROOT CAUSE PER TAXONOMY fact above, states one as confirmed
  when that fact says "Not available", OR fails to state ANY category
  as confirmed when that fact DOES give one (i.e. still presenting it
  as merely a "leading hypothesis" among several, when the taxonomy
  fact requires calling it confirmed).

- Creates unnecessary competing hypotheses in a clear case.

- Uses causal certainty stronger than the evidence allows.

- Describes a cause's evidence using a qualitative word ("some evidence",
  "fairly likely", "strongly suggests", etc.) that does not match that
  cause's actual grade in the investigation plan's evidence_map ("strong",
  "moderate", "weak" or "negligible").

- In an ambiguous case, presents the competing hypotheses as equally
  uncertain when their evidence_map grades actually differ.

- Names a different failure domain as the largest/most dominant one
  than the DOMINANT FAILURE DOMAIN fact given above. This is a plain
  factual/arithmetic check, not a judgment call — the correct domain
  has already been calculated by summing the real counts, so any
  report that names a different domain as biggest is simply wrong
  and must fail validation.


=========================================================
CONFIRMED ROOT CAUSE OVERRIDE (check this BEFORE clear/ambiguous below)
=========================================================

If CONFIRMED ROOT CAUSE PER TAXONOMY above gives an actual category,
the clear/ambiguous checks below do NOT apply — instead, the report
should state that category as the confirmed root cause (see the FAIL
rule above). Do not fail a report merely for calling that specific
category confirmed, even in what would otherwise be an ambiguous case.


=========================================================
CLEAR CASE
=========================================================

A clear case (with no taxonomy override) should:

- Have one selected leading hypothesis.
- Describe it as leading or best-supported.
- NOT describe it as confirmed.
- NOT claim other causes are impossible.


=========================================================
AMBIGUOUS CASE
=========================================================

An ambiguous case (with no taxonomy override) must:

- Preserve all selected paths as competing hypotheses.
- Explicitly state that current evidence cannot
  conclusively distinguish them.


Return exactly:

Validation Status: PASS or FAIL
Unsupported Claims: <list unsupported claims, or None>
Validation Reason: <short explanation>
"""

    # The critic's raw answer is just returned as plain text — it's
    # investigation_service.py's job (elsewhere) to check whether this
    # text starts with "VALIDATION STATUS: PASS" or not.
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text


# =========================================================
# Recommendation Agent
# =========================================================

# Writes the final "what should a human do next" checklist, once a report
# has been generated and passed validation. Kept deliberately conservative
# — it can only suggest reviewing/checking things, never taking automatic
# action (see "Do not perform autonomous remediation" etc. below).
def generate_recommendations(
    question: str,
    observed_transaction_evidence: dict,
    response_code_analysis: dict,
    rag_evidence: list,
    llm_summary: str,
    investigation_plan: dict,
    # Same taxonomy-confirmed-category fact used above — so
    # recommendations focus on the confirmed category's own runbook
    # guidance instead of generically saying "investigate every
    # selected path" when the taxonomy has already settled the question.
    confirmed_root_cause_summary: str | None = None,
):
    # Pull the two facts this prompt actually needs out of the larger
    # investigation_plan dictionary.
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

CONFIRMED ROOT CAUSE PER TAXONOMY (pre-computed, deterministic — when
present, this category is already settled; focus recommendations on
it directly instead of treating every selected path as still needing
investigation):
{confirmed_root_cause_summary or "Not available — no domain was dominant enough to confirm a category via the taxonomy."}

FAILURE-REASON ANALYSIS:
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
CONFIRMED ROOT CAUSE OVERRIDE (check this BEFORE clear/ambiguous below)
=========================================================

If CONFIRMED ROOT CAUSE PER TAXONOMY above gives an actual category,
recommend actions for THAT category specifically (using its own
approved runbook guidance from RETRIEVED APPROVED RUNBOOK EVIDENCE),
not a generic "investigate every selected path" list — the taxonomy
has already settled which category this is, regardless of what
ASSESSMENT says.


=========================================================
CLEAR CASE
=========================================================

If assessment == "clear" AND no taxonomy override applies:

- Focus recommendations on the selected path.
- Do not create unrelated investigation paths.
- Do not require human escalation solely because other ML
  classes have non-zero probabilities.


=========================================================
AMBIGUOUS CASE
=========================================================

If assessment == "ambiguous" AND no taxonomy override applies:

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

# Decides WHICH root-cause path(s) the rest of the pipeline should focus
# on investigating, and writes a short plain-English reason why. Note:
# it does NOT decide clear-vs-ambiguous itself — that's already been
# calculated deterministically elsewhere (see agent_graph.py's confidence
# thresholds) before this function is even called; it only explains and
# selects paths consistent with that already-made decision.
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

FAILURE-REASON ANALYSIS:
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
  transaction facts, failure-reason evidence, and ML signals.


Return ONLY valid JSON:

{{
  "assessment": "clear" or "ambiguous",
  "selected_paths": ["diagnosis_key"],
  "reason": "short investigation rationale"
}}
"""

    # Unlike the other 3 functions in this file (which return plain
    # paragraphs of text), this prompt asks the model to return actual
    # JSON — a structured {assessment, selected_paths, reason} object —
    # so the rest of the pipeline can read specific fields out of it
    # instead of having to re-parse free-form prose.
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    raw_output = response.output_text.strip()

    # Everything below is just carefully unpacking and sanity-checking
    # that JSON — the model is instructed to only return "clear" or
    # "ambiguous" and only known diagnosis-map keys, but nothing stops it
    # from occasionally returning something malformed or slightly off, so
    # every value gets validated/cleaned up here before being trusted.
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

        # If the model returned anything other than exactly these two
        # words, fall back to the safer "ambiguous" default.
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

        # Drop any path name the model invented that isn't actually one
        # of the 4 real diagnosis categories in diagnosis_map.py.
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

    # If the model's response wasn't valid JSON at all (rare, but
    # possible), don't crash the whole investigation — just fall back to
    # the safest possible answer: ambiguous, no paths selected, and an
    # honest note explaining what went wrong.
    except json.JSONDecodeError:
        return {
            "assessment": "ambiguous",
            "selected_paths": [],
            "reason": (
                "The Investigation Intelligence Agent "
                "returned invalid structured output."
            ),
        }
