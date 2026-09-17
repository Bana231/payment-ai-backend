# This file defines the actual step-by-step pipeline (the "LangGraph")
# that a root-cause investigation runs through — think of it like a
# flowchart where each function below is one box, and the wiring at the
# very bottom of the file (graph_builder.add_node / add_edge) draws the
# arrows connecting those boxes in the exact order they really run.
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END

from datetime import datetime

from diagnosis_map import get_diagnosis_map
from failure_taxonomy import failure_reason_code, failure_reason_details
from feature_extractor import extract_diagnostic_features
from maintenance_windows import check_window, parse_network_window
from ml_diagnosis import diagnose
from rag.retriever import retrieve_relevant_chunks
from supabase_store import list_transactions

from llm_service import (
    generate_investigation_summary,
    validate_investigation_summary,
    generate_recommendations,
    plan_investigation,
)


# =========================================================
# Shared State
# =========================================================

# This is the one "shopping cart" that gets passed from step to step —
# every node function below reads some of these fields and adds new ones
# as it finishes its job. `total=False` means no field is required to
# exist yet at any given point (early on, most of these are still empty).
class InvestigationState(TypedDict, total=False):
    question: str

    transactions: List[Dict[str, Any]]
    failed_transactions: List[Dict[str, Any]]

    observed_transaction_evidence: Dict[str, Any]

    diagnostic_features: Dict[str, Any]
    ml_diagnosis: Dict[str, Any]

    response_code_counts: Dict[str, int]
    response_code_analysis: Dict[str, Any]

    dominant_failure_codes: List[str]
    root_cause_hypothesis: List[str]

    # Domain-level rollup (see analyze_response_codes): groups the exact
    # reason codes above (e.g. "F06.01") up to their parent domain (e.g.
    # "F06") and totals them, so the LLM steps below get one clear,
    # pre-computed "biggest domain" answer instead of having to add up
    # scattered per-code numbers themselves.
    domain_failure_totals: Dict[str, int]
    dominant_domain: str
    dominant_domain_summary: str

    # Sub-root-cause (see analyze_response_codes): one level MORE specific
    # than dominant_domain above — the exact reason code (e.g. "F06.01",
    # not just "F06") that dominates, but ONLY filled in when one single
    # code clearly accounts for most of the failures. If no single code
    # is that dominant (failures are spread across several reasons), both
    # of these stay None rather than guessing.
    sub_root_cause_code: Optional[str]
    sub_root_cause_summary: Optional[str]

    # Confirmed root cause (see analyze_response_codes' Step 4): the
    # taxonomy's OWN definition of which category the dominant domain
    # belongs to (e.g. "F08" -> "payment_service_issue" per
    # diagnosis_map.py), used to override the ML model's independent
    # guess whenever one domain clearly dominates. This is a fixed
    # DEFINITION, not a statistical estimate, so it takes priority over
    # ml_diagnosis.predicted_cause when both are present.
    confirmed_root_cause: Optional[str]
    confirmed_root_cause_summary: Optional[str]

    # Network maintenance-window check (see check_network_maintenance_window
    # below): whichever card network accounts for a clear majority of
    # failures, and — when RAG similarity search finds that network's
    # documented downtime passage — whether the batch's failure timestamps
    # actually fall inside it. Like confirmed_root_cause above, this is a
    # deterministic, plain-code check, not an LLM judgement call.
    dominant_network: Optional[str]
    dominant_network_share: float
    maintenance_window_check: Dict[str, Any]
    maintenance_window_summary: Optional[str]

    diagnosis_assessment: Dict[str, Any]
    investigation_plan: Dict[str, Any]

    rag_query: str
    rag_evidence: List[Dict[str, Any]]

    llm_summary: str
    validation_result: str
    recommendations: str

    human_escalation_required: bool


# =========================================================
# Coverage-vs-Confidence Policy
#
# This system is a selective classifier with a reject option, not a system
# tuned for speed: rather than always emitting a single root-cause verdict,
# it only commits to one when the ML evidence clears an explicit confidence
# bar (MINIMUM_CLEAR_PROBABILITY, MINIMUM_CLEAR_GAP below), and defers every
# other case to human review. Moving that bar trades coverage (the share of
# investigations the system resolves on its own) against confidence (the
# accuracy of the ones it does resolve) — see evaluate_confidence_coverage.py
# for the empirical coverage/accuracy this specific bar yields on held-out
# data (92.5% coverage, 94.59% accuracy on accepted cases at time of writing).
#
# Why this use case sits on the confidence side of that trade-off: an
# incorrect payment root-cause handed to an on-call engineer as settled fact
# can send remediation in the wrong direction and prolong the real incident —
# a cost that compounds. A short human-review delay on an ambiguous case does
# not. Payment-ops investigation is therefore treated as tolerating delay
# over error: MINIMUM_CLEAR_PROBABILITY/MINIMUM_CLEAR_GAP are set high enough
# that a "clear" verdict should be safe to act on without a human in the
# loop, at the cost of routing more borderline cases to human review than a
# looser bar would.
#
# "Good vs weak evidence" is not itself binary, even though the accept/
# reject decision above is: grade_evidence_strength() below formally grades
# every candidate cause's evidence (not just the top one) against the
# uninformative baseline (1 / number of causes), so an "ambiguous" verdict
# still reports which alternatives are genuinely competitive versus which are
# noise, rather than collapsing all non-accepted causes into one flat label.
# =========================================================

MINIMUM_CLEAR_PROBABILITY = 0.70
MINIMUM_CLEAR_GAP = 0.30

# Evidence-strength bands, expressed as "lift" over the uninformative
# baseline (1 / number of candidate causes) rather than a fixed number, so
# the grading stays meaningful if the number of diagnosis classes changes.
WEAK_EVIDENCE_LIFT = 1.0
MODERATE_EVIDENCE_LIFT = 2.0


def grade_evidence_strength(
    probability: float,
    is_top_cause: bool,
    probability_gap: float,
    num_causes: int,
) -> str:
    """Grade one cause's evidence on a coverage-vs-confidence scale.

    "strong" is exactly the accept region of the coverage/confidence policy
    above (this cause is what a "clear" verdict would select). Everything
    else is graded by how far its probability sits above the uninformative
    baseline, so a rejected case still distinguishes a genuinely competing
    alternative from noise instead of reporting both as equally "ambiguous".
    """

    # This cause is the model's top pick AND clears both confidence bars
    # above — this is the one and only way to earn a "strong" grade.
    if (
        is_top_cause
        and probability >= MINIMUM_CLEAR_PROBABILITY
        and probability_gap >= MINIMUM_CLEAR_GAP
    ):
        return "strong"

    # "Uninformative baseline" = what a purely random guess would score —
    # e.g. with 4 possible causes, randomly guessing gets it right 25% of
    # the time. Everything below compares this cause's real probability
    # against that random-guess baseline instead of against a fixed number.
    baseline = 1 / num_causes if num_causes else 0

    # Barely better (or worse) than a random guess — basically no signal.
    if not baseline or probability <= baseline * WEAK_EVIDENCE_LIFT:
        return "negligible"

    # Better than random, but still well short of "strong" — a real but
    # modest signal.
    if probability <= baseline * MODERATE_EVIDENCE_LIFT:
        return "weak"

    return "moderate"

# =========================================================
# RAG Path Relevance
# =========================================================

# A scoring dictionary used later (see calculate_path_relevance) to decide
# how relevant a retrieved runbook passage is to a specific diagnosis
# path. For each path (e.g. "merchant_issue"), it lists phrases that would
# likely appear in a genuinely relevant passage, each worth a point value
# — the more/stronger phrases a passage contains, the more relevant it's
# considered to that path. This is plain keyword matching, not AI.
PATH_RELEVANCE_TERMS = {
    "merchant_issue": {
        "merchant configuration": 3,
        "merchant authorization": 3,
        "invalid merchant": 3,
        "merchant blocked": 3,
        "merchant unavailable": 3,
        "merchant offline": 3,
        "merchant network down": 3,
        "merchant category": 3,
        "mcc blocked": 3,
        "specific merchant": 2,
        "particular merchant": 2,
        "affected merchant": 2,
        "merchant-specific": 2,
        "across merchants": 1,
    },

    "issuer_issue": {
        "issuer decline": 3,
        "issuer-decline": 3,
        "response code 05": 3,
        "authorization reason": 3,
        "invalid account": 3,
        "account status": 3,
        "account information": 2,
        "issuer-related": 2,
    },

    "network_switch_issue": {
        "network connectivity": 3,
        "switch unavailable": 3,
        "issuer or switch unavailable": 3,
        "issuer unavailable": 3,
        "issuer availability": 3,
        "payment network": 2,
        "network incident": 2,
        "gateway incident": 2,
        "switch availability": 2,
    },

    "payment_service_issue": {
        "payment service": 3,
        "service failure": 3,
        "processing service": 3,
        "processing component": 3,
        "authorization service": 2,
        "internal service": 2,
        "service availability": 2,
    },
}


# =========================================================
# Node 1
# Transaction Analysis
# =========================================================

# First real step of the pipeline: load the transactions to investigate
# (whatever was already put in state, else the latest 500 from Supabase),
# then keep only the FAILED ones — everything downstream only cares about
# failures.
def analyze_transactions(
    state: InvestigationState,
):
    input_transactions = state.get(
        "transactions",
        list_transactions(500),
    )

    failed_transactions = [
        txn
        for txn in input_transactions
        if txn.get("status") == "FAILED"
    ]

    # Running tallies built up one failed transaction at a time below:
    # how many failures per merchant, per reason code, per service.
    merchant_failure_counts = {}
    reason_code_counts = {}
    service_failure_counts = {}

    for txn in failed_transactions:

        merchant = txn.get(
            "merchant",
            "Unknown",
        )

        merchant_failure_counts[merchant] = (
            merchant_failure_counts.get(
                merchant,
                0,
            )
            + 1
        )

        reason_code = txn.get(
            "reason_code"
        )

        if reason_code:
            reason_code_counts[reason_code] = (
                reason_code_counts.get(
                    reason_code,
                    0,
                )
                + 1
            )

        service = txn.get(
            "service",
            "Unknown",
        )

        service_failure_counts[service] = (
            service_failure_counts.get(
                service,
                0,
            )
            + 1
        )

    failure_count = len(
        failed_transactions
    )

    unique_affected_merchants = len(
        merchant_failure_counts
    )

    # Find whichever single merchant has the most failures, and how many
    # — this is the "one merchant is causing most of the pain" signal.
    if merchant_failure_counts:

        most_affected_merchant = max(
            merchant_failure_counts,
            key=merchant_failure_counts.get,
        )

        max_merchant_failure_count = (
            merchant_failure_counts[
                most_affected_merchant
            ]
        )

    else:

        most_affected_merchant = None
        max_merchant_failure_count = 0

    # What fraction of ALL failures belong to that single top merchant —
    # e.g. 0.8 means "80% of failures are this one merchant", a strong
    # hint the problem is merchant-specific rather than network-wide.
    merchant_failure_concentration_ratio = (
        max_merchant_failure_count
        / failure_count
        if failure_count
        else 0
    )

    # Bundle everything computed above into one evidence dictionary that
    # gets handed to the LLM later so it can quote real numbers instead
    # of guessing.
    observed_transaction_evidence = {
        "failure_count":
            failure_count,

        "merchant_failure_counts":
            merchant_failure_counts,

        "unique_affected_merchants":
            unique_affected_merchants,

        "most_affected_merchant":
            most_affected_merchant,

        "max_merchant_failure_count":
            max_merchant_failure_count,

        "merchant_failure_concentration_ratio":
            round(
                merchant_failure_concentration_ratio,
                3,
            ),

        "reason_code_counts":
            reason_code_counts,

        "service_failure_counts":
            service_failure_counts,

        "note": (
            "These values are directly calculated "
            "from the current failed transactions."
        ),
    }

    return {
        "failed_transactions":
            failed_transactions,

        "observed_transaction_evidence":
            observed_transaction_evidence,
    }


# =========================================================
# Node 2
# Feature Extraction
# =========================================================

# Second step: turn the raw failed-transaction list from Node 1 into the
# fixed 14-number feature vector the ML model understands (see
# feature_extractor.py for exactly what each number means).
def extract_features(
    state: InvestigationState,
):
    diagnostic_features = (
        extract_diagnostic_features(
            state.get(
                "failed_transactions",
                [],
            )
        )
    )

    return {
        "diagnostic_features":
            diagnostic_features
    }


# =========================================================
# Node 3
# ML Diagnosis
# =========================================================

# Third step: hand Node 2's 14 numbers to the trained Random Forest (see
# ml_diagnosis.py's diagnose()) and store back whatever it predicts.
def run_ml_diagnosis(
    state: InvestigationState,
):
    ml_result = diagnose(
        state.get(
            "diagnostic_features",
            {},
        )
    )

    return {
        "ml_diagnosis":
            ml_result
    }


# =========================================================
# Node 4
# Quantitative Diagnosis Assessment
# =========================================================

# Fourth step: decide whether the ML model's answer is confident enough to
# treat as "clear" (one obvious cause) or whether it's "ambiguous" (needs
# the LLM to weigh multiple candidate causes). This is the
# coverage-vs-confidence policy check using MINIMUM_CLEAR_PROBABILITY /
# MINIMUM_CLEAR_GAP defined near the top of this file.
def assess_ml_diagnosis(
    state: InvestigationState,
):
    ml_result = state.get(
        "ml_diagnosis",
        {},
    )

    probabilities = ml_result.get(
        "probabilities",
        {},
    )

    top_probability = ml_result.get(
        "top_probability",
        0,
    )

    probability_gap = ml_result.get(
        "probability_gap",
        0,
    )

    # Sort every possible cause from most to least likely so we can pull
    # out "the top 2 candidates" next.
    ranked_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_paths = [
        label
        for label, _
        in ranked_probabilities[:2]
    ]

    top_cause = (
        ranked_probabilities[0][0]
        if ranked_probabilities
        else None
    )

    num_causes = len(probabilities) or 1

    # Grade how strong the evidence is for EACH possible cause (not just
    # the winner) — used later so the report can say "moderate evidence
    # for X" instead of just a bare number.
    evidence_map = {
        label: {
            "probability": probability,
            "evidence_strength": grade_evidence_strength(
                probability,
                is_top_cause=(label == top_cause),
                probability_gap=probability_gap,
                num_causes=num_causes,
            ),
        }
        for label, probability in probabilities.items()
    }

    # The actual policy check: both the top probability AND the gap to
    # the runner-up must clear their thresholds for us to call this
    # "clear" — high confidence alone isn't enough if a close second
    # place cause exists.
    if (
        top_probability
        >= MINIMUM_CLEAR_PROBABILITY
        and probability_gap
        >= MINIMUM_CLEAR_GAP
    ):

        assessment = "clear"

        selected_paths = (
            top_paths[:1]
            if top_paths
            else []
        )

        reason = (
            "ML diagnosis is sufficiently separated "
            "for the prototype clear-path policy: "
            f"top probability={top_probability:.3f}, "
            f"probability gap={probability_gap:.3f}."
        )

    else:

        assessment = "ambiguous"

        selected_paths = top_paths

        reason = (
            "ML diagnosis does not meet the prototype "
            "clear-path thresholds: "
            f"top probability={top_probability:.3f}, "
            f"probability gap={probability_gap:.3f}. "
            "Multiple candidate paths should be investigated."
        )

    return {
        "diagnosis_assessment": {
            "assessment":
                assessment,

            "selected_paths":
                selected_paths,

            "reason":
                reason,

            "minimum_clear_probability":
                MINIMUM_CLEAR_PROBABILITY,

            "minimum_clear_gap":
                MINIMUM_CLEAR_GAP,

            "evidence_map":
                evidence_map,
        }
    }


# =========================================================
# Node 5
# Response Code Analysis
# =========================================================

def analyze_response_codes(
    state: InvestigationState,
):
    # ---------------------------------------------------------------
    # STEP 1: Count how many failed transactions used each exact
    # reason code (e.g. "F06.01", "F08.02"). This is a simple tally —
    # every failed transaction adds 1 to its own reason code's count.
    # ---------------------------------------------------------------
    failed_transactions = state.get(
        "failed_transactions",
        [],
    )

    response_code_counts = {}

    for txn in failed_transactions:

        code = failure_reason_code(txn)

        if not code:
            continue

        response_code_counts[code] = (
            response_code_counts.get(
                code,
                0,
            )
            + 1
        )

    response_code_analysis = {}

    for code, count in (
        response_code_counts.items()
    ):

        code_info = failure_reason_details(code)

        response_code_analysis[code] = {
            "count":
                count,

            "meaning":
                code_info["display_name"],

            "category":
                code_info["domain_name"],
        }

    # ---------------------------------------------------------------
    # STEP 2: Roll the exact reason codes up to their parent DOMAIN
    # (the first 3 characters, e.g. "F06.01" and "F06.02" both belong
    # to domain "F06"). This matters because a domain's failures are
    # often spread across several specific reason codes — e.g. 217
    # "F06.01" + 74 "F06.02" + 4 "F06.03" = 295 total F06 failures.
    # Computing the domain totals here, once, deterministically, and
    # handing them to the LLM directly means it never has to add up
    # scattered per-code numbers itself to figure out which domain is
    # biggest.
    # ---------------------------------------------------------------
    domain_failure_totals: Dict[str, int] = {}

    for code, count in response_code_counts.items():

        domain_code = code[:3]

        domain_failure_totals[domain_code] = (
            domain_failure_totals.get(
                domain_code,
                0,
            )
            + count
        )

    # dominant_domain is simply whichever domain has the highest total
    # above — this is a plain "which number is biggest" calculation
    # done once in Python, so neither the report-writing LLM nor the
    # critic LLM ever has to work it out (and potentially get it wrong).
    if domain_failure_totals:

        dominant_domain = max(
            domain_failure_totals,
            key=domain_failure_totals.get,
        )

    else:

        dominant_domain = None

    if response_code_counts:

        max_count = max(
            response_code_counts.values()
        )

        dominant_failure_codes = [
            code
            for code, count
            in response_code_counts.items()
            if count == max_count
        ]

    else:

        dominant_failure_codes = []

    root_cause_hypothesis = []

    for code in dominant_failure_codes:

        code_info = failure_reason_details(code)
        root_cause_hypothesis.append(
            f"Observed failures are associated with failure reason "
            f"{code}: {code_info['display_name']}."
        )

    # A single, unambiguous plain-English sentence stating the biggest
    # domain and its share of all failures — this is the sentence that
    # gets handed straight to both the report-writer and the critic so
    # there is one deterministic, pre-computed fact they can both check
    # their own claims against, instead of re-deriving it themselves.
    dominant_domain_summary = None
    # Declared here (not just inside the "if dominant_domain" block below)
    # because Step 4 further down also needs this same share number.
    dominant_domain_share = 0

    if dominant_domain:

        total_failures = sum(
            domain_failure_totals.values()
        )

        dominant_domain_share = (
            domain_failure_totals[dominant_domain] /
            total_failures
            if total_failures
            else 0
        )

        dominant_domain_summary = (
            f"Domain {dominant_domain} accounts for "
            f"{domain_failure_totals[dominant_domain]} of "
            f"{total_failures} failed transactions "
            f"({dominant_domain_share * 100:.1f}%), the largest "
            "share of any single domain."
        )

    # ---------------------------------------------------------------
    # STEP 3: Sub-root-cause — one level MORE specific than the domain
    # above. dominant_failure_codes (from Step 2) already names the
    # single EXACT reason code with the most failures (e.g. "F06.01",
    # not just "F06"). We only call this a confirmed "sub root cause"
    # when it's a clean, unambiguous winner: exactly one code tied for
    # the top spot (no tie with another code), AND that one code alone
    # covers at least half of all failures. If failures are spread out
    # instead (no single code dominant enough), this stays None rather
    # than naming a code that isn't actually confirmed by the data.
    # ---------------------------------------------------------------
    sub_root_cause_code = None
    sub_root_cause_summary = None

    if (
        len(dominant_failure_codes) == 1
        and response_code_counts
    ):

        candidate_code = dominant_failure_codes[0]
        total_reason_failures = sum(
            response_code_counts.values()
        )
        candidate_share = (
            response_code_counts[candidate_code] /
            total_reason_failures
            if total_reason_failures
            else 0
        )

        if candidate_share >= 0.5:

            sub_root_cause_code = candidate_code
            candidate_info = failure_reason_details(candidate_code)

            sub_root_cause_summary = (
                f"The specific failure reason {candidate_code} "
                f"({candidate_info['display_name']}) accounts for "
                f"{response_code_counts[candidate_code]} of "
                f"{total_reason_failures} failed transactions "
                f"({candidate_share * 100:.1f}%), a clear majority — "
                "confirmed as the sub root cause, not just a guess."
            )

    # ---------------------------------------------------------------
    # STEP 4: Confirmed root cause, PER YOUR TAXONOMY — not per the ML
    # model's independent guess. The ML classifier (Node 3/4 above) can
    # disagree with what the taxonomy itself says a domain means (e.g. it
    # once favored "issuer_issue" for an F08-dominant batch, even though
    # F08 is defined in diagnosis_map.py as belonging to
    # payment_service_issue) — the ML is a statistical estimate that can
    # be wrong or unreliable at unusual scales, but the taxonomy mapping
    # is a fixed, deterministic DEFINITION, not a guess. So: whenever one
    # domain clearly dominates (same >=50% bar as sub-root-cause above)
    # AND that domain maps to exactly one of the 4 root-cause categories,
    # THAT is reported as the confirmed root cause — overriding, not
    # supplementing, the ML's predicted_cause in the final report. If no
    # domain is that dominant, or it doesn't map to any category, this
    # stays None and the ML signal remains the only available answer.
    # ---------------------------------------------------------------
    confirmed_root_cause = None
    confirmed_root_cause_summary = None

    if dominant_domain and dominant_domain_share >= 0.5:

        diagnosis_map = get_diagnosis_map()

        matching_categories = [
            category_key
            for category_key, category_info in diagnosis_map.items()
            if dominant_domain in category_info.get("failure_domains", [])
        ]

        # Only confirm when the domain maps to EXACTLY one category — if
        # it were somehow in two categories' domain lists at once (not
        # currently possible with today's taxonomy, but checked anyway),
        # that would no longer be an unambiguous answer.
        if len(matching_categories) == 1:

            confirmed_root_cause = matching_categories[0]
            category_name = diagnosis_map[confirmed_root_cause]["name"]

            confirmed_root_cause_summary = (
                f"Domain {dominant_domain} maps to \"{category_name}\" "
                f"in the failure taxonomy (diagnosis_map.py), and "
                f"accounts for {dominant_domain_share * 100:.1f}% of "
                "failures — confirmed as the root cause per the "
                "taxonomy's own definition, regardless of what the ML "
                "model's probabilities alone suggest."
            )

    return {
        "response_code_counts":
            response_code_counts,

        "response_code_analysis":
            response_code_analysis,

        "dominant_failure_codes":
            dominant_failure_codes,

        "root_cause_hypothesis":
            root_cause_hypothesis,

        # New fields carrying the domain-level rollup described above.
        "domain_failure_totals":
            domain_failure_totals,

        "dominant_domain":
            dominant_domain,

        "dominant_domain_summary":
            dominant_domain_summary,

        # New fields carrying the sub-root-cause described above.
        "sub_root_cause_code":
            sub_root_cause_code,

        "sub_root_cause_summary":
            sub_root_cause_summary,

        # New fields carrying the taxonomy-confirmed root cause described
        # above — takes priority over ml_diagnosis.predicted_cause when
        # present (see investigation_service.py's run_investigation).
        "confirmed_root_cause":
            confirmed_root_cause,

        "confirmed_root_cause_summary":
            confirmed_root_cause_summary,
    }


# Same >=50% share bar used everywhere else in this file (sub_root_cause,
# confirmed_root_cause) for calling something "clearly dominant" rather
# than just "the biggest of several similar numbers".
MINIMUM_DOMINANT_NETWORK_SHARE = 0.5


# =========================================================
# Node 5b
# Network Maintenance-Window Check
# =========================================================

# Ties a batch's dominant card network to its documented scheduled
# downtime, deterministically. Two separate jobs, kept separate:
#
#   1. RAG (retrieve_relevant_chunks, now backed by the pgvector
#      knowledge_chunks table — see rag/retriever.py) finds WHICH
#      passage, if any, talks about this network's maintenance window,
#      the same way every other RAG lookup in this file finds relevant
#      knowledge — by embedding similarity, not a hardcoded document
#      lookup.
#   2. maintenance_windows.py turns that passage into structured data and
#      checks the batch's exact failure timestamps against it in plain
#      code — the same reliability standard as confirmed_root_cause
#      above. The LLM is never asked to judge whether a timestamp falls
#      inside a window.
def check_network_maintenance_window(
    state: InvestigationState,
):
    failed_transactions = state.get(
        "failed_transactions",
        [],
    )

    network_failure_counts: Dict[str, int] = {}

    for txn in failed_transactions:

        network = txn.get("network")

        if not network:
            continue

        network_failure_counts[network] = (
            network_failure_counts.get(
                network,
                0,
            )
            + 1
        )

    dominant_network = None
    dominant_network_share = 0.0

    if network_failure_counts:

        dominant_network = max(
            network_failure_counts,
            key=network_failure_counts.get,
        )

        dominant_network_share = (
            network_failure_counts[dominant_network]
            / len(failed_transactions)
        )

    maintenance_window_check: Dict[str, Any] = {}
    maintenance_window_summary = None

    # Only bother looking up a maintenance window when one network
    # clearly dominates the batch — with failures spread across several
    # networks, "the batch is dominated by network X" wouldn't even be a
    # true premise to check a window against.
    if (
        dominant_network
        and dominant_network_share >= MINIMUM_DOMINANT_NETWORK_SHARE
    ):

        rag_hits = retrieve_relevant_chunks(
            f"{dominant_network} scheduled maintenance downtime window",
            top_k=1,
        )

        passage_text = (
            rag_hits[0]["content"]
            if rag_hits
            else ""
        )

        parsed_window = parse_network_window(passage_text)

        # Sanity check: the retrieved passage must actually be ABOUT the
        # dominant network. Similarity search can occasionally surface
        # the closest match even when nothing truly relevant exists —
        # this guards against silently checking timestamps against the
        # wrong network's window.
        if (
            parsed_window
            and parsed_window["network"].lower()
            == dominant_network.lower()
        ):

            in_window_count = 0
            network_failures = 0

            for txn in failed_transactions:

                if txn.get("network") != dominant_network:
                    continue

                network_failures += 1

                raw_created_at = txn.get("created_at")

                if not raw_created_at:
                    continue

                created_at = datetime.fromisoformat(
                    str(raw_created_at).replace("Z", "+00:00")
                )

                if check_window(parsed_window, created_at)["in_window"]:
                    in_window_count += 1

            in_window_share = (
                in_window_count / network_failures
                if network_failures
                else 0
            )

            maintenance_window_check = {
                "network": dominant_network,
                "country": parsed_window["country"],
                "network_failure_count": network_failures,
                "in_window_count": in_window_count,
                "in_window_share": round(in_window_share, 3),
                "source_passage": passage_text,
            }

            if in_window_share >= MINIMUM_DOMINANT_NETWORK_SHARE:

                maintenance_window_summary = (
                    f"{dominant_network} accounts for "
                    f"{dominant_network_share * 100:.1f}% of failed "
                    f"transactions in this batch, and {in_window_count} "
                    f"of those {network_failures} {dominant_network} "
                    f"failures ({in_window_share * 100:.1f}%) occurred "
                    f"during {dominant_network}'s documented scheduled "
                    f"maintenance window for {parsed_window['country']} "
                    "(see retrieved runbook knowledge) — confirmed as a "
                    "contributing factor, not a coincidence."
                )

            else:

                maintenance_window_summary = (
                    f"{dominant_network} accounts for "
                    f"{dominant_network_share * 100:.1f}% of failed "
                    f"transactions in this batch, but only "
                    f"{in_window_count} of {network_failures} "
                    f"{dominant_network} failures "
                    f"({in_window_share * 100:.1f}%) occurred during "
                    f"{dominant_network}'s documented scheduled "
                    "maintenance window — the failures are not "
                    "explained by scheduled maintenance."
                )

    return {
        "dominant_network":
            dominant_network,

        "dominant_network_share":
            round(dominant_network_share, 3),

        "maintenance_window_check":
            maintenance_window_check,

        "maintenance_window_summary":
            maintenance_window_summary,
    }


# =========================================================
# Node 6
# Investigation Intelligence Agent
# =========================================================

# Sixth step: ask an LLM (see llm_service.py's plan_investigation) to
# write a short plan/reasoning for ONLY the candidate cause(s) Node 4
# selected — this keeps the LLM focused instead of reasoning about causes
# that were already ruled out.
def investigation_intelligence_agent(
    state: InvestigationState,
):
    diagnosis_map = (
        get_diagnosis_map()
    )

    assessment = state.get(
        "diagnosis_assessment",
        {},
    )

    selected_paths = assessment.get(
        "selected_paths",
        [],
    )

    # Shrink the full diagnosis map down to just the causes still in play.
    allowed_diagnosis_map = {
        path: diagnosis_map[path]
        for path in selected_paths
        if path in diagnosis_map
    }

    agent_plan = plan_investigation(
        question=
            state["question"],

        observed_transaction_evidence=
            state.get(
                "observed_transaction_evidence",
                {},
            ),

        ml_diagnosis=
            state.get(
                "ml_diagnosis",
                {},
            ),

        response_code_analysis=
            state.get(
                "response_code_analysis",
                {},
            ),

        diagnosis_map=
            allowed_diagnosis_map,
    )

    investigation_plan = {
        "assessment":
            assessment.get(
                "assessment",
                "ambiguous",
            ),

        "selected_paths":
            selected_paths,

        "reason":
            assessment.get(
                "reason",
                "",
            ),

        "agent_reasoning":
            agent_plan.get(
                "reason",
                "",
            ),

        # Formal per-cause evidence grading (coverage-vs-confidence policy),
        # not just the accepted/selected paths, so downstream reporting can
        # describe why a rejected alternative is "moderate" evidence versus
        # "negligible" rather than lumping every non-selected cause together.
        "evidence_map":
            assessment.get(
                "evidence_map",
                {},
            ),
    }

    return {
        "investigation_plan":
            investigation_plan
    }


# =========================================================
# RAG Relevance Calculation
# =========================================================

# Helper (not a graph node itself, called by the RAG nodes below): scores
# one retrieved knowledge-base passage against each candidate diagnosis
# path, using the PATH_RELEVANCE_TERMS keyword weights defined near the
# top of this file — higher score means the passage talks about things
# relevant to that path.
def calculate_path_relevance(
    content: str,
    selected_paths: List[str],
):
    content_lower = content.lower()

    path_scores = {}

    for path in selected_paths:

        terms = PATH_RELEVANCE_TERMS.get(
            path,
            {},
        )

        score = 0
        matched_terms = []

        for phrase, weight in terms.items():

            if phrase.lower() in content_lower:

                score += weight
                matched_terms.append(
                    phrase
                )

        path_scores[path] = {
            "score":
                score,

            "matched_terms":
                matched_terms,
        }

    strongest_score = max(
        (
            result["score"]
            for result
            in path_scores.values()
        ),
        default=0,
    )

    matched_paths = [
        path
        for path, result
        in path_scores.items()
        if result["score"] > 0
    ]

    return {
        "strongest_score":
            strongest_score,

        "matched_paths":
            matched_paths,

        "path_scores":
            path_scores,
    }


# =========================================================
# RAG Filtering
# =========================================================

# Helper (also not a graph node): given every passage the RAG search
# returned, keep only the ones actually relevant to a selected diagnosis
# path (using calculate_path_relevance above) so irrelevant runbook text
# doesn't get fed to the report-writing LLM.
def filter_rag_evidence(
    rag_evidence: List[Dict[str, Any]],
    selected_paths: List[str],
):
    if not selected_paths:
        return rag_evidence

    scored_evidence = []

    for evidence in rag_evidence:

        content = evidence.get(
            "content",
            "",
        )

        relevance = (
            calculate_path_relevance(
                content,
                selected_paths,
            )
        )

        path_score = relevance[
            "strongest_score"
        ]

        # Only keep passages that scored at least 2 points of relevance
        # — below that, a passage is more likely a coincidental keyword
        # match than genuinely useful evidence.
        if path_score >= 2:

            scored_evidence.append(
                {
                    **evidence,

                    "path_relevance_score":
                        path_score,

                    "matched_paths":
                        relevance[
                            "matched_paths"
                        ],
                }
            )

    # If nothing scored well enough to keep, fall back to just the first
    # 4 raw results rather than returning nothing at all.
    if not scored_evidence:
        return rag_evidence[:4]

    # Otherwise sort the kept passages best-first and only pass the top 4
    # along, so the LLM prompt doesn't get flooded with weak evidence.
    scored_evidence.sort(
        key=lambda evidence: (
            evidence.get(
                "path_relevance_score",
                0,
            ),
            evidence.get(
                "score",
                0,
            ),
        ),
        reverse=True,
    )

    return scored_evidence[:4]


# =========================================================
# Node 7
# Path-Aware RAG
# =========================================================

# Seventh step: search the knowledge base (runbooks + policies) for
# passages relevant to whichever diagnosis path(s) are still in play,
# rather than a plain keyword search on the raw question — this is what
# "path-aware" means here.
def retrieve_rag_evidence(
    state: InvestigationState,
):
    question = state[
        "question"
    ]

    investigation_plan = state.get(
        "investigation_plan",
        {},
    )

    selected_paths = (
        investigation_plan.get(
            "selected_paths",
            [],
        )
    )

    diagnosis_map = (
        get_diagnosis_map()
    )

    # Build a short human-readable description of each candidate path
    # (name, description, relevant domains/services) to steer the RAG
    # search query below.
    path_descriptions = []

    for path in selected_paths:

        path_info = diagnosis_map.get(
            path,
            {},
        )

        path_descriptions.append(
            (
                f"Diagnosis path: "
                f"{path_info.get('name', path)}\n"

                f"Description: "
                f"{path_info.get('description', '')}\n"

                f"Relevant failure domains: "
                f"{path_info.get('failure_domains', [])}\n"

                f"Relevant services: "
                f"{path_info.get('services', [])}"
            )
        )

    # If we have candidate paths, build a focused RAG query that names
    # them explicitly; otherwise fall back to just searching the raw
    # question text.
    if path_descriptions:

        focus_text = "\n\n".join(
            path_descriptions
        )

        rag_query = (
            "Original payment investigation question:\n"
            f"{question}\n\n"

            "Candidate diagnostic paths are listed below. "
            "These definitions determine investigation scope "
            "but are NOT evidence that a condition occurred.\n\n"

            f"{focus_text}\n\n"

            "Retrieve approved payment runbook guidance and "
            "reason-code knowledge relevant to these paths."
        )

    else:

        rag_query = question

    # Actually run the similarity search (rag_service.py) for the top 8
    # candidate passages, then narrow those down to the best 4 using the
    # path-relevance filter above.
    raw_rag_evidence = (
        retrieve_relevant_chunks(
            rag_query,
            top_k=8,
        )
    )

    rag_evidence = (
        filter_rag_evidence(
            raw_rag_evidence,
            selected_paths,
        )
    )

    return {
        "rag_query":
            rag_query,

        "rag_evidence":
            rag_evidence,
    }


# =========================================================
# Node 8
# Investigation Reasoning
# =========================================================

def generate_llm_summary(
    state: InvestigationState,
):
    # dominant_domain_summary is the one deterministic, pre-computed
    # sentence naming the biggest failure domain and its exact share
    # (see analyze_response_codes above). Handing it straight to the
    # report-writing LLM means it states the correct dominant domain
    # instead of trying to add up scattered per-code numbers itself.
    llm_summary = (
        generate_investigation_summary(
            question=
                state["question"],

            observed_transaction_evidence=
                state.get(
                    "observed_transaction_evidence",
                    {},
                ),

            response_code_analysis=
                state.get(
                    "response_code_analysis",
                    {},
                ),

            ml_diagnosis=
                state.get(
                    "ml_diagnosis",
                    {},
                ),

            investigation_plan=
                state.get(
                    "investigation_plan",
                    {},
                ),

            rag_evidence=
                state.get(
                    "rag_evidence",
                    [],
                ),

            dominant_domain_summary=
                state.get(
                    "dominant_domain_summary",
                ),

            # Same deterministic-fact idea as dominant_domain_summary,
            # but one level more specific (see analyze_response_codes'
            # Step 3) — only set when a single exact reason code clearly
            # dominates, so the report can confidently name it.
            sub_root_cause_summary=
                state.get(
                    "sub_root_cause_summary",
                ),

            # The taxonomy's OWN definition of the root cause category
            # (see analyze_response_codes' Step 4) — takes priority over
            # the ML model's predicted_cause when present.
            confirmed_root_cause_summary=
                state.get(
                    "confirmed_root_cause_summary",
                ),

            # Deterministic network-downtime-overlap fact (see
            # check_network_maintenance_window above) — same "pre-computed,
            # do not re-derive" treatment as confirmed_root_cause_summary.
            maintenance_window_summary=
                state.get(
                    "maintenance_window_summary",
                ),
        )
    )

    return {
        "llm_summary":
            llm_summary
    }


# =========================================================
# Node 9
# Validation / Critic
# =========================================================

def validate_investigation(
    state: InvestigationState,
):
    # Same dominant_domain_summary fact is also handed to the critic, so
    # it can directly compare "does the report's stated dominant cause
    # match this pre-computed fact" instead of only checking looser
    # things like "is this claim grounded in some evidence".
    validation_result = (
        validate_investigation_summary(
            question=
                state["question"],

            llm_summary=
                state.get(
                    "llm_summary",
                    "",
                ),

            observed_transaction_evidence=
                state.get(
                    "observed_transaction_evidence",
                    {},
                ),

            response_code_analysis=
                state.get(
                    "response_code_analysis",
                    {},
                ),

            ml_diagnosis=
                state.get(
                    "ml_diagnosis",
                    {},
                ),

            investigation_plan=
                state.get(
                    "investigation_plan",
                    {},
                ),

            rag_evidence=
                state.get(
                    "rag_evidence",
                    [],
                ),

            dominant_domain_summary=
                state.get(
                    "dominant_domain_summary",
                ),

            sub_root_cause_summary=
                state.get(
                    "sub_root_cause_summary",
                ),

            confirmed_root_cause_summary=
                state.get(
                    "confirmed_root_cause_summary",
                ),

            maintenance_window_summary=
                state.get(
                    "maintenance_window_summary",
                ),
        )
    )

    return {
        "validation_result":
            validation_result
    }


# =========================================================
# Validation Routing
# =========================================================

# Looks at the critic LLM's verdict text from Node 9 and decides which
# node runs next: if it starts with "VALIDATION STATUS: PASS", the report
# is trustworthy enough to go to recommendation_agent; anything else
# (FAIL, missing, malformed) routes to the validation_failed fallback
# below instead of showing a possibly-wrong report to the user.
def route_after_validation(
    state: InvestigationState,
):
    validation_result = state.get(
        "validation_result",
        "",
    )

    normalized = (
        validation_result
        .strip()
        .upper()
    )

    if normalized.startswith(
        "VALIDATION STATUS: PASS"
    ):
        return "recommendation"

    return "validation_failed"


# =========================================================
# Validation Failure Fallback
# =========================================================

# Dead-end node reached only when the critic rejected the report — instead
# of returning the untrusted LLM summary, this replaces it with a fixed
# "escalate to a human" message so a bad report never reaches the user.
def validation_failed(
    state: InvestigationState,
):
    return {
        "recommendations": (
            "Recommended Actions:\n"
            "1. Escalate the investigation for human review.\n"
            "2. Review the validation findings and collect "
            "additional evidence before relying on the "
            "AI-generated hypothesis.\n\n"
            "Investigation Scope:\n"
            "Assessment: validation_failed\n\n"
            "Guardrail Status: HUMAN_REVIEW_REQUIRED"
        ),

        "human_escalation_required":
            True,
    }


# =========================================================
# Node 10
# Recommendation Agent
# =========================================================

# Final success-path step: ask an LLM (generate_recommendations) for
# concrete next actions, then append a fixed "Investigation Scope" /
# "Guardrail Status: PASS" footer so every successful report ends with
# the same machine-readable summary block.
def recommendation_agent(
    state: InvestigationState,
):
    investigation_plan = state.get(
        "investigation_plan",
        {},
    )

    selected_paths = investigation_plan.get(
        "selected_paths",
        [],
    )

    assessment = investigation_plan.get(
        "assessment",
        "ambiguous",
    )

    recommendation_body = (
        generate_recommendations(
            question=
                state["question"],

            observed_transaction_evidence=
                state.get(
                    "observed_transaction_evidence",
                    {},
                ),

            response_code_analysis=
                state.get(
                    "response_code_analysis",
                    {},
                ),

            rag_evidence=
                state.get(
                    "rag_evidence",
                    [],
                ),

            llm_summary=
                state.get(
                    "llm_summary",
                    "",
                ),

            investigation_plan=
                investigation_plan,

            confirmed_root_cause_summary=
                state.get(
                    "confirmed_root_cause_summary",
                ),

            maintenance_window_summary=
                state.get(
                    "maintenance_window_summary",
                ),
        )
    )

    primary_paths_text = (
        ", ".join(selected_paths)
        if selected_paths
        else "none"
    )

    recommendations = (
        f"{recommendation_body}\n\n"
        f"Investigation Scope:\n"
        f"Primary Paths: {primary_paths_text}\n"
        f"Assessment: {assessment}\n\n"
        f"Guardrail Status: PASS"
    )

    # Even on the success path, still flag "ambiguous" cases for a human
    # to double-check — a clean guardrail pass doesn't mean the ML model
    # was actually confident about a single cause.
    human_escalation_required = (
        assessment == "ambiguous"
    )

    return {
        "recommendations":
            recommendations,

        "human_escalation_required":
            human_escalation_required,
    }


# =========================================================
# Insufficient Evidence Fallback
# =========================================================

# Dead-end node reached when RAG search (Node 7) found nothing usable —
# returns a fixed "not enough evidence, escalate to a human" bundle
# instead of ever calling the report-writing LLM on thin evidence.
def insufficient_evidence(
    state: InvestigationState,
):
    return {
        "llm_summary": (
            "Insufficient relevant knowledge-base "
            "evidence was found to produce a reliable "
            "investigation report."
        ),

        "validation_result": (
            "Validation Status: FAIL\n"
            "Unsupported Claims: None\n"
            "Validation Reason: Available evidence "
            "is insufficient for reliable diagnosis."
        ),

        "recommendations": (
            "Recommended Actions:\n"
            "1. Escalate for human investigation.\n"
            "2. Collect additional transaction, log, "
            "runbook and operational evidence.\n\n"
            "Investigation Scope:\n"
            "Assessment: insufficient_evidence\n\n"
            "Guardrail Status: HUMAN_REVIEW_REQUIRED"
        ),

        "human_escalation_required":
            True,
    }


# =========================================================
# RAG Routing
# =========================================================

# Decides which node runs after RAG retrieval (Node 7): if nothing came
# back, or the single best match is too weak a score (<0.35 similarity),
# go straight to the insufficient_evidence fallback instead of letting the
# report-writing LLM hallucinate off thin evidence.
def route_after_rag(
    state: InvestigationState,
):
    rag_evidence = state.get(
        "rag_evidence",
        [],
    )

    if not rag_evidence:
        return "insufficient_evidence"

    best_score = max(
        evidence.get(
            "score",
            0,
        )
        for evidence
        in rag_evidence
    )

    if best_score >= 0.35:
        return "llm_reasoning"

    return "insufficient_evidence"


# =========================================================
# Build Graph
# =========================================================

# Create the empty LangGraph "flowchart" object, typed to the
# InvestigationState shape defined near the top of this file — every node
# below both reads from and writes into that shared state.
graph_builder = StateGraph(
    InvestigationState
)


# Register every node function above under a short name — this is just
# giving each step of the flowchart a label; it doesn't wire up the
# actual order yet (that's the add_edge calls further down).
graph_builder.add_node(
    "transaction_analysis",
    analyze_transactions,
)

graph_builder.add_node(
    "feature_extraction",
    extract_features,
)

graph_builder.add_node(
    "ml_diagnosis",
    run_ml_diagnosis,
)

graph_builder.add_node(
    "diagnosis_assessment",
    assess_ml_diagnosis,
)

graph_builder.add_node(
    "response_code_analysis",
    analyze_response_codes,
)

graph_builder.add_node(
    "maintenance_window_check",
    check_network_maintenance_window,
)

graph_builder.add_node(
    "investigation_intelligence",
    investigation_intelligence_agent,
)

graph_builder.add_node(
    "rag_retrieval",
    retrieve_rag_evidence,
)

graph_builder.add_node(
    "llm_reasoning",
    generate_llm_summary,
)

graph_builder.add_node(
    "validation",
    validate_investigation,
)

graph_builder.add_node(
    "validation_failed",
    validation_failed,
)

graph_builder.add_node(
    "recommendation",
    recommendation_agent,
)

graph_builder.add_node(
    "insufficient_evidence",
    insufficient_evidence,
)


# =========================================================
# Workflow
# =========================================================

# THIS is the real, authoritative execution order of the whole pipeline:
# START -> transaction_analysis -> feature_extraction -> ml_diagnosis ->
# diagnosis_assessment -> response_code_analysis -> maintenance_window_check
# -> investigation_intelligence -> rag_retrieval -> (branch: llm_reasoning OR
# insufficient_evidence/END) -> validation -> (branch: recommendation/END
# OR validation_failed/END). A plain add_edge is an unconditional "always
# go here next"; add_conditional_edges below picks the next node at
# runtime based on a routing function's return value.
graph_builder.add_edge(
    START,
    "transaction_analysis",
)

graph_builder.add_edge(
    "transaction_analysis",
    "feature_extraction",
)

graph_builder.add_edge(
    "feature_extraction",
    "ml_diagnosis",
)

graph_builder.add_edge(
    "ml_diagnosis",
    "diagnosis_assessment",
)

graph_builder.add_edge(
    "diagnosis_assessment",
    "response_code_analysis",
)

graph_builder.add_edge(
    "response_code_analysis",
    "maintenance_window_check",
)

graph_builder.add_edge(
    "maintenance_window_check",
    "investigation_intelligence",
)

graph_builder.add_edge(
    "investigation_intelligence",
    "rag_retrieval",
)


# Branch point 1: after RAG retrieval, route_after_rag decides whether
# there's enough evidence to continue to llm_reasoning, or whether the
# pipeline should short-circuit straight to the insufficient_evidence
# dead end (which itself leads to END below).
graph_builder.add_conditional_edges(
    "rag_retrieval",
    route_after_rag,
    {
        "llm_reasoning":
            "llm_reasoning",

        "insufficient_evidence":
            "insufficient_evidence",
    },
)


graph_builder.add_edge(
    "llm_reasoning",
    "validation",
)


# Branch point 2: after the critic LLM runs, route_after_validation
# decides whether the report passed (go to recommendation) or failed (go
# to the validation_failed dead end instead).
graph_builder.add_conditional_edges(
    "validation",
    route_after_validation,
    {
        "recommendation":
            "recommendation",

        "validation_failed":
            "validation_failed",
    },
)


# All three possible endings (successful recommendation, validation
# failure, or insufficient evidence) terminate the graph the same way.
graph_builder.add_edge(
    "recommendation",
    END,
)

graph_builder.add_edge(
    "validation_failed",
    END,
)

graph_builder.add_edge(
    "insufficient_evidence",
    END,
)


# =========================================================
# Compile
# =========================================================

# Freeze the flowchart above into a runnable object — this is the single
# object api.py actually calls (as investigation_graph.invoke(...)) to run
# a real investigation end to end.
investigation_graph = (
    graph_builder.compile()
)
