from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from data import transactions as default_transactions
from data import response_codes
from diagnosis_map import get_diagnosis_map
from feature_extractor import extract_diagnostic_features
from ml_diagnosis import diagnose
from rag.retriever import retrieve_relevant_chunks

from llm_service import (
    generate_investigation_summary,
    validate_investigation_summary,
    generate_recommendations,
    plan_investigation,
)


# =========================================================
# Shared State
# =========================================================

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

    diagnosis_assessment: Dict[str, Any]
    investigation_plan: Dict[str, Any]

    rag_query: str
    rag_evidence: List[Dict[str, Any]]

    llm_summary: str
    validation_result: str
    recommendations: str

    human_escalation_required: bool


# =========================================================
# Prototype ML Policy
# =========================================================

MINIMUM_CLEAR_PROBABILITY = 0.70
MINIMUM_CLEAR_GAP = 0.30


# =========================================================
# RAG Path Relevance
# =========================================================

PATH_RELEVANCE_TERMS = {
    "merchant_issue": {
        "merchant configuration": 3,
        "merchant authorization": 3,
        "invalid merchant": 3,
        "merchant blocked": 3,
        "specific merchant": 2,
        "particular merchant": 2,
        "affected merchant": 2,
        "merchant-specific": 2,
        "across merchants": 1,
    },

    "issuer_issue": {
        "issuer decline": 3,
        "issuer-decline": 3,
        "do not honor": 3,
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
        "response code 91": 3,
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

def analyze_transactions(
    state: InvestigationState,
):
    input_transactions = state.get(
        "transactions",
        default_transactions,
    )

    failed_transactions = [
        txn
        for txn in input_transactions
        if txn.get("status") == "FAILED"
    ]

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

    merchant_failure_concentration_ratio = (
        max_merchant_failure_count
        / failure_count
        if failure_count
        else 0
    )

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
        }
    }


# =========================================================
# Node 5
# Response Code Analysis
# =========================================================

def analyze_response_codes(
    state: InvestigationState,
):
    failed_transactions = state.get(
        "failed_transactions",
        [],
    )

    response_code_counts = {}

    for txn in failed_transactions:

        code = txn.get(
            "response_code"
        )

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

        code_info = response_codes.get(
            code,
            {
                "meaning":
                    "Unknown response code",

                "category":
                    "unknown",
            },
        )

        response_code_analysis[code] = {
            "count":
                count,

            "meaning":
                code_info["meaning"],

            "category":
                code_info["category"],
        }

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

        code_info = response_codes.get(
            code
        )

        if code_info:

            root_cause_hypothesis.append(
                f"Observed failures are associated "
                f"with response code {code}: "
                f"{code_info['meaning']}."
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
    }


# =========================================================
# Node 6
# Investigation Intelligence Agent
# =========================================================

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
    }

    return {
        "investigation_plan":
            investigation_plan
    }


# =========================================================
# RAG Relevance Calculation
# =========================================================

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

    if not scored_evidence:
        return rag_evidence[:4]

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

                f"Relevant response codes: "
                f"{path_info.get('response_codes', [])}\n"

                f"Relevant services: "
                f"{path_info.get('services', [])}"
            )
        )

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
        )
    )

    return {
        "validation_result":
            validation_result
    }


# =========================================================
# Validation Routing
# =========================================================

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

graph_builder = StateGraph(
    InvestigationState
)


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
    "investigation_intelligence",
)

graph_builder.add_edge(
    "investigation_intelligence",
    "rag_retrieval",
)


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

investigation_graph = (
    graph_builder.compile()
)