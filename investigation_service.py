import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent_graph import investigation_graph


# =========================================================
# Investigation History Storage
# =========================================================

HISTORY_FILE = Path(
    "investigation_history.json"
)


# =========================================================
# Synthetic Investigation Scenarios
#
# These are prototype transaction groups used when the
# frontend sends only a natural-language question.
#
# Explicit transactions supplied by the API always take
# priority over these scenario datasets.
# =========================================================


AMBIGUOUS_TRANSACTIONS = [
    {
        "transaction_id": "TXN1001",
        "amount": 120.50,
        "currency": "USD",
        "merchant": "Store Alpha",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TXN1003",
        "amount": 220.00,
        "currency": "USD",
        "merchant": "Store Gamma",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "TXN1004",
        "amount": 149.99,
        "currency": "USD",
        "merchant": "Store Delta",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TXN1005",
        "amount": 89.40,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "004",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TXN1006",
        "amount": 315.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "005",
        "service": "authorization-service",
    },
]


ISSUER_TRANSACTIONS = [
    {
        "transaction_id": "ISS2001",
        "amount": 100.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2002",
        "amount": 125.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2003",
        "amount": 80.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2004",
        "amount": 210.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2005",
        "amount": 55.00,
        "currency": "USD",
        "merchant": "Merchant E",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2006",
        "amount": 190.00,
        "currency": "USD",
        "merchant": "Merchant F",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
]


MERCHANT_TRANSACTIONS = [
    {
        "transaction_id": "MER3001",
        "amount": 90.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": "004",
        "service": "payment-gateway",
    },
    {
        "transaction_id": "MER3002",
        "amount": 110.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": "004",
        "service": "payment-gateway",
    },
    {
        "transaction_id": "MER3003",
        "amount": 145.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": "005",
        "service": "payment-gateway",
    },
    {
        "transaction_id": "MER3004",
        "amount": 65.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": "005",
        "service": "payment-gateway",
    },
]


NETWORK_TRANSACTIONS = [
    {
        "transaction_id": "NET4001",
        "amount": 75.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "NET4002",
        "amount": 130.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "NET4003",
        "amount": 245.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "NET4004",
        "amount": 55.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
]


PAYMENT_SERVICE_TRANSACTIONS = [
    {
        "transaction_id": "PAY5001",
        "amount": 60.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "PAY5002",
        "amount": 125.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "PAY5003",
        "amount": 185.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "PAY5004",
        "amount": 220.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": None,
        "service": "payment-gateway",
    },
]


# =========================================================
# Payment-Domain Guardrail
# =========================================================

PAYMENT_DOMAIN_TERMS = {
    "payment",
    "payments",
    "transaction",
    "transactions",
    "authorization",
    "authorisation",
    "issuer",
    "merchant",
    "gateway",
    "switch",
    "decline",
    "declines",
    "failed",
    "failure",
    "failures",
    "response code",
    "reason code",
    "card",
    "incident",
}


def is_payment_domain_question(
    question: str,
) -> bool:
    question_lower = question.lower()

    return any(
        term in question_lower
        for term in PAYMENT_DOMAIN_TERMS
    )


# =========================================================
# Question Intent Classification
#
# We distinguish direct observational questions from
# root-cause investigation questions.
# =========================================================

FACTUAL_RESPONSE_CODE_TERMS = [
    "what response code",
    "which response code",
    "what response codes",
    "which response codes",
    "response codes are causing",
    "response code causing",
    "most common response code",
    "most frequent response code",
    "dominant response code",
    "response code distribution",
]


FACTUAL_MERCHANT_TERMS = [
    "which merchant",
    "most affected merchant",
    "merchant has most",
    "merchant has the most",
    "merchant failure count",
]


FACTUAL_FAILURE_TERMS = [
    "how many failures",
    "how many transactions failed",
    "failure count",
    "number of failures",
]


def classify_question_intent(
    question: str,
) -> str:
    """
    Deterministic prototype intent router.

    Possible values:
        factual_response_codes
        factual_merchants
        factual_failure_count
        root_cause
    """

    question_lower = (
        question
        .lower()
        .strip()
    )

    if any(
        term in question_lower
        for term in FACTUAL_RESPONSE_CODE_TERMS
    ):
        return "factual_response_codes"

    if any(
        term in question_lower
        for term in FACTUAL_MERCHANT_TERMS
    ):
        return "factual_merchants"

    if any(
        term in question_lower
        for term in FACTUAL_FAILURE_TERMS
    ):
        return "factual_failure_count"

    return "root_cause"


# =========================================================
# Response-Code Knowledge
# =========================================================

RESPONSE_CODE_MEANINGS = {
    "05": {
        "meaning": "Do not honor",
        "category": "issuer_decline",
    },
    "91": {
        "meaning": "Issuer or switch unavailable",
        "category": "network_or_issuer_unavailable",
    },
    "12": {
        "meaning": "Invalid transaction",
        "category": "transaction_or_configuration_issue",
    },
}


# =========================================================
# Deterministic Factual Answers
# =========================================================

def build_factual_answer(
    intent: str,
    transactions: List[Dict[str, Any]],
) -> Dict[str, Any]:

    failed_transactions = [
        transaction
        for transaction in transactions
        if str(
            transaction.get(
                "status",
                "",
            )
        ).upper() == "FAILED"
    ]

    failure_count = len(
        failed_transactions
    )

    response_code_counts: Dict[str, int] = {}
    merchant_counts: Dict[str, int] = {}

    for transaction in failed_transactions:

        response_code = str(
            transaction.get(
                "response_code",
                "",
            )
        ).strip()

        if response_code:
            response_code_counts[
                response_code
            ] = (
                response_code_counts.get(
                    response_code,
                    0,
                )
                + 1
            )

        merchant = str(
            transaction.get(
                "merchant",
                "Unknown",
            )
        )

        merchant_counts[
            merchant
        ] = (
            merchant_counts.get(
                merchant,
                0,
            )
            + 1
        )


    # -----------------------------------------------------
    # Factual Response-Code Question
    # -----------------------------------------------------

    if intent == "factual_response_codes":

        if not response_code_counts:
            return {
                "answer": (
                    "No response codes were found in the "
                    "current failed transaction set."
                ),

                "response_code_counts": {},

                "dominant_response_code": None,
            }


        dominant_code = max(
            response_code_counts,
            key=response_code_counts.get,
        )

        dominant_count = (
            response_code_counts[
                dominant_code
            ]
        )

        dominant_ratio = (
            dominant_count / failure_count
            if failure_count
            else 0
        )

        code_info = (
            RESPONSE_CODE_MEANINGS.get(
                dominant_code,
                {
                    "meaning": "Unknown",
                    "category": "unknown",
                },
            )
        )

        return {
            "answer": (
                f"Response code {dominant_code} "
                f"({code_info['meaning']}) is the most frequent "
                f"response code in the current failed transaction set. "
                f"It appears in {dominant_count} of {failure_count} "
                f"failed transactions "
                f"({dominant_ratio * 100:.1f}%). "
                "This is an observed frequency and does not by itself "
                "prove the underlying root cause."
            ),

            "response_code_counts":
                response_code_counts,

            "dominant_response_code": {
                "code":
                    dominant_code,

                "count":
                    dominant_count,

                "ratio":
                    round(
                        dominant_ratio,
                        4,
                    ),

                "meaning":
                    code_info[
                        "meaning"
                    ],

                "category":
                    code_info[
                        "category"
                    ],
            },
        }


    # -----------------------------------------------------
    # Factual Merchant Question
    # -----------------------------------------------------

    if intent == "factual_merchants":

        if not merchant_counts:
            return {
                "answer": (
                    "No affected merchants were found in the "
                    "current failed transaction set."
                ),

                "merchant_failure_counts": {},

                "most_affected_merchant": None,
            }


        most_affected = max(
            merchant_counts,
            key=merchant_counts.get,
        )

        most_affected_count = (
            merchant_counts[
                most_affected
            ]
        )

        concentration = (
            most_affected_count / failure_count
            if failure_count
            else 0
        )

        return {
            "answer": (
                f"{most_affected} is the most affected merchant "
                f"with {most_affected_count} of {failure_count} "
                f"failed transactions "
                f"({concentration * 100:.1f}%). "
                "This concentration is an observed transaction fact "
                "and does not by itself establish a merchant-side "
                "root cause."
            ),

            "merchant_failure_counts":
                merchant_counts,

            "most_affected_merchant":
                most_affected,

            "concentration_ratio":
                round(
                    concentration,
                    4,
                ),
        }


    # -----------------------------------------------------
    # Factual Failure-Count Question
    # -----------------------------------------------------

    if intent == "factual_failure_count":

        return {
            "answer": (
                f"The current transaction set contains "
                f"{failure_count} failed transactions."
            ),

            "failure_count":
                failure_count,
        }


    raise ValueError(
        f"Unsupported factual intent: {intent}"
    )


# =========================================================
# Synthetic Scenario Routing
# =========================================================

def select_synthetic_scenario(
    question: str,
) -> Dict[str, Any]:
    """
    Prototype-only deterministic routing from the natural
    language question to a synthetic transaction scenario.

    Used only for root-cause investigations when explicit
    transactions are not supplied.
    """

    question_lower = (
        question.lower()
    )


    # -----------------------------------------------------
    # Network / Switch
    # -----------------------------------------------------

    network_terms = [
        "switch unavailable",
        "switch issue",
        "network issue",
        "network failure",
        "network connectivity",
        "code 91",
        "response code 91",
        "issuer or switch unavailable",
    ]

    if any(
        term in question_lower
        for term in network_terms
    ):
        return {
            "scenario_id":
                "network_switch_issue",

            "scenario_reason": (
                "Question explicitly references network, "
                "switch or response-code 91 conditions."
            ),

            "transactions":
                NETWORK_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Merchant
    # -----------------------------------------------------

    merchant_terms = [
        "invalid merchant",
        "merchant configuration",
        "merchant issue",
        "merchant problem",
        "specific merchant",
        "store omega",
        "reason code 004",
    ]

    if any(
        term in question_lower
        for term in merchant_terms
    ):
        return {
            "scenario_id":
                "merchant_issue",

            "scenario_reason": (
                "Question explicitly references merchant-specific "
                "or merchant-configuration conditions."
            ),

            "transactions":
                MERCHANT_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Payment Service / Gateway
    # -----------------------------------------------------

    service_terms = [
        "payment service",
        "payment gateway",
        "gateway issue",
        "gateway failure",
        "processing service",
        "internal service",
    ]

    if any(
        term in question_lower
        for term in service_terms
    ):
        return {
            "scenario_id":
                "payment_service_issue",

            "scenario_reason": (
                "Question explicitly references payment-service "
                "or gateway processing conditions."
            ),

            "transactions":
                PAYMENT_SERVICE_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Issuer
    # -----------------------------------------------------

    issuer_terms = [
        "issuer decline",
        "issuer issue",
        "issuer problem",
        "do not honor",
        "do not honour",
        "code 05",
        "response code 05",
        "authorization decline",
        "authorisation decline",
    ]

    if any(
        term in question_lower
        for term in issuer_terms
    ):
        return {
            "scenario_id":
                "issuer_issue",

            "scenario_reason": (
                "Question explicitly references issuer decline "
                "or response-code 05 conditions."
            ),

            "transactions":
                ISSUER_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Generic Root-Cause Investigation
    # -----------------------------------------------------

    return {
        "scenario_id":
            "mixed_ambiguous",

        "scenario_reason": (
            "No specific synthetic failure scenario was requested. "
            "Using the mixed payment-failure dataset."
        ),

        "transactions":
            AMBIGUOUS_TRANSACTIONS,
    }


# =========================================================
# History Helpers
# =========================================================

def load_investigation_history() -> List[Dict[str, Any]]:

    if not HISTORY_FILE.exists():
        return []

    try:
        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            history = json.load(
                file
            )

        if isinstance(
            history,
            list,
        ):
            return history

    except (
        json.JSONDecodeError,
        OSError,
    ):
        pass

    return []


def save_investigation_history(
    history: List[Dict[str, Any]],
) -> None:

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False,
        )


def add_history_record(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    record = {
        "investigation_id":
            result[
                "investigation_id"
            ],

        "created_at":
            result[
                "created_at"
            ],

        "question":
            result[
                "question"
            ],

        "status":
            result[
                "status"
            ],

        "scenario_id":
            result.get(
                "scenario_id"
            ),

        "predicted_cause":
            result[
                "ml_diagnosis"
            ].get(
                "predicted_cause"
            ),

        "assessment":
            result[
                "diagnosis_assessment"
            ].get(
                "assessment"
            ),

        "selected_paths":
            result[
                "diagnosis_assessment"
            ].get(
                "selected_paths",
                [],
            ),

        "top_probability":
            result[
                "ml_diagnosis"
            ].get(
                "top_probability",
                0,
            ),

        "probability_gap":
            result[
                "ml_diagnosis"
            ].get(
                "probability_gap",
                0,
            ),

        "validation_passed":
            result[
                "validation"
            ].get(
                "passed",
                False,
            ),

        "human_escalation_required":
            result.get(
                "human_escalation_required",
                True,
            ),

        "failure_count":
            result[
                "observed_evidence"
            ].get(
                "failure_count",
                0,
            ),
    }

    history = (
        load_investigation_history()
    )

    history.insert(
        0,
        record,
    )

    history = history[:100]

    save_investigation_history(
        history
    )

    return record


def get_investigation_history(
    limit: int = 20,
) -> List[Dict[str, Any]]:

    history = (
        load_investigation_history()
    )

    safe_limit = max(
        1,
        min(
            limit,
            100,
        ),
    )

    return history[
        :safe_limit
    ]


# =========================================================
# Main Investigation Service
# =========================================================

def run_investigation(
    question: str,
    transactions: Optional[
        List[Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:

    # -----------------------------------------------------
    # Domain Guardrail
    # -----------------------------------------------------

    if not is_payment_domain_question(
        question
    ):
        raise ValueError(
            "OUT_OF_SCOPE: This investigation service "
            "only supports payment operations, transaction "
            "failures, merchants, issuers, payment gateways "
            "and payment-network incidents."
        )


    # -----------------------------------------------------
    # Question Intent
    # -----------------------------------------------------

    question_intent = (
        classify_question_intent(
            question
        )
    )


    # -----------------------------------------------------
    # Factual / Observational Question
    #
    # These questions do not require the root-cause ML/RAG
    # pipeline.
    # -----------------------------------------------------

    if (
        question_intent !=
        "root_cause"
    ):

        if transactions is not None:

            factual_transactions = (
                transactions
            )

            scenario_id = (
                "explicit_transaction_input"
            )

        else:

            factual_transactions = (
                AMBIGUOUS_TRANSACTIONS
            )

            scenario_id = (
                "current_mixed_dataset"
            )


        factual_result = (
            build_factual_answer(
                intent=
                    question_intent,

                transactions=
                    factual_transactions,
            )
        )


        investigation_id = (
            "INV-"
            + uuid4()
            .hex[:8]
            .upper()
        )


        created_at = (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )


        return {
            "investigation_id":
                investigation_id,

            "created_at":
                created_at,

            "question":
                question,

            "question_intent":
                question_intent,

            "status":
                "FACTUAL_ANSWER",

            "scenario_id":
                scenario_id,

            "answer":
                factual_result[
                    "answer"
                ],

            "factual_result":
                factual_result,

            "human_escalation_required":
                False,

            "root_cause_analysis_performed":
                False,

            "note": (
                "This question was answered directly from "
                "observed transaction evidence. The ML "
                "root-cause investigation pipeline was not "
                "required."
            ),
        }


    # -----------------------------------------------------
    # Root-Cause Investigation Evidence Source
    # -----------------------------------------------------

    if transactions is not None:

        selected_transactions = (
            transactions
        )

        scenario_id = (
            "explicit_transaction_input"
        )

        scenario_reason = (
            "Transaction records were supplied explicitly "
            "by the caller."
        )

    else:

        scenario = (
            select_synthetic_scenario(
                question
            )
        )

        selected_transactions = (
            scenario[
                "transactions"
            ]
        )

        scenario_id = (
            scenario[
                "scenario_id"
            ]
        )

        scenario_reason = (
            scenario[
                "scenario_reason"
            ]
        )


    # -----------------------------------------------------
    # Graph Input
    # -----------------------------------------------------

    graph_input: Dict[
        str,
        Any,
    ] = {
        "question":
            question,

        "transactions":
            selected_transactions,
    }


    # -----------------------------------------------------
    # Run Complete LangGraph Workflow
    # -----------------------------------------------------

    result = (
        investigation_graph.invoke(
            graph_input
        )
    )


    # -----------------------------------------------------
    # Investigation Identity
    # -----------------------------------------------------

    investigation_id = (
        "INV-"
        + uuid4()
        .hex[:8]
        .upper()
    )


    created_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


    # -----------------------------------------------------
    # ML
    # -----------------------------------------------------

    ml_diagnosis = (
        result.get(
            "ml_diagnosis",
            {},
        )
    )


    probabilities = (
        ml_diagnosis.get(
            "probabilities",
            {},
        )
    )


    # -----------------------------------------------------
    # Diagnosis Assessment
    # -----------------------------------------------------

    diagnosis_assessment = (
        result.get(
            "diagnosis_assessment",
            {},
        )
    )


    assessment = (
        diagnosis_assessment.get(
            "assessment",
            "unknown",
        )
    )


    selected_paths = (
        diagnosis_assessment.get(
            "selected_paths",
            [],
        )
    )


    # -----------------------------------------------------
    # Investigation Plan
    # -----------------------------------------------------

    investigation_plan = (
        result.get(
            "investigation_plan",
            {},
        )
    )


    # -----------------------------------------------------
    # Clean RAG Evidence
    # -----------------------------------------------------

    rag_evidence = []


    for index, evidence in enumerate(
        result.get(
            "rag_evidence",
            [],
        ),
        start=1,
    ):

        rag_evidence.append(
            {
                "id":
                    index,

                "content":
                    evidence.get(
                        "content",
                        "",
                    ),

                "semantic_score":
                    evidence.get(
                        "score",
                        0,
                    ),

                "path_relevance_score":
                    evidence.get(
                        "path_relevance_score",
                        0,
                    ),

                "matched_paths":
                    evidence.get(
                        "matched_paths",
                        [],
                    ),
            }
        )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    human_escalation_required = (
        result.get(
            "human_escalation_required",
            True,
        )
    )


    validation_result = (
        result.get(
            "validation_result",
            "",
        )
    )


    validation_passed = (
        validation_result
        .strip()
        .upper()
        .startswith(
            "VALIDATION STATUS: PASS"
        )
    )


    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    if (
        human_escalation_required
        and
        assessment ==
        "ambiguous"
    ):

        status = (
            "HUMAN_REVIEW_REQUIRED"
        )

    elif (
        human_escalation_required
    ):

        status = (
            "ESCALATED"
        )

    elif (
        assessment ==
        "clear"
    ):

        status = (
            "AI_ASSISTED_DIAGNOSIS"
        )

    else:

        status = (
            "INVESTIGATION_COMPLETE"
        )


    # -----------------------------------------------------
    # Clean API Response
    # -----------------------------------------------------

    response = {
        "investigation_id":
            investigation_id,

        "created_at":
            created_at,

        "question":
            question,

        "question_intent":
            "root_cause",

        "status":
            status,

        # ---------------------------------------------
        # Scenario Metadata
        # ---------------------------------------------

        "scenario_id":
            scenario_id,

        "scenario_reason":
            scenario_reason,

        # ---------------------------------------------
        # Observed Evidence
        # ---------------------------------------------

        "observed_evidence":
            result.get(
                "observed_transaction_evidence",
                {},
            ),

        # ---------------------------------------------
        # ML Diagnosis
        # ---------------------------------------------

        "ml_diagnosis": {
            "predicted_cause":
                ml_diagnosis.get(
                    "predicted_cause"
                ),

            "probabilities":
                probabilities,

            "top_probability":
                ml_diagnosis.get(
                    "top_probability",
                    0,
                ),

            "second_probability":
                ml_diagnosis.get(
                    "second_probability",
                    0,
                ),

            "probability_gap":
                ml_diagnosis.get(
                    "probability_gap",
                    0,
                ),
        },

        # ---------------------------------------------
        # Diagnosis Assessment
        # ---------------------------------------------

        "diagnosis_assessment": {
            "assessment":
                assessment,

            "selected_paths":
                selected_paths,

            "reason":
                diagnosis_assessment.get(
                    "reason",
                    "",
                ),

            "minimum_clear_probability":
                diagnosis_assessment.get(
                    "minimum_clear_probability"
                ),

            "minimum_clear_gap":
                diagnosis_assessment.get(
                    "minimum_clear_gap"
                ),
        },

        # ---------------------------------------------
        # Investigation Intelligence
        # ---------------------------------------------

        "investigation_plan": {
            "selected_paths":
                investigation_plan.get(
                    "selected_paths",
                    [],
                ),

            "agent_reasoning":
                investigation_plan.get(
                    "agent_reasoning",
                    "",
                ),
        },

        # ---------------------------------------------
        # Response Codes
        # ---------------------------------------------

        "response_code_analysis":
            result.get(
                "response_code_analysis",
                {},
            ),

        # ---------------------------------------------
        # RAG Evidence
        # ---------------------------------------------

        "evidence":
            rag_evidence,

        # ---------------------------------------------
        # Investigation Report
        # ---------------------------------------------

        "investigation_report":
            result.get(
                "llm_summary",
                "",
            ),

        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        "validation": {
            "result":
                validation_result,

            "passed":
                validation_passed,
        },

        # ---------------------------------------------
        # Recommendations
        # ---------------------------------------------

        "recommendations":
            result.get(
                "recommendations",
                "",
            ),

        # ---------------------------------------------
        # Human Escalation
        # ---------------------------------------------

        "human_escalation_required":
            human_escalation_required,

        "root_cause_analysis_performed":
            True,
    }


    # -----------------------------------------------------
    # Persist Root-Cause Investigations
    # -----------------------------------------------------

    add_history_record(
        response
    )


    return response