from investigation_service import run_investigation


# =========================================================
# Scenario 1
# Ambiguous Merchant vs Issuer
#
# Expected:
# - ambiguous
# - merchant_issue + issuer_issue
# - human review required
# =========================================================

AMBIGUOUS_TRANSACTIONS = [
    {
        "transaction_id": "AMB1001",
        "amount": 120.50,
        "currency": "USD",
        "merchant": "Store Alpha",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "AMB1002",
        "amount": 220.00,
        "currency": "USD",
        "merchant": "Store Gamma",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "AMB1003",
        "amount": 149.99,
        "currency": "USD",
        "merchant": "Store Delta",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "AMB1004",
        "amount": 89.40,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "004",
        "service": "authorization-service",
    },
    {
        "transaction_id": "AMB1005",
        "amount": 315.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "005",
        "service": "authorization-service",
    },
]


# =========================================================
# Scenario 2
# Clear Issuer Issue
#
# Expected:
# - clear
# - issuer_issue only
# - no human escalation
# =========================================================

CLEAR_ISSUER_TRANSACTIONS = [
    {
        "transaction_id": "ISS1001",
        "amount": 100.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS1002",
        "amount": 125.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS1003",
        "amount": 80.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS1004",
        "amount": 210.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS1005",
        "amount": 55.00,
        "currency": "USD",
        "merchant": "Merchant E",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS1006",
        "amount": 190.00,
        "currency": "USD",
        "merchant": "Merchant F",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
]


# =========================================================
# Pretty Scenario Output
# =========================================================

def print_scenario_result(
    name: str,
    result: dict,
):
    ml = result.get(
        "ml_diagnosis",
        {},
    )

    assessment = result.get(
        "diagnosis_assessment",
        {},
    )

    plan = result.get(
        "investigation_plan",
        {},
    )

    validation = result.get(
        "validation",
        {},
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"SCENARIO: {name}"
    )

    print(
        "=" * 70
    )

    print(
        "\nSTATUS:"
    )

    print(
        result.get(
            "status"
        )
    )


    print(
        "\nML PREDICTED CAUSE:"
    )

    print(
        ml.get(
            "predicted_cause"
        )
    )


    print(
        "\nML PROBABILITIES:"
    )

    probabilities = ml.get(
        "probabilities",
        {},
    )

    for cause, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"  {cause}: "
            f"{probability * 100:.1f}%"
        )


    print(
        "\nTOP PROBABILITY:"
    )

    print(
        f"{ml.get('top_probability', 0) * 100:.1f}%"
    )


    print(
        "\nPROBABILITY GAP:"
    )

    print(
        f"{ml.get('probability_gap', 0) * 100:.1f}%"
    )


    print(
        "\nASSESSMENT:"
    )

    print(
        assessment.get(
            "assessment"
        )
    )


    print(
        "\nSELECTED PATHS:"
    )

    print(
        assessment.get(
            "selected_paths"
        )
    )


    print(
        "\nINTELLIGENCE AGENT REASONING:"
    )

    print(
        plan.get(
            "agent_reasoning"
        )
    )


    print(
        "\nVALIDATION PASSED:"
    )

    print(
        validation.get(
            "passed"
        )
    )


    print(
        "\nHUMAN ESCALATION:"
    )

    print(
        result.get(
            "human_escalation_required"
        )
    )


    print(
        "\nRECOMMENDATIONS:"
    )

    print(
        result.get(
            "recommendations"
        )
    )


# =========================================================
# Run Scenario 1
# =========================================================

ambiguous_result = run_investigation(
    question=(
        "Why are these payment "
        "transactions failing?"
    ),
    transactions=AMBIGUOUS_TRANSACTIONS,
)


print_scenario_result(
    "Ambiguous Merchant vs Issuer",
    ambiguous_result,
)


# =========================================================
# Run Scenario 2
# =========================================================

issuer_result = run_investigation(
    question=(
        "Why are these payment "
        "transactions failing?"
    ),
    transactions=CLEAR_ISSUER_TRANSACTIONS,
)


print_scenario_result(
    "Clear Issuer Issue",
    issuer_result,
)


# =========================================================
# Automated Assertions
#
# These prove the routing behaviour.
# =========================================================

print(
    "\n"
    + "=" * 70
)

print(
    "AUTOMATED CHECKS"
)

print(
    "=" * 70
)


# ---------------------------------------------------------
# Ambiguous Case
# ---------------------------------------------------------

ambiguous_assessment = (
    ambiguous_result[
        "diagnosis_assessment"
    ]
)

assert (
    ambiguous_assessment[
        "assessment"
    ]
    == "ambiguous"
), (
    "Ambiguous scenario was "
    "not classified as ambiguous."
)


assert set(
    ambiguous_assessment[
        "selected_paths"
    ]
) == {
    "merchant_issue",
    "issuer_issue",
}, (
    "Ambiguous scenario did not "
    "select Merchant + Issuer paths."
)


assert (
    ambiguous_result[
        "human_escalation_required"
    ]
    is True
), (
    "Ambiguous scenario should "
    "require human review."
)


# ---------------------------------------------------------
# Clear Issuer Case
# ---------------------------------------------------------

issuer_assessment = (
    issuer_result[
        "diagnosis_assessment"
    ]
)


assert (
    issuer_assessment[
        "assessment"
    ]
    == "clear"
), (
    "Issuer scenario was "
    "not classified as clear."
)


assert (
    issuer_assessment[
        "selected_paths"
    ]
    == ["issuer_issue"]
), (
    "Clear issuer scenario did not "
    "select only issuer_issue."
)


assert (
    issuer_result[
        "human_escalation_required"
    ]
    is False
), (
    "Clear issuer scenario should "
    "not require human escalation."
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert (
    ambiguous_result[
        "validation"
    ][
        "passed"
    ]
    is True
), (
    "Ambiguous scenario report "
    "failed validation."
)


assert (
    issuer_result[
        "validation"
    ][
        "passed"
    ]
    is True
), (
    "Issuer scenario report "
    "failed validation."
)


print(
    "\nPASS:"
)

print(
    "Ambiguous case -> "
    "multi-path investigation + "
    "human review"
)


print(
    "PASS:"
)

print(
    "Clear issuer case -> "
    "single-path investigation + "
    "no unnecessary escalation"
)


print(
    "PASS:"
)

print(
    "Both generated reports passed "
    "the validation guardrail"
)


print(
    "\nALL END-TO-END CHECKS PASSED"
)