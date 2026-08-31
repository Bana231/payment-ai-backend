from typing import List, Dict, Any

from failure_taxonomy import failure_domain_code


def extract_diagnostic_features(
    transactions: List[Dict[str, Any]],
):
    failed_transactions = [
        txn
        for txn in transactions
        if txn["status"] == "FAILED"
    ]

    failure_count = len(failed_transactions)

    # Legacy feature names remain stable for the existing trained model, but
    # their values are derived from the approved F01–F09 taxonomy—not ISO
    # response codes such as 91 or 68.
    code_05_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F01"
    )

    # F05/F06 (merchant/acquirer connectivity, POS/terminal) are deliberately
    # excluded here and counted only in reason_005_count below. Counting them
    # in both this network-side feature and the merchant-side one diluted
    # evidence for both hypotheses whenever F05/F06 dominated a batch.
    code_91_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) in {"F02", "F03"}
    )

    # Reason-code features
    reason_001_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F01"
    )

    reason_002_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F08"
    )

    reason_004_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F04"
    )

    reason_005_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) in {"F05", "F06"}
    )

    # Service-level features
    authorization_service_failures = sum(
        1
        for txn in failed_transactions
        if txn["service"] == "authorization-service"
    )

    payment_gateway_failures = sum(
        1
        for txn in failed_transactions
        if txn["service"] == "payment-gateway"
    )

    # Merchant spread
    unique_affected_merchants = len(
        {
            txn["merchant"]
            for txn in failed_transactions
        }
    )

    # Ratios
    issuer_decline_ratio = (
        code_05_count / failure_count
        if failure_count
        else 0
    )

    network_failure_ratio = (
        code_91_count / failure_count
        if failure_count
        else 0
    )

    account_issue_ratio = (
        (reason_001_count + reason_002_count) / failure_count
        if failure_count
        else 0
    )

    merchant_issue_ratio = (
        (
            reason_004_count
            + reason_005_count
        ) / failure_count
        if failure_count
        else 0
    )

    return {
        "failure_count": failure_count,

        "code_05_count": code_05_count,
        "code_91_count": code_91_count,

        "reason_001_count": reason_001_count,
        "reason_002_count": reason_002_count,
        "reason_004_count": reason_004_count,
        "reason_005_count": reason_005_count,

        "authorization_service_failures":
            authorization_service_failures,

        "payment_gateway_failures":
            payment_gateway_failures,

        "unique_affected_merchants":
            unique_affected_merchants,

        "issuer_decline_ratio":
            issuer_decline_ratio,

        "network_failure_ratio":
            network_failure_ratio,

        "account_issue_ratio":
            account_issue_ratio,

        "merchant_issue_ratio":
            merchant_issue_ratio,
    }
