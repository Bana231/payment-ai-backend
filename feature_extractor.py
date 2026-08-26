from typing import List, Dict, Any


def extract_diagnostic_features(
    transactions: List[Dict[str, Any]],
):
    failed_transactions = [
        txn
        for txn in transactions
        if txn["status"] == "FAILED"
    ]

    failure_count = len(failed_transactions)

    # Response-code features
    code_05_count = sum(
        1
        for txn in failed_transactions
        if txn["response_code"] == "05"
    )

    code_91_count = sum(
        1
        for txn in failed_transactions
        if txn["response_code"] == "91"
    )

    # Reason-code features
    reason_001_count = sum(
        1
        for txn in failed_transactions
        if txn.get("reason_code") == "001"
    )

    reason_002_count = sum(
        1
        for txn in failed_transactions
        if txn.get("reason_code") == "002"
    )

    reason_004_count = sum(
        1
        for txn in failed_transactions
        if txn.get("reason_code") == "004"
    )

    reason_005_count = sum(
        1
        for txn in failed_transactions
        if txn.get("reason_code") == "005"
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
        (reason_004_count + reason_005_count) / failure_count
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