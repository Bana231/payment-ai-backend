from typing import List, Dict, Any

from failure_taxonomy import failure_domain_code


# This is the "translator" between raw transaction rows and the 14 numbers
# the ML model actually understands. It never predicts anything itself —
# it just counts and calculates ratios from the transactions it's given,
# then hands that fixed-shape dictionary of 14 numbers to ml_diagnosis.py.
def extract_diagnostic_features(
    transactions: List[Dict[str, Any]],
):
    # Only failed transactions matter for diagnosing a failure pattern —
    # successful ones are ignored for every count/ratio below.
    failed_transactions = [
        txn
        for txn in transactions
        if txn["status"] == "FAILED"
    ]

    failure_count = len(failed_transactions)

    # Legacy feature names remain stable for the existing trained model, but
    # their values are derived from the approved F01–F09 taxonomy—not ISO
    # response codes such as 91 or 68.
    # How many failures were issuer-side decisions (domain F01)?
    code_05_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F01"
    )

    # F05/F06 (merchant/acquirer connectivity, POS/terminal) are deliberately
    # excluded here and counted only in reason_005_count below. Counting them
    # in both this network-side feature and the merchant-side one diluted
    # evidence for both hypotheses whenever F05/F06 dominated a batch.
    # How many failures were network/card-scheme availability issues
    # (domains F02 and F03)?
    code_91_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) in {"F02", "F03"}
    )

    # Reason-code features
    # Same as code_05_count above — issuer decisions (F01), counted again
    # under a different feature name because that's what the trained model
    # expects (see FEATURE_NAMES in ml_diagnosis.py).
    reason_001_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F01"
    )

    # How many failures were authentication/risk-gateway blocks (F08)?
    reason_002_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F08"
    )

    # How many failures were merchant-acceptance problems (F04)?
    reason_004_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) == "F04"
    )

    # How many failures were merchant/acquirer connectivity or POS/terminal
    # problems (F05 and F06 combined)?
    reason_005_count = sum(
        1
        for txn in failed_transactions
        if failure_domain_code(txn) in {"F05", "F06"}
    )

    # Service-level features
    # How many failed transactions were handled by the authorization
    # service specifically, regardless of why they failed?
    authorization_service_failures = sum(
        1
        for txn in failed_transactions
        if txn["service"] == "authorization-service"
    )

    # Same idea, but for the payment-gateway service.
    payment_gateway_failures = sum(
        1
        for txn in failed_transactions
        if txn["service"] == "payment-gateway"
    )

    # These two are turned into RATIOS below (not left as raw counts)
    # because most failure domains route the bulk of their failures
    # through "authorization-service" regardless of which domain is
    # actually at fault (e.g. F06/merchant-side failures are ~60%
    # authorization-service, almost identical to F01/issuer-side at
    # ~64%). A big RAW count there mostly just reflects "this was a big
    # batch", not "payment-service is the cause". The ratio (share of
    # THIS batch's own failures) is the part that's actually
    # informative, regardless of how large or small the batch is.
    authorization_service_ratio = (
        authorization_service_failures / failure_count
        if failure_count
        else 0
    )

    payment_gateway_ratio = (
        payment_gateway_failures / failure_count
        if failure_count
        else 0
    )

    # Merchant spread
    # How many DIFFERENT merchants had at least one failure? A low number
    # (failures piling up at 1-2 merchants) points toward a merchant-side
    # issue; a high number (spread evenly) points away from it.
    unique_affected_merchants = len(
        {
            txn["merchant"]
            for txn in failed_transactions
        }
    )

    # Ratios
    # From here down, every value is a percentage (0 to 1) rather than a
    # raw count, so the model can compare batches of very different sizes
    # fairly — e.g. "70% of failures were issuer declines" means the same
    # thing whether there were 10 failures or 1,000.
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

    # Package everything into one flat dictionary — this exact shape (same
    # 14 keys, same order expected) is what ml_diagnosis.py's FEATURE_NAMES
    # list turns into the numeric vector the model actually runs on.
    return {
        "failure_count": failure_count,

        "code_05_count": code_05_count,
        "code_91_count": code_91_count,

        "reason_001_count": reason_001_count,
        "reason_002_count": reason_002_count,
        "reason_004_count": reason_004_count,
        "reason_005_count": reason_005_count,

        "authorization_service_ratio":
            authorization_service_ratio,

        "payment_gateway_ratio":
            payment_gateway_ratio,

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
