"""Approved payment-failure taxonomy used for analysis and presentation.

ISO response codes may still be stored for protocol compatibility, but they are
not the primary business reason shown to users or used for domain grouping.
"""

FAILURE_DOMAINS = {
    "F01": "Issuer decision",
    "F02": "Issuer availability",
    "F03": "Card-network availability",
    "F04": "Merchant acceptance",
    "F05": "Merchant / acquirer connectivity",
    "F06": "POS / terminal",
    "F07": "Gateway / processor",
    "F08": "Authentication / risk gateway",
    "F09": "Invalid request",
}

FAILURE_REASONS = {
    "F01.01": "Insufficient funds", "F01.02": "Account restricted",
    "F01.03": "Limit exceeded", "F01.04": "Card expired",
    "F01.05": "Suspected fraud", "F01.06": "Issuer decline",
    "F02.01": "Issuer host down", "F02.02": "Issuer timeout",
    "F02.03": "Issuer maintenance",
    "F03.01": "Visa unavailable", "F03.02": "Mastercard unavailable",
    "F03.03": "RuPay unavailable", "F03.04": "Scheme route timeout",
    "F04.01": "Merchant inactive", "F04.02": "MCC blocked",
    "F04.03": "Terminal not enabled",
    "F05.01": "Merchant network down", "F05.02": "Acquirer link down",
    "F06.01": "POS offline", "F06.02": "POS timeout",
    "F06.03": "Terminal message invalid",
    "F07.01": "Gateway down", "F07.02": "Processor timeout",
    "F07.03": "Processing error",
    "F08.01": "3DS challenge failed", "F08.02": "Merchant risk block",
    "F08.03": "Acquirer risk block",
    "F09.01": "Malformed request", "F09.02": "Missing required field",
    "F09.03": "Unsupported currency",
}


def failure_reason_code(transaction: dict) -> str | None:
    code = transaction.get("failure_reason_code") or transaction.get("reason_code")
    return str(code).upper() if code else None


def failure_domain_code(transaction: dict) -> str | None:
    code = transaction.get("failure_domain_code") or failure_reason_code(transaction)
    domain = str(code).upper()[:3] if code else ""
    return domain if domain in FAILURE_DOMAINS else None


def failure_reason_details(code: str | None) -> dict:
    normalized = str(code).upper() if code else ""
    domain = normalized[:3]
    return {
        "code": normalized or "Unknown",
        "display_name": FAILURE_REASONS.get(normalized, "Unknown failure reason"),
        "domain_code": domain if domain in FAILURE_DOMAINS else None,
        "domain_name": FAILURE_DOMAINS.get(domain, "Unknown domain"),
    }
