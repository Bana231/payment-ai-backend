"""Approved payment-failure taxonomy used for analysis and presentation.

ISO response codes may still be stored for protocol compatibility, but they are
not the primary business reason shown to users or used for domain grouping.
"""

# The 9 broad "categories" a failure can fall into (F01-F09). Every specific
# reason code below (like "F01.03") belongs to exactly one of these — the
# first 3 characters of a reason code are always its domain code.
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

# The full list of specific, human-readable failure reasons (e.g. "F01.03"
# = "Limit exceeded"). This is the exact wording shown anywhere a failure
# reason appears in the app — transactions, reports, investigations.
FAILURE_REASONS = {
    "F01.01": "Insufficient funds", "F01.02": "Account restricted",
    "F01.03": "Limit exceeded", "F01.04": "Card expired",
    "F01.05": "Suspected fraud", "F01.06": "Issuer decline",
    "F02.01": "Issuer host down", "F02.02": "Issuer timeout",
    "F02.03": "Issuer maintenance",
    "F03.01": "Visa unavailable", "F03.02": "Mastercard unavailable",
    "F03.03": "American Express unavailable", "F03.04": "Discover unavailable",
    "F03.05": "Diners Club unavailable", "F03.06": "RuPay unavailable",
    "F03.07": "JCB unavailable", "F03.08": "UnionPay unavailable",
    "F03.09": "BC Card unavailable", "F03.10": "NAPAS unavailable",
    "F03.11": "eftpos unavailable", "F03.12": "Maestro unavailable",
    "F03.13": "Cartes Bancaires unavailable", "F03.14": "Girocard unavailable",
    "F03.15": "Bancontact unavailable", "F03.16": "Mir unavailable",
    "F03.17": "Troy unavailable", "F03.18": "Verve unavailable",
    "F03.19": "Meeza unavailable", "F03.20": "Elo unavailable",
    "F03.21": "Scheme route timeout",
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


# Reads whichever failure-reason field a transaction actually has (there are
# two possible column names for historical reasons) and returns it in a
# consistent uppercase form, e.g. "F01.03". Returns None for a transaction
# that didn't fail (it has no reason code at all).
def failure_reason_code(transaction: dict) -> str | None:
    code = transaction.get("failure_reason_code") or transaction.get("reason_code")
    return str(code).upper() if code else None


# Same idea, but returns just the broad domain (first 3 characters, e.g.
# "F01") instead of the full specific reason. Used whenever code only cares
# about "which of the 9 categories" rather than the exact reason.
def failure_domain_code(transaction: dict) -> str | None:
    code = transaction.get("failure_domain_code") or failure_reason_code(transaction)
    domain = str(code).upper()[:3] if code else ""
    return domain if domain in FAILURE_DOMAINS else None


# Turns a bare code like "F01.03" into a full, ready-to-display bundle:
# its human-readable meaning, its domain code, and the domain's own name.
# Never crashes on a bad/missing code — falls back to "Unknown" text so
# callers don't need their own error handling for this.
def failure_reason_details(code: str | None) -> dict:
    normalized = str(code).upper() if code else ""
    domain = normalized[:3]
    return {
        "code": normalized or "Unknown",
        "display_name": FAILURE_REASONS.get(normalized, "Unknown failure reason"),
        "domain_code": domain if domain in FAILURE_DOMAINS else None,
        "domain_name": FAILURE_DOMAINS.get(domain, "Unknown domain"),
    }
