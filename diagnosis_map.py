# This file is a lookup table (a dictionary), not a program that "runs" —
# it just holds the 5 possible root-cause categories the whole system can
# ever diagnose, and which failure domains (F01-F09) and services belong
# to each one. Other files import DIAGNOSIS_MAP and use it to translate a
# domain code like "F01" into a human category like "Issuer-related issue".

DIAGNOSIS_MAP = {
    # Category 1: something went wrong on the card-issuing bank's side
    # (e.g. insufficient funds, card expired, issuer's system down).
    "issuer_issue": {
        "name": "Issuer-related issue",
        "description": (
            "Transaction failures associated with issuer declines "
            "or issuer-side processing."
        ),
        "failure_domains": ["F01"],
        "services": ["authorization-service"],
    },

    # Category 2: something went wrong with the card network/switch itself
    # (e.g. Visa/Mastercard's own systems being unavailable).
    "network_switch_issue": {
        "name": "Network / switch issue",
        "description": (
            "Transaction failures associated with payment-network, "
            "switch, or issuer/scheme availability."
        ),
        # F05/F06 (merchant/acquirer connectivity, POS/terminal) belong to
        # merchant_issue only — see feature_extractor.py's code_91_count.
        # Listing them here too diluted evidence for both hypotheses.
        "failure_domains": ["F02", "F03"],
        "services": ["payment-gateway"],
    },

    # Category 3: something went wrong specific to one merchant (e.g. their
    # terminal is offline, or their network connection is down).
    "merchant_issue": {
        "name": "Merchant-specific issue",
        "description": (
            "Failures concentrated around a specific merchant "
            "or merchant integration."
        ),
        "failure_domains": ["F04", "F05", "F06"],
        "services": [],
    },

    # Category 4: something went wrong inside our own payment processing
    # software/infrastructure (e.g. the gateway crashed, a bad request).
    "payment_service_issue": {
        "name": "Payment-service issue",
        "description": (
            "Failures associated with an internal payment service "
            "or processing component."
        ),
        "failure_domains": ["F07", "F08", "F09"],
        "services": [],
    },

    # Fallback category: used when nothing about the evidence points
    # clearly at any of the four real categories above.
    "unknown": {
        "name": "Unknown / other issue",
        "description": (
            "The available diagnostic features do not clearly "
            "support one of the known failure categories."
        ),
        "failure_domains": [],
        "services": [],
    },
}


# Simple "getter" function — just hands back the dictionary above. Exists
# so other files can call a function instead of importing the raw
# dictionary directly, which keeps things consistent if this ever needs
# to become more than a static lookup later.
def get_diagnosis_map():
    """
    Return the supported payment failure diagnosis categories.

    This map defines the diagnostic space available to the
    investigation workflow. It is not itself an ML model.
    """
    return DIAGNOSIS_MAP
