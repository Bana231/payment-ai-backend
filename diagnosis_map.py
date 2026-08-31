DIAGNOSIS_MAP = {
    "issuer_issue": {
        "name": "Issuer-related issue",
        "description": (
            "Transaction failures associated with issuer declines "
            "or issuer-side processing."
        ),
        "failure_domains": ["F01"],
        "services": ["authorization-service"],
    },

    "network_switch_issue": {
        "name": "Network / switch issue",
        "description": (
            "Transaction failures associated with payment-network, "
            "switch, or downstream connectivity."
        ),
        "failure_domains": ["F02", "F03", "F05", "F06"],
        "services": ["payment-gateway"],
    },

    "merchant_issue": {
        "name": "Merchant-specific issue",
        "description": (
            "Failures concentrated around a specific merchant "
            "or merchant integration."
        ),
        "failure_domains": ["F04", "F05", "F06"],
        "services": [],
    },

    "payment_service_issue": {
        "name": "Payment-service issue",
        "description": (
            "Failures associated with an internal payment service "
            "or processing component."
        ),
        "failure_domains": ["F07", "F08", "F09"],
        "services": [],
    },

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


def get_diagnosis_map():
    """
    Return the supported payment failure diagnosis categories.

    This map defines the diagnostic space available to the
    investigation workflow. It is not itself an ML model.
    """
    return DIAGNOSIS_MAP
