# ---------------------------------------------------------
# Synthetic payment transaction data
# ---------------------------------------------------------

transactions = [
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
        "transaction_id": "TXN1002",
        "amount": 75.00,
        "currency": "USD",
        "merchant": "Store Beta",
        "status": "SUCCESS",
        "response_code": "00",
        "reason_code": None,
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


# ---------------------------------------------------------
# Payment response codes
# ---------------------------------------------------------

response_codes = {
    "00": {
        "meaning": "Approved",
        "category": "success",
    },

    "05": {
        "meaning": "Do not honor",
        "category": "issuer_decline",
    },

    "91": {
        "meaning": "Issuer or switch unavailable",
        "category": "network_or_issuer_unavailable",
    },
}


# ---------------------------------------------------------
# FAS authorization reason codes
#
# These are kept separate from payment response codes.
# ---------------------------------------------------------

reason_codes = {
    "001": {
        "meaning": "Invalid Account",
        "category": "account_issue",
    },

    "002": {
        "meaning": "Account Status",
        "category": "account_issue",
    },

    "004": {
        "meaning": "Invalid Merchant",
        "category": "merchant_issue",
    },

    "005": {
        "meaning": "Merchant Blocked for Auth",
        "category": "merchant_issue",
    },
}