from feature_extractor import extract_diagnostic_features
from ml_diagnosis import diagnose


clear_case_transactions = [
    {
        "transaction_id": "TEST1001",
        "amount": 100.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1002",
        "amount": 125.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1003",
        "amount": 80.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1004",
        "amount": 210.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1005",
        "amount": 55.00,
        "currency": "USD",
        "merchant": "Merchant E",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "TEST1006",
        "amount": 190.00,
        "currency": "USD",
        "merchant": "Merchant F",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
]


features = extract_diagnostic_features(
    clear_case_transactions
)

diagnosis = diagnose(
    features
)


print("\n==============================")
print("CLEAR CASE FEATURES")
print("==============================")
print(features)


print("\n==============================")
print("ML DIAGNOSIS")
print("==============================")
print(diagnosis)


top_probability = diagnosis.get(
    "top_probability",
    0,
)

probability_gap = diagnosis.get(
    "probability_gap",
    0,
)


print("\n==============================")
print("PROTOTYPE ASSESSMENT")
print("==============================")


if (
    top_probability >= 0.70
    and probability_gap >= 0.30
):
    print("Assessment: CLEAR")
    print(
        "Selected path:",
        diagnosis.get(
            "predicted_cause"
        )
    )

else:
    print("Assessment: AMBIGUOUS")