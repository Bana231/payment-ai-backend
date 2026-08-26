import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "failure_count",
    "code_05_count",
    "code_91_count",
    "reason_001_count",
    "reason_002_count",
    "reason_004_count",
    "reason_005_count",
    "authorization_service_failures",
    "payment_gateway_failures",
    "unique_affected_merchants",
    "issuer_decline_ratio",
    "network_failure_ratio",
    "account_issue_ratio",
    "merchant_issue_ratio",
]


# ---------------------------------------------------------
# 1. Load synthetic labelled incidents
# ---------------------------------------------------------

with open(
    "synthetic_ml_dataset.json",
    "r",
    encoding="utf-8",
) as file:
    dataset = json.load(file)


# ---------------------------------------------------------
# 2. Build X and y
# ---------------------------------------------------------

X = []
y = []

for incident in dataset:
    feature_vector = [
        incident["features"][feature_name]
        for feature_name in FEATURE_NAMES
    ]

    X.append(feature_vector)
    y.append(incident["label"])


# ---------------------------------------------------------
# 3. Same 80/20 train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. Get predictions AND probabilities
# ---------------------------------------------------------

predictions = model.predict(X_test)
probability_rows = model.predict_proba(X_test)


# ---------------------------------------------------------
# 6. Build one diagnostic record per test incident
# ---------------------------------------------------------

results = []

for actual, predicted, probabilities in zip(
    y_test,
    predictions,
    probability_rows,
):
    probability_map = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    ranked = sorted(
        probability_map.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_probability = ranked[0][1]

    second_probability = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    probability_gap = (
        top_probability - second_probability
    )

    results.append(
        {
            "actual": str(actual),
            "predicted": str(predicted),
            "correct": actual == predicted,
            "top_probability": top_probability,
            "probability_gap": probability_gap,
        }
    )


# ---------------------------------------------------------
# 7. Evaluate an acceptance policy
#
# Accepted:
#     AI diagnosis is clear enough for the AI-assisted
#     investigation/reporting path.
#
# Escalated:
#     Diagnosis is ambiguous -> human review.
# ---------------------------------------------------------

def evaluate_policy(
    minimum_probability,
    minimum_gap,
):
    accepted = [
        result
        for result in results
        if (
            result["top_probability"]
            >= minimum_probability
            and result["probability_gap"]
            >= minimum_gap
        )
    ]

    escalated = [
        result
        for result in results
        if result not in accepted
    ]

    correct_accepted = sum(
        result["correct"]
        for result in accepted
    )

    incorrect_accepted = (
        len(accepted) - correct_accepted
    )

    coverage = (
        len(accepted) / len(results)
        if results
        else 0
    )

    accepted_accuracy = (
        correct_accepted / len(accepted)
        if accepted
        else 0
    )

    escalation_rate = (
        len(escalated) / len(results)
        if results
        else 0
    )

    return {
        "minimum_probability":
            minimum_probability,

        "minimum_gap":
            minimum_gap,

        "accepted":
            len(accepted),

        "escalated":
            len(escalated),

        "coverage":
            coverage,

        "accepted_accuracy":
            accepted_accuracy,

        "incorrect_accepted":
            incorrect_accepted,

        "escalation_rate":
            escalation_rate,
    }


# ---------------------------------------------------------
# 8. Try several policies
#
# These are evaluation thresholds, NOT production values.
# ---------------------------------------------------------

policies = [
    (0.00, 0.00),
    (0.50, 0.10),
    (0.60, 0.20),
    (0.70, 0.30),
    (0.80, 0.40),
    (0.90, 0.50),
]


print("\nCONFIDENCE / COVERAGE EVALUATION")
print("=" * 90)

print(
    f"{'Min Prob':<10}"
    f"{'Min Gap':<10}"
    f"{'Accepted':<11}"
    f"{'Escalated':<12}"
    f"{'Coverage':<12}"
    f"{'Accuracy':<12}"
    f"{'Wrong Accepted':<15}"
)

print("-" * 90)


for minimum_probability, minimum_gap in policies:

    evaluation = evaluate_policy(
        minimum_probability,
        minimum_gap,
    )

    print(
        f"{evaluation['minimum_probability']:<10.2f}"
        f"{evaluation['minimum_gap']:<10.2f}"
        f"{evaluation['accepted']:<11}"
        f"{evaluation['escalated']:<12}"
        f"{evaluation['coverage']:<12.2%}"
        f"{evaluation['accepted_accuracy']:<12.2%}"
        f"{evaluation['incorrect_accepted']:<15}"
    )


# ---------------------------------------------------------
# 9. Show ambiguous examples
# ---------------------------------------------------------

ambiguous_results = sorted(
    results,
    key=lambda result: (
        result["probability_gap"],
        result["top_probability"],
    ),
)


print("\nMOST AMBIGUOUS TEST INCIDENTS")
print("=" * 60)


for index, result in enumerate(
    ambiguous_results[:10],
    start=1,
):
    print(
        f"\nIncident {index}"
    )

    print(
        "Actual:",
        result["actual"],
    )

    print(
        "Predicted:",
        result["predicted"],
    )

    print(
        "Correct:",
        result["correct"],
    )

    print(
        "Top probability:",
        round(
            result["top_probability"],
            3,
        ),
    )

    print(
        "Probability gap:",
        round(
            result["probability_gap"],
            3,
        ),
    )