from sklearn.ensemble import RandomForestClassifier

from supabase_store import list_training_examples


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


def load_training_data():
    dataset = list_training_examples()
    if not dataset:
        raise RuntimeError("Supabase ml_training_examples is empty.")

    x_train = []
    y_train = []

    for incident in dataset:
        feature_vector = [
            incident["features"][feature_name]
            for feature_name in FEATURE_NAMES
        ]

        x_train.append(feature_vector)
        y_train.append(incident["label"])

    return x_train, y_train


x_train, y_train = load_training_data()


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(
    x_train,
    y_train,
)


def diagnose(features: dict):
    feature_vector = [
        features[feature_name]
        for feature_name in FEATURE_NAMES
    ]

    probabilities = model.predict_proba(
        [feature_vector]
    )[0]

    diagnosis_probabilities = {
        str(label): round(float(probability), 4)
        for label, probability
        in zip(model.classes_, probabilities)
    }

    predicted_label = str(
        model.predict(
            [feature_vector]
        )[0]
    )

    sorted_probabilities = sorted(
        diagnosis_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_probability = sorted_probabilities[0][1]

    second_probability = (
        sorted_probabilities[1][1]
        if len(sorted_probabilities) > 1
        else 0
    )

    probability_gap = round(
        top_probability - second_probability,
        4,
    )

    return {
        "predicted_cause": predicted_label,
        "probabilities": diagnosis_probabilities,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "probability_gap": probability_gap,
    }
