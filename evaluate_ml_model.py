import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
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
# 1. Load the synthetic labelled incident dataset
# ---------------------------------------------------------

with open(
    "synthetic_ml_dataset.json",
    "r",
    encoding="utf-8",
) as file:
    dataset = json.load(file)


# ---------------------------------------------------------
# 2. Convert incidents into X (features) and Y (labels)
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


print("Total incidents:", len(X))


# ---------------------------------------------------------
# 3. Split into training and testing data
#
# 80% -> model learns from these
# 20% -> model never sees these during training
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("Training incidents:", len(X_train))
print("Testing incidents:", len(X_test))


# ---------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# 5. Ask the model to diagnose the unseen test incidents
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 6. Measure overall accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print("\nOverall Accuracy:")
print(f"{accuracy:.2%}")


# ---------------------------------------------------------
# 7. Detailed class-level evaluation
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 8. Confusion Matrix
#
# Rows    = actual diagnosis
# Columns = model prediction
# ---------------------------------------------------------

labels = sorted(set(y))

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

print("Diagnosis Classes:")

for index, label in enumerate(labels):
    print(f"{index}: {label}")


print("\nConfusion Matrix:")
print(matrix)


# ---------------------------------------------------------
# 9. Count incorrect diagnoses
# ---------------------------------------------------------

incorrect_predictions = sum(
    actual != predicted
    for actual, predicted in zip(
        y_test,
        y_pred,
    )
)

print(
    "\nIncorrect diagnoses:",
    incorrect_predictions,
)

print(
    "Correct diagnoses:",
    len(y_test) - incorrect_predictions,
)


# ---------------------------------------------------------
# 10. Inspect model probability for a few test incidents
# ---------------------------------------------------------

probabilities = model.predict_proba(X_test)

print("\nSample Diagnostic Predictions:")

for index in range(min(5, len(X_test))):

    probability_map = {
        str(label): round(float(probability), 3)
        for label, probability in zip(
            model.classes_,
            probabilities[index],
        )
    }

    sorted_probabilities = sorted(
        probability_map.items(),
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
        3,
    )

    print(f"\nTest Incident {index + 1}")
    print("Actual cause:   ", y_test[index])
    print("Predicted cause:", str(y_pred[index]))
    print("Probabilities:  ", probability_map)
    print("Top probability:", top_probability)
    print("Probability gap:", probability_gap)