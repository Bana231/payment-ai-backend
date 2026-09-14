import math

from sklearn.ensemble import RandomForestClassifier

from supabase_store import list_training_examples


# The exact 14 feature names, IN THIS EXACT ORDER. This order must always
# match feature_extractor.py's output dictionary and every row in the
# training dataset — the model has no idea what each number "means", it
# only knows "position 1 is failure_count, position 2 is code_05_count",
# etc. If this order ever changed without also changing the training
# data, the model would silently learn the wrong relationships.
FEATURE_NAMES = [
    "failure_count",
    "code_05_count",
    "code_91_count",
    "reason_001_count",
    "reason_002_count",
    "reason_004_count",
    "reason_005_count",
    # These two are RATIOS, not raw counts, because most failure domains
    # route roughly 60% of their failures through "authorization-service"
    # regardless of which domain is actually at fault — a big raw count
    # there mostly just means "this was a big batch", not "payment
    # service is the cause". The ratio (this batch's own share) is the
    # part that's actually informative, independent of batch size. Every
    # training row must carry this same ratio shape.
    "authorization_service_ratio",
    "payment_gateway_ratio",
    "unique_affected_merchants",
    "issuer_decline_ratio",
    "network_failure_ratio",
    "account_issue_ratio",
    "merchant_issue_ratio",
]


# Which of the 14 features above are raw COUNTS (failure_count,
# code_05_count, etc.) as opposed to ratios that are already 0-1
# regardless of batch size. This distinction matters for
# _scale_feature_vector below: a real production question can pull in
# hundreds of failures, while every historical training example so far
# only ever had 8-161 — a raw count of 679 is a number the model has
# literally never seen and can't place sensibly among its learned
# splits, even though the underlying pattern (e.g. "82% of failures are
# one reason code") is exactly the kind of thing the model should still
# recognize regardless of how big the batch is.
COUNT_FEATURE_NAMES = {
    "failure_count",
    "code_05_count",
    "code_91_count",
    "reason_001_count",
    "reason_002_count",
    "reason_004_count",
    "reason_005_count",
    "unique_affected_merchants",
}


# Squashes every raw-count feature through log1p (log(1 + x)) before the
# model ever sees it, while leaving the already-scale-free ratio features
# untouched. log1p turns "679 vs 25" (a huge, never-seen-before jump) into
# "6.5 vs 3.3" (a much smaller, well-inside-the-learned-range difference)
# — so the model's learned splits still apply sensibly no matter how big
# the actual batch was. This is applied identically to both the training
# data (below) and every live prediction (see diagnose further down), so
# training and prediction always agree on what a "feature" means.
def _scale_feature_vector(
    feature_vector: list,
) -> list:
    return [
        math.log1p(value) if name in COUNT_FEATURE_NAMES else value
        for name, value in zip(FEATURE_NAMES, feature_vector)
    ]


# Pulls every labelled training example out of Supabase and turns each one
# into (a list of 14 numbers, the correct answer) — this is the "textbook"
# the model studies before it can make predictions.
def load_training_data():
    dataset = list_training_examples()
    if not dataset:
        raise RuntimeError("Supabase ml_training_examples is empty.")

    x_train = []
    y_train = []

    for incident in dataset:
        # Pull this example's 14 feature values out in the FEATURE_NAMES
        # order above, ignoring whatever order they happen to be stored in.
        feature_vector = [
            incident["features"][feature_name]
            for feature_name in FEATURE_NAMES
        ]

        # Scale it the same way a live prediction will be scaled (see
        # _scale_feature_vector above) before the model ever trains on it.
        x_train.append(_scale_feature_vector(feature_vector))
        y_train.append(incident["label"])

    return x_train, y_train


# The actual trained model lives here, as a variable the whole module can
# see. It starts empty and gets filled in by retrain_model() below.
model = None


def retrain_model() -> None:
    """(Re)load training data from Supabase and fit a fresh Random Forest.

    Called once at import time, and again whenever a new labelled example is
    added (e.g. from confirmed/overridden human review feedback), so the live
    model reflects it immediately instead of only on the next process restart.
    """
    global model

    # Step 1: get the full "textbook" of labelled examples.
    x_train, y_train = load_training_data()

    # Step 2: create a brand new, untrained Random Forest — 200 individual
    # decision trees that will each vote on every prediction. random_state
    # just makes the "randomness" reproducible run to run.
    fitted_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )

    # Step 3: actually train it — this is where the model "learns" the
    # relationship between the 14 numbers and the 4 possible root causes.
    fitted_model.fit(
        x_train,
        y_train,
    )

    # Step 4: swap the old model out for the newly trained one.
    model = fitted_model


# Train the model once immediately, as soon as this file is first imported,
# so it's ready to answer questions right away.
retrain_model()


# Given one transaction batch's 14 computed features, ask the trained
# model which of the 4 root causes it thinks is most likely, and how
# confident it is.
def diagnose(features: dict):
    # Turn the incoming feature dictionary into the same ordered list of
    # 14 numbers the model was trained on.
    feature_vector = [
        features[feature_name]
        for feature_name in FEATURE_NAMES
    ]

    # Scale it exactly the same way the training data was scaled (see
    # _scale_feature_vector above), so a batch far bigger than anything
    # in training (e.g. 679 failures) still maps onto ranges the model
    # has actually learned from, instead of being an unseen raw number.
    scaled_feature_vector = _scale_feature_vector(feature_vector)

    # Ask the model for a probability for EVERY possible cause (not just
    # its single best guess) — e.g. {"issuer_issue": 0.06, "merchant_issue":
    # 0.74, ...}. predict_proba expects a list of examples, hence the
    # extra [ ] around scaled_feature_vector even though we're only asking
    # about one batch — and it returns one row of probabilities per
    # example, so [0] takes that first (and only) row back out.
    probabilities = model.predict_proba(
        [scaled_feature_vector]
    )[0]

    # Pair each cause's name up with its rounded probability.
    diagnosis_probabilities = {
        str(label): round(float(probability), 4)
        for label, probability
        in zip(model.classes_, probabilities)
    }

    # The model's single best-guess label (always just the highest
    # probability one from above, computed separately by scikit-learn).
    predicted_label = str(
        model.predict(
            [scaled_feature_vector]
        )[0]
    )

    # Sort causes from most to least likely, so we can easily pull out
    # "the top one" and "the second one" next.
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

    # How far ahead the top guess is from the runner-up — a big gap means
    # the model is confident there's a clear winner; a small gap means two
    # causes are basically tied, which is a strong hint the case is
    # genuinely ambiguous (see agent_graph.py's confidence-policy check).
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
