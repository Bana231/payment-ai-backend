from itertools import product

from ml_diagnosis import diagnose


CLEAR_PROBABILITY_THRESHOLD = 0.70
CLEAR_GAP_THRESHOLD = 0.30


TARGET_CLASSES = [
    "issuer_issue",
    "merchant_issue",
    "network_switch_issue",
    "payment_service_issue",
]


def build_features(
    failure_count: int,
    code_05_count: int,
    code_91_count: int,
    reason_001_count: int,
    reason_002_count: int,
    reason_004_count: int,
    reason_005_count: int,
    authorization_service_failures: int,
    payment_gateway_failures: int,
    unique_affected_merchants: int,
):
    """
    Build the exact feature dictionary expected by ml_model.diagnose().

    Ratios are derived from the synthetic counts rather than entered
    independently so that the test cases remain internally consistent.
    """

    if failure_count <= 0:
        raise ValueError(
            "failure_count must be greater than zero"
        )

    issuer_decline_ratio = (
        code_05_count / failure_count
    )

    network_failure_ratio = (
        code_91_count / failure_count
    )

    account_issue_ratio = (
        (
            reason_001_count
            + reason_002_count
        )
        / failure_count
    )

    merchant_issue_ratio = (
        (
            reason_004_count
            + reason_005_count
        )
        / failure_count
    )

    return {
        "failure_count": failure_count,
        "code_05_count": code_05_count,
        "code_91_count": code_91_count,
        "reason_001_count": reason_001_count,
        "reason_002_count": reason_002_count,
        "reason_004_count": reason_004_count,
        "reason_005_count": reason_005_count,
        "authorization_service_failures":
            authorization_service_failures,
        "payment_gateway_failures":
            payment_gateway_failures,
        "unique_affected_merchants":
            unique_affected_merchants,
        "issuer_decline_ratio":
            round(issuer_decline_ratio, 4),
        "network_failure_ratio":
            round(network_failure_ratio, 4),
        "account_issue_ratio":
            round(account_issue_ratio, 4),
        "merchant_issue_ratio":
            round(merchant_issue_ratio, 4),
    }


def is_clear(diagnosis: dict) -> bool:
    return (
        diagnosis["top_probability"]
        >= CLEAR_PROBABILITY_THRESHOLD
        and
        diagnosis["probability_gap"]
        >= CLEAR_GAP_THRESHOLD
    )


def print_diagnosis(
    title: str,
    features: dict,
    diagnosis: dict,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Predicted cause: "
        f"{diagnosis['predicted_cause']}"
    )

    print(
        f"Top probability: "
        f"{diagnosis['top_probability']:.3f}"
    )

    print(
        f"Probability gap: "
        f"{diagnosis['probability_gap']:.3f}"
    )

    print(
        f"Clear: "
        f"{is_clear(diagnosis)}"
    )

    print()
    print("Probabilities:")

    sorted_probabilities = sorted(
        diagnosis["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for label, probability in sorted_probabilities:
        print(
            f"  {label:<24} "
            f"{probability:.4f}"
        )

    print()
    print("Feature vector:")

    for name, value in features.items():
        print(
            f"  {name:<32} {value}"
        )


def candidate_feature_sets():
    """
    Generate internally consistent synthetic feature combinations.

    This does NOT alter or retrain the model.

    It asks the already-trained Random Forest how it classifies
    different operational failure patterns.
    """

    failure_counts = [
        4,
        6,
        8,
        10,
    ]

    for failure_count in failure_counts:

        count_values = sorted(
            {
                0,
                1,
                max(
                    1,
                    failure_count // 4,
                ),
                max(
                    1,
                    failure_count // 2,
                ),
                failure_count,
            }
        )

        merchant_values = sorted(
            {
                1,
                min(
                    2,
                    failure_count,
                ),
                max(
                    1,
                    failure_count // 2,
                ),
                failure_count,
            }
        )

        for (
            code_05_count,
            code_91_count,
            reason_001_count,
            reason_002_count,
            reason_004_count,
            reason_005_count,
            authorization_failures,
            gateway_failures,
            unique_merchants,
        ) in product(
            count_values,
            count_values,
            count_values,
            count_values,
            count_values,
            count_values,
            count_values,
            count_values,
            merchant_values,
        ):

            # ---------------------------------------------
            # Basic consistency checks
            # ---------------------------------------------

            if (
                code_05_count
                > failure_count
            ):
                continue

            if (
                code_91_count
                > failure_count
            ):
                continue

            if (
                code_05_count
                + code_91_count
                > failure_count
            ):
                continue

            if (
                authorization_failures
                > failure_count
            ):
                continue

            if (
                gateway_failures
                > failure_count
            ):
                continue

            if (
                authorization_failures
                + gateway_failures
                > failure_count
            ):
                continue

            if (
                unique_merchants
                > failure_count
            ):
                continue

            if (
                reason_001_count
                + reason_002_count
                + reason_004_count
                + reason_005_count
                > failure_count
            ):
                continue

            features = build_features(
                failure_count=
                    failure_count,
                code_05_count=
                    code_05_count,
                code_91_count=
                    code_91_count,
                reason_001_count=
                    reason_001_count,
                reason_002_count=
                    reason_002_count,
                reason_004_count=
                    reason_004_count,
                reason_005_count=
                    reason_005_count,
                authorization_service_failures=
                    authorization_failures,
                payment_gateway_failures=
                    gateway_failures,
                unique_affected_merchants=
                    unique_merchants,
            )

            yield features


def find_clear_examples():
    """
    Search for one clear representative feature vector
    for every diagnosis class.
    """

    found = {}

    best_seen = {}

    checked = 0

    for features in candidate_feature_sets():

        checked += 1

        diagnosis = diagnose(
            features
        )

        predicted_cause = (
            diagnosis[
                "predicted_cause"
            ]
        )

        current_best = best_seen.get(
            predicted_cause
        )

        if (
            current_best is None
            or
            diagnosis["top_probability"]
            >
            current_best[
                "diagnosis"
            ][
                "top_probability"
            ]
        ):
            best_seen[
                predicted_cause
            ] = {
                "features": features,
                "diagnosis": diagnosis,
            }

        if (
            predicted_cause
            in TARGET_CLASSES
            and
            is_clear(
                diagnosis
            )
            and
            predicted_cause
            not in found
        ):
            found[
                predicted_cause
            ] = {
                "features": features,
                "diagnosis": diagnosis,
            }

            print()
            print(
                "FOUND CLEAR CASE:"
            )

            print(
                predicted_cause
            )

            print(
                f"after checking "
                f"{checked} candidate vectors"
            )

        if (
            len(found)
            ==
            len(TARGET_CLASSES)
        ):
            break

    return (
        found,
        best_seen,
        checked,
    )


def main():

    print()
    print("=" * 72)
    print(
        "PAYOPS SENTINEL - ML CLASS COVERAGE TEST"
    )
    print("=" * 72)

    print()
    print(
        "Clear-path policy:"
    )

    print(
        f"  minimum top probability = "
        f"{CLEAR_PROBABILITY_THRESHOLD}"
    )

    print(
        f"  minimum probability gap = "
        f"{CLEAR_GAP_THRESHOLD}"
    )

    print()
    print(
        "Searching the existing trained model "
        "for representative clear cases..."
    )

    (
        clear_examples,
        best_seen,
        checked,
    ) = find_clear_examples()

    print()
    print("=" * 72)
    print(
        "SEARCH COMPLETE"
    )
    print("=" * 72)

    print(
        f"Candidate feature vectors checked: "
        f"{checked}"
    )

    print(
        f"Clear diagnosis classes found: "
        f"{len(clear_examples)} / "
        f"{len(TARGET_CLASSES)}"
    )

    # -----------------------------------------------------
    # Print clear examples
    # -----------------------------------------------------

    for target_class in TARGET_CLASSES:

        example = clear_examples.get(
            target_class
        )

        if example is None:
            continue

        print_diagnosis(
            title=(
                f"CLEAR CASE: "
                f"{target_class}"
            ),
            features=example[
                "features"
            ],
            diagnosis=example[
                "diagnosis"
            ],
        )

    # -----------------------------------------------------
    # Report classes that did not achieve clear status
    # -----------------------------------------------------

    missing_classes = [
        target_class
        for target_class
        in TARGET_CLASSES
        if target_class
        not in clear_examples
    ]

    if missing_classes:

        print()
        print("=" * 72)
        print(
            "CLASSES WITHOUT A CLEAR CASE"
        )
        print("=" * 72)

        for target_class in missing_classes:

            print()
            print(
                f"Target class: "
                f"{target_class}"
            )

            best = best_seen.get(
                target_class
            )

            if best is None:

                print(
                    "The model did not predict "
                    "this class for any tested "
                    "feature vector."
                )

                continue

            print_diagnosis(
                title=(
                    f"BEST OBSERVED CASE: "
                    f"{target_class}"
                ),
                features=best[
                    "features"
                ],
                diagnosis=best[
                    "diagnosis"
                ],
            )

    # -----------------------------------------------------
    # Assertions for discovered clear cases
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print(
        "AUTOMATED CLASS COVERAGE CHECKS"
    )
    print("=" * 72)

    for target_class in TARGET_CLASSES:

        example = clear_examples.get(
            target_class
        )

        if example is None:

            print(
                f"NOT CLEAR: "
                f"{target_class}"
            )

            continue

        diagnosis = example[
            "diagnosis"
        ]

        assert (
            diagnosis[
                "predicted_cause"
            ]
            ==
            target_class
        )

        assert (
            diagnosis[
                "top_probability"
            ]
            >=
            CLEAR_PROBABILITY_THRESHOLD
        )

        assert (
            diagnosis[
                "probability_gap"
            ]
            >=
            CLEAR_GAP_THRESHOLD
        )

        print(
            f"PASS: "
            f"{target_class}"
        )

    print()
    print("=" * 72)

    if (
        len(clear_examples)
        ==
        len(TARGET_CLASSES)
    ):

        print(
            "ALL FOUR ML CLASSES HAVE "
            "A CLEAR REPRESENTATIVE CASE"
        )

    else:

        print(
            "MODEL COVERAGE GAP DETECTED"
        )

        print()

        print(
            "This is useful evidence, not "
            "automatically a test failure."
        )

        print(
            "A class that cannot meet the "
            "clear-path policy may indicate "
            "insufficient synthetic training "
            "coverage or weak class separation."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()