import random
import json


random.seed(42)


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def noisy_count(base, spread, minimum=0):
    return max(
        minimum,
        base + random.randint(-spread, spread),
    )


def generate_issuer_issue():
    failure_count = random.randint(10, 35)

    code_05_count = random.randint(
        int(failure_count * 0.45),
        int(failure_count * 0.85),
    )

    code_91_count = random.randint(
        0,
        int(failure_count * 0.25),
    )

    reason_001_count = random.randint(
        1,
        max(1, int(failure_count * 0.35)),
    )

    reason_002_count = random.randint(
        1,
        max(1, int(failure_count * 0.30)),
    )

    # Add occasional merchant-like noise
    reason_004_count = random.randint(
        0,
        max(1, int(failure_count * 0.10)),
    )

    reason_005_count = random.randint(
        0,
        max(1, int(failure_count * 0.10)),
    )

    authorization_service_failures = random.randint(
        int(failure_count * 0.45),
        int(failure_count * 0.90),
    )

    payment_gateway_failures = random.randint(
        int(failure_count * 0.10),
        int(failure_count * 0.45),
    )

    unique_affected_merchants = random.randint(2, 10)

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
            round(code_05_count / failure_count, 3),
        "network_failure_ratio":
            round(code_91_count / failure_count, 3),
        "account_issue_ratio":
            round(
                (
                    reason_001_count
                    + reason_002_count
                )
                / failure_count,
                3,
            ),
        "merchant_issue_ratio":
            round(
                (
                    reason_004_count
                    + reason_005_count
                )
                / failure_count,
                3,
            ),
    }


def generate_network_issue():
    failure_count = random.randint(10, 35)

    code_91_count = random.randint(
        int(failure_count * 0.45),
        int(failure_count * 0.85),
    )

    # Some issuer declines can happen at the same time
    code_05_count = random.randint(
        0,
        int(failure_count * 0.30),
    )

    reason_001_count = random.randint(
        0,
        max(1, int(failure_count * 0.08)),
    )

    reason_002_count = random.randint(
        0,
        max(1, int(failure_count * 0.08)),
    )

    reason_004_count = random.randint(
        0,
        max(1, int(failure_count * 0.08)),
    )

    reason_005_count = random.randint(
        0,
        max(1, int(failure_count * 0.08)),
    )

    authorization_service_failures = random.randint(
        int(failure_count * 0.10),
        int(failure_count * 0.50),
    )

    payment_gateway_failures = random.randint(
        int(failure_count * 0.45),
        int(failure_count * 0.90),
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
            random.randint(3, 15),
        "issuer_decline_ratio":
            round(code_05_count / failure_count, 3),
        "network_failure_ratio":
            round(code_91_count / failure_count, 3),
        "account_issue_ratio":
            round(
                (
                    reason_001_count
                    + reason_002_count
                )
                / failure_count,
                3,
            ),
        "merchant_issue_ratio":
            round(
                (
                    reason_004_count
                    + reason_005_count
                )
                / failure_count,
                3,
            ),
    }


def generate_merchant_issue():
    failure_count = random.randint(8, 30)

    reason_004_count = random.randint(
        1,
        max(1, int(failure_count * 0.40)),
    )

    reason_005_count = random.randint(
        1,
        max(1, int(failure_count * 0.40)),
    )

    # Merchant issues may still produce issuer-style declines
    code_05_count = random.randint(
        int(failure_count * 0.30),
        int(failure_count * 0.75),
    )

    code_91_count = random.randint(
        0,
        int(failure_count * 0.20),
    )

    reason_001_count = random.randint(
        0,
        max(1, int(failure_count * 0.10)),
    )

    reason_002_count = random.randint(
        0,
        max(1, int(failure_count * 0.10)),
    )

    authorization_service_failures = random.randint(
        int(failure_count * 0.35),
        int(failure_count * 0.85),
    )

    payment_gateway_failures = random.randint(
        int(failure_count * 0.05),
        int(failure_count * 0.40),
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

        # Merchant issue should usually be concentrated,
        # but occasionally multiple merchants can show noise.
        "unique_affected_merchants":
            random.randint(1, 3),

        "issuer_decline_ratio":
            round(code_05_count / failure_count, 3),
        "network_failure_ratio":
            round(code_91_count / failure_count, 3),
        "account_issue_ratio":
            round(
                (
                    reason_001_count
                    + reason_002_count
                )
                / failure_count,
                3,
            ),
        "merchant_issue_ratio":
            round(
                (
                    reason_004_count
                    + reason_005_count
                )
                / failure_count,
                3,
            ),
    }


def generate_payment_service_issue():
    failure_count = random.randint(10, 35)

    # Service incidents can produce mixed response codes
    code_05_count = random.randint(
        int(failure_count * 0.10),
        int(failure_count * 0.45),
    )

    code_91_count = random.randint(
        int(failure_count * 0.10),
        int(failure_count * 0.45),
    )

    reason_001_count = random.randint(
        0,
        max(1, int(failure_count * 0.12)),
    )

    reason_002_count = random.randint(
        0,
        max(1, int(failure_count * 0.12)),
    )

    reason_004_count = random.randint(
        0,
        max(1, int(failure_count * 0.12)),
    )

    reason_005_count = random.randint(
        0,
        max(1, int(failure_count * 0.12)),
    )

    authorization_service_failures = random.randint(
        int(failure_count * 0.25),
        int(failure_count * 0.70),
    )

    payment_gateway_failures = random.randint(
        int(failure_count * 0.25),
        int(failure_count * 0.70),
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
            random.randint(2, 12),
        "issuer_decline_ratio":
            round(code_05_count / failure_count, 3),
        "network_failure_ratio":
            round(code_91_count / failure_count, 3),
        "account_issue_ratio":
            round(
                (
                    reason_001_count
                    + reason_002_count
                )
                / failure_count,
                3,
            ),
        "merchant_issue_ratio":
            round(
                (
                    reason_004_count
                    + reason_005_count
                )
                / failure_count,
                3,
            ),
    }


def add_label_noise(dataset, noise_rate=0.05):
    """
    Deliberately mislabel a small percentage of synthetic incidents.

    This simulates imperfect historical labels and makes the
    evaluation less artificially clean.
    """

    labels = [
        "issuer_issue",
        "network_switch_issue",
        "merchant_issue",
        "payment_service_issue",
    ]

    noisy_count = int(
        len(dataset) * noise_rate
    )

    noisy_indexes = random.sample(
        range(len(dataset)),
        noisy_count,
    )

    for index in noisy_indexes:
        current_label = dataset[index]["label"]

        alternative_labels = [
            label
            for label in labels
            if label != current_label
        ]

        dataset[index]["label"] = random.choice(
            alternative_labels
        )


def build_dataset(
    samples_per_class=100,
    label_noise_rate=0.05,
):
    dataset = []

    generators = {
        "issuer_issue":
            generate_issuer_issue,

        "network_switch_issue":
            generate_network_issue,

        "merchant_issue":
            generate_merchant_issue,

        "payment_service_issue":
            generate_payment_service_issue,
    }

    for label, generator in generators.items():

        for _ in range(samples_per_class):

            dataset.append(
                {
                    "features": generator(),
                    "label": label,
                }
            )

    # Introduce some deliberately imperfect labels.
    add_label_noise(
        dataset,
        noise_rate=label_noise_rate,
    )

    random.shuffle(dataset)

    return dataset


if __name__ == "__main__":

    dataset = build_dataset(
        samples_per_class=100,
        label_noise_rate=0.05,
    )

    with open(
        "synthetic_ml_dataset.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2,
        )

    print(
        "Generated incidents:",
        len(dataset),
    )

    print(
        "Synthetic label noise:",
        "5%",
    )