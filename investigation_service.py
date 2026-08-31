import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent_graph import investigation_graph
from failure_taxonomy import (
    FAILURE_REASONS,
    failure_domain_code,
    failure_reason_code,
    failure_reason_details,
)
from supabase_store import (
    list_investigations,
    list_transactions,
    list_transactions_in_window,
    save_investigation,
)


# =========================================================
# Synthetic Investigation Scenarios
#
# These are prototype transaction groups used when the
# frontend sends only a natural-language question.
#
# Explicit transactions supplied by the API always take
# priority over these scenario datasets.
# =========================================================


AMBIGUOUS_TRANSACTIONS = [
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


ISSUER_TRANSACTIONS = [
    {
        "transaction_id": "ISS2001",
        "amount": 100.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2002",
        "amount": 125.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2003",
        "amount": 80.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2004",
        "amount": 210.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2005",
        "amount": 55.00,
        "currency": "USD",
        "merchant": "Merchant E",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "002",
        "service": "authorization-service",
    },
    {
        "transaction_id": "ISS2006",
        "amount": 190.00,
        "currency": "USD",
        "merchant": "Merchant F",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": "001",
        "service": "authorization-service",
    },
]


MERCHANT_TRANSACTIONS = [
    {
        "transaction_id": "MER3001",
        "amount": 90.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "58",
        "reason_code": "merchant_offline",
        "service": "payment-gateway",
    },
    {
        "transaction_id": "MER3002",
        "amount": 110.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "58",
        "reason_code": "merchant_offline",
        "service": "payment-gateway",
    },
    {
        "transaction_id": "MER3003",
        "amount": 145.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "57",
        "reason_code": "mcc_blocked",
        "service": "payment-gateway",
    },
    {
        "transaction_id": "MER3004",
        "amount": 65.00,
        "currency": "USD",
        "merchant": "Store Omega",
        "status": "FAILED",
        "response_code": "57",
        "reason_code": "mcc_blocked",
        "service": "payment-gateway",
    },
]


NETWORK_TRANSACTIONS = [
    {
        "transaction_id": "NET4001",
        "amount": 75.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "NET4002",
        "amount": 130.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "NET4003",
        "amount": 245.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "NET4004",
        "amount": 55.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
]


PAYMENT_SERVICE_TRANSACTIONS = [
    {
        "transaction_id": "PAY5001",
        "amount": 60.00,
        "currency": "USD",
        "merchant": "Merchant A",
        "status": "FAILED",
        "response_code": "05",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "PAY5002",
        "amount": 125.00,
        "currency": "USD",
        "merchant": "Merchant B",
        "status": "FAILED",
        "response_code": "91",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "PAY5003",
        "amount": 185.00,
        "currency": "USD",
        "merchant": "Merchant C",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": None,
        "service": "payment-gateway",
    },
    {
        "transaction_id": "PAY5004",
        "amount": 220.00,
        "currency": "USD",
        "merchant": "Merchant D",
        "status": "FAILED",
        "response_code": "12",
        "reason_code": None,
        "service": "payment-gateway",
    },
]


# =========================================================
# Payment-Domain Guardrail
# =========================================================

PAYMENT_DOMAIN_TERMS = {
    "payment",
    "payments",
    "transaction",
    "transactions",
    "authorization",
    "authorisation",
    "issuer",
    "merchant",
    "gateway",
    "switch",
    "decline",
    "declines",
    "failed",
    "failure",
    "failures",
    "response code",
    "reason code",
    "card",
    "incident",
}


def is_payment_domain_question(
    question: str,
) -> bool:
    question_lower = question.lower()

    return any(
        term in question_lower
        for term in PAYMENT_DOMAIN_TERMS
    )


# =========================================================
# Question Intent Classification
#
# We distinguish direct observational questions from
# root-cause investigation questions.
# =========================================================

FACTUAL_RESPONSE_CODE_TERMS = [
    "what response code",
    "which response code",
    "what response codes",
    "which response codes",
    "response codes are causing",
    "response code causing",
    "most common response code",
    "most frequent response code",
    "dominant response code",
    "response code distribution",
]


FACTUAL_MERCHANT_TERMS = [
    "which merchant",
    "most affected merchant",
    "merchant has most",
    "merchant has the most",
    "merchant failure count",
]


FACTUAL_FAILURE_TERMS = [
    "how many failures",
    "how many transactions failed",
    "failure count",
    "number of failures",
    "how many transaction did fail",
    "how many transaction failed",
    "how many transactions did fail",
]


MONTH_NAMES = {
    name: index
    for index, name in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ),
        start=1,
    )
}


def local_from_utc(
    instant: datetime,
    offset_minutes: int,
) -> datetime:
    """Convert a UTC-aware instant to the caller's local wall-clock time.

    offset_minutes follows JavaScript's Date.getTimezoneOffset() convention:
    minutes to ADD to local time to get UTC. Transactions are stored in UTC,
    but a question like "4th September" means the caller's calendar day, not
    the UTC calendar day — those can differ by a day depending on the time of
    day and the caller's offset from UTC.
    """
    return instant - timedelta(minutes=offset_minutes)


def utc_from_local_date(
    local_date: datetime,
    offset_minutes: int,
) -> datetime:
    """Convert a naive local calendar date/time into the equivalent UTC instant."""
    return local_date.replace(tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)


def build_calendar_window(
    year: int,
    month: int,
    label: str,
    offset_minutes: int = 0,
) -> Dict[str, Any]:
    start = utc_from_local_date(datetime(year, month, 1), offset_minutes)
    if month == 12:
        end = utc_from_local_date(datetime(year + 1, 1, 1), offset_minutes)
    else:
        end = utc_from_local_date(datetime(year, month + 1, 1), offset_minutes)
    return {
        "start": start,
        "end": end,
        "label": label,
    }


def build_calendar_day_window(
    year: int,
    month: int,
    day: int,
    label: str,
    offset_minutes: int = 0,
) -> Dict[str, Any]:
    try:
        start = utc_from_local_date(datetime(year, month, day), offset_minutes)
    except ValueError:
        return {
            "clarification": (
                f"{label} is not a valid calendar date. Please provide a valid date."
            ),
        }
    return {
        "start": start,
        "end": start + timedelta(days=1),
        "label": label,
    }


def transaction_years_for_month(
    transactions: List[Dict[str, Any]],
    month: int,
    offset_minutes: int = 0,
) -> List[int]:
    years: set[int] = set()
    for transaction in transactions:
        raw_created_at = transaction.get("created_at")
        if not raw_created_at:
            continue
        try:
            created_at = datetime.fromisoformat(
                str(raw_created_at).replace("Z", "+00:00")
            )
        except ValueError:
            continue
        local_created_at = local_from_utc(created_at, offset_minutes)
        if local_created_at.month == month:
            years.add(local_created_at.year)
    return sorted(years)


def transaction_years_for_day_month(
    transactions: List[Dict[str, Any]],
    day: int,
    month: int,
    offset_minutes: int = 0,
) -> List[int]:
    years: set[int] = set()
    for transaction in transactions:
        raw_created_at = transaction.get("created_at")
        if not raw_created_at:
            continue
        try:
            created_at = datetime.fromisoformat(
                str(raw_created_at).replace("Z", "+00:00")
            )
        except ValueError:
            continue
        local_created_at = local_from_utc(created_at, offset_minutes)
        if local_created_at.month == month and local_created_at.day == day:
            years.add(local_created_at.year)
    return sorted(years)


def resolve_question_date_window(
    question: str,
    *,
    offset_minutes: int = 0,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Resolve explicit and relative date language into a UTC time window.

    A bare month is returned for data-aware resolution by the caller, which can
    select its only available year or request clarification when needed.
    """

    current_time_utc = now or datetime.now(timezone.utc)
    if current_time_utc.tzinfo is None:
        current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)
    current_time = local_from_utc(current_time_utc, offset_minutes)

    question_lower = question.lower()
    year: Optional[int] = None
    month: Optional[int] = None
    label: Optional[str] = None

    if "this month" in question_lower:
        year = current_time.year
        month = current_time.month
        label = current_time.strftime("%B %Y")
    elif "last month" in question_lower:
        month = current_time.month - 1 or 12
        year = current_time.year if current_time.month > 1 else current_time.year - 1
        label = datetime(year, month, 1).strftime("%B %Y")
    else:
        explicit_day = re.search(
            r"\b(3[01]|[12]\d|0?[1-9])(?:st|nd|rd|th)?\s+("
            + "|".join(MONTH_NAMES)
            + r")\s*,?\s+(20\d{2})\b",
            question_lower,
        )
        explicit_month = re.search(
            r"\b(" + "|".join(MONTH_NAMES) + r")\s+(20\d{2})\b",
            question_lower,
        )
        if explicit_day:
            day = int(explicit_day.group(1))
            month_name = explicit_day.group(2)
            month = MONTH_NAMES[month_name]
            year = int(explicit_day.group(3))
            return build_calendar_day_window(
                year,
                month,
                day,
                f"{day} {month_name.title()} {year}",
                offset_minutes,
            )
        if explicit_month:
            month_name = explicit_month.group(1)
            month = MONTH_NAMES[month_name]
            year = int(explicit_month.group(2))
            label = f"{month_name.title()} {year}"
        else:
            bare_day = re.search(
                r"\b(3[01]|[12]\d|0?[1-9])(?:st|nd|rd|th)?\s+("
                + "|".join(MONTH_NAMES)
                + r")\b",
                question_lower,
            )
            if bare_day:
                day = int(bare_day.group(1))
                month_name = bare_day.group(2)
                return {
                    "bare_day": day,
                    "bare_month": month_name,
                    "month": MONTH_NAMES[month_name],
                }
            bare_month = re.search(
                r"\b(" + "|".join(MONTH_NAMES) + r")\b",
                question_lower,
            )
            if bare_month:
                month_name = bare_month.group(1)
                return {
                    "bare_month": month_name,
                    "month": MONTH_NAMES[month_name],
                }

    if year is None or month is None or label is None:
        return {}

    return build_calendar_window(
        year,
        month,
        label,
        offset_minutes,
    )


def resolve_available_date_window(
    question: str,
    transactions: List[Dict[str, Any]],
    offset_minutes: int = 0,
) -> Dict[str, Any]:
    """Resolve incomplete month/day language against persisted transaction dates."""

    date_window = resolve_question_date_window(question, offset_minutes=offset_minutes)
    if date_window.get("bare_day"):
        day = date_window["bare_day"]
        month = date_window["month"]
        month_name = date_window["bare_month"].title()
        matching_years = transaction_years_for_day_month(
            transactions,
            day,
            month,
            offset_minutes,
        )
        if len(matching_years) == 1:
            year = matching_years[0]
            return build_calendar_day_window(
                year,
                month,
                day,
                f"{day} {month_name} {year}",
                offset_minutes,
            )
        if len(matching_years) > 1:
            return {
                "clarification": (
                    f"Transactions exist for {day} {month_name} in "
                    + ", ".join(str(year) for year in matching_years)
                    + ". Please specify which year you mean."
                ),
            }
        return {
            "empty": True,
            "label": f"{day} {month_name} (no saved transactions)",
        }

    if date_window.get("bare_month"):
        month = date_window["month"]
        month_name = date_window["bare_month"].title()
        matching_years = transaction_years_for_month(transactions, month, offset_minutes)
        if len(matching_years) == 1:
            year = matching_years[0]
            return build_calendar_window(year, month, f"{month_name} {year}", offset_minutes)
        if len(matching_years) > 1:
            return {
                "clarification": (
                    f"Transactions exist for {month_name} in "
                    + ", ".join(str(year) for year in matching_years)
                    + ". Please specify which year you mean."
                ),
            }
        return {
            "empty": True,
            "label": f"{month_name} (no saved transactions)",
        }

    return date_window


def filter_transactions_to_window(
    transactions: List[Dict[str, Any]],
    start: datetime,
    end: datetime,
) -> List[Dict[str, Any]]:
    filtered_transactions: List[Dict[str, Any]] = []
    for transaction in transactions:
        raw_created_at = transaction.get("created_at")
        if not raw_created_at:
            continue
        try:
            created_at = datetime.fromisoformat(
                str(raw_created_at).replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if start <= created_at < end:
            filtered_transactions.append(transaction)
    return filtered_transactions


def select_root_cause_evidence(
    question: str,
    transactions: List[Dict[str, Any]],
    offset_minutes: int = 0,
) -> tuple[List[Dict[str, Any]], str]:
    """Apply explicit reason/domain and date scope before ML analysis.

    A specific requested failure reason must never silently fall back to the
    whole transaction set, because that produces misleading repeated reports.
    """

    question_upper = question.upper()
    requested_codes = set(
        re.findall(r"\bF0[1-9](?:\.\d{2})?\b", question_upper)
    )
    requested_reasons = {
        code
        for code, display_name in FAILURE_REASONS.items()
        if display_name.lower() in question.lower()
    }

    if requested_reasons:
        requested_codes.update(requested_reasons)

    selected = transactions
    scopes: List[str] = []

    exact_reasons = {code for code in requested_codes if "." in code}
    requested_domains = {code for code in requested_codes if "." not in code}

    if exact_reasons:
        selected = [
            transaction
            for transaction in selected
            if failure_reason_code(transaction) in exact_reasons
        ]
        scopes.append("failure reason " + ", ".join(sorted(exact_reasons)))
    elif requested_domains:
        selected = [
            transaction
            for transaction in selected
            if failure_domain_code(transaction) in requested_domains
        ]
        scopes.append("failure domain " + ", ".join(sorted(requested_domains)))

    date_window = resolve_available_date_window(
        question,
        transactions,
        offset_minutes,
    )
    if date_window.get("clarification"):
        return [], date_window["clarification"]
    if date_window.get("empty"):
        return [], "Applied question scope: " + date_window["label"] + "."
    if date_window.get("start"):
        selected = filter_transactions_to_window(
            selected,
            date_window["start"],
            date_window["end"],
        )
        scopes.append(date_window["label"])

    if scopes:
        return selected, "Applied question scope: " + "; ".join(scopes) + "."
    return selected, "No specific scope was requested; using the latest transaction set."


def classify_question_intent(
    question: str,
) -> str:
    """
    Deterministic prototype intent router.

    Possible values:
        factual_response_codes
        factual_merchants
        factual_failure_count
        factual_failure_count_with_reasons
        root_cause
    """

    question_lower = (
        question
        .lower()
        .strip()
    )

    if any(
        term in question_lower
        for term in FACTUAL_RESPONSE_CODE_TERMS
    ):
        return "factual_response_codes"

    if any(
        term in question_lower
        for term in FACTUAL_MERCHANT_TERMS
    ):
        return "factual_merchants"

    is_failure_count_question = any(
        term in question_lower
        for term in FACTUAL_FAILURE_TERMS
    ) or (
        "how many" in question_lower
        and "transaction" in question_lower
        and "fail" in question_lower
    )

    if is_failure_count_question:
        if any(
            term in question_lower
            for term in (
                "reason",
                "reasons",
                "response code",
                "response codes",
            )
        ):
            return "factual_failure_count_with_reasons"
        return "factual_failure_count"

    return "root_cause"


# =========================================================
# Deterministic Factual Answers
# =========================================================

def build_factual_answer(
    intent: str,
    transactions: List[Dict[str, Any]],
    scope_label: str = "current transactions",
) -> Dict[str, Any]:

    failed_transactions = [
        transaction
        for transaction in transactions
        if str(
            transaction.get(
                "status",
                "",
            )
        ).upper() == "FAILED"
    ]

    failure_count = len(
        failed_transactions
    )

    response_code_counts: Dict[str, int] = {}
    merchant_counts: Dict[str, int] = {}

    for transaction in failed_transactions:

        response_code = failure_reason_code(transaction)

        if response_code:
            response_code_counts[
                response_code
            ] = (
                response_code_counts.get(
                    response_code,
                    0,
                )
                + 1
            )

        merchant = str(
            transaction.get(
                "merchant",
                "Unknown",
            )
        )

        merchant_counts[
            merchant
        ] = (
            merchant_counts.get(
                merchant,
                0,
            )
            + 1
        )


    # -----------------------------------------------------
    # Factual Response-Code Question
    # -----------------------------------------------------

    if intent == "factual_response_codes":

        if not response_code_counts:
            return {
                "answer": (
                    "No failure reasons were found in the "
                    f"failed transaction set for {scope_label}."
                ),

                "response_code_counts": {},

                "dominant_response_code": None,
            }


        dominant_code = max(
            response_code_counts,
            key=response_code_counts.get,
        )

        dominant_count = (
            response_code_counts[
                dominant_code
            ]
        )

        dominant_ratio = (
            dominant_count / failure_count
            if failure_count
            else 0
        )

        code_info = (
            failure_reason_details(dominant_code)
        )

        return {
            "answer": (
                f"Failure reason {dominant_code} "
                f"({code_info['display_name']}) is the most frequent failure "
                f"reason in the failed transaction set for {scope_label}. "
                f"It appears in {dominant_count} of {failure_count} "
                f"failed transactions "
                f"({dominant_ratio * 100:.1f}%). "
                "This is an observed frequency and does not by itself "
                "prove the underlying root cause."
            ),

            "response_code_counts":
                response_code_counts,

            "dominant_response_code": {
                "code":
                    dominant_code,

                "count":
                    dominant_count,

                "ratio":
                    round(
                        dominant_ratio,
                        4,
                    ),

                "meaning":
                    code_info["display_name"],

                "category":
                    code_info["domain_name"],
            },
        }


    # -----------------------------------------------------
    # Factual Merchant Question
    # -----------------------------------------------------

    if intent == "factual_merchants":

        if not merchant_counts:
            return {
                "answer": (
                    "No affected merchants were found in the "
                    f"failed transaction set for {scope_label}."
                ),

                "merchant_failure_counts": {},

                "most_affected_merchant": None,
            }


        most_affected = max(
            merchant_counts,
            key=merchant_counts.get,
        )

        most_affected_count = (
            merchant_counts[
                most_affected
            ]
        )

        concentration = (
            most_affected_count / failure_count
            if failure_count
            else 0
        )

        return {
            "answer": (
                f"{most_affected} is the most affected merchant "
                f"with {most_affected_count} of {failure_count} "
                f"failed transactions "
                f"({concentration * 100:.1f}%) for {scope_label}. "
                "This concentration is an observed transaction fact "
                "and does not by itself establish a merchant-side "
                "root cause."
            ),

            "merchant_failure_counts":
                merchant_counts,

            "most_affected_merchant":
                most_affected,

            "concentration_ratio":
                round(
                    concentration,
                    4,
                ),
        }


    # -----------------------------------------------------
    # Factual Failure Count + Reasons Question
    # -----------------------------------------------------

    if intent == "factual_failure_count_with_reasons":

        if not response_code_counts:
            return {
                "answer": (
                    f"The transaction set for {scope_label} contains "
                    "0 failed transactions, so there are no failure reasons to report."
                ),
                "failure_count": 0,
                "response_code_counts": {},
                "dominant_response_code": None,
            }

        dominant_code = max(
            response_code_counts,
            key=response_code_counts.get,
        )
        dominant_count = response_code_counts[dominant_code]
        dominant_ratio = dominant_count / failure_count if failure_count else 0
        dominant_info = failure_reason_details(dominant_code)

        reason_summary = ", ".join(
            f"{code} ({failure_reason_details(code)['display_name']}): {count}"
            for code, count in sorted(
                response_code_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )

        return {
            "answer": (
                f"The transaction set for {scope_label} contains "
                f"{failure_count} failed transactions. Observed failure reasons: "
                f"{reason_summary}."
            ),
            "failure_count": failure_count,
            "response_code_counts": response_code_counts,
            "dominant_response_code": {
                "code": dominant_code,
                "count": dominant_count,
                "ratio": round(dominant_ratio, 4),
                "meaning": dominant_info["display_name"],
                "category": dominant_info["domain_name"],
            },
        }


    # -----------------------------------------------------
    # Factual Failure-Count Question
    # -----------------------------------------------------

    if intent == "factual_failure_count":

        return {
            "answer": (
                f"The transaction set for {scope_label} contains "
                f"{failure_count} failed transactions."
            ),

            "failure_count":
                failure_count,
        }


    raise ValueError(
        f"Unsupported factual intent: {intent}"
    )


# =========================================================
# Synthetic Scenario Routing
# =========================================================

def select_synthetic_scenario(
    question: str,
) -> Dict[str, Any]:
    """
    Prototype-only deterministic routing from the natural
    language question to a synthetic transaction scenario.

    Used only for root-cause investigations when explicit
    transactions are not supplied.
    """

    question_lower = (
        question.lower()
    )


    # -----------------------------------------------------
    # Network / Switch
    # -----------------------------------------------------

    network_terms = [
        "switch unavailable",
        "switch issue",
        "network issue",
        "network failure",
        "network connectivity",
        "code 91",
        "issuer availability",
        "issuer or switch unavailable",
    ]

    if any(
        term in question_lower
        for term in network_terms
    ):
        return {
            "scenario_id":
                "network_switch_issue",

            "scenario_reason": (
                "Question explicitly references network, "
                "switch or availability conditions."
            ),

            "transactions":
                NETWORK_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Merchant
    # -----------------------------------------------------

    merchant_terms = [
        "invalid merchant",
        "merchant configuration",
        "merchant issue",
        "merchant problem",
        "specific merchant",
        "store omega",
        "reason code 004",
    ]

    if any(
        term in question_lower
        for term in merchant_terms
    ):
        return {
            "scenario_id":
                "merchant_issue",

            "scenario_reason": (
                "Question explicitly references merchant-specific "
                "or merchant-configuration conditions."
            ),

            "transactions":
                MERCHANT_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Payment Service / Gateway
    # -----------------------------------------------------

    service_terms = [
        "payment service",
        "payment gateway",
        "gateway issue",
        "gateway failure",
        "processing service",
        "internal service",
    ]

    if any(
        term in question_lower
        for term in service_terms
    ):
        return {
            "scenario_id":
                "payment_service_issue",

            "scenario_reason": (
                "Question explicitly references payment-service "
                "or gateway processing conditions."
            ),

            "transactions":
                PAYMENT_SERVICE_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Issuer
    # -----------------------------------------------------

    issuer_terms = [
        "issuer decline",
        "issuer issue",
        "issuer problem",
        "code 05",
        "response code 05",
        "authorization decline",
        "authorisation decline",
    ]

    if any(
        term in question_lower
        for term in issuer_terms
    ):
        return {
            "scenario_id":
                "issuer_issue",

            "scenario_reason": (
                "Question explicitly references issuer decline "
                "or issuer-decision conditions."
            ),

            "transactions":
                ISSUER_TRANSACTIONS,
        }


    # -----------------------------------------------------
    # Generic Root-Cause Investigation
    # -----------------------------------------------------

    return {
        "scenario_id":
            "mixed_ambiguous",

        "scenario_reason": (
            "No specific synthetic failure scenario was requested. "
            "Using the mixed payment-failure dataset."
        ),

        "transactions":
            AMBIGUOUS_TRANSACTIONS,
    }


# =========================================================
# History Helpers
# =========================================================

def add_history_record(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    if not result.get(
        "root_cause_analysis_performed",
        False,
    ):
        factual_result = result.get(
            "factual_result",
            {},
        )

        return save_investigation(
            {
                "investigation_id": result["investigation_id"],
                "created_at": result["created_at"],
                "question": result["question"],
                "status": result["status"],
                "scenario_id": result.get("scenario_id"),
                "predicted_cause": None,
                "assessment": "factual",
                "selected_paths": [],
                "top_probability": 0,
                "probability_gap": 0,
                "validation_passed": True,
                "human_escalation_required": False,
                "failure_count": factual_result.get("failure_count", 0),
            }
        )

    record = {
        "investigation_id":
            result[
                "investigation_id"
            ],

        "created_at":
            result[
                "created_at"
            ],

        "question":
            result[
                "question"
            ],

        "status":
            result[
                "status"
            ],

        "scenario_id":
            result.get(
                "scenario_id"
            ),

        "predicted_cause":
            result[
                "ml_diagnosis"
            ].get(
                "predicted_cause"
            ),

        "assessment":
            result[
                "diagnosis_assessment"
            ].get(
                "assessment"
            ),

        "selected_paths":
            result[
                "diagnosis_assessment"
            ].get(
                "selected_paths",
                [],
            ),

        "top_probability":
            result[
                "ml_diagnosis"
            ].get(
                "top_probability",
                0,
            ),

        "probability_gap":
            result[
                "ml_diagnosis"
            ].get(
                "probability_gap",
                0,
            ),

        "validation_passed":
            result[
                "validation"
            ].get(
                "passed",
                False,
            ),

        "human_escalation_required":
            result.get(
                "human_escalation_required",
                True,
            ),

        "failure_count":
            result[
                "observed_evidence"
            ].get(
                "failure_count",
                0,
            ),
    }

    return save_investigation(record)


def get_investigation_history(
    limit: int = 20,
) -> List[Dict[str, Any]]:

    safe_limit = max(
        1,
        min(
            limit,
            100,
        ),
    )

    return list_investigations(safe_limit)


# =========================================================
# Main Investigation Service
# =========================================================

def run_investigation(
    question: str,
    transactions: Optional[
        List[Dict[str, Any]]
    ] = None,
    timezone_offset_minutes: int = 0,
) -> Dict[str, Any]:

    # -----------------------------------------------------
    # Domain Guardrail
    # -----------------------------------------------------

    if not is_payment_domain_question(
        question
    ):
        raise ValueError(
            "OUT_OF_SCOPE: This investigation service "
            "only supports payment operations, transaction "
            "failures, merchants, issuers, payment gateways "
            "and payment-network incidents."
        )


    # -----------------------------------------------------
    # Question Intent
    # -----------------------------------------------------

    question_intent = (
        classify_question_intent(
            question
        )
    )


    # -----------------------------------------------------
    # Factual / Observational Question
    #
    # These questions do not require the root-cause ML/RAG
    # pipeline.
    # -----------------------------------------------------

    if (
        question_intent !=
        "root_cause"
    ):

        date_window = resolve_available_date_window(
            question,
            transactions
            if transactions is not None
            else list_transactions(10_000),
            timezone_offset_minutes,
        )

        clarification = date_window.get(
            "clarification",
        )

        scope_label = date_window.get(
            "label",
            "current transactions",
        )

        if clarification:

            factual_result = {
                "answer": clarification,
            }

            scenario_id = "date_window_clarification"

        elif transactions is not None:

            factual_transactions = (
                transactions
            )

            if date_window.get("empty"):
                factual_transactions = []
            elif date_window:
                factual_transactions = (
                    filter_transactions_to_window(
                        factual_transactions,
                        date_window["start"],
                        date_window["end"],
                    )
                )

            scenario_id = (
                "explicit_transaction_input"
            )

        else:

            if date_window.get("empty"):
                factual_transactions = []
            elif date_window:
                factual_transactions = (
                    list_transactions_in_window(
                        date_window["start"],
                        date_window["end"],
                    )
                )
            else:
                factual_transactions = list_transactions(500)

            scenario_id = (
                "live_supabase_transactions_window"
                if date_window
                else "live_supabase_transactions"
            )


        if not clarification:
            factual_result = (
                build_factual_answer(
                    intent=
                        question_intent,

                    transactions=
                        factual_transactions,

                    scope_label=scope_label,
                )
            )


        investigation_id = (
            "INV-"
            + uuid4()
            .hex[:8]
            .upper()
        )


        created_at = (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )


        factual_response = {
            "investigation_id":
                investigation_id,

            "created_at":
                created_at,

            "question":
                question,

            "question_intent":
                question_intent,

            "status":
                "FACTUAL_ANSWER",

            "scenario_id":
                scenario_id,

            "answer":
                factual_result[
                    "answer"
                ],

            "factual_result":
                factual_result,

            "human_escalation_required":
                False,

            "root_cause_analysis_performed":
                False,

            "note": (
                "This question was answered directly from "
                "observed transaction evidence. The ML "
                "root-cause investigation pipeline was not "
                "required."
            ),
        }

        add_history_record(factual_response)

        return factual_response


    # -----------------------------------------------------
    # Root-Cause Investigation Evidence Source
    # -----------------------------------------------------

    if transactions is not None:

        selected_transactions, scope_reason = (
            select_root_cause_evidence(
                question,
                transactions,
                timezone_offset_minutes,
            )
        )

        scenario_id = (
            "explicit_transaction_input"
        )

        scenario_reason = (
            "Transaction records were supplied explicitly by the caller. "
            + scope_reason
        )

    else:

        available_transactions = list_transactions(10_000)
        selected_transactions, scope_reason = (
            select_root_cause_evidence(
                question,
                available_transactions,
                timezone_offset_minutes,
            )
        )

        scenario_id = "live_supabase_transactions"

        scenario_reason = (
            "Using persisted synthetic transactions from Supabase. "
            + scope_reason
        )


    # -----------------------------------------------------
    # Graph Input
    # -----------------------------------------------------

    graph_input: Dict[
        str,
        Any,
    ] = {
        "question":
            question,

        "transactions":
            selected_transactions,
    }


    # -----------------------------------------------------
    # Run Complete LangGraph Workflow
    # -----------------------------------------------------

    result = (
        investigation_graph.invoke(
            graph_input
        )
    )


    # -----------------------------------------------------
    # Investigation Identity
    # -----------------------------------------------------

    investigation_id = (
        "INV-"
        + uuid4()
        .hex[:8]
        .upper()
    )


    created_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


    # -----------------------------------------------------
    # ML
    # -----------------------------------------------------

    ml_diagnosis = (
        result.get(
            "ml_diagnosis",
            {},
        )
    )


    probabilities = (
        ml_diagnosis.get(
            "probabilities",
            {},
        )
    )


    # -----------------------------------------------------
    # Diagnosis Assessment
    # -----------------------------------------------------

    diagnosis_assessment = (
        result.get(
            "diagnosis_assessment",
            {},
        )
    )


    assessment = (
        diagnosis_assessment.get(
            "assessment",
            "unknown",
        )
    )


    selected_paths = (
        diagnosis_assessment.get(
            "selected_paths",
            [],
        )
    )


    # -----------------------------------------------------
    # Investigation Plan
    # -----------------------------------------------------

    investigation_plan = (
        result.get(
            "investigation_plan",
            {},
        )
    )


    # -----------------------------------------------------
    # Clean RAG Evidence
    # -----------------------------------------------------

    rag_evidence = []


    for index, evidence in enumerate(
        result.get(
            "rag_evidence",
            [],
        ),
        start=1,
    ):

        rag_evidence.append(
            {
                "id":
                    index,

                "content":
                    evidence.get(
                        "content",
                        "",
                    ),

                "semantic_score":
                    evidence.get(
                        "score",
                        0,
                    ),

                "path_relevance_score":
                    evidence.get(
                        "path_relevance_score",
                        0,
                    ),

                "matched_paths":
                    evidence.get(
                        "matched_paths",
                        [],
                    ),
            }
        )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    human_escalation_required = (
        result.get(
            "human_escalation_required",
            True,
        )
    )


    validation_result = (
        result.get(
            "validation_result",
            "",
        )
    )


    validation_passed = (
        validation_result
        .strip()
        .upper()
        .startswith(
            "VALIDATION STATUS: PASS"
        )
    )


    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    if (
        human_escalation_required
        and
        assessment ==
        "ambiguous"
    ):

        status = (
            "HUMAN_REVIEW_REQUIRED"
        )

    elif (
        human_escalation_required
    ):

        status = (
            "ESCALATED"
        )

    elif (
        assessment ==
        "clear"
    ):

        status = (
            "AI_ASSISTED_DIAGNOSIS"
        )

    else:

        status = (
            "INVESTIGATION_COMPLETE"
        )


    # -----------------------------------------------------
    # Clean API Response
    # -----------------------------------------------------

    response = {
        "investigation_id":
            investigation_id,

        "created_at":
            created_at,

        "question":
            question,

        "question_intent":
            "root_cause",

        "status":
            status,

        # ---------------------------------------------
        # Scenario Metadata
        # ---------------------------------------------

        "scenario_id":
            scenario_id,

        "scenario_reason":
            scenario_reason,

        # ---------------------------------------------
        # Observed Evidence
        # ---------------------------------------------

        "observed_evidence":
            result.get(
                "observed_transaction_evidence",
                {},
            ),

        # ---------------------------------------------
        # ML Diagnosis
        # ---------------------------------------------

        "ml_diagnosis": {
            "predicted_cause":
                ml_diagnosis.get(
                    "predicted_cause"
                ),

            "probabilities":
                probabilities,

            "top_probability":
                ml_diagnosis.get(
                    "top_probability",
                    0,
                ),

            "second_probability":
                ml_diagnosis.get(
                    "second_probability",
                    0,
                ),

            "probability_gap":
                ml_diagnosis.get(
                    "probability_gap",
                    0,
                ),
        },

        # ---------------------------------------------
        # Diagnosis Assessment
        # ---------------------------------------------

        "diagnosis_assessment": {
            "assessment":
                assessment,

            "selected_paths":
                selected_paths,

            "reason":
                diagnosis_assessment.get(
                    "reason",
                    "",
                ),

            "minimum_clear_probability":
                diagnosis_assessment.get(
                    "minimum_clear_probability"
                ),

            "minimum_clear_gap":
                diagnosis_assessment.get(
                    "minimum_clear_gap"
                ),

            # Formal probability-based grading (coverage-vs-confidence
            # policy) for every candidate cause, not just the accepted or
            # selected ones — see agent_graph.py's grade_evidence_strength().
            "evidence_map":
                diagnosis_assessment.get(
                    "evidence_map",
                    {},
                ),
        },

        # ---------------------------------------------
        # Investigation Intelligence
        # ---------------------------------------------

        "investigation_plan": {
            "selected_paths":
                investigation_plan.get(
                    "selected_paths",
                    [],
                ),

            "agent_reasoning":
                investigation_plan.get(
                    "agent_reasoning",
                    "",
                ),

            "evidence_map":
                investigation_plan.get(
                    "evidence_map",
                    {},
                ),
        },

        # ---------------------------------------------
        # Response Codes
        # ---------------------------------------------

        "response_code_analysis":
            result.get(
                "response_code_analysis",
                {},
            ),

        # ---------------------------------------------
        # RAG Evidence
        # ---------------------------------------------

        "evidence":
            rag_evidence,

        # ---------------------------------------------
        # Investigation Report
        # ---------------------------------------------

        "investigation_report":
            result.get(
                "llm_summary",
                "",
            ),

        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        "validation": {
            "result":
                validation_result,

            "passed":
                validation_passed,
        },

        # ---------------------------------------------
        # Recommendations
        # ---------------------------------------------

        "recommendations":
            result.get(
                "recommendations",
                "",
            ),

        # ---------------------------------------------
        # Human Escalation
        # ---------------------------------------------

        "human_escalation_required":
            human_escalation_required,

        "root_cause_analysis_performed":
            True,
    }


    # -----------------------------------------------------
    # Persist Root-Cause Investigations
    # -----------------------------------------------------

    add_history_record(
        response
    )


    return response
