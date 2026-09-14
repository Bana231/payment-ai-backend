import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent_graph import (
    MINIMUM_CLEAR_GAP,
    MINIMUM_CLEAR_PROBABILITY,
    investigation_graph,
)
from diagnosis_map import DIAGNOSIS_MAP
from failure_taxonomy import (
    FAILURE_DOMAINS,
    FAILURE_REASONS,
    failure_domain_code,
    failure_reason_code,
    failure_reason_details,
)
from supabase_store import (
    get_transaction,
    list_investigations,
    list_transactions,
    list_transactions_in_window,
    save_investigation,
)


# =========================================================
# Payment-Domain Guardrail
# =========================================================

# A quick, cheap "is this even a payments question" filter — if none of
# these words appear anywhere in the question, we refuse to run the full
# investigation pipeline on it (see is_payment_domain_question below).
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
    "network",
    "country",
    "countries",
    "region",
    "regions",
    "incident",
}


# Returns True if the question contains at least one payments-related
# word from the set above (case-insensitive), False otherwise.
def is_payment_domain_question(
    question: str,
) -> bool:
    question_lower = question.lower()

    return any(
        term in question_lower
        for term in PAYMENT_DOMAIN_TERMS
    )


# =========================================================
# Single-Transaction Lookup
#
# A question that names a specific synthetic transaction ID
# (e.g. "why did SIM-05BCD473606F fail") is about that one
# transaction, not an aggregate batch. Detecting the ID lets
# us answer directly from its already-known failure code
# instead of pulling hundreds of unrelated transactions into
# the ML root-cause pipeline.
# =========================================================

# Matches a synthetic transaction ID like "SIM-05BCD473606F" anywhere in
# free text (case-insensitive) — this is how we detect "the user is
# asking about ONE specific transaction" versus an aggregate question.
TRANSACTION_ID_PATTERN = re.compile(
    r"\bSIM-[0-9A-F]{12}\b",
    re.IGNORECASE,
)


# Pulls the first transaction ID out of the question text, or None if the
# question doesn't mention one.
def extract_transaction_id(
    question: str,
) -> Optional[str]:
    match = TRANSACTION_ID_PATTERN.search(question)
    return match.group(0).upper() if match else None


# Given a failure domain code (e.g. "F06"), find which of the 4 ML
# diagnosis paths (issuer_issue, merchant_issue, etc.) that domain belongs
# to, by checking DIAGNOSIS_MAP's own domain lists. Falls back to
# "unknown" if nothing matches.
def diagnosis_key_for_domain(
    domain_code: Optional[str],
) -> str:
    if not domain_code:
        return "unknown"

    for key, info in DIAGNOSIS_MAP.items():
        if domain_code in info.get("failure_domains", []):
            return key

    return "unknown"


# Builds the response shown when a question names a transaction ID that
# doesn't exist in the database at all.
def build_transaction_not_found_response(
    question: str,
    transaction_id: str,
) -> Dict[str, Any]:
    investigation_id = "INV-" + uuid4().hex[:8].upper()
    created_at = datetime.now(timezone.utc).isoformat()

    answer = (
        f"No transaction with ID {transaction_id} was found in the "
        "stored synthetic transaction records."
    )

    response = {
        "investigation_id": investigation_id,
        "created_at": created_at,
        "question": question,
        "question_intent": "single_transaction_lookup",
        "status": "FACTUAL_ANSWER",
        "scenario_id": "single_transaction_not_found",
        "answer": answer,
        "factual_result": {"answer": answer},
        "human_escalation_required": False,
        "root_cause_analysis_performed": False,
        "note": (
            "This question named a specific transaction ID, which was "
            "looked up directly and not found."
        ),
    }

    add_history_record(response)
    return response


# Builds the direct-answer response for a question that named a
# transaction ID which WAS found — this is what bypasses the whole
# ML/LLM pipeline for single-transaction questions, answering straight
# from that one row's own stored status/failure code instead of an
# aggregate analysis.
def build_transaction_lookup_response(
    question: str,
    transaction: Dict[str, Any],
) -> Dict[str, Any]:
    transaction_id = transaction.get("transaction_id", "")
    status = (transaction.get("status") or "").upper()

    investigation_id = "INV-" + uuid4().hex[:8].upper()
    created_at = datetime.now(timezone.utc).isoformat()

    # Branch A: the named transaction actually succeeded — no root-cause
    # analysis makes sense, so just say so and stop here.
    if status != "FAILED":
        answer = (
            f"Transaction {transaction_id} did not fail — its recorded "
            f"status is {status or 'UNKNOWN'}."
        )

        response = {
            "investigation_id": investigation_id,
            "created_at": created_at,
            "question": question,
            "question_intent": "single_transaction_lookup",
            "status": "FACTUAL_ANSWER",
            "scenario_id": "single_transaction_lookup_no_failure",
            "answer": answer,
            "factual_result": {"answer": answer},
            "human_escalation_required": False,
            "root_cause_analysis_performed": False,
            "note": (
                "This question named a specific transaction ID that did "
                "not fail; no root-cause diagnosis was needed."
            ),
        }

        add_history_record(response)
        return response

    # Branch B: the transaction DID fail — look up its exact failure
    # reason code and figure out which of the 4 diagnosis categories that
    # code belongs to, so we can answer as if this were a (single-row)
    # root-cause investigation.
    reason_code = failure_reason_code(transaction)
    domain_code = failure_domain_code(transaction)
    diagnosis_key = diagnosis_key_for_domain(domain_code)
    reason_details = failure_reason_details(reason_code)
    merchant = transaction.get("merchant") or "Unknown merchant"

    # A one-transaction version of the same "observed_transaction_evidence"
    # shape the full pipeline builds for a whole batch, just with every
    # count fixed at 1.
    observed_evidence = {
        "failure_count": 1,
        "merchant_failure_counts": {merchant: 1},
        "unique_affected_merchants": 1,
        "most_affected_merchant": merchant,
        "max_merchant_failure_count": 1,
        "merchant_failure_concentration_ratio": 1.0,
        "reason_code_counts": {reason_code: 1} if reason_code else {},
        "service_failure_counts": {},
        "note": (
            f"Single-transaction lookup for {transaction_id}; no "
            "aggregate batch was pulled."
        ),
    }

    # Sub-branch B1: the failure code exists but doesn't map to any of
    # our 4 known diagnosis categories — flag for human review rather
    # than guessing a cause.
    if diagnosis_key == "unknown":
        response = {
            "investigation_id": investigation_id,
            "created_at": created_at,
            "question": question,
            "question_intent": "root_cause",
            "status": "HUMAN_REVIEW_REQUIRED",
            "scenario_id": "single_transaction_lookup",
            "scenario_reason": (
                f"Question named transaction {transaction_id} directly; "
                "its failure code did not map to a known diagnosis "
                "category."
            ),
            "observed_evidence": observed_evidence,
            "ml_diagnosis": {
                "predicted_cause": None,
                "probabilities": {},
                "top_probability": 0,
                "second_probability": 0,
                "probability_gap": 0,
            },
            "diagnosis_assessment": {
                "assessment": "ambiguous",
                "selected_paths": [],
                "reason": (
                    f"Transaction {transaction_id}'s failure reason code "
                    f"({reason_code or 'unknown'}) does not map to any "
                    "known diagnosis category."
                ),
                "minimum_clear_probability": MINIMUM_CLEAR_PROBABILITY,
                "minimum_clear_gap": MINIMUM_CLEAR_GAP,
                "evidence_map": {},
            },
            "investigation_plan": {
                "selected_paths": [],
                "agent_reasoning": (
                    "Deterministic lookup found the transaction but its "
                    "failure code is not represented in the diagnosis "
                    "taxonomy."
                ),
                "evidence_map": {},
            },
            "response_code_analysis": (
                {
                    reason_code: {
                        "count": 1,
                        "meaning": reason_details["display_name"],
                        "category": reason_details["domain_name"],
                    }
                }
                if reason_code
                else {}
            ),
            # This branch's failure code doesn't map to a known
            # diagnosis category at all, so there's nothing to confirm
            # as a sub root cause either — always None/None here, kept
            # only so the response shape matches every other root-cause
            # response (the frontend always expects this key to exist).
            "sub_root_cause": {
                "code": None,
                "summary": None,
            },
            # Same reasoning as sub_root_cause above — this branch's
            # failure code has no known category at all, so there's
            # nothing to confirm per the taxonomy either.
            "confirmed_root_cause": {
                "category": None,
                "summary": None,
            },
            "evidence": [],
            "investigation_report": (
                f"Transaction {transaction_id} failed with reason code "
                f"{reason_code or 'unknown'}, which does not map to a "
                "known diagnosis category. Human review is required."
            ),
            "validation": {
                "result": "VALIDATION STATUS: PASS (deterministic single-transaction lookup)",
                "passed": True,
            },
            "recommendations": (
                "Escalate for manual review — the failure code on this "
                "transaction is not represented in the diagnosis taxonomy."
            ),
            "human_escalation_required": True,
            "root_cause_analysis_performed": True,
        }

        add_history_record(response)
        return response

    # Sub-branch B2: the failure code DOES map to a known category — build
    # a "fake" 100%-confidence ML result (since there's really only one
    # transaction, there's nothing to be uncertain about) so the rest of
    # the response can reuse the normal root-cause response shape.
    causes = [key for key in DIAGNOSIS_MAP if key != "unknown"]
    probabilities = {
        cause: (1.0 if cause == diagnosis_key else 0.0)
        for cause in causes
    }
    evidence_map = {
        cause: {
            "probability": probabilities[cause],
            "evidence_strength": (
                "strong" if cause == diagnosis_key else "negligible"
            ),
        }
        for cause in causes
    }
    diagnosis_info = DIAGNOSIS_MAP[diagnosis_key]

    llm_summary = (
        f"Transaction {transaction_id} at merchant \"{merchant}\" failed "
        f"with reason code {reason_code} ({reason_details['display_name']}), "
        f"which falls under the {diagnosis_info['name']} category "
        f"({diagnosis_info['description']}). Because this transaction's "
        "failure code is already known, no aggregate pattern analysis "
        "across other transactions was necessary."
    )

    recommendations = (
        f"Treat this as a {diagnosis_info['name'].lower()}: review "
        + (
            ", ".join(diagnosis_info["services"])
            if diagnosis_info["services"]
            else "the relevant service"
        )
        + f" logs around this transaction's timestamp for reason code {reason_code}."
    )

    response = {
        "investigation_id": investigation_id,
        "created_at": created_at,
        "question": question,
        "question_intent": "root_cause",
        "status": "AI_ASSISTED_DIAGNOSIS",
        "scenario_id": "single_transaction_lookup",
        "scenario_reason": (
            f"Question named transaction {transaction_id} directly; it "
            "was looked up by ID instead of using an aggregate batch."
        ),
        "observed_evidence": observed_evidence,
        "ml_diagnosis": {
            "predicted_cause": diagnosis_key,
            "probabilities": probabilities,
            "top_probability": 1.0,
            "second_probability": 0.0,
            "probability_gap": 1.0,
        },
        "diagnosis_assessment": {
            "assessment": "clear",
            "selected_paths": [diagnosis_key],
            "reason": (
                f"Transaction {transaction_id}'s failure reason code "
                f"{reason_code} maps directly to {diagnosis_info['name']} "
                "in the failure taxonomy."
            ),
            "minimum_clear_probability": MINIMUM_CLEAR_PROBABILITY,
            "minimum_clear_gap": MINIMUM_CLEAR_GAP,
            "evidence_map": evidence_map,
        },
        "investigation_plan": {
            "selected_paths": [diagnosis_key],
            "agent_reasoning": (
                "Deterministic lookup: the question named a specific "
                "transaction ID with an already-known failure code, so "
                "the taxonomy mapping was used directly instead of the "
                "aggregate ML pipeline."
            ),
            "evidence_map": evidence_map,
        },
        "response_code_analysis": (
            {
                reason_code: {
                    "count": 1,
                    "meaning": reason_details["display_name"],
                    "category": reason_details["domain_name"],
                }
            }
            if reason_code
            else {}
        ),
        # A single transaction's own failure code is trivially its "sub
        # root cause" — 100% of the (one) observed failure, so it's
        # always confirmed here, matching the shape the aggregate
        # root-cause path uses (see run_investigation below).
        "sub_root_cause": {
            "code": reason_code,
            "summary": (
                f"The specific failure reason {reason_code} "
                f"({reason_details['display_name']}) is this "
                "transaction's own recorded failure code — confirmed, "
                "since this is a single-transaction lookup."
            )
            if reason_code
            else None,
        },
        # Same idea one level up: this one transaction's domain maps
        # directly (via diagnosis_key_for_domain above) to exactly one of
        # the 4 taxonomy categories, so that category is trivially
        # confirmed too — there's no ML uncertainty to weigh against,
        # since there's only one transaction being looked at.
        "confirmed_root_cause": {
            "category": diagnosis_key,
            "summary": (
                f"This transaction's domain maps to \"{diagnosis_info['name']}\" "
                "in the failure taxonomy (diagnosis_map.py) — confirmed, "
                "since this is a single-transaction lookup."
            ),
        },
        "evidence": [],
        "investigation_report": llm_summary,
        "validation": {
            "result": "VALIDATION STATUS: PASS (deterministic single-transaction lookup)",
            "passed": True,
        },
        "recommendations": recommendations,
        "human_escalation_required": False,
        "root_cause_analysis_performed": True,
    }

    add_history_record(response)
    return response


# =========================================================
# Question Intent Classification
#
# We distinguish direct observational questions from
# root-cause investigation questions.
# =========================================================

# Each of these TERM lists below is a set of phrases that, if found in
# the question, hint the user wants a specific FACT (a count, a name, a
# ranking) rather than a full root-cause investigation — see
# classify_question_intent further down for how they're actually used.
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


FACTUAL_NETWORK_TERMS = [
    "which network",
    "which networks",
    "most affected network",
    "most affected networks",
    "network has most",
    "network has the most",
    "network failure count",
    "network failed most",
    "networks failed most",
]


FACTUAL_COUNTRY_TERMS = [
    "which country",
    "which countries",
    "which region",
    "which regions",
    "most affected country",
    "most affected countries",
    "most affected region",
    "most affected regions",
    "country has most",
    "country has the most",
    "region has most",
    "region has the most",
    "country failure count",
    "countries affected",
    "regions affected",
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


# Maps a lowercase month name to its 1-12 number, so "september" -> 9
# when parsing dates typed in plain English.
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


# Builds a UTC start/end window covering one whole calendar month (e.g.
# "September 2026") in the caller's local time, for questions like "this
# month" or "in September".
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


# Same idea as build_calendar_window above, but for exactly ONE day (e.g.
# "on the 14th of September") instead of a whole month.
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


# If a question says "in September" without a year, this looks at the
# actual stored transactions to figure out which year(s) had data in
# that month — so we don't have to just assume "this year".
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


# Same as transaction_years_for_month above, but narrowed to one exact
# day+month combination (e.g. "14th September") instead of a whole month.
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

    # Case 1: relative month words — "this month" / "last month".
    if "this month" in question_lower:
        year = current_time.year
        month = current_time.month
        label = current_time.strftime("%B %Y")
    elif "last month" in question_lower:
        month = current_time.month - 1 or 12
        year = current_time.year if current_time.month > 1 else current_time.year - 1
        label = datetime(year, month, 1).strftime("%B %Y")
    else:
        # Case 2: a fully explicit date like "14 September 2026".
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
        # Case 3: a month + year but no day, e.g. "September 2026".
        if explicit_month:
            month_name = explicit_month.group(1)
            month = MONTH_NAMES[month_name]
            year = int(explicit_month.group(2))
            label = f"{month_name.title()} {year}"
        else:
            # Case 4: a day + month but no year at all, e.g. "14
            # September" — the caller must resolve which year using the
            # actual transaction data (see transaction_years_for_day_month).
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
            # Case 5: just a month name alone, e.g. "in September".
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

            # A day number with no month at all ("on the 13th", "on 13th")
            # — assume the caller's current local month/year, matching how
            # "today"/"this month" already default to the current period.
            day_only = re.search(
                r"\bon\s+(?:the\s+)?(3[01]|[12]\d|0?[1-9])(?:st|nd|rd|th)\b",
                question_lower,
            )
            if day_only:
                day = int(day_only.group(1))
                return build_calendar_day_window(
                    current_time.year,
                    current_time.month,
                    day,
                    f"{day} {current_time.strftime('%B %Y')}",
                    offset_minutes,
                )

    if year is None or month is None or label is None:
        return {}

    return build_calendar_window(
        year,
        month,
        label,
        offset_minutes,
    )


# Converts a 12-hour clock hour + "am"/"pm" into plain 24-hour form, e.g.
# (2, "pm") -> 14.
def _to_24_hour(hour: int, meridiem: str) -> int:
    return (hour % 12) + (12 if meridiem == "pm" else 0)


def resolve_time_of_day_window(question: str) -> Dict[str, Any]:
    """Parse an explicit clock-time range from the question, local time.

    Recognizes "between 2pm and 3pm", "from 2 to 3pm", "2-3pm" (the first
    time borrows the second's am/pm when omitted), "from 2:48am for about
    one hour" (a start time plus an explicit duration), and a single point
    like "at 2pm" (a one-hour window starting there). Returns {} when no
    clock time is mentioned, so callers fall back to day-level scoping only.
    """

    question_lower = question.lower()

    # Try a two-sided range first: "between 2pm and 3pm" / "from 2 to 3pm"
    # / "2-3pm". The optional "(?:for\s+)?" right before "and|to" tolerates
    # a stray leftover word there (e.g. "from 2:48 AM for to 3:48 AM" —
    # someone starting to type a duration, then switching to an end-time
    # instead), so the range is still recognized instead of the whole
    # window falling back to scoping by the entire day.
    range_match = re.search(
        r"\b(?:between|from)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*"
        r"(?:for\s+)?(?:and|to)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        question_lower,
    ) or re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*-\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        question_lower,
    )

    if range_match:
        start_hour, start_minute, start_meridiem, end_hour, end_minute, end_meridiem = (
            range_match.groups()
        )
        start_meridiem = start_meridiem or end_meridiem
        end_meridiem = end_meridiem or start_meridiem
        if not start_meridiem or not end_meridiem:
            return {}
        return {
            "start_hour": _to_24_hour(int(start_hour), start_meridiem),
            "start_minute": int(start_minute or 0),
            "end_hour": _to_24_hour(int(end_hour), end_meridiem),
            "end_minute": int(end_minute or 0),
            "label": (
                f"{start_hour}{(':' + start_minute) if start_minute else ''}{start_meridiem} "
                f"to {end_hour}{(':' + end_minute) if end_minute else ''}{end_meridiem}"
            ),
        }

    # "from 2:48 AM for about one hour" / "after 2:48am for 30 minutes" —
    # a start time plus an explicit DURATION, instead of a second clock
    # time. This is checked separately from the two-sided range above
    # (which needs a second clock time) and the open-ended "after"/
    # "before" case below (which has no duration).
    #
    # The duration number itself can be a digit ("30 minutes") OR a
    # spelled-out word ("about one hour", "a couple hours" isn't
    # supported, but "an hour"/"a hour" is via the 1-word map below) —
    # people write small durations like this in plain English far more
    # often than as digits.
    duration_word_values = {
        "a": 1, "an": 1,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12,
    }
    duration_match = re.search(
        r"\b(?:from|after|starting at|starting from)\s+"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+for\s+(?:about\s+)?"
        r"(\d+(?:\.\d+)?|" + "|".join(duration_word_values) + r")\s*"
        r"(hour|hours|hr|hrs|minute|minutes|min|mins)\b",
        question_lower,
    )
    if duration_match:
        hour, minute, meridiem, duration_value, duration_unit = (
            duration_match.groups()
        )
        start_hour_24 = _to_24_hour(int(hour), meridiem)
        start_total_minutes = start_hour_24 * 60 + int(minute or 0)

        # Either a plain digit string ("30") or one of the spelled-out
        # words above ("one") — resolve whichever form matched to a
        # number before doing arithmetic on it.
        duration_number = (
            duration_word_values[duration_value]
            if duration_value in duration_word_values
            else float(duration_value)
        )

        duration_minutes = duration_number * (
            60 if duration_unit.startswith("hour") or duration_unit.startswith("hr")
            else 1
        )

        end_total_minutes = start_total_minutes + duration_minutes

        return {
            "start_hour": start_hour_24,
            "start_minute": int(minute or 0),
            # Modulo 24/60 here so a duration that crosses midnight
            # (e.g. starting 11pm for 3 hours) still lands on a valid
            # clock time — _apply_time_of_day_window already rolls the
            # END date to the next day whenever end <= start, exactly
            # like it does for a normal two-sided range.
            "end_hour": int(end_total_minutes // 60) % 24,
            "end_minute": int(end_total_minutes % 60),
            "label": (
                f"{hour}{(':' + minute) if minute else ''}{meridiem} "
                f"for {duration_value} {duration_unit}"
            ),
        }

    # Otherwise try a single point in time: "at 2pm" — treated as a
    # 1-hour window starting there.
    point_match = re.search(
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        question_lower,
    )
    if point_match:
        hour, minute, meridiem = point_match.groups()
        start_hour_24 = _to_24_hour(int(hour), meridiem)
        return {
            "start_hour": start_hour_24,
            "start_minute": int(minute or 0),
            "end_hour": (start_hour_24 + 1) % 24,
            "end_minute": int(minute or 0),
            "label": f"{hour}{(':' + minute) if minute else ''}{meridiem}",
        }

    # Open-ended: "after 1:30pm" (that time through end of day) or
    # "before 1:30pm" (start of day through that time).
    after_match = re.search(
        r"\bafter\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        question_lower,
    )
    if after_match:
        hour, minute, meridiem = after_match.groups()
        start_hour_24 = _to_24_hour(int(hour), meridiem)
        return {
            "start_hour": start_hour_24,
            "start_minute": int(minute or 0),
            "open_ended": "after",
            "label": f"after {hour}{(':' + minute) if minute else ''}{meridiem}",
        }

    before_match = re.search(
        r"\bbefore\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        question_lower,
    )
    if before_match:
        hour, minute, meridiem = before_match.groups()
        end_hour_24 = _to_24_hour(int(hour), meridiem)
        return {
            "end_hour": end_hour_24,
            "end_minute": int(minute or 0),
            "open_ended": "before",
            "label": f"before {hour}{(':' + minute) if minute else ''}{meridiem}",
        }

    return {}


# Combines a day-level window (from resolve_question_date_window) with a
# time-of-day window (from resolve_time_of_day_window) into one final
# start/end range — e.g. "yesterday between 2pm and 3pm" needs both
# pieces combined together.
def _apply_time_of_day_window(
    question: str,
    day_window: Dict[str, Any],
    offset_minutes: int,
) -> Dict[str, Any]:
    if day_window.get("clarification") or day_window.get("empty"):
        return day_window

    time_window = resolve_time_of_day_window(question)
    if not time_window:
        return day_window

    # If no day was specified at all, default to "today" in the caller's
    # local time.
    if day_window.get("start"):
        day_start_local = local_from_utc(day_window["start"], offset_minutes)
        day_label = day_window.get("label", "")
    else:
        day_start_local = local_from_utc(
            datetime.now(timezone.utc),
            offset_minutes,
        ).replace(hour=0, minute=0, second=0, microsecond=0)
        day_label = "today"

    # Handle the 3 shapes a time window can take: open-ended "after X"
    # (X through end of that day), open-ended "before X" (start of day
    # through X), or a normal two-sided start/end range.
    open_ended = time_window.get("open_ended")

    if open_ended == "after":
        start_local = day_start_local.replace(
            hour=time_window["start_hour"],
            minute=time_window["start_minute"],
            second=0,
            microsecond=0,
        )
        end_local = day_start_local + timedelta(days=1)
    elif open_ended == "before":
        start_local = day_start_local
        end_local = day_start_local.replace(
            hour=time_window["end_hour"],
            minute=time_window["end_minute"],
            second=0,
            microsecond=0,
        )
    else:
        start_local = day_start_local.replace(
            hour=time_window["start_hour"],
            minute=time_window["start_minute"],
            second=0,
            microsecond=0,
        )
        end_local = day_start_local.replace(
            hour=time_window["end_hour"],
            minute=time_window["end_minute"],
            second=0,
            microsecond=0,
        )
        if end_local <= start_local:
            end_local += timedelta(days=1)

    return {
        "start": utc_from_local_date(start_local, offset_minutes),
        "end": utc_from_local_date(end_local, offset_minutes),
        "label": (
            f"{day_label}, {time_window['label']}" if day_label else time_window["label"]
        ),
    }


def resolve_available_date_window(
    question: str,
    transactions: List[Dict[str, Any]],
    offset_minutes: int = 0,
) -> Dict[str, Any]:
    """Resolve incomplete month/day language against persisted transaction dates,
    then narrow further to an explicit clock-time range if the question has one.
    """

    return _apply_time_of_day_window(
        question,
        _resolve_day_window(question, transactions, offset_minutes),
        offset_minutes,
    )


def _resolve_day_window(
    question: str,
    transactions: List[Dict[str, Any]],
    offset_minutes: int = 0,
) -> Dict[str, Any]:
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


# Keeps only the transactions whose created_at timestamp falls inside
# [start, end) — the actual filtering step that applies whatever
# date/time window was resolved above.
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
    # Pick up any explicit failure code (like "F06" or "F06.01") typed
    # directly in the question...
    requested_codes = set(
        re.findall(r"\bF0[1-9](?:\.\d{2})?\b", question_upper)
    )
    # ...or a failure reason typed out by its plain-English name instead
    # of its code (e.g. "issuer unavailable").
    requested_reasons = {
        code
        for code, display_name in FAILURE_REASONS.items()
        if display_name.lower() in question.lower()
    }

    # A question can also name the DOMAIN by its descriptive name instead
    # of a code or exact reason — e.g. "authentication/risk gateway"
    # instead of "F08" or "3DS challenge failed". Domain names contain a
    # "/" (e.g. "Authentication / risk gateway") that people often type
    # without the surrounding spaces ("authentication/risk gateway"), so
    # both sides are normalized to "no spaces around /" before comparing.
    normalized_question = re.sub(
        r"\s*/\s*",
        "/",
        question.lower(),
    )
    requested_domain_names = {
        domain_code
        for domain_code, domain_name in FAILURE_DOMAINS.items()
        if re.sub(r"\s*/\s*", "/", domain_name.lower()) in normalized_question
    }

    if requested_reasons:
        requested_codes.update(requested_reasons)

    if requested_domain_names:
        requested_codes.update(requested_domain_names)

    selected = transactions
    scopes: List[str] = []

    # An exact reason code (has a "." like "F06.01") narrows tighter than
    # a bare domain code (just "F06") — prefer the exact one if both were
    # somehow mentioned.
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

    # Same idea, but for a specific network/country named in the question
    # (e.g. "RuPay failures in India").
    all_networks = {
        transaction.get("network")
        for transaction in transactions
        if transaction.get("network")
    }
    all_countries = {
        transaction.get("country")
        for transaction in transactions
        if transaction.get("country")
    }

    matched_networks = extract_mentioned_values(question, all_networks)
    matched_countries = extract_mentioned_values(question, all_countries)

    if matched_networks:
        selected = [
            transaction
            for transaction in selected
            if transaction.get("network") in matched_networks
        ]
        scopes.append(" / ".join(sorted(matched_networks)) + " network")

    if matched_countries:
        selected = [
            transaction
            for transaction in selected
            if transaction.get("country") in matched_countries
        ]
        scopes.append(" / ".join(sorted(matched_countries)))

    # Finally, apply whatever date/time window the question implied.
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

    # Keep the evidence set deterministic and make the most recently stored
    # matching authorization the first record considered by the investigation.
    # Supabase normally returns this order already, but an explicit sort keeps
    # it correct after every reason/domain/date filter and for caller-supplied
    # transaction lists.
    selected.sort(
        key=lambda transaction: str(transaction.get("created_at") or ""),
        reverse=True,
    )

    if scopes:
        return selected, "Applied question scope: " + "; ".join(scopes) + "."
    return selected, "No specific scope was requested; using the latest transaction set."


# Looks at keyword hints in the question to decide which of the "quick
# factual answer" branches applies (see build_factual_answer below), or
# whether this needs the full ML/LLM root-cause pipeline instead.
def classify_question_intent(
    question: str,
) -> str:
    """
    Deterministic prototype intent router.

    Possible values:
        factual_response_codes
        factual_merchants
        factual_networks
        factual_countries
        factual_networks_and_countries
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

    mentions_network = any(
        term in question_lower
        for term in FACTUAL_NETWORK_TERMS
    )

    mentions_country = any(
        term in question_lower
        for term in FACTUAL_COUNTRY_TERMS
    )

    if mentions_network and mentions_country:
        # A compound question like "which region and which network failed
        # most" needs both halves answered together, not just one.
        return "factual_networks_and_countries"

    if mentions_network:
        return "factual_networks"

    if mentions_country:
        return "factual_countries"

    # A "how many failed" style question, with or without asking for the
    # breakdown by reason as well.
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

# Some countries are stored under a short form ("USA", "UK", "UAE") that
# people don't always type verbatim — someone asking about "US" failures
# means the country stored as "USA", but "usa" never literally appears in
# the text "US". This maps each affected stored value to the other ways
# people commonly write it, so any of those alternate spellings count as
# a match too, not just the one exact stored string. These are safe to
# match case-INsensitively — they're multi-word or punctuated enough that
# they don't collide with ordinary English.
COUNTRY_ALIASES = {
    "USA": ["u.s.", "u.s.a.", "united states", "united states of america", "america"],
    "UK": ["united kingdom", "great britain", "britain"],
    "UAE": ["united arab emirates"],
    "South Korea": ["korea", "republic of korea"],
}

# Separate from COUNTRY_ALIASES above: short aliases that ARE ordinary
# English words in lowercase ("us" as in "help us") and would cause false
# matches if checked case-insensitively — but as a bare, capitalized "US"
# they unambiguously mean the country abbreviation. So these are only
# ever checked against the ORIGINAL question text, requiring the exact
# case shown here, never lowercased.
COUNTRY_CASE_SENSITIVE_ALIASES = {
    "USA": ["US"],
}


def extract_mentioned_values(
    question: str,
    candidate_values: set,
) -> List[str]:
    """Case-insensitive, whole-word match of any candidate string (a network
    or country name actually present in the transaction data) mentioned in
    the question. Used to scope a factual answer to "RuPay" or "India"
    without hardcoding a network/country list that could drift out of sync
    with the live data. Also checks COUNTRY_ALIASES /
    COUNTRY_CASE_SENSITIVE_ALIASES above, so a common alternate spelling
    ("US" for "USA") still counts as a match.
    """

    question_lower = question.lower()
    matched = []

    for value in candidate_values:
        if not value:
            continue

        found = False

        # Case-insensitive forms: the stored value itself, plus any
        # known unambiguous alternate spellings for it.
        forms_to_check = [str(value)] + COUNTRY_ALIASES.get(str(value), [])

        for form in forms_to_check:
            pattern = r"\b" + re.escape(form.lower()) + r"\b"
            if re.search(pattern, question_lower):
                found = True
                break

        # Case-SENSITIVE forms: checked against the original question,
        # exact case only, so a lowercase everyday word ("us") never
        # falsely triggers the country filter.
        if not found:
            for form in COUNTRY_CASE_SENSITIVE_ALIASES.get(str(value), []):
                pattern = r"\b" + re.escape(form) + r"\b"
                if re.search(pattern, question):
                    found = True
                    break

        if found:
            matched.append(str(value))

    return matched


# Answers one of the "quick factual" intents (network/country/merchant/
# response-code counts, etc.) directly from the transaction list, without
# ever touching the ML model or an LLM — deterministic counting only.
def build_factual_answer(
    intent: str,
    transactions: List[Dict[str, Any]],
    scope_label: str = "current transactions",
    question: str = "",
) -> Dict[str, Any]:

    # If the question names a specific network/country (e.g. "RuPay",
    # "India"), narrow the transaction set down to just that before
    # counting anything — otherwise a mention of one network's failures
    # would get diluted by every other network's numbers too.
    all_networks = {
        transaction.get("network")
        for transaction in transactions
        if transaction.get("network")
    }
    all_countries = {
        transaction.get("country")
        for transaction in transactions
        if transaction.get("country")
    }

    matched_networks = extract_mentioned_values(question, all_networks)
    matched_countries = extract_mentioned_values(question, all_countries)

    scoped_transactions = transactions
    filter_labels: List[str] = []

    if matched_networks:
        scoped_transactions = [
            transaction
            for transaction in scoped_transactions
            if transaction.get("network") in matched_networks
        ]
        filter_labels.append(" / ".join(sorted(matched_networks)) + " network")

    if matched_countries:
        scoped_transactions = [
            transaction
            for transaction in scoped_transactions
            if transaction.get("country") in matched_countries
        ]
        filter_labels.append(" / ".join(sorted(matched_countries)))

    if filter_labels:
        scope_label = scope_label + " (filtered to " + ", ".join(filter_labels) + ")"

    failed_transactions = [
        transaction
        for transaction in scoped_transactions
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

    # Running tallies filled in below, one failed transaction at a time —
    # exactly which of these actually gets used depends on `intent`.
    response_code_counts: Dict[str, int] = {}
    merchant_counts: Dict[str, int] = {}
    network_counts: Dict[str, int] = {}
    country_counts: Dict[str, int] = {}

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

        network = str(
            transaction.get(
                "network",
                "Unknown",
            )
        )

        network_counts[
            network
        ] = (
            network_counts.get(
                network,
                0,
            )
            + 1
        )

        country = str(
            transaction.get(
                "country",
                "Unknown",
            )
        )

        country_counts[
            country
        ] = (
            country_counts.get(
                country,
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
    # Factual Network Question
    # -----------------------------------------------------

    if intent == "factual_networks":

        if not network_counts:
            return {
                "answer": (
                    "No affected card networks were found in the "
                    f"failed transaction set for {scope_label}."
                ),

                "network_failure_counts": {},

                "most_affected_network": None,
            }


        most_affected_network = max(
            network_counts,
            key=network_counts.get,
        )

        most_affected_network_count = (
            network_counts[
                most_affected_network
            ]
        )

        network_concentration = (
            most_affected_network_count / failure_count
            if failure_count
            else 0
        )

        return {
            "answer": (
                f"{most_affected_network} is the most affected card network "
                f"with {most_affected_network_count} of {failure_count} "
                f"failed transactions "
                f"({network_concentration * 100:.1f}%) for {scope_label}. "
                "This concentration is an observed transaction fact "
                "and does not by itself establish a network-side "
                "root cause."
            ),

            "network_failure_counts":
                network_counts,

            "most_affected_network":
                most_affected_network,

            "network_concentration_ratio":
                round(
                    network_concentration,
                    4,
                ),
        }


    # -----------------------------------------------------
    # Factual Country Question
    # -----------------------------------------------------

    if intent == "factual_countries":

        if not country_counts:
            return {
                "answer": (
                    "No affected countries were found in the "
                    f"failed transaction set for {scope_label}."
                ),

                "country_failure_counts": {},

                "most_affected_country": None,
            }


        most_affected_country = max(
            country_counts,
            key=country_counts.get,
        )

        most_affected_country_count = (
            country_counts[
                most_affected_country
            ]
        )

        country_concentration = (
            most_affected_country_count / failure_count
            if failure_count
            else 0
        )

        return {
            "answer": (
                f"{most_affected_country} is the most affected country "
                f"with {most_affected_country_count} of {failure_count} "
                f"failed transactions "
                f"({country_concentration * 100:.1f}%) for {scope_label}. "
                "This concentration is an observed transaction fact "
                "and does not by itself establish a geography-side "
                "root cause."
            ),

            "country_failure_counts":
                country_counts,

            "most_affected_country":
                most_affected_country,

            "country_concentration_ratio":
                round(
                    country_concentration,
                    4,
                ),
        }


    # -----------------------------------------------------
    # Factual Network + Country Question (compound)
    # -----------------------------------------------------

    if intent == "factual_networks_and_countries":

        if not network_counts and not country_counts:
            return {
                "answer": (
                    "No affected networks or countries were found in the "
                    f"failed transaction set for {scope_label}."
                ),

                "network_failure_counts": {},
                "most_affected_network": None,
                "country_failure_counts": {},
                "most_affected_country": None,
            }


        most_affected_network = (
            max(network_counts, key=network_counts.get)
            if network_counts
            else None
        )

        most_affected_country = (
            max(country_counts, key=country_counts.get)
            if country_counts
            else None
        )

        network_sentence = (
            f"{most_affected_network} is the most affected network "
            f"with {network_counts[most_affected_network]} of {failure_count} "
            "failed transactions"
            if most_affected_network
            else "No affected networks were found"
        )

        country_sentence = (
            f"{most_affected_country} is the most affected country "
            f"with {country_counts[most_affected_country]} of {failure_count} "
            "failed transactions"
            if most_affected_country
            else "no affected countries were found"
        )

        return {
            "answer": (
                f"{network_sentence}, and {country_sentence.lower() if most_affected_country else country_sentence} "
                f"for {scope_label}. These concentrations are observed "
                "transaction facts and do not by themselves establish a "
                "root cause."
            ),

            "network_failure_counts":
                network_counts,

            "most_affected_network":
                most_affected_network,

            "country_failure_counts":
                country_counts,

            "most_affected_country":
                most_affected_country,

            "network_concentration_ratio":
                round(
                    network_counts[most_affected_network] / failure_count,
                    4,
                )
                if most_affected_network and failure_count
                else None,

            "country_concentration_ratio":
                round(
                    country_counts[most_affected_country] / failure_count,
                    4,
                )
                if most_affected_country and failure_count
                else None,
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
# History Helpers
# =========================================================

# Saves one finished investigation's summary into Supabase's
# investigations table, so it shows up in "recent questions" history and
# the human-feedback review flow. Builds a different, smaller record for
# a factual (deterministic, no-ML) answer versus a full root-cause one.
def add_history_record(
    result: Dict[str, Any],
    diagnostic_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    # Factual answers don't have ML predictions/evidence to store — just
    # log the basics.
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

        "diagnostic_features":
            diagnostic_features
            or {},
    }

    return save_investigation(record)


# Fetches recent past investigations for the "history" view, clamping
# the requested limit to a sane 1-100 range so a bad request can't ask
# for an absurd number of rows.
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

# THE main entry point of the whole backend — api.py's /api/investigate
# route calls this one function with the user's question and gets back
# the full response. It tries each short-circuit (single transaction,
# non-payments question, factual question) in order before finally
# falling through to the full LangGraph ML/LLM pipeline at the bottom.
def run_investigation(
    question: str,
    transactions: Optional[
        List[Dict[str, Any]]
    ] = None,
    timezone_offset_minutes: int = 0,
) -> Dict[str, Any]:

    # -----------------------------------------------------
    # Single-Transaction Lookup
    #
    # A question naming a specific transaction ID is about
    # that one transaction, not an aggregate batch — answer
    # it directly from the transaction's own known failure
    # code instead of pulling hundreds of unrelated rows
    # into the ML root-cause pipeline.
    # -----------------------------------------------------

    transaction_id = extract_transaction_id(question)

    if transaction_id is not None:
        looked_up_transaction = get_transaction(transaction_id)

        if looked_up_transaction is None:
            return build_transaction_not_found_response(
                question,
                transaction_id,
            )

        return build_transaction_lookup_response(
            question,
            looked_up_transaction,
        )


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

        # Case A: caller (e.g. a test/UI batch view) already supplied the
        # exact transaction list to use — just apply the date window to it.
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

        # Case B: no transaction list was supplied — fetch straight from
        # Supabase ourselves, either scoped to the resolved date window or
        # (if no window applies) just the latest 500.
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

                    question=question,
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
        # Sub Root Cause
        #
        # One level more specific than the 4-category ML root cause
        # above (issuer/merchant/network/payment_service): the exact
        # failure reason code (e.g. "F06.01"), but ONLY when a single
        # code clearly covers a majority of failures (see
        # analyze_response_codes' Step 3 in agent_graph.py). This is
        # deterministic — pure counting, not an ML guess — so it can be
        # confirmed even when the broader root-cause assessment above is
        # still "ambiguous". `code` is None when nothing was that
        # dominant, so the frontend can simply check for that instead of
        # trying to interpret an empty string.
        # ---------------------------------------------

        "sub_root_cause": {
            "code":
                result.get(
                    "sub_root_cause_code",
                ),

            "summary":
                result.get(
                    "sub_root_cause_summary",
                ),
        },

        # ---------------------------------------------
        # Confirmed Root Cause (per taxonomy)
        #
        # The TAXONOMY's own definition of which of the 4 categories the
        # dominant domain belongs to (see analyze_response_codes' Step 4
        # in agent_graph.py) — a fixed definition, not the ML model's
        # independent guess, so it takes priority over
        # ml_diagnosis.predicted_cause whenever it's set. `category` is
        # None when no domain was dominant enough (<50% share) or it
        # didn't map cleanly to one category, in which case the ML
        # signal above remains the only available answer.
        # ---------------------------------------------

        "confirmed_root_cause": {
            "category":
                result.get(
                    "confirmed_root_cause",
                ),

            "summary":
                result.get(
                    "confirmed_root_cause_summary",
                ),
        },

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
        response,
        diagnostic_features=result.get(
            "diagnostic_features",
            {},
        ),
    )


    return response
