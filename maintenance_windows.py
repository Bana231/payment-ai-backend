# maintenance_windows.py
#
# Turns ONE retrieved RAG passage from the "scheduled-network-maintenance-
# windows" knowledge document into structured data, then checks a
# transaction's exact timestamp against it in plain code. This is the
# same reliability standard already used for sub_root_cause/
# confirmed_root_cause in agent_graph.py: vector similarity search's job
# (see rag/retriever.py) is only to find the right passage — WHICH times
# actually apply, and whether an incident's timestamp falls inside them,
# is computed deterministically here, never left to an LLM to eyeball.
#
# Timezone note: the source document states clock times with no
# timezone, and nothing else in this codebase maps a country to a
# timezone. Every comparison here is done in UTC time-of-day (the same
# timezone `transactions.created_at` is stored in) — not real per-country
# local time.
import re
from datetime import datetime, time, timedelta
from typing import Optional, TypedDict, Union


WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
WEEKEND_DAYS = {"Saturday", "Sunday"}


class ParsedWindow(TypedDict):
    network: str
    country: str
    base_start: time
    base_end: time
    # One of the strings "daily" / "weekdays" / "weekends", or an
    # explicit set of weekday names (e.g. {"Tuesday", "Friday"}) for a
    # "on X and Y only" clause.
    applicable_days: Union[str, set]
    excluded_days: set  # from "No downtime on <Day>."
    day_of_week_extensions: dict  # weekday name -> (start, end)
    day_of_month_extensions: dict  # day-of-month int -> (start, end)
    last_weekday_extensions: dict  # weekday name -> (start, end)


class WindowCheckResult(TypedDict):
    in_window: bool
    applicable_start: Optional[time]
    applicable_end: Optional[time]
    reason: str


# A bare, unnamed time pattern re-used (as literal text, not a shared
# capture group) inside several regexes below — each regex below gives
# its own "start"/"end" names to its own two copies of this pattern, so
# there's no duplicate-group-name conflict.
_TIME_PATTERN = r"\d{1,2}:\d{2} [AP]M"

_BASE_RE = re.compile(
    r"Downtime for (?P<network>.+?) in (?:the )?(?P<country>.+?) is "
    rf"(?P<start>{_TIME_PATTERN}) to (?P<end>{_TIME_PATTERN})"
    r"(?: (?P<applicability>daily|on .+? only))?\."
)

_EXCLUDE_DAY_RE = re.compile(r"No downtime on (?P<day>\w+)\.")
_EXCLUDE_WEEKENDS_RE = re.compile(r"No downtime on weekends\.")

_DAY_EXTENSION_RE = re.compile(
    r"On (?P<day>\w+), downtime extends to "
    rf"(?P<start>{_TIME_PATTERN}) to (?P<end>{_TIME_PATTERN})"
)

_MONTH_DAY_EXTENSION_RE = re.compile(
    r"On the (?P<ordinal>\d+)(?:st|nd|rd|th) of each month, "
    rf"downtime extends to (?P<start>{_TIME_PATTERN}) to (?P<end>{_TIME_PATTERN})"
)

_LAST_WEEKDAY_EXTENSION_RE = re.compile(
    r"On the last (?P<day>\w+) of each month, "
    rf"downtime extends to (?P<start>{_TIME_PATTERN}) to (?P<end>{_TIME_PATTERN})"
)


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%I:%M %p").time()


def parse_network_window(passage_text: str) -> Optional[ParsedWindow]:
    """Parse one network's downtime paragraph into structured data.

    Returns None if `passage_text` doesn't match the document's base
    "Downtime for X in Y is ... to ..." sentence at all — e.g. RAG
    retrieved an unrelated passage. The caller treats that as "no
    maintenance-window fact available" rather than guessing.
    """
    base_match = _BASE_RE.search(passage_text)

    if not base_match:
        return None

    applicability_text = (
        base_match.group("applicability") or "daily"
    ).strip()

    if applicability_text == "daily":
        applicable_days: Union[str, set] = "daily"

    elif applicability_text == "on weekdays only":
        applicable_days = "weekdays"

    elif applicability_text == "on weekends only":
        applicable_days = "weekends"

    else:
        # "on <Day1>[, <Day2>...] and <DayN> only" — pull every weekday
        # name out from between "on" and "only".
        day_list_text = re.sub(r"^on\s+", "", applicability_text)
        day_list_text = re.sub(r"\s+only$", "", day_list_text)

        applicable_days = {
            day.strip()
            for day in re.split(r",|\band\b", day_list_text)
            if day.strip() in WEEKDAYS
        }

    excluded_days = {
        match.group("day")
        for match in _EXCLUDE_DAY_RE.finditer(passage_text)
        if match.group("day") in WEEKDAYS
    }

    if _EXCLUDE_WEEKENDS_RE.search(passage_text):
        excluded_days |= WEEKEND_DAYS

    day_of_week_extensions = {
        match.group("day"): (
            _parse_time(match.group("start")),
            _parse_time(match.group("end")),
        )
        for match in _DAY_EXTENSION_RE.finditer(passage_text)
        if match.group("day") in WEEKDAYS
    }

    day_of_month_extensions = {
        int(match.group("ordinal")): (
            _parse_time(match.group("start")),
            _parse_time(match.group("end")),
        )
        for match in _MONTH_DAY_EXTENSION_RE.finditer(passage_text)
    }

    last_weekday_extensions = {
        match.group("day"): (
            _parse_time(match.group("start")),
            _parse_time(match.group("end")),
        )
        for match in _LAST_WEEKDAY_EXTENSION_RE.finditer(passage_text)
        if match.group("day") in WEEKDAYS
    }

    return {
        "network": base_match.group("network").strip(),
        "country": base_match.group("country").strip(),
        "base_start": _parse_time(base_match.group("start")),
        "base_end": _parse_time(base_match.group("end")),
        "applicable_days": applicable_days,
        "excluded_days": excluded_days,
        "day_of_week_extensions": day_of_week_extensions,
        "day_of_month_extensions": day_of_month_extensions,
        "last_weekday_extensions": last_weekday_extensions,
    }


def _is_last_occurrence_of_weekday(moment: datetime) -> bool:
    """True if `moment`'s date is the LAST time its weekday occurs this
    month (e.g. the last Sunday of the month) — one week later would
    roll into next month."""
    return (moment + timedelta(days=7)).month != moment.month


def check_window(parsed: ParsedWindow, moment: datetime) -> WindowCheckResult:
    """Deterministically check whether `moment` falls inside the
    network's documented maintenance window on that date, accounting for
    excluded days and day-of-week / day-of-month extensions.
    """
    weekday = WEEKDAYS[moment.weekday()]
    is_weekend = weekday in WEEKEND_DAYS
    applicable_days = parsed["applicable_days"]

    day_has_no_downtime = (
        weekday in parsed["excluded_days"]
        or (applicable_days == "weekdays" and is_weekend)
        or (applicable_days == "weekends" and not is_weekend)
        or (
            isinstance(applicable_days, set)
            and weekday not in applicable_days
        )
    )

    if day_has_no_downtime:
        return {
            "in_window": False,
            "applicable_start": None,
            "applicable_end": None,
            "reason": (
                f"{parsed['network']} has no scheduled downtime on "
                f"{weekday}s."
            ),
        }

    # Start from the base daily window, then let a matching extension
    # (day-of-week, day-of-month, or last-weekday-of-month — the document
    # never combines more than one on the same network) fully replace it
    # for this specific date.
    start, end = parsed["base_start"], parsed["base_end"]
    window_label = "standard daily window"

    if weekday in parsed["day_of_week_extensions"]:
        start, end = parsed["day_of_week_extensions"][weekday]
        window_label = f"extended {weekday} window"

    elif moment.day in parsed["day_of_month_extensions"]:
        start, end = parsed["day_of_month_extensions"][moment.day]
        window_label = f"extended day-{moment.day}-of-month window"

    elif (
        weekday in parsed["last_weekday_extensions"]
        and _is_last_occurrence_of_weekday(moment)
    ):
        start, end = parsed["last_weekday_extensions"][weekday]
        window_label = f"extended last-{weekday}-of-month window"

    moment_time = moment.time()

    if start <= end:
        in_window = start <= moment_time <= end
    else:
        # Defensive: none of today's documented windows cross midnight,
        # but handle it correctly if one ever does.
        in_window = moment_time >= start or moment_time <= end

    return {
        "in_window": in_window,
        "applicable_start": start,
        "applicable_end": end,
        "reason": (
            f"{moment_time.strftime('%I:%M %p')} UTC on {weekday} "
            f"{'falls inside' if in_window else 'falls outside'} "
            f"{parsed['network']}'s {window_label} "
            f"({start.strftime('%I:%M %p')}-{end.strftime('%I:%M %p')} UTC)."
        ),
    }
