"""Supabase persistence helpers for PayOps Sentinel.

The backend deliberately uses the service-role key only on the server.  Never
expose this key through the browser or commit it to source control.
"""

import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()
load_dotenv(".env.supabase")


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not service_role_key:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY in .env."
        )

    return create_client(url, service_role_key)


def save_investigation(record: Dict[str, Any]) -> Dict[str, Any]:
    response = get_supabase().table("investigations").insert(record).execute()
    return response.data[0]


def list_investigations(limit: int) -> List[Dict[str, Any]]:
    response = (
        get_supabase()
        .table("investigations")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data


def get_transaction(transaction_id: str) -> Dict[str, Any] | None:
    """Look up one persisted transaction for investigation context."""
    response = (
        get_supabase()
        .table("transactions")
        .select("*")
        .eq("transaction_id", transaction_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


# PostgREST enforces its own server-side max-rows cap (1000 by default)
# regardless of what `.limit()` asks for, so pulling more than that requires
# paging through with `.range()` until a page comes back short.
SUPABASE_PAGE_SIZE = 1000


def list_transactions(limit: int) -> List[Dict[str, Any]]:
    all_transactions: List[Dict[str, Any]] = []
    offset = 0

    while len(all_transactions) < limit:
        page_size = min(SUPABASE_PAGE_SIZE, limit - len(all_transactions))
        response = (
            get_supabase()
            .table("transactions")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = response.data
        all_transactions.extend(page)

        if len(page) < page_size:
            break
        offset += page_size

    return all_transactions


def list_transactions_in_window(
    start: datetime,
    end: datetime,
) -> List[Dict[str, Any]]:
    all_transactions: List[Dict[str, Any]] = []
    offset = 0

    while True:
        response = (
            get_supabase()
            .table("transactions")
            .select("*")
            .gte("created_at", start.isoformat())
            .lt("created_at", end.isoformat())
            .order("created_at", desc=True)
            .range(offset, offset + SUPABASE_PAGE_SIZE - 1)
            .execute()
        )
        page = response.data
        all_transactions.extend(page)

        if len(page) < SUPABASE_PAGE_SIZE:
            break
        offset += SUPABASE_PAGE_SIZE

    return all_transactions


def list_knowledge_documents() -> List[Dict[str, Any]]:
    return get_supabase().table("knowledge_documents").select("*").order("slug").execute().data


def list_training_examples() -> List[Dict[str, Any]]:
    return get_supabase().table("ml_training_examples").select("label, features").execute().data
