"""Supabase persistence helpers for PayOps Sentinel.

The backend deliberately uses the service-role key only on the server.  Never
expose this key through the browser or commit it to source control.
"""

import os
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


def list_transactions(limit: int) -> List[Dict[str, Any]]:
    response = (
        get_supabase()
        .table("transactions")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data
