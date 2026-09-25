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


# Read the two secret values (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) out of
# whichever .env file has them — .env first, then .env.supabase as a
# fallback, so either file (or both) can hold the real credentials.
load_dotenv()
load_dotenv(".env.supabase")


# Builds the one shared connection object used to talk to Supabase.
# @lru_cache(maxsize=1) means this only actually runs ONCE no matter how
# many times it's called — every later call just re-uses the same
# already-built connection instead of creating a new one each time.
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


# Everything below this point is a small, single-purpose function that
# reads or writes one specific table. None of them contain any business
# logic — they're just "talk to the database" helpers that the rest of
# the app calls instead of writing raw Supabase queries everywhere.


# Saves one finished investigation record (the full root-cause / factual
# answer) into the investigations table, and hands back the row Supabase
# actually stored (including anything the database filled in itself).
def save_investigation(record: Dict[str, Any]) -> Dict[str, Any]:
    response = get_supabase().table("investigations").insert(record).execute()
    return response.data[0]


# Looks up one specific past investigation by its ID. Returns None if no
# investigation with that ID exists, instead of raising an error.
def get_investigation(investigation_id: str) -> Dict[str, Any] | None:
    response = (
        get_supabase()
        .table("investigations")
        .select("*")
        .eq("investigation_id", investigation_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


# Adds one new labelled example to the ML training set — this is how a
# human's "confirmed"/"overridden" feedback on an investigation turns
# into something the model can actually learn from (see ml_diagnosis.py's
# retrain_model, which gets called right after this).
def add_ml_training_example(label: str, features: Dict[str, Any]) -> Dict[str, Any]:
    response = (
        get_supabase()
        .table("ml_training_examples")
        .insert({"label": label, "features": features})
        .execute()
    )
    return response.data[0]


# Updates an existing investigation with a human reviewer's decision
# (confirmed / overridden / rejected, plus their notes). Returns None if
# the investigation_id doesn't exist, instead of raising an error.
def submit_investigation_feedback(
    investigation_id: str,
    feedback: Dict[str, Any],
) -> Dict[str, Any] | None:
    response = (
        get_supabase()
        .table("investigations")
        .update(feedback)
        .eq("investigation_id", investigation_id)
        .execute()
    )
    return response.data[0] if response.data else None


# Fetches the most recent investigations, newest first, up to `limit` of
# them — this is what powers the "recent questions" / history views.
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


# Every investigation currently awaiting an admin decision (see
# api.py's /feedback and /feedback/resolve routes) — oldest first, so an
# admin works through the queue in the order things actually came in.
# Capped at 200: a prototype-scale queue, not meant to page past that.
def list_pending_investigations() -> List[Dict[str, Any]]:
    response = (
        get_supabase()
        .table("investigations")
        .select("*")
        .eq("review_status", "pending_admin_approval")
        .order("human_feedback_at")
        .limit(200)
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


# Fetches up to `limit` transactions, newest first. Because of the 1000-row
# cap explained above, this can't just ask Supabase for everything in one
# go once `limit` is bigger than 1000 — instead it fetches in pages of
# 1000, gluing the pages together, and stops early either once it has
# enough rows or once a page comes back with fewer rows than requested
# (meaning there's nothing left in the table).
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


# Same paging idea as list_transactions above, but instead of "give me the
# first N", this asks for "give me every transaction created between
# these two exact timestamps" — used whenever a question is scoped to a
# specific date/time window (e.g. "yesterday", "after 1:30pm").
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


# Fetches every document in the RAG knowledge base (runbooks + policies),
# alphabetically by slug — this is the raw material rag_loader.py chunks
# and embeds for retrieval.
def list_knowledge_documents() -> List[Dict[str, Any]]:
    return get_supabase().table("knowledge_documents").select("*").order("slug").execute().data


# Adds a brand-new knowledge document, OR overwrites an existing one that
# already has the same slug (that's what "upsert ... on_conflict=slug"
# means) — this is what the Knowledge Base page's "+" button calls.
def create_knowledge_document(record: Dict[str, Any]) -> Dict[str, Any]:
    response = (
        get_supabase()
        .table("knowledge_documents")
        .upsert(record, on_conflict="slug")
        .execute()
    )
    return response.data[0]


# Permanently removes one knowledge document by its slug — this is what
# the Knowledge Base page's delete button calls. Returns the deleted
# row (so the caller can confirm what was actually removed), or None if
# no document with that slug existed.
def delete_knowledge_document(slug: str) -> Dict[str, Any] | None:
    response = (
        get_supabase()
        .table("knowledge_documents")
        .delete()
        .eq("slug", slug)
        .execute()
    )
    return response.data[0] if response.data else None


# Fetches every labelled example the ML model can learn from — this is
# the full "textbook" ml_diagnosis.py's retrain_model() reads before
# fitting the Random Forest.
def list_training_examples() -> List[Dict[str, Any]]:
    return get_supabase().table("ml_training_examples").select("label, features").execute().data
