# This file is the "front door" of the whole backend — it's the only file
# that defines actual web addresses (routes) the frontend can call, like
# /api/investigate or /api/transactions. Every route here is deliberately
# thin: it just reads the incoming request, calls into another file that
# does the real work (investigation_service.py, supabase_store.py, etc.),
# and hands the result back as JSON.
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from investigation_service import (
    get_investigation_history,
    is_cause_evidence_supported,
    run_investigation,
)
from ml_diagnosis import retrain_model
from rag.retriever import refresh_index
from supabase_store import (
    add_ml_training_example,
    count_failed_transactions,
    count_transactions,
    create_knowledge_document,
    delete_knowledge_document,
    get_investigation,
    list_knowledge_documents,
    list_pending_investigations,
    list_transactions,
    submit_investigation_feedback,
)


# =========================================================
# FastAPI
# =========================================================

# Creates the actual web server application. Everything below registers
# routes onto this one `app` object.
app = FastAPI(
    title="PayOps Sentinel API",
    description=(
        "Agentic AI payment incident "
        "investigation backend."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

# By default, a browser blocks a webpage from calling an API running on a
# different port (e.g. the frontend on :8080 calling this backend on
# :8000) — this explicitly allows exactly those specific local addresses
# to call in, and nothing else.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Paths
# =========================================================

# NOTE: this folder-based knowledge base is LEGACY/UNUSED. The functions
# further down that read from it (load_knowledge_documents and friends)
# are never actually called by the live /api/knowledge-base route below —
# that route reads from Supabase instead. Kept only for reference.
KNOWLEDGE_BASE_DIR = Path(
    "knowledge_base"
)


# =========================================================
# Request Models
# =========================================================
# Each class below defines exactly what a request body must look like for
# one route — FastAPI automatically rejects a request that doesn't match
# (e.g. missing a required field) before our own code even runs.

# One synthetic transaction, as sent by a caller providing its own
# transaction list instead of using whatever's already stored in Supabase.
class TransactionInput(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    merchant: str
    status: str
    response_code: str
    reason_code: Optional[str] = None
    service: str


# The body of a POST /api/investigate request — a plain-English question,
# optionally an explicit list of transactions to investigate instead of
# pulling from Supabase, and the caller's timezone (so date words like
# "today" resolve to the asker's actual calendar day).
class InvestigationRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language payment "
            "investigation question."
        ),
    )

    transactions: Optional[
        List[TransactionInput]
    ] = None

    timezone_offset_minutes: int = Field(
        default=0,
        description=(
            "The caller's UTC offset in minutes, JS Date.getTimezoneOffset() "
            "convention. Lets date language like 'this month' or '4th "
            "September' resolve against the caller's calendar day rather "
            "than the UTC calendar day the data happens to be stored in."
        ),
    )


# The body of a human reviewer's feedback on one investigation — did they
# confirm the AI's diagnosis, override it with a different cause, or
# reject it outright?
class InvestigationFeedbackInput(BaseModel):
    decision: str = Field(
        ...,
        description="One of: confirmed, overridden, rejected.",
    )
    cause: Optional[str] = Field(
        default=None,
        description=(
            "The diagnosis key the reviewer confirms or overrides to "
            "(e.g. issuer_issue). Required for 'confirmed'/'overridden'."
        ),
    )
    notes: Optional[str] = None
    reviewer: Optional[str] = None
    role: str = Field(
        default="user",
        description=(
            "Which profile is submitting this review: 'user' or 'admin'. "
            "No password behind this — it's a workflow gate, not real "
            "authentication. Admin can lock a review even when the "
            "evidence doesn't clearly support the chosen cause; a user "
            "submission that doesn't clear the evidence bar goes to "
            "pending_admin_approval instead of locking immediately."
        ),
    )


# The body of an admin's decision on a pending (not-yet-locked) review —
# approve the user's original proposal as-is, override it with a
# different cause, or reject it and send the investigation back to
# needing review.
class AdminResolutionInput(BaseModel):
    admin_action: str = Field(
        ...,
        description="One of: approve, override, reject.",
    )
    cause: Optional[str] = Field(
        default=None,
        description="Required when admin_action is 'override'.",
    )
    notes: Optional[str] = None
    reviewer: str = Field(
        ...,
        description="The admin's name, for the audit trail.",
    )


# The body of a "create/update a knowledge base document" request — used
# by the Knowledge Base page's "+" button.
class KnowledgeDocumentInput(BaseModel):
    title: str = Field(..., min_length=1)
    category: str = Field(default="Policy", min_length=1)
    tags: List[str] = Field(default_factory=list)
    content: str = Field(..., min_length=1)
    slug: Optional[str] = Field(
        default=None,
        description=(
            "URL-safe identifier. Auto-derived from the title when omitted. "
            "Posting an existing slug updates that document in place."
        ),
    )


# =========================================================
# Health
# =========================================================

# The simplest possible route — just proves the server is alive and
# responding. Used by monitoring/uptime checks, not the app itself.
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "payops-sentinel",
    }


# =========================================================
# Investigation
# =========================================================

# The main entry point of the whole system — takes a plain-English
# question and returns either a direct factual answer or a full
# root-cause investigation report. All the actual thinking happens in
# investigation_service.run_investigation(); this route just calls it
# and translates its one special error (out-of-scope questions) into a
# proper HTTP error the frontend can show nicely.
@app.post("/api/investigate")
def investigate(
    request: InvestigationRequest,
) -> Dict[str, Any]:

    # If the caller supplied their own transaction list, convert each one
    # from a validated Pydantic object into a plain dictionary. Otherwise
    # leave this as None, meaning "pull transactions from Supabase instead".
    transactions = None

    if request.transactions is not None:
        transactions = [
            transaction.model_dump()
            for transaction
            in request.transactions
        ]

    try:
        return run_investigation(
            question=request.question,
            transactions=transactions,
            timezone_offset_minutes=request.timezone_offset_minutes,
        )

    except ValueError as error:
        message = str(error)

        # run_investigation raises a plain ValueError starting with this
        # exact prefix when a question isn't about payments at all (e.g.
        # "what's the weather") — turn that into a proper 422 response
        # instead of a generic crash.
        if message.startswith(
            "OUT_OF_SCOPE:"
        ):
            raise HTTPException(
                status_code=422,
                detail=message.replace(
                    "OUT_OF_SCOPE:",
                    "",
                ).strip(),
            )

        raise


# =========================================================
# Investigation History
# =========================================================

# Returns the most recent past investigations — this is what powers the
# "recent questions" list and any history/audit views in the frontend.
@app.get("/api/investigations")
def investigation_history(
    limit: int = 20,
) -> Dict[str, Any]:

    investigations = (
        get_investigation_history(
            limit=limit
        )
    )

    return {
        "count": len(
            investigations
        ),
        "investigations":
            investigations,
    }


# Every investigation currently awaiting an admin decision — powers the
# Approvals page's queue. Oldest first, so an admin works through it in
# arrival order.
@app.get("/api/investigations/pending")
def pending_investigations() -> Dict[str, Any]:

    investigations = (
        list_pending_investigations()
    )

    return {
        "count": len(
            investigations
        ),
        "investigations":
            investigations,
    }


# Lets a human reviewer confirm, override, or reject a specific
# investigation's diagnosis. Three rules on top of the plain save:
#   1. One review per investigation — a second submission on an already
#      reviewed (locked or pending) investigation is rejected, not
#      silently overwritten.
#   2. A 'rejected' decision (no cause chosen) always locks immediately —
#      there's nothing evidence-checkable about "no valid cause".
#   3. A 'confirmed'/'overridden' decision only locks immediately when
#      the reviewer is Admin, or the chosen cause is graded strong/
#      moderate in this investigation's persisted evidence_map. A User
#      picking a cause the evidence doesn't support gets saved as
#      pending_admin_approval instead — not trained on until an admin
#      resolves it via the /feedback/resolve route below.
@app.post("/api/investigations/{investigation_id}/feedback")
def submit_investigation_feedback_route(
    investigation_id: str,
    feedback: InvestigationFeedbackInput,
) -> Dict[str, Any]:

    # Basic input validation before touching the database at all.
    if feedback.decision not in {"confirmed", "overridden", "rejected"}:
        raise HTTPException(
            status_code=422,
            detail="decision must be one of: confirmed, overridden, rejected.",
        )

    if feedback.decision in {"confirmed", "overridden"} and not feedback.cause:
        raise HTTPException(
            status_code=422,
            detail="cause is required when decision is confirmed or overridden.",
        )

    if feedback.role not in {"user", "admin"}:
        raise HTTPException(
            status_code=422,
            detail="role must be one of: user, admin.",
        )

    existing = get_investigation(investigation_id)

    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=f"No investigation found with id {investigation_id}.",
        )

    if existing.get("review_status") is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This investigation has already been reviewed "
                f"(status: {existing['review_status']}) — a second "
                "submission is not allowed. See /feedback/resolve if "
                "it's still pending admin approval."
            ),
        )

    locks_immediately = (
        feedback.decision == "rejected"
        or feedback.role == "admin"
        or is_cause_evidence_supported(
            existing.get("evidence_map", {}),
            feedback.cause,
        )
    )

    base_update = {
        "human_feedback_decision": feedback.decision,
        "human_feedback_cause": feedback.cause,
        "human_feedback_notes": feedback.notes,
        "human_feedback_reviewer": feedback.reviewer,
        "human_feedback_role": feedback.role,
        "human_feedback_at": datetime.now(timezone.utc).isoformat(),
    }

    if locks_immediately:
        updated = submit_investigation_feedback(
            investigation_id,
            {
                **base_update,
                "review_status": "locked",
            },
        )
    else:
        # Saved, but not final — a User proposed a cause the evidence
        # doesn't clearly support, so it waits for an Admin decision
        # instead of locking (and training) unchecked.
        grade = (
            existing.get("evidence_map", {})
            .get(feedback.cause, {})
            .get("evidence_strength", "not evaluated")
        )
        updated = submit_investigation_feedback(
            investigation_id,
            {
                **base_update,
                "review_status": "pending_admin_approval",
                "pending_reason": (
                    f"Evidence for '{feedback.cause}' is graded "
                    f"'{grade}' — below the strong/moderate bar for "
                    "locking without admin approval."
                ),
            },
        )

    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"No investigation found with id {investigation_id}.",
        )

    # Close the loop: a confirmed/overridden human decision is a genuine
    # labelled example, so feed it back into the training set and retrain
    # immediately — but only once it's actually locked. A pending review
    # never trains the model until an admin resolves it. A single added
    # example has a modest effect on a 400+-example Random Forest — this
    # is a real feedback loop, not an instant fix for one question.
    training_example_added = False
    diagnostic_features = updated.get("diagnostic_features")

    if (
        locks_immediately
        and feedback.decision in {"confirmed", "overridden"}
        and diagnostic_features
    ):
        add_ml_training_example(feedback.cause, diagnostic_features)
        retrain_model()
        training_example_added = True

    updated["ml_training_example_added"] = training_example_added

    return updated


# Lets an Admin resolve a pending (not-yet-locked) review: approve the
# User's original proposal as-is, override it with a different cause, or
# reject it outright and send the investigation back to needing review.
# This is the ONLY way a pending review's evidence-unsupported cause ever
# reaches the ML training set — and even then, only via 'approve' or
# 'override', never 'reject'.
@app.post("/api/investigations/{investigation_id}/feedback/resolve")
def resolve_investigation_feedback_route(
    investigation_id: str,
    resolution: AdminResolutionInput,
) -> Dict[str, Any]:

    if resolution.admin_action not in {"approve", "override", "reject"}:
        raise HTTPException(
            status_code=422,
            detail="admin_action must be one of: approve, override, reject.",
        )

    if resolution.admin_action == "override" and not resolution.cause:
        raise HTTPException(
            status_code=422,
            detail="cause is required when admin_action is 'override'.",
        )

    existing = get_investigation(investigation_id)

    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=f"No investigation found with id {investigation_id}.",
        )

    if existing.get("review_status") != "pending_admin_approval":
        raise HTTPException(
            status_code=409,
            detail=(
                "This investigation is not awaiting admin approval "
                f"(status: {existing.get('review_status')})."
            ),
        )

    now = datetime.now(timezone.utc).isoformat()
    final_cause: Optional[str] = None

    if resolution.admin_action == "approve":

        # The User's original proposal stands, exactly as submitted —
        # this is the moment it finally locks and can train the model.
        update = {
            "review_status": "locked",
            "admin_decision": "approved",
            "admin_reviewer": resolution.reviewer,
            "admin_decided_at": now,
        }
        final_cause = existing.get("human_feedback_cause")

    elif resolution.admin_action == "override":

        # Admin's own cause replaces the User's proposal. The original
        # proposal is preserved separately so the audit trail still
        # shows what the User actually proposed, not just Admin's answer.
        update = {
            "review_status": "locked",
            "admin_decision": "overridden",
            "admin_reviewer": resolution.reviewer,
            "admin_decided_at": now,
            "original_proposed_cause": existing.get("human_feedback_cause"),
            "human_feedback_cause": resolution.cause,
            "human_feedback_decision": "overridden",
            "human_feedback_notes": resolution.notes,
        }
        final_cause = resolution.cause

    else:  # reject

        # Neither answer stands — reset back to "not yet reviewed" so a
        # fresh review can be submitted, rather than leaving a rejected
        # proposal stuck in place. Nothing here ever trains the model.
        update = {
            "review_status": None,
            "admin_decision": "rejected",
            "admin_reviewer": resolution.reviewer,
            "admin_decided_at": now,
            "pending_reason": None,
            "human_feedback_decision": None,
            "human_feedback_cause": None,
            "human_feedback_notes": None,
            "human_feedback_reviewer": None,
            "human_feedback_role": None,
            "human_feedback_at": None,
        }

    updated = submit_investigation_feedback(
        investigation_id,
        update,
    )

    training_example_added = False
    diagnostic_features = (updated or {}).get("diagnostic_features")

    if (
        resolution.admin_action in {"approve", "override"}
        and final_cause
        and diagnostic_features
    ):
        add_ml_training_example(final_cause, diagnostic_features)
        retrain_model()
        training_example_added = True

    updated["ml_training_example_added"] = training_example_added

    return updated


# Returns the stored synthetic transactions. `limit` has no upper cap —
# `count` is always the TRUE total row count in the table (a separate,
# cheap count="exact" query), never just "however many rows we happened
# to return", so it keeps climbing correctly past any request size.
# Powers the Dashboard and Transactions pages.
@app.get("/api/transactions")
def transaction_history(
    limit: int = 100,
) -> Dict[str, Any]:
    """Return persisted synthetic authorization records from Supabase."""

    safe_limit = max(1, limit)
    transactions = list_transactions(safe_limit)

    return {
        "count": count_transactions(),
        "failed_count": count_failed_transactions(),
        "transactions": transactions,
    }


# =========================================================
# Knowledge Base Helpers
# =========================================================
# UNUSED / LEGACY: everything in this section (down to
# load_knowledge_documents) was written for an earlier version of the app
# that read knowledge-base documents from local .txt/.md files on disk.
# The live /api/knowledge-base route below no longer calls any of these —
# it reads documents from Supabase instead (via list_knowledge_documents
# in supabase_store.py). Left here for reference only; safe to ignore.

def clean_title(
    file_path: Path,
    content: str,
) -> str:
    """
    Prefer the first Markdown heading when available.
    Otherwise derive a readable title from the filename.
    """

    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            title = (
                stripped
                .lstrip("#")
                .strip()
            )

            if title:
                return title

    return (
        file_path.stem
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


def build_summary(
    content: str,
) -> str:
    """
    Create a short preview from the first useful line.
    """

    useful_lines = []

    for line in content.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        useful_lines.append(
            stripped
        )

        if len(
            " ".join(
                useful_lines
            )
        ) >= 180:
            break

    summary = " ".join(
        useful_lines
    )

    if len(summary) > 220:
        return (
            summary[:217]
            + "..."
        )

    return summary


def infer_category(
    file_path: Path,
) -> str:
    """
    Derive a display category from the knowledge-base folder.
    """

    try:
        relative = (
            file_path.relative_to(
                KNOWLEDGE_BASE_DIR
            )
        )

        if len(
            relative.parts
        ) > 1:
            return (
                relative.parts[0]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )

    except ValueError:
        pass

    filename = (
        file_path.stem.lower()
    )

    if "reason" in filename:
        return "Reason Codes"

    if "response" in filename:
        return "Response Codes"

    if "runbook" in filename:
        return "Runbook"

    if "merchant" in filename:
        return "Merchant"

    if "issuer" in filename:
        return "Issuer"

    if "network" in filename:
        return "Network"

    return "Payment Operations"


def build_tags(
    file_path: Path,
    category: str,
) -> List[str]:
    values = {
        "rag",
        "payment-ops",
        category.lower().replace(
            " ",
            "-",
        ),
    }

    filename = (
        file_path.stem.lower()
    )

    for term in [
        "issuer",
        "merchant",
        "network",
        "gateway",
        "response",
        "reason",
        "authorization",
        "runbook",
    ]:
        if term in filename:
            values.add(term)

    return sorted(values)


def load_knowledge_documents() -> List[
    Dict[str, Any]
]:
    """
    Read the real local knowledge-base documents.

    These are the same backend source files used by
    the prototype knowledge retrieval layer.
    """

    if not KNOWLEDGE_BASE_DIR.exists():
        return []

    supported_extensions = {
        ".txt",
        ".md",
    }

    documents = []

    files = sorted(
        file_path
        for file_path
        in KNOWLEDGE_BASE_DIR.rglob("*")
        if (
            file_path.is_file()
            and
            file_path.suffix.lower()
            in supported_extensions
        )
    )

    for index, file_path in enumerate(
        files,
        start=1,
    ):
        try:
            content = (
                file_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            UnicodeDecodeError,
            OSError,
        ):
            continue

        category = (
            infer_category(
                file_path
            )
        )

        relative_path = str(
            file_path.relative_to(
                KNOWLEDGE_BASE_DIR
            )
        )

        documents.append(
            {
                "id":
                    f"KB-{index:03d}",

                "path":
                    relative_path,

                "title":
                    clean_title(
                        file_path,
                        content,
                    ),

                "summary":
                    build_summary(
                        content
                    ),

                "category":
                    category,

                "tags":
                    build_tags(
                        file_path,
                        category,
                    ),

                "content":
                    content,
            }
        )

    return documents


# =========================================================
# Knowledge Base API
# =========================================================

# Returns every RAG knowledge-base document (runbooks + policies) stored
# in Supabase — this is what the Knowledge Base page displays, and it's
# also the same source rag/retriever.py embeds for investigation retrieval.
@app.get("/api/knowledge-base")
def knowledge_base() -> Dict[str, Any]:

    documents = list_knowledge_documents()

    return {
        "count":
            len(documents),

        "source":
            "knowledge_base",

        "rag_grounding_source":
            True,

        "documents":
            documents,
    }


# Turns a human-readable title like "My New Policy!" into a safe,
# URL-friendly identifier like "my-new-policy" — lowercase, no
# punctuation, words joined with hyphens.
def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower())
    return slug.strip("-") or "document"


# Adds a new knowledge-base document (or overwrites an existing one with
# the same slug) — this is what the Knowledge Base page's "+" button
# calls. After saving, it immediately rebuilds the in-memory RAG search
# index (refresh_index) so the new content is searchable right away
# instead of only after a server restart.
@app.post("/api/knowledge-base")
def create_knowledge_base_document(
    document: KnowledgeDocumentInput,
) -> Dict[str, Any]:

    slug = document.slug or slugify(document.title)

    record = create_knowledge_document(
        {
            "slug": slug,
            "title": document.title,
            "category": document.category,
            "tags": document.tags,
            "content": document.content,
        }
    )

    # New/edited content is invisible to RAG retrieval until the in-memory
    # chunk index is rebuilt — it's only computed once at process import time.
    refresh_index()

    return record


# Permanently removes one knowledge-base document by its slug — this is
# what the Knowledge Base page's delete button calls. Rebuilds the RAG
# search index afterward too, the same way creating a document does, so
# the deleted content stops showing up in investigations right away
# instead of only after a server restart.
@app.delete("/api/knowledge-base/{slug}")
def delete_knowledge_base_document(
    slug: str,
) -> Dict[str, Any]:

    deleted_record = delete_knowledge_document(slug)

    if deleted_record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No knowledge document found with slug '{slug}'.",
        )

    refresh_index()

    return deleted_record
