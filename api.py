from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from investigation_service import (
    get_investigation_history,
    run_investigation,
)
from supabase_store import list_knowledge_documents, list_transactions


# =========================================================
# FastAPI
# =========================================================

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

KNOWLEDGE_BASE_DIR = Path(
    "knowledge_base"
)


# =========================================================
# Request Models
# =========================================================

class TransactionInput(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    merchant: str
    status: str
    response_code: str
    reason_code: Optional[str] = None
    service: str


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


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "payops-sentinel",
    }


# =========================================================
# Investigation
# =========================================================

@app.post("/api/investigate")
def investigate(
    request: InvestigationRequest,
) -> Dict[str, Any]:

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
        )

    except ValueError as error:
        message = str(error)

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


@app.get("/api/transactions")
def transaction_history(
    limit: int = 100,
) -> Dict[str, Any]:
    """Return persisted synthetic authorization records from Supabase."""

    safe_limit = max(1, min(limit, 500))
    transactions = list_transactions(safe_limit)

    return {
        "count": len(transactions),
        "transactions": transactions,
    }


# =========================================================
# Knowledge Base Helpers
# =========================================================

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
