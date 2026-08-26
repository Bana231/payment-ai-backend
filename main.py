import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_graph import investigation_graph


# Load environment variables
load_dotenv()

print(
    "OpenAI key loaded:",
    bool(os.getenv("OPENAI_API_KEY")),
)


# Create FastAPI application
app = FastAPI(
    title="Payment Incident Investigation Copilot"
)


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Payment AI Backend is working",
    }


# Health-check endpoint
@app.get("/api/health")
def health_check():
    return {
        "backend": "online",
        "service": "payment-ai-backend",
    }


# Request model
class InvestigationRequest(BaseModel):
    question: str


# Main investigation endpoint
@app.post("/api/investigate")
def investigate(request: InvestigationRequest):

    # Send the user question into LangGraph
    result = investigation_graph.invoke(
        {
            "question": request.question
        }
    )

    # Return the complete investigation result
    return {
        "question": result.get(
            "question",
            request.question,
        ),

        "status": "completed",

        "failed_transaction_count": len(
            result.get(
                "failed_transactions",
                [],
            )
        ),

        "response_code_counts": result.get(
            "response_code_counts",
            {},
        ),

        "response_code_analysis": result.get(
            "response_code_analysis",
            {},
        ),

        "dominant_failure_codes": result.get(
            "dominant_failure_codes",
            [],
        ),

        "root_cause_hypothesis": result.get(
            "root_cause_hypothesis",
            [],
        ),

        "rag_evidence": result.get(
            "rag_evidence",
            [],
        ),

        "llm_summary": result.get(
            "llm_summary",
            "",
        ),

        "validation_result": result.get(
            "validation_result",
            "",
        ),

        "recommendations": result.get(
            "recommendations",
            "",
        ),

        "human_escalation_required": result.get(
            "human_escalation_required",
            False,
        ),

        "failed_transactions": result.get(
            "failed_transactions",
            [],
        ),
    }