from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .core.main_engine import HADSECEEngineV7
import datetime
import uuid

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  # React app's default port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global list to store decision history (for demonstration purposes)
decision_history = []

# Global list to store Human-in-the-Loop requests
hil_requests = []

class Transaction(BaseModel):
    transaction_amount: int
    destination_country_risk: str
    purpose: str
    security_risk_level: str

class TextDescription(BaseModel):
    description: str

class HILRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    original_decision: dict
    proposed_delta: dict | None = None # Proposed delta can be null initially
    evidence_snippets: str
    status: str = "pending"
    human_decision: dict | None = None
    human_reasoning: str | None = None

class HILDecision(BaseModel):
    request_id: str
    decision: str  # "approved" or "rejected"
    reasoning: str | None = None

@app.post("/evaluate")
def evaluate_transaction(transaction: Transaction):
    engine = HADSECEEngineV7()
    decision = engine.constitutional_decision_flow(transaction.dict())
    
    # Store the decision with a timestamp
    decision_with_timestamp = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transaction_input": transaction.dict(),
        "decision_output": decision
    }
    decision_history.append(decision_with_timestamp)

    if decision.get("require_interpretation"):
        hil_request = HILRequest(
            original_decision=decision,
            proposed_delta=None, # AI doesn't propose a delta, human does
            evidence_snippets=f"Transaction Input: {transaction.json()}"
        )
        hil_requests.append(hil_request)
        return {"status": "HIL_REQUIRED", "hil_request_id": hil_request.id, "original_decision": decision}

    return decision

@app.get("/get-decision-history")
def get_decision_history():
    return decision_history

@app.post("/evaluate-text")
def evaluate_text_description(text_description: TextDescription):
    engine = HADSECEEngineV7()
    evaluation_result = engine.evaluate_text(text_description.description)

    if evaluation_result.get("require_interpretation"):
        hil_request = HILRequest(
            original_decision=evaluation_result,
            proposed_delta=None, # AI doesn't propose a delta, human does
            evidence_snippets=f"Text Input: {text_description.description}"
        )
        hil_requests.append(hil_request)
        return {"status": "HIL_REQUIRED", "hil_request_id": hil_request.id, "original_decision": evaluation_result}

    return {"evaluation": evaluation_result}

@app.post("/log-error")
def log_error(error: dict):
    print("--- FRONTEND ERROR ---")
    print(error)
    print("----------------------")
    return {"status": "error logged"}

@app.post("/create-hil-request", response_model=HILRequest)
def create_hil_request(request: HILRequest):
    hil_requests.append(request)
    return request

@app.get("/get-hil-requests", response_model=list[HILRequest])
def get_hil_requests():
    return [req for req in hil_requests if req.status == "pending"]

@app.post("/submit-hil-decision")
def submit_hil_decision(decision: HILDecision):
    for req in hil_requests:
        if req.id == decision.request_id:
            if req.status != "pending":
                raise HTTPException(status_code=400, detail="HIL request already processed")
            req.status = decision.decision  # "approved" or "rejected"
            req.human_decision = {"decision": decision.decision}
            req.human_reasoning = decision.reasoning
            return {"status": "HIL decision recorded"}
    raise HTTPException(status_code=404, detail="HIL request not found")

@app.get("/")
def read_root():
    return {"message": "Eidetic Continuum Engine is running."}