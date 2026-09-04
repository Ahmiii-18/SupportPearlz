from typing import List, Literal
from pydantic import BaseModel, Field

class GroundedResponse(BaseModel):
    answer: str = Field(
        ..., 
        description="Direct answer to the user's inquiry based strictly on the provided context."
    )
    sources: List[str] = Field(
        default_factory=list, 
        description="List of document citations explicitly utilized (e.g. ['warranty_policy.md §3.2', 'product_manual.pdf p.14'])."
    )
    confidence: Literal["high", "partial", "none"] = Field(
        ..., 
        description="Self-evaluated accuracy confidence based strictly on context availability."
    )
    answered: bool = Field(
        ..., 
        description="Set to True if query was answered from context; False if refused/unanswerable."
    )