from typing import List
from pydantic import BaseModel, Field

class AgentResponse(BaseModel):
    answer: str = Field(
        description="The detailed response to provide to the user based on context."
    )
    sources: List[str] = Field(
        default_factory=list, 
        description="List of document source names utilized to construct the answer."
    )
    confidence: str = Field(
        default="High", 
        description="Confidence level of the response: 'High', 'Medium', or 'Low'."
    )