from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Literal

from ..config import groq_client

router = APIRouter(tags=["Chemistry Tutor"])


class TutorMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChemTutorChatRequest(BaseModel):
    messages: List[TutorMessage]
    context_smiles: Optional[str] = None


@router.post("/chem_tutor_chat")
async def chem_tutor_chat(req: ChemTutorChatRequest):

    system_prompt = """
You are an organic chemistry tutor. You explain clearly, keep answers short,
and help students understand mechanisms and SMILES-based reactions.
"""

    groq_messages = [{"role": "system", "content": system_prompt}]

    for m in req.messages:
        groq_messages.append({"role": m.role, "content": m.content})

    if req.context_smiles:
        groq_messages.append({
            "role": "user",
            "content": f"Reaction context: {req.context_smiles}",
        })

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=groq_messages,
        temperature=0.4,
    )

    return {"answer": response.choices[0].message.content.strip()}
