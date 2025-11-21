from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from ..config import groq_client

router = APIRouter(tags=["Voice Chemist"])


class VoiceChemistQuery(BaseModel):
    question: str
    context_smiles: Optional[str] = None


@router.post("/voice_chemist")
async def voice_chemist(query: VoiceChemistQuery):

    prompt = f"""
You are a helpful chemistry professor. Explain in a clear, spoken style:

Student question:
{query.question}
"""

    if query.context_smiles:
        prompt += f"\nRelevant SMILES: {query.context_smiles}"

    prompt += """
Keep the answer 3–6 spoken sentences.
Avoid long equations.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return {"answer": response.choices[0].message.content.strip()}
