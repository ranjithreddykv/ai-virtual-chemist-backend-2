from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem

router = APIRouter(tags=["3D Structure Generator"])


class SmilesInput(BaseModel):
    smiles: str


@router.post("/generate_3d", response_class=PlainTextResponse)
async def generate_3d_model(input_data: SmilesInput):
    try:
        mol = Chem.MolFromSmiles(input_data.smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        return Chem.MolToMolBlock(mol)
    except Exception as e:
        return f"Error generating 3D structure: {str(e)}"
