from fastapi import APIRouter, HTTPException
from rdkit import Chem
from typing import Optional

from ..api_models import PredictionInput, FullCRMPathOutput
from ..inference import predict_elementary_step_and_next_smiles
from ..core.model_loader import load_model

router = APIRouter(tags=["Mechanism Predictor"])


@router.post("/predict_crm_path", response_model=FullCRMPathOutput)
async def predict_full_reaction_path(input_data: PredictionInput):

    current_smiles = input_data.smiles_string

    try:
        initial_catalyst_mol = Chem.MolFromSmiles(current_smiles.split(".")[0])
        initial_catalyst_smiles = (
            Chem.MolToSmiles(initial_catalyst_mol) if initial_catalyst_mol else None
        )
    except:
        initial_catalyst_smiles = None

    if not initial_catalyst_smiles:
        raise HTTPException(status_code=400, detail="Invalid SMILES format.")

    reaction_steps = []
    next_smiles = current_smiles

    for i in range(4):
        try:
            step_output = predict_elementary_step_and_next_smiles(current_smiles)
            reaction_steps.append(step_output)
            next_smiles = step_output.next_reaction_smiles

            # Stop if reductive elimination
            if step_output.predicted_class_index == 3:
                break

            # Stop if catalyst regenerated
            parts = next_smiles.split(".")
            if parts:
                regenerated_mol = Chem.MolFromSmiles(parts[0])
                if regenerated_mol:
                    regenerated = Chem.MolToSmiles(regenerated_mol)
                    if regenerated == initial_catalyst_smiles:
                        break

            current_smiles = next_smiles

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error at step {i+1}: {e}")

    # final product
    parts = next_smiles.split(".")
    final_product = parts[1] if len(parts) > 1 else "Unknown Product"

    return FullCRMPathOutput(
        initial_smiles=input_data.smiles_string,
        final_smiles_system=next_smiles,
        final_product_smiles=final_product,
        reaction_steps=reaction_steps,
    )
