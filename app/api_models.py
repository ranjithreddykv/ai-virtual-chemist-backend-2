# app/api_models.py

from pydantic import BaseModel, Field
from typing import List

# Mapping from your ReactAIvate notebook
INDEX_TO_LABEL = {
    0:'oxidative_addition',
    1:'boronate_formation',
    2:'boron transmetallation',    
    3:'reductive_elemination',
    4:'amine_coordination',
    5:'acid_base_deprotonation',                  
    6:'transmetallation',
    7:'ood'
}

class PredictionInput(BaseModel):
    """Input schema for the CRM prediction endpoint."""
    smiles_string: str = Field(
        ..., 
        description="SMILES string of the reaction components (catalyst.substrate.reagent.base)",
        examples=["CC(C)(C)[P+]([Pd])(c1ccccc1-c1ccccc1)C(C)(C)C.Clc1ccccn1.OB(O)c1cccc2ccccc12.[Cs]O"]
    )

class ClassificationOutput(BaseModel):
    """Output schema for a single prediction step."""
    predicted_elementary_step: str = Field(..., description="Predicted chemical reaction step.")
    predicted_class_index: int = Field(..., description="Predicted class index (0-7).")
    predicted_reactive_atoms_indices: List[int] = Field(..., description="List of 0-indexed atom indices predicted to be reactive.")
    next_reaction_smiles: str = Field(..., description="SMILES string of the chemical system after the predicted step (CRM).")

class FullCRMPathOutput(BaseModel):
    """Output schema for the full four-step CRM path prediction."""
    initial_smiles: str = Field(..., description="Initial SMILES string submitted.")
    final_smiles_system: str = Field(..., description="SMILES string of the system after the last predicted step.")
    final_product_smiles: str = Field(..., description="The final product SMILES isolated from the system.")
    reaction_steps: List[ClassificationOutput] = Field(..., description="Details of each predicted elementary reaction step.")