from fastapi import APIRouter
from ..api_models import PredictionInput
from ..utils.mechanism_visualizer import generate_step_visual
from ..utils.mechanism_explainer import generate_teacher_style_explanation
from .reactions import predict_full_reaction_path

router = APIRouter(tags=["Mechanism Explanation"])


@router.post("/predict_crm_path_explained")
async def predict_full_reaction_path_explained(input_data: PredictionInput):

    crm_output = await predict_full_reaction_path(input_data)

    visuals = [generate_step_visual(step) for step in crm_output.reaction_steps]

    explanation = generate_teacher_style_explanation(
        initial=crm_output.initial_smiles,
        product=crm_output.final_product_smiles,
        steps=crm_output.reaction_steps,
    )

    return {
        "initial_smiles": crm_output.initial_smiles,
        "final_smiles_system": crm_output.final_smiles_system,
        "final_product_smiles": crm_output.final_product_smiles,
        "steps": [
            {
                "model_step": crm_output.reaction_steps[i].dict(),
                "visual_svg": visuals[i],
            }
            for i in range(len(crm_output.reaction_steps))
        ],
        "teacher_explanation": explanation,
    }
