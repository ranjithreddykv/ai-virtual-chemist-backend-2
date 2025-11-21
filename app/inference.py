# app/inference.py

import torch
import os
import numpy as np
import dgl
# Correct RDKit imports
from rdkit import Chem
from rdkit.Chem import rdDepictor, AllChem
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, smiles_to_bigraph

# Import model and necessary CRM_path/other utility functions
from model import AttentiveFPPredictor_rxn
from utils import CRM_path # Assuming CRM_path is now defined in your utils.py
from app.api_models import INDEX_TO_LABEL, ClassificationOutput

# --- Configuration for Loading ---
MODEL_PATH = "final_trained_ReactAIvate_model"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
ATOM_FEATURIZER = AttentiveFPAtomFeaturizer(atom_data_field='hv')
BOND_FEATURIZER = AttentiveFPBondFeaturizer(bond_data_field='he')
N_FEATS = ATOM_FEATURIZER.feat_size('hv')
E_FEATS = BOND_FEATURIZER.feat_size('he')

MODEL = None

def load_model():
    """Loads the model once and moves it to the appropriate device."""
    global MODEL
    if MODEL is None:
        # Define model parameters to match training
        model_params = {
            "node_feat_size": N_FEATS,
            "edge_feat_size": E_FEATS,
            "num_layers": 2,
            "num_timesteps": 1,
            "graph_feat_size": 200,
            "n_tasks": 8,
            "dropout": 0.1
        }
        
        try:
            model = AttentiveFPPredictor_rxn(**model_params)
            # Ensure the weights file exists
            if not os.path.exists(MODEL_PATH):
                 # Added detailed error handling
                raise FileNotFoundError(f"Model weights file not found: {MODEL_PATH}")

            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            MODEL = model
            print("--- ML Model Loaded Successfully ---")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    return MODEL

def preprocess_smiles(smiles_string: str):
    """Preprocesses the SMILES string into a DGL graph."""
    g = smiles_to_bigraph(
        smiles_string, 
        node_featurizer=ATOM_FEATURIZER,
        edge_featurizer=BOND_FEATURIZER, 
        canonical_atom_order=False 
    )
    
    # We must match the expected node feature tensor size (N_FEATS + 1 for the reactive atom label column)
    # At inference, the reactive atom column (the N+1 column) is predicted and doesn't exist yet, 
    # so we must only pass the N_FEATS (39) base features to the model's GNN.
    
    bg = dgl.batch([g]).to(DEVICE)
    atom_feats = bg.ndata['hv'].to(DEVICE)
    bond_feats = bg.edata['he'].to(DEVICE)

    return bg, atom_feats, bond_feats

def predict_elementary_step_and_next_smiles(smiles_string: str) -> ClassificationOutput:
    """
    Runs the full prediction pipeline for a single step and calculates the next SMILES.
    """
    model = load_model()
    bg, atom_feats, bond_feats = preprocess_smiles(smiles_string)

    with torch.no_grad():
        # Get class prediction and reactive atom prediction
        class_label_p, atom_weights, reactive_atom_label_p, graph_feat = model(
            bg, atom_feats, bond_feats, get_node_weight=True
        )

    # 1. Classification (Elementary Step)
    predicted_class_index = class_label_p.cpu().argmax(dim=1).item()
    predicted_step_label = INDEX_TO_LABEL.get(predicted_class_index, "Unknown Step")

    # 2. Reactive Atom Prediction
    threshold = 0.5
    y_pred_node = reactive_atom_label_p.detach().cpu().numpy()
    # Find indices where probability > threshold
    predicted_reactive_atoms_indices = [
        int(index) for index, pred in enumerate(y_pred_node) if pred.item() >= threshold
    ]

    # 3. Calculate the next SMILES using the CRM_path logic
    next_smiles = CRM_path(smiles_string, predicted_class_index)

    return ClassificationOutput(
        predicted_elementary_step=predicted_step_label,
        predicted_class_index=predicted_class_index,
        predicted_reactive_atoms_indices=predicted_reactive_atoms_indices,
        next_reaction_smiles=next_smiles
    )