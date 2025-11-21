import pandas as pd
import random
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from rdkit import Chem
import torch
import os
import random
import numpy as np
import ast

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, display
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from rdkit import RDLogger 
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages

def extra_non_reactive_class(dff):
    all_labels_app = []
    for i in range(dff.shape[0]):
        string_list = dff['smiles'][i]
        spp =string_list.split(".")
        random.shuffle(spp)
        spp = spp.pop(0)
        #spp = ('.').join(spp)
        all_labels = [spp,[],7]
        all_labels_app.append(all_labels)
    all_labels_app = pd.DataFrame(all_labels_app, columns = dff.columns)
    return all_labels_app
    

# Function to retrieve values at specified positions
def get_values_at_positions(my_tuple, positions_list):
    return [my_tuple[pos] for pos in positions_list if pos < len(my_tuple)]
    
def atom_finder(smiles, ids):
    mol = Chem.MolFromSmiles(smiles)
    if len(ids) == 0:
        shuffled_smiles = Chem.MolToSmiles(mol, doRandom=True)
        return shuffled_smiles, ids
    else: 
        atoms_interested = ast.literal_eval(ids)
        shuffled_smiles = Chem.MolToSmiles(mol, doRandom=True) #'CC(C)=O.CC(C)(C)c1ccccc1' #here we just have to give shuffled or randomized smiles
        shuffled_mol = Chem.MolFromSmiles(shuffled_smiles)
        shuffled_ids  = shuffled_mol.GetSubstructMatch(mol)
        new_ids = get_values_at_positions(shuffled_ids, atoms_interested)
        return shuffled_smiles, new_ids

def smiles_augmentation(df, num_aug, augmentation = True): 
    if augmentation == True:
      information = []
      for i in range(df.shape[0]):
          for _ in range(num_aug):
              smiles, ids = atom_finder(df['smiles'][i], df['atom_mapped'][i])
              react_random_list = [smiles, ids, df['class_label'][i]]
              information.append(react_random_list)
    else:
      information = []
      for i in range(df.shape[0]):
            react_random_list = [df['smiles'][i], ast.literal_eval(df['atom_mapped'][i]), df['class_label'][i]]
            information.append(react_random_list)    
    
    return information
        
def concat_feature_reactive_atom(graph_feat, changed_atoms):
    smiles_list = []
    target_list = []
    for i in range(len(changed_atoms)):
        node_lebel_tensor = torch.zeros(graph_feat[i].ndata['hv'].shape[0])
        #node_lebel_tensor[ast.literal_eval(changed_atoms[i][1])] = 1
        node_lebel_tensor[changed_atoms[i][1]] = 1
        node_lebel_tensor = node_lebel_tensor.unsqueeze(1)
        smiles_list.append(changed_atoms[i][0])
        target_list.append(changed_atoms[i][2])
        

        graph_feat[i].ndata['hv'] = torch.cat([graph_feat[i].ndata['hv'], node_lebel_tensor], dim=1)
    # Convert the list of numbers to a list of tensors
    target_tensor_list = [torch.tensor([x], dtype=torch.float32) for x in target_list]
    return list(zip(smiles_list, graph_feat, target_tensor_list))
    
def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))


    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels
    
    
def Canon_SMILES_similarity(smiles_list):
    # Convert all SMILES strings to molecular objects
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Create an array of canonical smiles strings
    canonical_smiles_array = np.array([Chem.MolToSmiles(mol) if mol else None for mol in mol_list])

    # Use broadcasting to compare canonical smiles strings
    matrix = (canonical_smiles_array[:, None] != canonical_smiles_array).astype(np.float32)

    return torch.tensor(matrix)
    
    
# utils.py (ADD THIS FUNCTION to the bottom of your existing utils.py content)
# This is the function extracted from your uploaded notebook.

# The necessary reaction templates (must be defined in utils.py or imported/passed)
# For simplicity, these templates are defined inside the function's closure or assumed global/local in the module.

def ox_addition_template_select(mol):
    # Oxidative addition mechanism
    if mol.GetSubstructMatch(Chem.MolFromSmarts('[#6][I:1]')):
        ox_addition_temp = '[Mg,Fe,Ni,Cu,Pd,Pt:2].[*:3][I:1]>>[Mg,Fe,Ni,Cu,Pd,Pt:2]([I:1])[*:3]'
    elif mol.GetSubstructMatch(Chem.MolFromSmarts('[#6][Br:1]')):
        ox_addition_temp = '[Mg,Fe,Ni,Cu,Pd,Pt:2].[*:3][Br:1]>>[Mg,Fe,Ni,Cu,Pd,Pt:2]([Br:1])[*:3]'
    elif mol.GetSubstructMatch(Chem.MolFromSmarts('[#6][Cl:1]')):
        ox_addition_temp = '[Mg,Fe,Ni,Cu,Pd,Pt:2].[*:3][Cl:1]>>[Mg,Fe,Ni,Cu,Pd,Pt:2]([Cl:1])[*:3]'
    elif mol.GetSubstructMatch(Chem.MolFromSmarts('[#6][F:1]')):
        ox_addition_temp = '[Mg,Fe,Ni,Cu,Pd,Pt:2].[*:3][F:1]>>[Mg,Fe,Ni,Cu,Pd,Pt:2]([F:1])[*:3]'
    else: # Fallback or error handling
        return None 
    
    return AllChem.ReactionFromSmarts(ox_addition_temp)

reductive_elimination_template = '[#6:1][Mg,Fe,Ni,Cu,Pd,Pt:2][#6,#7:3]>>[#6:1][#6,#7:3].[Mg,Fe,Ni,Cu,Pd,Pt:2]'
REDUCTIVE_ELIMINATION = AllChem.ReactionFromSmarts(reductive_elimination_template)

transmetallation_template = '[Mg,Fe,Ni,Cu,Pd,Pt:1][F,Cl,Br,I:2].[#6:5][B:3]>>[Mg,Fe,Ni,Cu,Pd,Pt:1][#6:5].[F,Cl,Br,I:2][B:3]'
TRANS_METALATION = AllChem.ReactionFromSmarts(transmetallation_template)

MI_template_kumada = '[Mg,Fe,Ni,Cu,Pd,Pt:1][F,Cl,Br,I:2].[#6:5][Mg,Fe:3]>>[Mg,Fe,Ni,Cu,Pd,Pt:1][#6:5].[F,Cl,Br,I:2][Mg,Fe:3]'
MI_KUMADA = AllChem.ReactionFromSmarts(MI_template_kumada)

boronate_template = '[Na,K,Rb,Cs:1][O:2].[#6:3][B:4]>>[O:2][B-:4][#6:3].[Na+,K+,Rb+,Cs+:1]'
BORONATE = AllChem.ReactionFromSmarts(boronate_template)

am_add_template = '[Pd:1].[N;!H0:2]>>[Pd:1][N+:2]'
AM_ADD = AllChem.ReactionFromSmarts(am_add_template)

ac_ba_template = '[N:5][Mg,Fe,Ni,Cu,Pd,Pt:3][F,Cl,Br,I:4].[Na,K,Rb,Cs:1][O:2]>>[Na,K,Rb,Cs:1][F,Cl,Br,I:4].[0:2].[N:5][Mg,Fe,Ni,Cu,Pd,Pt:3]'
AC_BA = AllChem.ReactionFromSmarts(ac_ba_template)

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def CRM_path(reaction_path, predicted_class):
    # This logic implements the reaction transformations based on the predicted class index.
    # IMPORTANT: The current implementation has error handling that returns a generic SMILES ('CCCCCC.CCCCCC.CCCCCC') 
    # which may mask issues in your template matching.
    
    reactants  = reaction_path.split('.')
    
    try:
        if predicted_class == 0:
            # Oxidative Addition
            sub1, sub2 = reactants[0], reactants[1]
            mol1, mol2 = Chem.MolFromSmiles(sub1), Chem.MolFromSmiles(sub2)
            
            ox_addition_template = ox_addition_template_select(mol2)
            if ox_addition_template is None: raise ValueError("Oxidation Addition template not found.")

            ox_add_mol = ox_addition_template.RunReactants((mol1, mol2))
            ox_add_sm = Chem.MolToSmiles(ox_add_mol[0][0])
            
            # Recombine the rest of the components
            return '.'.join([ox_add_sm] + reactants[2:])

        elif predicted_class == 4:
            # Amine coordination
            sub1, sub2 = reactants[0], reactants[1]
            mol1, mol2 = Chem.MolFromSmiles(sub1), Chem.MolFromSmiles(sub2)
            
            am_add_mol = AM_ADD.RunReactants((mol1, mol2))
            am_add_sm0 = Chem.MolToSmiles(am_add_mol[0][0])
            
            return '.'.join([am_add_sm0] + reactants[2:])

        elif predicted_class == 5:
            # Acid/Base deprotonation
            sub1, sub2 = reactants[0], reactants[1]
            mol1, mol2 = Chem.MolFromSmiles(sub1), Chem.MolFromSmiles(sub2)
            
            neut_am_mol = neutralize_atoms(mol1)
            
            ac_ba_mol = AC_BA.RunReactants((neut_am_mol, mol2)) 
            ac_ba_sm0 = Chem.MolToSmiles(ac_ba_mol[0][0])
            ac_ba_sm1 = Chem.MolToSmiles(ac_ba_mol[0][1])
            ac_ba_sm2 = Chem.MolToSmiles(ac_ba_mol[0][2])
            
            # Ordering is critical: ac_ba_sm2, ac_ba_sm0, ac_ba_sm1 (as per notebook)
            return '.'.join([ac_ba_sm2, ac_ba_sm0, ac_ba_sm1] + reactants[3:])

        elif predicted_class == 3:
            # Reductive Elimination
            sub1 = reactants[0]
            mol1 = Chem.MolFromSmiles(sub1)
            
            red_eli_mol = REDUCTIVE_ELIMINATION.RunReactants((mol1, ))
            red_eli_sm0 = Chem.MolToSmiles(red_eli_mol[0][0])
            red_eli_sm1 = Chem.MolToSmiles(red_eli_mol[0][1])
            
            # Ordering: red_eli_sm1 (Product), red_eli_sm0 (New Catalyst State)
            # Notebook ordering: red_eli_sm1, red_eli_sm0, reactants[1:]
            return '.'.join([red_eli_sm1, red_eli_sm0] + reactants[1:])

        elif predicted_class == 2:
            # Boron transmetallation
            sub1, sub2 = reactants[0], reactants[1]
            mol1, mol2 = Chem.MolFromSmiles(sub1), Chem.MolFromSmiles(sub2)
            
            tr_met_mol = TRANS_METALATION.RunReactants((mol1, mol2))
            tr_met_sm0 = Chem.MolToSmiles(tr_met_mol[0][0])
            tr_met_sm1 = Chem.MolToSmiles(tr_met_mol[0][1])
            
            # Ordering: tr_met_sm0 (New Complex), tr_met_sm1 (Side Product)
            return '.'.join([tr_met_sm0, tr_met_sm1] + reactants[2:])
            
        elif predicted_class == 1:
            # Boronate formation
            # Requires 3 components: catalyst.aryl_halide.boronic_acid/amine/grignard_reagent.base 
            # The logic extracts sub2 (aryl_halide) and sub3 (boronic_acid/amine/grignard_reagent)
            
            if len(reactants) < 3:
                 raise ValueError("Boronate formation requires at least 3 reactants.")
                 
            sub2, sub3 = reactants[1], reactants[2]
            mol2, mol3 = Chem.MolFromSmiles(sub2), Chem.MolFromSmiles(sub3)
            
            br_met_mol = BORONATE.RunReactants((mol3, mol2)) # mol3 (base) and mol2 (reagent)
            br_met_sm0 = Chem.MolToSmiles(br_met_mol[0][0])  # Boronate
            br_met_sm1 = Chem.MolToSmiles(br_met_mol[0][1])  # Metal ion
            
            # Ordering: reactants[0] (Catalyst), br_met_sm0 (Boronate Reagent), br_met_sm1 (Metal Ion)
            return '.'.join([reactants[0], br_met_sm0, br_met_sm1] + reactants[3:])

        elif predicted_class == 6:
            # Kumada Transmetallation (using MI_KUMADA template)
            sub1, sub2 = reactants[0], reactants[1]
            mol1, mol2 = Chem.MolFromSmiles(sub1), Chem.MolFromSmiles(sub2)
            
            tr_met_mol = MI_KUMADA.RunReactants((mol1, mol2))
            tr_met_sm0 = Chem.MolToSmiles(tr_met_mol[0][0])
            tr_met_sm1 = Chem.MolToSmiles(tr_met_mol[0][1])
            
            return '.'.join([tr_met_sm0, tr_met_sm1] + reactants[2:])

        elif predicted_class == 7:
            # OOD (Out-of-Distribution/No-Reaction)
            return reaction_path
            
    except Exception as e:
        print(f"Reaction step prediction failed for class {predicted_class}: {e}")
        # Return a simple, consistent, non-terminating SMILES string if the reaction fails
        return 'CCCCCC.CCCCCC.CCCCCC' 
        
    return reaction_path