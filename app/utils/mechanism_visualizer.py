# mechanism_visualizer.py

from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64


def smiles_to_svg(smiles: str, size=(350, 300)):
    """
    Converts a SMILES string to an SVG image.
    Returns raw SVG string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg


def generate_step_visual(step_output):
    """
    Takes a single reaction step output (Pydantic model)
    and returns SVG for next_reaction_smiles.
    """
    try:
        smiles = step_output.next_reaction_smiles
        return smiles_to_svg(smiles)
    except Exception:
        return None
