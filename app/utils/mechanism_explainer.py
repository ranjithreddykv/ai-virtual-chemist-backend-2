# mechanism_explainer.py

import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_teacher_style_explanation(initial, product, steps):

    # Format readable list of step predictions
    step_text = "\n".join([
        f"- **Step {i+1} ({s.predicted_elementary_step})**: "
        f"class {s.predicted_class_index}, next state: `{s.next_reaction_smiles}`"
        for i, s in enumerate(steps)
    ])

    # FINAL UPGRADED, SUBSTRATE-SPECIFIC MECHANISM PROMPT
    prompt = f"""
You are an organic chemistry professor. Your task is to explain a catalytic
reaction mechanism in a clean, academic, textbook-quality style.

CRITICAL REQUIREMENTS:
- You MUST analyze the given SMILES structures.
- You MUST tailor the mechanism to these specific molecules.
- You MUST NOT output a generic mechanism.
- You MUST discuss steric, electronic, and functional group effects.
- No emojis or decorative symbols.
- No hallucinated bonds.

----------------------------------------------------------
## Input Structures

**Reactant:** `{initial}`  
**Product:** `{product}`  

Analyze the following for EACH:
- Functional groups  
- Electron-donating and withdrawing substituents  
- Steric hindrance around reactive centers  
- Leaving group quality  
- Alkyne, aryl, heteroatom, or coordination sites  
- Any group that will affect oxidative addition, coordination, acidity, or reductive elimination

----------------------------------------------------------
## Overview

Write a 3–4 line overview describing:
- What transformation occurs BETWEEN THESE specific molecules
- Structural changes
- The role of palladium/metal catalyst for THESE substrates
- Why these substrates are reactive or challenging

----------------------------------------------------------
## Step-by-Step Mechanism

Use each predicted step below to build substrate-specific explanations.
Here are the steps for reference:

{step_text}

For each predicted step, use the format:

### Step X — Predicted Step Name

**What happens:**
- Describe the structural change based on the ACTUAL SMILES.
- Identify which atoms form or break bonds.
- Describe changes in metal coordination.
- Discuss steric/electronic factors specific to THIS substrate.

**Why this step occurs:**
- Give mechanistic reasoning (electron flow, ligand effects, acidity, etc.)
- MUST be based on the real groups present in the SMILES.

----------------------------------------------------------
## Key Mechanistic Principles (Applied to THIS reaction)

Explain ONLY what applies to the given SMILES, such as:
- Ease/difficulty of oxidative addition due to substituents
- Acid-base activation of alkynes (if present)
- Transmetalation feasibility based on electronics
- Reductive elimination influenced by sterics or group orientation
- Catalyst cycling
- Coordination effects specific to the given substrates

DO NOT mention processes not relevant to these structures.

----------------------------------------------------------
## Bond Changes Summary

List ONLY changes that occur between the input structures:

**New bonds formed:**
- Specify atom-to-atom connections formed.

**Bonds broken:**
- Specify atom-to-atom connections broken.

**Coordination/electronic changes:**
- Only oxidation state changes that actually apply.
- Ligand changes relevant to THESE molecules.

----------------------------------------------------------
## Why the Reaction Terminates

Explain why the catalytic cycle stops for THESE substrates:
- Catalyst regeneration
- All reactive partners consumed
- No remaining functional handles

----------------------------------------------------------
## Final Insight

A 2–3 line conclusion describing:
- What is unique about THIS reaction
- How substituents helped or hindered the mechanism
- Why this example illustrates organometallic logic

----------------------------------------------------------

GENERAL RULES:
- NO generic textbook derivative explanations.
- NO speculative chemistry unrelated to the SMILES.
- NO invented groups or incorrect atoms.
- DO NOT write filler content.
- Keep writing clean, crisp, and academic.

Produce the final answer in well-structured Markdown.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
