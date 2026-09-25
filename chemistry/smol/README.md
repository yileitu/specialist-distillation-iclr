# Terminology

- SMILES: A standard ASCII representation of a molecule. Separate multiple molecules with `.`; for example, `CC(=O)O.O` represents acetic acid and water.
  - Canonical SMILES: A molecule can have multiple valid SMILES representations. Canonicalization maps them to one normalized representation.
- IUPAC name: A standardized, human-readable chemical name, such as `2-methylprop-1-ene`.
- InChI: The International Chemical Identifier, a unique textual identifier for a chemical substance.

# Fourteen subtasks

- `retrosynthesis`: Predict reactants from a product SMILES. Output: `<SMILES></SMILES>`.
- `property_prediction-sider`: Predict whether a compound has an adverse effect. Output: `<BOOLEAN></BOOLEAN>`.
- `property_prediction-lipo`: Predict lipophilicity as LogD. Output: `<NUMBER></NUMBER>`.
- `property_prediction-hiv`: Predict whether a compound inhibits HIV replication. Output: `<BOOLEAN></BOOLEAN>`.
- `property_prediction-esol`: Predict aqueous solubility as LogS. Output: `<NUMBER></NUMBER>`.
- `property_prediction-clintox`: Predict clinical-trial toxicity. Output: `<BOOLEAN></BOOLEAN>`.
- `property_prediction-bbbp`: Predict blood-brain barrier permeability. Output: `<BOOLEAN></BOOLEAN>`.
- `name_conversion-s2i`: Convert SMILES to an IUPAC name. Output: `<IUPAC></IUPAC>`.
- `name_conversion-s2f`: Convert SMILES to a molecular formula. Output: `<MOLFORMULA></MOLFORMULA>`.
- `name_conversion-i2s`: Convert an IUPAC name to SMILES. Output: `<SMILES></SMILES>`.
- `name_conversion-i2f`: Convert an IUPAC name to a molecular formula. Output: `<MOLFORMULA></MOLFORMULA>`.
- `molecule_generation`: Generate a SMILES string from requested molecular properties. Output: `<SMILES></SMILES>`.
- `molecule_captioning`: Describe a molecule from its SMILES string in natural language; no output tag is used.
- `forward_synthesis`: Predict a product from reactant SMILES. Output: `<SMILES></SMILES>`.
