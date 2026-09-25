ALL_TASKS=[
    "retrosynthesis",
    "property_prediction-sider",
    "property_prediction-lipo",
    "property_prediction-hiv",
    "property_prediction-esol",
    "property_prediction-clintox",
    "property_prediction-bbbp",
    "name_conversion-s2i",
    "name_conversion-s2f",
    "name_conversion-i2s",
    "name_conversion-i2f",
    "molecule_generation",
    "molecule_captioning",
    "forward_synthesis",
]

SMILES_TASKS=[
    "retrosynthesis",
    "name_conversion-i2s",
    "molecule_generation",
    "forward_synthesis",
]

BOOL_TASKS=[
    "property_prediction-sider",
    "property_prediction-hiv",
    "property_prediction-clintox",
    "property_prediction-bbbp",
]

NUM_TASKS=[
    "property_prediction-lipo",
    "property_prediction-esol",
]

MOLFORMULA_TASKS=[
    "name_conversion-s2f",
    "name_conversion-i2f",
]

IUPAC_TASKS=[
    "name_conversion-s2i",
]


# NOTE: The functions below are adapted from https://github.com/OSU-NLP-Group/LLM4Chem/blob/main/utils/smiles_canonicalization.py
from rdkit import Chem, RDLogger
from rdchiral.chiral import copy_chirality
from rdkit.Chem.AllChem import AssignStereochemistry


RDLogger.DisableLog('rdApp.*')


def canonicalize(smiles, isomeric=False, canonical=True, kekulize=False):
    # When canonicalizing a SMILES string, we typically want to
    # run Chem.RemoveHs(mol), but this will try to kekulize the mol
    # which is not required for canonical SMILES.  Instead, we make a
    # copy of the mol retaining only the information we desire (not explicit Hs)
    # Then, we sanitize the mol without kekulization.  copy_atom and copy_edit_mol
    # Are used to create this clean copy of the mol.
    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        if atom.GetIsAromatic() and atom.GetNoImplicit():
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            #elif atom.GetSymbol() == 'N':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
            #    new_atom.SetNumExplicitHs(-atom.GetImplicitValence())
            #elif atom.GetSymbol() == 'S':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
        return new_atom

    def copy_edit_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
            new_bond = new_mol.GetBondBetweenAtoms(a1, a2)
            new_bond.SetBondDir(bond.GetBondDir())
            new_bond.SetStereo(bond.GetStereo())
        for new_atom in new_mol.GetAtoms():
            atom = mol.GetAtomWithIdx(new_atom.GetIdx())
            copy_chirality(atom, new_atom)
        return new_mol

    smiles = smiles.replace(" ", "")
    tmp = Chem.MolFromSmiles(smiles, sanitize=False)
    tmp.UpdatePropertyCache()
    new_mol = copy_edit_mol(tmp)
    #Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    if not kekulize:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
    else:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)

    AssignStereochemistry(new_mol, cleanIt=False, force=True, flagPossibleStereoCenters=True)

    new_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=isomeric, canonical=canonical)
    return new_smiles


def canonicalize_molecule_smiles(smiles, return_none_for_error=True, skip_mol=False, sort_things=True, isomeric=True, kekulization=True, allow_empty_part=False):
    things = smiles.split('.')
    if skip_mol:
        new_things = things
    else:
        new_things = []
        for thing in things:
            try:
                if thing == '' and not allow_empty_part:
                    raise ValueError('SMILES contains empty part.')

                mol = Chem.MolFromSmiles(thing)
                assert mol is not None
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                thing_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                assert thing_smiles is not None
                can_in = thing_smiles
                can_out = canonicalize(thing_smiles, isomeric=isomeric)
                assert can_out is not None, can_in
                thing_smiles = can_out
                if kekulization:
                    thing_smiles = keku_mid = Chem.MolFromSmiles(thing_smiles)
                    assert keku_mid is not None, 'Before can: %s\nAfter can: %s' % (can_in, can_out)
                    thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=True, isomericSmiles=isomeric)
            except KeyboardInterrupt:
                raise
            except:
                if return_none_for_error:
                    return None
                else:
                    raise
            new_things.append(thing_smiles)
    if sort_things:
        new_things = sorted(new_things)
    new_things = '.'.join(new_things)
    return new_things


# NOTE: The functions below are adapted from https://github.com/OSU-NLP-Group/LLM4Chem/blob/main/prediction_extraction.py
import re


def extract_forward_synthesis(output_text):
    pattern = r"(\[|]|\[[^\]]+]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|;|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])+"

    output_text = output_text.strip()
    if re.match(pattern + '$', output_text):
        return output_text

    pos = output_text.rfind('Predicted product SMILES:')
    if pos != -1:
        output_text = output_text[pos + 1:].strip()
        return output_text

    pos = output_text.rfind(':')
    if pos != -1:
        m = re.match(pattern, output_text[pos + 1:].strip())
        if m is not None:
            output_text = output_text[pos + 1:].strip()
            return output_text[:m.span()[1]]

    match_begin = re.match(pattern, output_text)
    if match_begin is not None:
        return output_text[:(match_begin.span())[1]]

    return ''


def extract_retrosynthesis(output_text):
    pattern = r"(\[|]|\[[^\]]+]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|;|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])+"
    output_text = output_text.strip()

    m = re.match(pattern, output_text)
    if m is not None:
        span = m.span()
        if span[1] == len(output_text):
            return output_text.strip()
        elif span[1] > 10:
            return output_text[:span[1]]

    titles = ('Reactant SMILES:', 'the predicted reactant molecules based on the product SMILES:', 'Here are the predicted reactant molecules in SMILES format:', 'Reactants SMILES:', 'Reactants:', 'the predicted reactants for the given product SMILES:', 'Here are the predicted reactants for the given product molecule:', 'following reactants:', 'Here are the predicted reactants for the given product molecule using the SMILES representation:', 'The predicted reactants for the given product molecule are:', 'the predicted reactants based on the product SMILES:', 'the predicted reactants for the given product:', 'the predicted reactants for the given reaction:')
    found = False
    for title in titles:
        pos = output_text.lower().rfind(title.lower())
        if pos != -1:
            output_text = output_text[pos + len(title):].strip()
            found = True
            break

    if not found:
        for title in ('predicted reactants for each product SMILES:', 'the predicted reactants for each product:', 'the predicted reactants for each product molecule:', 'Here are the predicted reactants for the given product molecules:'):
            pos = output_text.lower().rfind(title.lower())
            if pos != -1:
                output_text = output_text[pos + len(title):].strip()
                pos = output_text.lower().find('product 2:')
                if pos != -1:
                    output_text = output_text[pos + len('product 2:'):].strip().strip('*').strip()
                    found = True

    if found:
        match_begin = re.match(pattern, output_text)
        if match_begin is not None:
            return output_text[:(match_begin.span())[1]]
    return ''


def extract_molecule_generation(output_text):
    pattern = r"(\[|]|\[[^\]]+]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|;|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])+"
    output_text = output_text.strip()

    pos = output_text.find('\n')
    if pos != -1:
        first_line = output_text[:pos].strip()
    else:
        first_line = output_text
    m = re.match(pattern + '$', first_line)
    if m is not None:
        return first_line

    found = False

    titles = ('SMILES for the second molecule: ', 'the second input:', ' is:', 'molecules:', 'description:', 'you described:', 'SMILES:', 'described in the input:', 'Here is the SMILES representation of the molecule:', 'you provided:', 'the molecule is:')
    for title in titles:
        pos = output_text.lower().find(title.lower())
        if pos != -1:
            output_text = output_text[pos + len(title):].strip()
            found = True
            break

    if not found:
        titles= (':',)

        for title in titles:
            pos = output_text.lower().rfind(title.lower())
            if pos != -1:
                output_text = output_text[pos + len(title):].strip()
                found = True
                break

    if found:
        match_begin = re.match(pattern, output_text)
        if match_begin is None:
            pos = output_text.find(':')
            output_text = output_text[pos + 1:].strip()
        match_begin = re.match(pattern, output_text)
        if match_begin is not None:
            return output_text[:(match_begin.span())[1]]
    return ''


def get_molecule_id(smiles, remove_duplicate=True):
    if remove_duplicate:
        assert ';' not in smiles
        all_inchi = set()
        for part in smiles.split('.'):
            inchi = get_molecule_id(part, remove_duplicate=False)
            all_inchi.add(inchi)
        all_inchi = tuple(sorted(all_inchi))
        return all_inchi
    else:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchi(mol)
