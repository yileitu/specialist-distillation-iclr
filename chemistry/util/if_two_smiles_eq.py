
from rdkit import Chem
from func_timeout import func_timeout
from func_timeout.exceptions import FunctionTimedOut, TimeoutError

def smiles_are_equivalent(smiles1, smiles2, timeout=5):
    """Check with a timeout whether two SMILES strings represent the same molecule."""
    if not smiles1 or not smiles2:
        return False

    # Try a direct string comparison first
    if smiles1 == smiles2:
        return True

    try:
        # Apply the timeout wrapper
        return _compare_with_timeout(smiles1, smiles2, timeout)
    except (FunctionTimedOut, TimeoutError):
        print(f"Timed out after {timeout} seconds while comparing SMILES")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def _compare_with_timeout(smiles1, smiles2, timeout):
    """Implement the underlying comparison."""

    def compare():
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return False

        canonical1 = Chem.MolToSmiles(mol1)
        canonical2 = Chem.MolToSmiles(mol2)

        return canonical1 == canonical2

    return func_timeout(timeout, compare)