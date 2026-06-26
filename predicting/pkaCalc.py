from molscrub import AcidBaseConjugator
from typing import Literal
from dataclasses import dataclass
from rdkit import Chem 
from rdkit.Chem import Mol
from molscrub.core import load_model
import joblib


PkaMethod = Literal["rules", "etr1"]

@dataclass()
class PkaResult:
    pka: float
    rxn_name: str
    rxn_atom: int



def calculate_pka(mol:Mol, method: PkaMethod = "rules", model_file: str = None) -> list[PkaResult]:
    conjugator = AcidBaseConjugator.from_default_data_files()

    if method == "rules":
        result = rules_pka(mol, conjugator)
    elif method == "etr1":
        if model_file is None:
            model = load_model()
        else:
            model = joblib.load(model_file)

        result = model_pka(mol, conjugator, model = model)
    else: 
        raise ValueError("Unrecognized pka method.")

    return result



def model_pka(mol: Mol, conjugator: AcidBaseConjugator, model):
    rxns = conjugator.get_rxn_info(mol)

    return [PkaResult(conjugator.calculate_pka(v["original"], v, model)[0], v["rxn_name"], v["protonated_atom"]) for v in rxns]

def rules_pka(mol: Mol, conjugator: AcidBaseConjugator): 

    rxns = conjugator.get_rxn_info(mol)

    return [PkaResult(v["rule_pka"], v["rxn_name"], v["protonated_atom"]) for v in rxns]