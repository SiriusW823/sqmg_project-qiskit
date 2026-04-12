from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings

# rdkit 2026.03.1 的 sascorer 內部已改用 rdFingerprintGenerator，
# 但在舊版路徑或中間版本可能印出 DEPRECATION WARNING。
# 此 import 對 conda-forge 版 rdkit 正常可用。
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from rdkit.Contrib.SA_Score import sascorer

from rdkit.Chem.Crippen import MolLogP, MolMR
import numpy as np
from typing import List

class FitnessCalculator():
    def __init__(self, task, distribution_learning=True):
        self.task = task
        self.distribution_learning = distribution_learning

    def calc_property(self, mol):
        if self.task == "qed":
            return Descriptors.qed(mol)
        elif self.task == "logP":
            return Descriptors.MolLogP(mol)
        elif self.task == "ClogP":
            return MolLogP(mol)
        elif self.task == "CMR":
            return MolMR(mol)
        elif self.task == "tpsa":
            return Descriptors.TPSA(mol)
        elif self.task in ["sascore", "SAscore"]:
            return sascorer.calculateScore(mol)

    def calc_score(self, smiles_dict: dict, condition_score=None):
        if self.task == "validity":
            total_samples = sum(smiles_dict.values())
            validity = (total_samples - smiles_dict.get(None, 0) - smiles_dict.get("None", 0)) / total_samples
            return validity, validity
        elif self.task == "uniqueness":
            smiles_dict_copy = smiles_dict.copy()
            smiles_dict_copy.pop("None", None)
            smiles_dict_copy.pop(None, None)
            total_valid_samples = sum(smiles_dict_copy.values())
            total_unique_smiles = len(smiles_dict_copy.keys())
            uniqueness = total_unique_smiles / total_valid_samples
            return uniqueness, uniqueness
        elif self.task in ["product_validity_uniqueness", "product_uniqueness_validity"]:
            total_samples = sum(smiles_dict.values())
            smiles_dict_copy = smiles_dict.copy()
            smiles_dict_copy.pop("None", None)
            smiles_dict_copy.pop(None, None)
            total_unique_smiles = len(smiles_dict_copy.keys())
            return total_unique_smiles / total_samples, total_unique_smiles / total_samples
        
        total_count = 0
        property_sum = 0
        property_pure_sum = 0
        for smiles, count in smiles_dict.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                continue
            else:
                total_count += count
                if condition_score:
                    mol_property = self.calc_property(mol)
                    property_sum += np.abs(mol_property - condition_score) * count
                    property_pure_sum += mol_property * count
                else:
                    property_sum += self.calc_property(mol) * count
                    property_pure_sum = property_sum
        return property_sum / total_count, property_pure_sum / total_count
    
    def generate_distribution(self, smiles_dict: dict):
        data_list = []
        for smiles, count in smiles_dict.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                data_list += [0] * count
            else:
                property = self.calc_property(mol)
                data_list += [property] * count
        return data_list
    
    def generate_property_distribution(self, smiles_dict: dict):
        data_list = []
        for smiles, count in smiles_dict.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                continue
            else:
                property = self.calc_property(mol)
                data_list += [property]
        return data_list

    def generate_property_dict(self, smiles_dict: dict):
        prop_dict = {}
        for smiles, count in smiles_dict.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                continue
            else:
                property = self.calc_property(mol)
                prop_dict.update({smiles: property})
        return prop_dict

class FitnessCalculatorWrapper():
    def __init__(self, task_list: List[str], condition:None):
        self.task_list = task_list
        self.condition_list = [float(x) if x not in [None, "None"] else None for x in condition]
        self.function_dict = {task: FitnessCalculator(task) for task in self.task_list}
        self.task_condition = {task: condition for task, condition in zip(self.task_list, self.condition_list)}
    
    def evaluate(self, smiles_dict):
        score_dict = dict()
        score_pure_dict = dict()
        for task in self.task_list:
            score, score_pure = self.function_dict[task].calc_score(smiles_dict, self.task_condition[task])
            score_dict.update({task: (score, None) })
            score_pure_dict.update({task: score_pure})
        return score_dict, score_pure_dict