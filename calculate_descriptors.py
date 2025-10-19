import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from PyBioMed import Pymolecule, PyProtein
from PyBioMed.PyProtein import PyProtein
from modlamp.descriptors import GlobalDescriptor
from tqdm import tqdm
from collections import Counter

def calculate_des(seq):
    des = {}
    protein_class = PyProtein.PyProtein(seq)
    des.update(protein_class.GetAAComp())           # 20
    des.update(protein_class.GetMoreauBrotoAuto())  # 240
    des.update(protein_class.GetQSO())              # 50
    des.update(protein_class.GetSOCN())             # 45
    des.update(protein_class.GetTriad())            # 343
    des.update(protein_class.GetCTD())              # 147
    des.update(protein_class.GetDPComp())           # 400
    return des

def add_suffix_for_duplicates(dicts, sources):
    """
    dicts: [dict1, dict2, ...]
    sources: ["Pymolecule", "PyProtein", "modlamp", ...]
    """
    all_keys = []
    for d in dicts:
        all_keys.extend(d.keys())

    counter = Counter(all_keys)
    final_dict = {}

    for d, source in zip(dicts, sources):
        for k, v in d.items():
            if counter[k] > 1:  # Duplicate column, add suffix
                new_key = f"{k}_{source}"
            else:
                new_key = k
            final_dict[new_key] = v
    return final_dict

def calculate_descriptors(smile_seq):
    smile_one, seq_one = smile_seq
    try:
        mol = Pymolecule.PyMolecule()
        mol.ReadMolFromSmile(smile_one)

        fp_SM = mol.GetAllDescriptor()
        des = calculate_des(seq_one)
        desc = GlobalDescriptor(seq_one)
        desc.calculate_all(amide=False)
        modlamp_feature = desc.featurenames
        one_dim_array = desc.descriptor.flatten()
        desc_list = one_dim_array.tolist()
        desc_dic = dict(zip(modlamp_feature, desc_list))

        fp = add_suffix_for_duplicates(
            [fp_SM, des, desc_dic],
            ["Pymolecule", "PyProtein", "modlamp"]
        )
    except Exception as e:
        return {"Error": f"{smile_one}: {str(e)}", "SMILES": smile_one, "SEQUENCE": seq_one}

    fp["SMILES"] = smile_one
    fp["SEQUENCE"] = seq_one
    return fp

if __name__ == "__main__":
    data = pd.read_csv("data/example.csv")
    all_smile_seq = list(zip(data["SMILES"], data["SEQUENCE"]))

    results = []
    for item in tqdm(all_smile_seq, desc="Calculating descriptors"):
        results.append(calculate_descriptors(item))

    df = pd.DataFrame(results)

    # Save
    df.to_csv("data/example_feature_result.csv", index=False)
    print("Feature calculation completed")
