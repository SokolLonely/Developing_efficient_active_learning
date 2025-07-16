import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import dscribe
from active_learning.utils import  smiles_to_rdkit_mol, rdkit_mol_to_ase, smiles_to_acsf, smiles_to_soap
from config import ROOT_DIR
import pandas as pd
from active_learning.data_prep import get_data
def get_max_len(df):
    lst = df['smiles'].tolist()
    outputSoap = smiles_to_soap(lst)
    mx = -1
    for i in range(len(outputSoap)):
         x = outputSoap[i].shape[0]
         mx = max(mx, x)
    return mx
a = get_data().head(10)
print('max len is')
print(get_max_len(a))

