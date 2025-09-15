
from typing import Any
import sys
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import torch
import h5py
import transformers
from config import ROOT_DIR
from sklearn.decomposition import PCA
from active_learning.utils import molecular_graph_featurizer as smiles_to_graph, scramble_features, smiles_to_ecfp, smiles_to_maccs, \
    get_tanimoto_matrix, check_featurizability, scramble_graphs, smiles_to_rdkit_mol, rdkit_mol_to_ase, smiles_to_acsf, smiles_to_soap, smiles_to_morfeus, smiles_to_chemberta_embeddings, smiles_to_chemgpt_embeddings


def canonicalize(smiles: str, sanitize: bool = True):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=sanitize))


def get_data(random_state: int = 42, dataset: str = 'ALDH1'):

    # read smiles from file and canonicalize them
    with open(os.path.join(ROOT_DIR, f'data/{dataset}/original/inactives.smi')) as f:
        inactives = [canonicalize(smi.strip().split()[0]) for smi in f.readlines()]
    with open(os.path.join(ROOT_DIR, f'data/{dataset}/original/actives.smi')) as f:
        actives = [canonicalize(smi.strip().split()[0]) for smi in f.readlines()]

    # remove duplicates:
    inactives = list(dict.fromkeys(inactives))
    actives = list(dict.fromkeys(actives))
    
    # remove intersecting molecules:
    intersecting_mols = np.intersect1d(inactives, actives)
    inactives = [smi for smi in inactives if smi not in intersecting_mols]
    actives = [smi for smi in actives if smi not in intersecting_mols]

    # remove molecules that have scaffolds that cannot be kekulized or featurized
    inactives_, actives_ = [], []
    for smi in tqdm(actives):
        try:
            if Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False)) is not None:
                if check_featurizability(smi):
                    actives_.append(smi)
        except:
            pass
    for smi in tqdm(inactives):
        try:
            if Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False)) is not None:
                if check_featurizability(smi):
                    inactives_.append(smi)
        except:
            pass

    # add to df
    df = pd.DataFrame({'smiles': inactives_ + actives_,
                       'y': [0] * len(inactives_) + [1] * len(actives_)})

    # shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print('get_data complete')
    return df


def split_data(df: pd.DataFrame, random_state: int = 42, screen_size: int = 50000, test_size: int = 10000,
               dataset: str = 'ALDH1') -> (pd.DataFrame, pd.DataFrame):

    from sklearn.model_selection import train_test_split
    df_screen, df_test = train_test_split(df, stratify=df['y'].tolist(), train_size=screen_size, test_size=test_size,
                                          random_state=random_state)

    # write to csv
    df_screen.to_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/screen.csv'), index=False)
    df_test.to_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/test.csv'), index=False)
    print('split_data complete')
    return df_screen, df_test


class MasterDataset:
    """ Dataset that holds all data in an indexable way """
    def __init__(self, name: str, df: pd.DataFrame = None, dataset: str = 'ALDH1', representation: str = 'ecfp', root: str = 'data',
                 overwrite: bool = False, scramble_x: bool = False, scramble_x_seed: int = 1) -> None:

        #assert representation in ['ecfp', 'graph', 'smiles', 'morfeus', 'only_morfeus', 'robert768', 'chemgpt', 'maccs', 'acsf'], f"'representation' must be 'ecfp' or 'graph', not {representation}"
        self.representation = representation
        self.pth = os.path.join(ROOT_DIR, root, dataset, name)
        print(self.pth)
        # If not done already, process all data. Else just load it
        if not os.path.exists(self.pth) or overwrite:
            os.makedirs(os.path.join(root, dataset, name), exist_ok=True)
            self.process(df)
            self.smiles_index, self.index_smiles, self.smiles, self.x, self.y, self.graphs = self.load()
        else:
            self.smiles_index, self.index_smiles, self.smiles, self.x, self.y, self.graphs = self.load()

        if scramble_x:
            if representation == 'ecfp':
                self.x = scramble_features(self.x, seed=scramble_x_seed)
            if representation == 'graph':
                self.graphs = scramble_graphs(self.graphs, seed=scramble_x_seed)
        print('init done')



    def process(self, df: pd.DataFrame) -> None:

        print('Processing data ... ', flush=True, file=sys.stderr)

        index_smiles = OrderedDict({i: smi for i, smi in enumerate(df.smiles)})
        smiles_index = OrderedDict({smi: i for i, smi in enumerate(df.smiles)})
        smiles = np.array(df.smiles.tolist())
        x = smiles_to_ecfp(smiles, silent=False)
        y = torch.tensor(df.y.tolist())
        #x_roberta_embedding = smiles_to_chemberta_embeddings(smiles, silent=False)
        print('chemberta started')
        x_llm_embedding = smiles_to_chemgpt_embeddings(smiles, silent=False)
        print('maccs started')
        #x_maccs = smiles_to_maccs(smiles, silent=False)
        #x_morfeus = smiles_to_morfeus(smiles, silent=False, path = self.pth)
        #graphs = [smiles_to_graph(smi, y=y.type(torch.LongTensor)) for smi, y in tqdm(zip(smiles, y))]
        #x_acsf = smiles_to_acsf(smiles, silent=False)
        # print('new part started') 
        # x_soap = smiles_to_soap(smiles)
        # print('soaps done')
        #x_acsf = smiles_to_acsf(smiles)
        # print('new part finished')
        
        #pca = PCA(n_components=512, svd_solver="randomized", random_state=42)
        #if len(x) < 512: #only for preprocess demo
         #   pca = PCA(n_components=len(x),  random_state=42)
        #x_512 = pca.fit_transform(x)
        #x_256 = x_512[:, :256]
        #x_128 = x_512[:, :128]

        torch.save(index_smiles, os.path.join(self.pth, 'index_smiles'))
        torch.save(smiles_index, os.path.join(self.pth, 'smiles_index'))
        torch.save(smiles, os.path.join(self.pth, 'smiles'))
        torch.save(x, os.path.join(self.pth, 'x'))
        #torch.save(x_512, os.path.join(self.pth, 'x_512'))
        #torch.save(x_256, os.path.join(self.pth, 'x_256'))
        #torch.save(x_128, os.path.join(self.pth, 'x_128'))
        torch.save(y, os.path.join(self.pth, 'y'))
        #x_ecfp = torch.load(os.path.join(self.pth, 'x'), weights_only=False)#[:var_len]
        #x_maccs = torch.load(os.path.join(self.pth, 'x_maccs'), weights_only=False)#[:var_len]
        #x = np.concatenate((x_ecfp,  x_maccs, ), axis=1)
        #x = np.pad(x, ((0, 0), (0, 9)), mode='constant')#pad to 1200 len on axis 1 for self-attention
        #pca768 = PCA(n_components=768, svd_solver="randomized", random_state=42)
        #if len(x) < 768: #for demo preprocessing
         #   pca768 = PCA(n_components=len(x), svd_solver="randomized", random_state=42)
        #mm_768 = pca768.fit_transform(x)
        #mm_512 = mm_768[:, :512]
        #mm_256 = mm_768[:, :256]
        #torch.save(mm_768, os.path.join(self.pth, 'mm_768'))
        #torch.save(mm_512, os.path.join(self.pth, 'mm_512'))
        #torch.save(mm_256, os.path.join(self.pth, 'mm_256'))
        torch.save(x_llm_embedding, os.path.join(self.pth, 'x_llm_embedding'))
        #torch.save(x_roberta_embedding, os.path.join(self.pth, 'x_roberta_embedding'))
        #torch.save(x_morfeus, os.path.join(self.pth, 'x_morfeus'))
        #torch.save(x_maccs, os.path.join(self.pth, 'x_maccs'))
        #torch.save(x_acsf, os.path.join(self.pth, 'x_acsf'))
        #torch.save(graphs, os.path.join(self.pth, 'graphs'))
        print('processing done')
    def load(self) -> (dict, dict, np.ndarray, np.ndarray, np.ndarray, list):
        from collections import OrderedDict
        print('Loading data ... ', flush=True, file=sys.stderr)
        index_smiles = torch.load(os.path.join(self.pth, 'index_smiles'), weights_only=False)
        var_len = len(index_smiles) #cuts arrays to length of index_smiles. Usually useless, but can be changed (len(index_smiles)/2 for example ) to test smaller screening space
        index_smiles = torch.load(os.path.join(self.pth, 'index_smiles'), weights_only=False)#[:var_len]
        index_smiles = OrderedDict(list(index_smiles.items())[:var_len])
        smiles_index = torch.load(os.path.join(self.pth, 'smiles_index'), weights_only=False)#[:var_len]
        smiles_index = OrderedDict(list(smiles_index.items())[:var_len])
        smiles = torch.load(os.path.join(self.pth, 'smiles',), weights_only=False)[:var_len]
        smiles = np.array(smiles.tolist())[:var_len]
        y = torch.load(os.path.join(self.pth, 'y'), weights_only=False)[:var_len]
        if self.representation == 'morfeus':
            x_ecfp = torch.load(os.path.join(self.pth, 'x'), weights_only=False)[:var_len]
            print(f"ecfp len = {len(x_ecfp)}")
            x_m = torch.load(os.path.join(self.pth, 'x_morfeus'), weights_only=False)#[:var_len]#.squeeze(-1)
            print(f"morfeus len = {len(x_m)}")
            x_m = x_m.squeeze(1)[:var_len]
            print(f"squeezed morfeus len = {len(x_m)}")
            x = np.concatenate((x_ecfp, x_m), axis = 1)[:var_len]
            print(f"Y LEN = {len(y)}")
        elif self.representation == 'only_morfeus':
            x = torch.load(os.path.join(self.pth, 'x_morfeus'), weights_only=False)
            x = x.squeeze(1)[:var_len]
            print(f"only morfeus len = {len(x)}")
        elif self.representation == 'robert768':
            x = torch.load(os.path.join(self.pth, 'x_roberta_embedding'), weights_only=False)[:var_len]
            #print(f"roberta len = {len(x)}")
        elif self.representation == 'chemgpt':
                x = torch.load(os.path.join(self.pth, 'x_llm_embedding'), weights_only=False)[:var_len]# 2048
        elif self.representation == 'maccs':
             x = torch.load(os.path.join(self.pth, 'x_maccs'), weights_only=False)[:var_len]
        elif self.representation == 'acsf':
            x = torch.load(os.path.join(self.pth, 'x_acsf'), weights_only=False)[:var_len]
        elif self.representation == 'mm':
            x_ecfp = torch.load(os.path.join(self.pth, 'x'), weights_only=False)[:var_len]
            x_maccs = torch.load(os.path.join(self.pth, 'x_maccs'), weights_only=False)[:var_len]
            x = np.concatenate((x_ecfp,  x_maccs, ), axis=1)
            x = np.pad(x, ((0, 0), (0, 9)), mode='constant')#pad to 1200 len on axis 1 for self-attention 
        elif self.representation == 'x_512':
             x = torch.load(os.path.join(self.pth, 'x_512'), weights_only=False)[:var_len]
        elif self.representation == 'x_256':
             x = torch.load(os.path.join(self.pth, 'x_256'), weights_only=False)[:var_len]
        elif self.representation == 'x_128':
             x = torch.load(os.path.join(self.pth, 'x_128'), weights_only=False)[:var_len]
        elif self.representation == 'mm_768':
            x =  torch.load(os.path.join(self.pth, 'mm_768'), weights_only=False)
        elif self.representation == 'mm_512':
            x =  torch.load(os.path.join(self.pth, 'mm_512'), weights_only=False)
        elif self.representation == 'mm_256':
            x =  torch.load(os.path.join(self.pth, 'mm_256'), weights_only=False)
        else:
            x = torch.load(os.path.join(self.pth, 'x'), weights_only=False)[:var_len]
        graphs = 1 #torch.load(os.path.join(self.pth, 'graphs'), weights_only=False)[:var_len]
        print('loading done')
        #return OrderedDict(list(smiles_index.items())[:20]), OrderedDict(list(index_smiles.items())[:20]), smiles[:20], x[:20], y[:20], graphs[:20]
        return smiles_index, index_smiles, smiles, x, y, graphs

    def __len__(self) -> int:
        return len(self.x)

    def all(self):
        return self[range(len(self.smiles))]
    def tokenize(self, smiles):#gets smiles[idx]
        from transformers import AutoTokenizer
        model_name = "DeepChem/ChemBERTa-77M-MLM"
        tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=2)
        lst = []
        for el in smiles:
            res = tokenizer(el, return_tensors="pt")
            lst.append(res)
        #print(lst[:10])
        return lst
    def __getitem__(self, idx):
        if type(idx) is int:
            idx = [idx]
        #print(f"idx: {idx}, dtype = {type(idx)}")
        #print(f"smiles: {self.smiles[idx]}")
        #if self.representation == 'ecfp' or self.representation == 'morfeus' or self.representation == 'only_morfeus' or self.representation == 'robert768' or self.representation == 'chemgpt' or self.representation == 'maccs':
            
        if self.representation == 'graph':
            return [self.graphs[i] for i in idx], self.y[idx], self.smiles[idx]
        elif self.representation == 'smiles':
            return self.tokenize(self.smiles[idx]), self.y[idx], self.smiles[idx]
        else:
            print(len(self.x), len(self.y), len(self.smiles)) # len(self.x_acsf))
            return self.x[idx], self.y[idx], self.smiles[idx]
       

def smi_to_scaff(smiles: str, includeChirality: bool = False):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=includeChirality)


def similarity_vectors(df_screen, df_test, root: str = 'data', dataset: str = 'ALDH1'):

    print("Computing Tanimoto matrix for all test molecules")
    S = get_tanimoto_matrix(df_test['smiles'].tolist(), verbose=True, scaffolds=False, zero_diag=True, as_vector=True)
    save_hdf5(1-S, f'{ROOT_DIR}/{root}/{dataset}/test/tanimoto_distance_vector')
    del S

    print("Computing Tanimoto matrix for all screen molecules")
    S = get_tanimoto_matrix(df_screen['smiles'].tolist(), verbose=True, scaffolds=False, zero_diag=True, as_vector=True)
    save_hdf5(1 - S, f'{ROOT_DIR}/{root}/{dataset}/screen/tanimoto_distance_vector')
    del S


def save_hdf5(obj: Any, filename: str):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('obj', data=obj)
    hf.close()


def load_hdf5(filename: str) -> Any:
    hf = h5py.File(filename, 'r')
    obj = np.array(hf.get('obj'))
    hf.close()

    return obj
