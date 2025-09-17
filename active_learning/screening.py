"""

This script contains the main active learning loop that runs all experiments.

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

from math import ceil
import re
import pandas as pd
import numpy as np
#from tqdm.auto import tqdm
import torch
from torch.utils.data import WeightedRandomSampler
from active_learning.nn import Ensemble#, RfEnsemble#, ChembertModel
from active_learning.data_prep import MasterDataset
from active_learning.data_handler import Handler
from active_learning.utils import Evaluate, to_torch_dataloader
from active_learning.acquisition import Acquisition, logits_to_pred
from transformers import AutoTokenizer, AutoModel

INFERENCE_BATCH_SIZE = 512
TRAINING_BATCH_SIZE = 64


def active_learning(n_start: int = 64, acquisition_method: str = 'exploration', max_screen_size: int = None,
                    batch_size: int = 16,n_layers = 3, n_hidden = 1024,  function = 'relu', epochs = 50, architecture: str = 'gcn', seed: int = 0, bias: str = 'random',
                    optimize_hyperparameters: bool = False, ensemble_size: int = 10, retrain: bool = True,
                    anchored: bool = True, dataset: str = 'ALDH1', scrambledx: bool = False, corrupt: int = False,
                    scrambledx_seed: int = 1) -> pd.DataFrame:

    # :param n_start: number of molecules to start out with
    # :param acquisition_method: acquisition method, as defined in active_learning.acquisition
    # :param max_screen_size: we stop when this number of molecules has been screened
    # :param batch_size: number of molecules to add every cycle
    # :param architecture: 'gcn', 'mlp', or 'rf' #rf is random forest
    # :param seed: int 1-20
    # :param bias: 'random', 'small', 'large'
    # :param optimize_hyperparameters: Bool
    # :param ensemble_size: number of models in the ensemble, default is 10
    # :param scrambledx: toggles randomizing the features
    # :param scrambledx_seed: seed for scrambling the features
    # :return: dataframe with results


    # Load the datasets
    arch_to_repr = {
        "mlp": "ecfp",
        "rf": "ecfp",
        "mlp2048": "ecfp",
        "amlp": "ecfp",
        "chemberta": "smiles",
        "morfeus_mlp": "morfeus",
        "only_morfeus": "only_morfeus",
        "robert768": "robert768",
        "chemgpt": "chemgpt",
        "maccs": "maccs",
        "acsf": "acsf",
        "mm": "mm",
        "mmnoatt": "mm",
        "x_512": "x_512",
        "x_256": "x_256",
        "x_128": "x_128",
        "mm_768": "mm_768",
        "mm_512": "mm_512",
        "mm_256": "mm_256",
    }
    representation = arch_to_repr.get(architecture, "graph")

# Special case for chembert
    if architecture == "chembert":
        ensemble_size = 1
    
    ds_screen = MasterDataset('screen', representation=representation, dataset=dataset, scramble_x=scrambledx,
                              scramble_x_seed=scrambledx_seed)
    ds_test = MasterDataset('test', representation=representation, dataset=dataset)
    
    # Initiate evaluation trackers
    eval_test, eval_screen, eval_train = Evaluate(), Evaluate(), Evaluate()
    handler = Handler(n_start=n_start, seed=seed, bias=bias, dataset=dataset, corrupt = corrupt)

    # Define some variables
    hits_discovered, total_mols_screened, all_train_smiles = [], [], []
    max_screen_size = len(ds_screen) if max_screen_size is None else max_screen_size

    # build test loader
    x_test, y_test, smiles_test = ds_test.all()
    #print(x_test[:5])#list of smiles strings
    test_loader = to_torch_dataloader(x_test, y_test,
                                      batch_size=INFERENCE_BATCH_SIZE,
                                      shuffle=False, pin_memory=True, architecture=architecture)

    n_cycles = ceil((max_screen_size - n_start) / batch_size)
    # exploration_factor = 1 / lambd^x. To achieve a factor of 1 at the last cycle: lambd = 1 / nth root of 2
    lambd = 1 / (2 ** (1/n_cycles))

    ACQ = Acquisition(method=acquisition_method, seed=seed, lambd=lambd)

    # While max_screen_size has not been achieved, do some active learning in cycles
    for cycle in range(n_cycles+1): 
        print(f"cycle: {cycle}")
        # Get the train and screen data for this cycle
        train_idx, screen_idx = handler()
        x_train, y_train, smiles_train = ds_screen[train_idx]
        y_train[handler.f_p] = 1 # Set the false positives to 1, so they are considered hits
        x_screen, y_screen, smiles_screen = ds_screen[screen_idx]
        if representation == 'smiles':
            x_screen = smiles_screen
            x_train = smiles_train

        # Update some tracking variables
        all_train_smiles.append(';'.join(smiles_train.tolist()))
        temp = sum(y_train).item()
        hits_discovered.append(temp-handler.n_false_labels)
        #print(f"hits_discovered: {hits_discovered}")
        hits = smiles_train[np.where(y_train == 1)] 
        total_mols_screened.append(len(y_train))

        if len(train_idx) >= max_screen_size:
            break

        if architecture != 'rf':
            # Get class weight to build a weighted random sampler to balance out this data
            class_weights = [1 - sum((y_train == 0) * 1) / len(y_train), 1 - sum((y_train == 1) * 1) / len(y_train)]
            weights = [class_weights[i] for i in y_train]
            sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)
            #if representation == 'ecfp':
            # Get the screen and train + balanced train loaders
            train_loader = to_torch_dataloader(x_train, y_train,
                                               batch_size=INFERENCE_BATCH_SIZE,
                                               shuffle=False, pin_memory=True, architecture=architecture)

            train_loader_balanced = to_torch_dataloader(x_train, y_train,#change this so it returns Dataset object
                                                        batch_size=TRAINING_BATCH_SIZE,
                                                        sampler=sampler,
                                                        shuffle=False, pin_memory=True, architecture=architecture)

            screen_loader = to_torch_dataloader(x_screen, y_screen,
                                                batch_size=INFERENCE_BATCH_SIZE,
                                                shuffle=False, pin_memory=True, architecture=architecture)
            #print(data)
            # Initiate and train the model (optimize if specified)
            print("Training model")
            if retrain or cycle == 0:
                M = Ensemble(seed=seed, ensemble_size=ensemble_size,epochs = epochs, architecture=architecture,n_layers=n_layers, n_hidden = n_hidden, function = function, anchored=anchored)
                if cycle == 0 and optimize_hyperparameters:
                    M.optimize_hyperparameters(x_train, y_train)
                M.train(train_loader_balanced, verbose=False)

            # Do inference of the train/test/screen data
            print("Train/test/screen inference")
            train_logits_N_K_C = M.predict(train_loader, architecture)
            eval_train.eval(train_logits_N_K_C, y_train, architecture)

            test_logits_N_K_C = M.predict(test_loader, architecture)
            eval_test.eval(test_logits_N_K_C, y_test, architecture)

            screen_logits_N_K_C = M.predict(screen_loader, architecture)
            eval_screen.eval(screen_logits_N_K_C, y_screen, architecture)
        # elif architecture == 'chemberta':
        #     print('training chembert')
        #     if retrain or cycle == 0:
        #         model_name = "deepchem/chemberta-77m-mlm"
        #         m = chembertmodel() automodel.from_pretrained(model_name, num_labels=2)
        #     print("train/test/screen inference")
        #     train_logits_n_k_c = m.predict(train_loader)
        #     eval_train.eval(train_logits_n_k_c, y_train)

        #     test_logits_n_k_c = m.predict(test_loader)
        #     eval_test.eval(test_logits_n_k_c, y_test)

        #     screen_logits_n_k_c = m.predict(screen_loader)
        #     eval_screen.eval(screen_logits_n_k_c, y_screen)
        else: #random forest
            print("Training model")
            if retrain or cycle == 0:
                M = RfEnsemble(seed=seed, ensemble_size=ensemble_size)
                # if cycle == 0 and optimize_hyperparameters:
                #     M.optimize_hyperparameters(x_train, y_train)
                M.train(x_train, y_train, verbose=False)

            # Do inference of the train/test/screen data
            print("Train/test/screen inference")
            train_logits_N_K_C = M.predict(x_train)
            eval_train.eval(train_logits_N_K_C, y_train)

            test_logits_N_K_C = M.predict(x_test)
            eval_test.eval(test_logits_N_K_C, y_test)

            screen_logits_N_K_C = M.predict(x_screen)
            eval_screen.eval(screen_logits_N_K_C, y_screen)

        # If this is the second to last cycle, update the batch size, so we end at max_screen_size
        if len(train_idx) + batch_size > max_screen_size:
            batch_size = max_screen_size - len(train_idx)

        # Select the molecules to add for the next cycle
        print("Sample acquisition")
        picks = ACQ.acquire(screen_logits_N_K_C, smiles_screen, hits=hits, n=batch_size)
        print(f"Picked {len(picks)} molecules (batch size = {batch_size})") # for some reason returns only one molecule, instead of 64
        #print(picks)
        #print(type(picks))
        handler.add(picks)

    # Add all results to a dataframe
    train_results = eval_train.to_dataframe("train_")
    test_results = eval_test.to_dataframe("test_")
    screen_results = eval_screen.to_dataframe('screen_')
    results = pd.concat([train_results, test_results, screen_results], axis=1)
    print(len(results), len(hits_discovered))
    results['hits_discovered'] = hits_discovered
    results['total_mols_screened'] = total_mols_screened
    results['all_train_smiles'] = all_train_smiles

    return results
