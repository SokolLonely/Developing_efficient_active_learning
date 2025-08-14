
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm.auto import tqdm
from active_learning.screening import active_learning
import itertools
import argparse


PARAMETERS = {'max_screen_size': [1000],
              'n_start': [64],
              'batch_size': [64, 32, 16],
              'architecture': ['gcn', 'mlp'],
              'dataset': ['ALDH1', 'PKM2', 'VDR'],
              'seed': list(range(10)),
              'bias': ['random', 'small', 'large'],
              'acquisition': ['random', 'exploration', 'exploitation', 'dynamic', 'dynamicbald', 'similarity', 'bald', 'mc_bald', 'dynamic_exploration_mc_bald']
              }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-acq', help="Acquisition function ('random', 'exploration', 'exploitation', 'dynamic', "
                                     "'similarity', 'bald', 'dynamicbald')", default='exploitation')
    parser.add_argument('-bias', help='The level of bias ("random", "small", "large")', default='random')
    parser.add_argument('-arch', help='The neural network architecture ("gcn", "mlp", "chembert", "only_morfeus")', default='mlp2048')
    parser.add_argument('-dataset', help='The dataset ("ALDH1", "PKM2", "VDR")', default='ALDH1')
    parser.add_argument('-retrain', help='Retrain the model every cycle', default='True')
    parser.add_argument('-batch_size', help='How many molecules we select each cycle', default=64)
    parser.add_argument('-n_start', help='How many molecules we have in our starting set (min=2)', default=64)
    parser.add_argument('-anchored', help='Anchor the weights', default='True')
    parser.add_argument('-scrambledx', help='Scramble the features', default='False')
    parser.add_argument('-scrambledx_seed', help='Seed for scrambling the features', default=1)
    parser.add_argument('-function', default='relu')
    parser.add_argument('-epochs', default = 20)
    parser.add_argument('-n_layers', default = 3)
    parser.add_argument('-size', default = 1)
    parser.add_argument('-max_screen_size', default=1000, type=int)
    args = parser.parse_args()

    PARAMETERS['acquisition'] = [args.acq]
    PARAMETERS['bias'] = [args.bias]
    PARAMETERS['dataset'] = [args.dataset]
    PARAMETERS['retrain'] = [eval(args.retrain)]
    PARAMETERS['architecture'] = [args.arch]
    PARAMETERS['batch_size'] = [int(args.batch_size)]
    PARAMETERS['n_start'] = [int(args.n_start)]
    PARAMETERS['n_start'] = [int(args.n_start)]
    PARAMETERS['anchored'] = [eval(args.anchored)]
    PARAMETERS['scrambledx'] = [eval(args.scrambledx)]
    PARAMETERS['scrambledx_seed'] = [int(args.scrambledx_seed)]
    PARAMETERS['function'] = [args.function]
    PARAMETERS['epochs'] = [int(args.epochs)]
    PARAMETERS['n_layers'] = [int(args.n_layers)]
    PARAMETERS['max_screen_size'] = [int(args.max_screen_size)]
    # LOG_FILE = f'{args.o}/{args.arch}_{args.acq}_{args.bias}_{args.batch_size}_simulation_results.csv'
    LOG_FILE = args.o

    # PARAMETERS['acquisition'] = ['random']
    # PARAMETERS['bias'] = ['random']
    # PARAMETERS['architecture'] = ['gcn']
    # LOG_FILE = f"results/{PARAMETERS['architecture'][0]}_{PARAMETERS['acquisition'][0]}_{PARAMETERS['bias'][0]}_simulation_results.csv"
   # print(PARAMETERS)
    experiments = [dict(zip(PARAMETERS.keys(), v)) for v in itertools.product(*PARAMETERS.values())]
    #print(experiments)
   # print(len(experiments))
    for experiment in experiments:#runs 10 times, with 10 different seeds

        results = active_learning(n_start=experiment['n_start'],
                                  bias=experiment['bias'],
                                  acquisition_method=experiment['acquisition'],
                                  max_screen_size=experiment['max_screen_size'],
                                  batch_size=experiment['batch_size'],
                                  architecture=experiment['architecture'],
                                  n_layers = experiment['n_layers'],
                                  epochs = experiment['epochs'],
                                  function = experiment['function'],
                                  seed=experiment['seed'],
                                  retrain=experiment['retrain'],
                                  anchored=experiment['anchored'],
                                  dataset=experiment['dataset'],
                                  scrambledx=experiment['scrambledx'],
                                  scrambledx_seed=experiment['scrambledx_seed'],
                                  ensemble_size=5, 
                                  #size = experiment['size'],
                                  optimize_hyperparameters=False)
        
        # Add the experimental settings to the outfile
        results['acquisition_method'] = experiment['acquisition']
        results['architecture'] = experiment['architecture']
        results['n_start'] = experiment['n_start']
        results['batch_size'] = experiment['batch_size']
        results['seed'] = experiment['seed']
        results['bias'] = experiment['bias']
        results['func'] = experiment['function']
        results['retrain'] = experiment['retrain']
        results['scrambledx'] = experiment['scrambledx']
        results['scrambledx_seed'] = experiment['scrambledx_seed']
        results['dataset'] = experiment['dataset']
        for e in results['hits_discovered']:
          print(e, end = ' ')
        results.to_csv(LOG_FILE, mode='a', index=False, header=False if os.path.isfile(LOG_FILE) else True)
        print(f'experiment {experiment} complete')
        break
