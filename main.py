import argparse
from utils.utils import *
import torch
import os
from solver import Solver
import random
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_list(x):
        if isinstance(x, list):
            return [tensor_to_list(i) for i in x]
        elif hasattr(x, 'tolist'):
            return x.tolist()
        else:
            return x  # already a number

def main(config):
    if (not os.path.exists(config.model_save_path)):
        os.mkdir(config.model_save_path)

    precisions, recalls, f_scores, rocs,prrocs,valiscores = [], [], [], [],[],[]
    thres,thresmean,meanstd,thresmedian=[],[],[],[]
    randomseedlist=[1000,1001,1002,1004]

    for i in randomseedlist:
        set_seed(i)
        config.mode='train'
        solver = Solver(vars(config))
        valiscore=solver.train()
        valiscores.append(valiscore)
        config.mode='test'
        solver = Solver(vars(config))
        precision, recall, f_score, roc,prroc = solver.test()
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        rocs.append(roc)

        results = {
            'valiscores': tensor_to_list(valiscores),
            'meanstd': tensor_to_list(meanstd),
            'f_scores': tensor_to_list(f_scores),
            'rocs': tensor_to_list(rocs),
        }

    current_path = os.getcwd()
    fullpath = current_path + '/results/'

    # Make sure the folder exists
    os.makedirs(fullpath, exist_ok=True)

    filename = f'{fullpath}_result_{config.dataset}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved at " + filename)

    return solver


if __name__ == '__main__':

    current_path = os.getcwd()
    print(current_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--model_save_path', type=str, default=current_path+'/checkpoints')
    config = parser.parse_args()
    config.val_ratio=0.1
    config.lr=1e-4
    config.noise_ratio=0.0
    
    if config.dataset == 'SMD':
        config.win_size = 128 # window size w for input
        config.stride = config.win_size//4 # stride s for input
        config.dlayer =4 # Number of AdaDetrending layers n_d
        config.movk =3 # Moving average kernel size k_avg
        config.freqk=3 # FreqCNN kernel size k_f
        config.lamda=1 # Regularization parameter lambda
        config.d_model = 256 # Latent dimension d_m
        config.flayer=4 # Number of FreqCNN layers n_f
        config.cnngroup=16 # Number of CNN layers within FreqCNN n_cnn
        
        
    if config.dataset == 'WADI':
        config.win_size = 128 # window size w for input
        config.stride = 32 # stride s for input
        config.dlayer =4 # Number of AdaDetrending layers n_d
        config.movk =3 # Moving average kernel size k_avg
        config.freqk=3 # FreqCNN kernel size k_f
        config.lamda=1 # Regularization parameter lambda
        config.d_model = 256 # Latent dimension d_m
        config.flayer=8 # Number of FreqCNN layers n_f
        config.cnngroup=4 # Number of CNN layers within FreqCNN n_cnn
        

    if config.dataset == 'MSL':
        config.win_size = 128
        config.stride = config.win_size//4
        config.dlayer =4
        config.movk = 3
        config.freqk=3
        config.lamda=1
        config.d_model = 512
        config.flayer=4
        config.cnngroup=16
        

    if config.dataset == 'SMAP':
        config.win_size = 128
        config.stride = config.win_size//4
        config.dlayer =1
        config.movk = 3
        config.freqk=3
        config.lamda=1
        config.d_model = 256
        config.flayer=4
        config.cnngroup=16 
        
        
    if config.dataset == 'SWaT':
        config.win_size = 128
        config.stride = config.win_size//4
        config.dlayer =4
        config.movk = 3
        config.freqk=3
        config.lamda=1
        config.d_model = 256
        config.flayer=4
        config.cnngroup=64 
        
    if config.dataset == 'PSM':
        config.win_size = 128
        config.stride = config.win_size//4
        config.dlayer =4
        config.movk = 3
        config.freqk=3
        config.lamda=1
        config.d_model =64
        config.flayer=4
        config.cnngroup=16
        
        
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
