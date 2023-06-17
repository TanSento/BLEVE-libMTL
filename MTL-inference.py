import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timeit
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

from LibMTL_Adapt.data_processing import get_data, BLEVEDataset
from LibMTL_Adapt.model import Encoder
from LibMTL_Adapt.metrics import HuberLoss, BLEVEMetrics, MAPE


import torch,  argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset

from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.architecture import *
# from LibMTL.metrics import AbsMetric
# from LibMTL.loss import AbsLoss
from LibMTL.weighting import *
from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device


def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')         
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--name', default='Multitask', type=str, help='name of the task')
    parser.add_argument('--large_data', default=False, action='store_true',
                        help='whether to use large data, True or False. If False, do not declare in command line arguments, else declare')
    parser.add_argument('--testing_size', type = int, default=7200, help='Size of testing data')
    return parser.parse_args()

def main(params):
    print(params)
    X_train_torch, X_test_torch, target_train, target_test, quantiles = get_data(mode = 'multitask', large=params.large_data)

    
    X_test_torch = X_test_torch[:params.testing_size]
    target_test = target_test[:params.testing_size]

    data_train = BLEVEDataset(X_train_torch, target_train, sev_tar=False)
    data_test = BLEVEDataset(X_test_torch, target_test, sev_tar=False)


    train_loader = DataLoader(data_train, batch_size=512, shuffle=True)

    print('length of X_test_torch:', len(X_test_torch))
    test_loader = DataLoader(data_test, batch_size=len(X_test_torch), shuffle=False) # PyTorch Dataloader knows how to concatenate to load labels in parallel, 
                                                                    # even as a dict, as long as our batch have indexing


       
    task_dict = {
                'posi_peaktime': {'metrics':['R2','MAPE','RMSE'], 
                                'metrics_fn': BLEVEMetrics(quantiles[0]),
                                'loss_fn': HuberLoss(),
                                'weight': [1,0,0]}, 
                'nega_peaktime': {'metrics':['R2','MAPE','RMSE'], 
                            'metrics_fn': BLEVEMetrics(quantiles[1]),
                            'loss_fn': HuberLoss(),
                            'weight': [1,0,0]},
                'arri_time': {'metrics':['R2','MAPE','RMSE'], 
                            'metrics_fn': BLEVEMetrics(quantiles[2]),
                            'loss_fn': HuberLoss(),
                            'weight': [1,0,0]},
                'posi_dur': {'metrics':['R2','MAPE','RMSE'], 
                            'metrics_fn': BLEVEMetrics(quantiles[3]),
                            'loss_fn': HuberLoss(),
                            'weight': [1,0,0]},
                'nega_dur': {'metrics':['R2','MAPE','RMSE'], 
                            'metrics_fn': BLEVEMetrics(quantiles[4]),
                            'loss_fn': HuberLoss(),
                            'weight': [1,0,0]},           
                'posi_pressure': {'metrics':['R2','MAPE','RMSE'], 
                            'metrics_fn': BLEVEMetrics(quantiles[5]),
                            'loss_fn': HuberLoss(),
                            'weight': [1,0,0]},
                'nega_pressure': {'metrics':['R2','MAPE','RMSE'], 
                            'metrics_fn': BLEVEMetrics(quantiles[6]),
                            'loss_fn': HuberLoss(),
                            'weight': [1,0,0]},
                'posi_impulse': {'metrics':['R2','MAPE','RMSE'], 
                            'metrics_fn': BLEVEMetrics(quantiles[7]),
                            'loss_fn': HuberLoss(),
                            'weight': [1,0,0]}
                }


    num_out_channels = {'posi_peaktime': 1, 'nega_peaktime': 1, 'arri_time': 1, 'posi_dur': 1, 'nega_dur': 1,
                        'posi_pressure': 1, 'nega_pressure': 1, 'posi_impulse': 1}
    
    

    # decoders = nn.ModuleDict({task: nn.Linear(256, 
    #                                             num_out_channels[task]) for task in list(task_dict.keys())})

    decoders = nn.ModuleDict({task: nn.Sequential(
                                                nn.Linear(256,256),
                                                nn.Mish(),
                                                nn.Dropout(0.1),
                                                nn.Linear(256, num_out_channels[task])
                                                ) 
                                                for task in list(task_dict.keys())})


    optim_param = {'optim': 'adam', 'lr': params.lr, 'weight_decay': 1e-5}


    ### Weighting Hyper-parameters
    if params.weighting == 'GradNorm':
        kwargs = {'weight_args': {'alpha':params.alpha}, 'arch_args': {}}

    elif params.weighting == 'Nash_MTL':   # rep_grad must be false
        kwargs = {'weight_args': {'update_weights_every':params.update_weights_every, 
                                  'optim_niter': params.optim_niter,
                                  'max_norm': params.max_norm}, 
                  'arch_args': {}}
        
    elif params.weighting == 'CAGrad':      # rep_grad must be false
        kwargs = {'weight_args': {'calpha':params.calpha, 
                                  'rescale': params.rescale}, 
                  'arch_args': {}}
    else:
        kwargs = {'weight_args': {}, 'arch_args': {}}
    ###
        
    
    # scheduler_param = {'scheduler': 'step'}


    BLEVENet = Trainer(task_dict=task_dict, 
                        weighting=eval(params.weighting), 
                        architecture=HPS, 
                        encoder_class=Encoder, 
                        decoders=decoders,
                        rep_grad=params.rep_grad,
                        multi_input=False,
                        optim_param=optim_param,
                        scheduler_param=params.scheduler,
                        **kwargs)
    
    
    BLEVENet.test(test_loader, mode = None, name=params.name, inference = True)

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # print(params.weighting)
    # set device
    # set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)