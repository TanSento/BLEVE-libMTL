import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timeit
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

from LibMTL_Adapt.data_processing import get_data, BLEVEDataset, BLEVEDatasetSingle
from LibMTL_Adapt.model import Encoder
from LibMTL_Adapt.metrics import HuberLoss, MAPE, RMSE


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
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')         # Hỏi Phúc affect and how to use ?
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    return parser.parse_args()

def main(params):
    print(params)
    X_train_torch, X_test_torch, target_train, target_test, quantile = get_data(mode='single', task_name='posi_peaktime')

    data_train = BLEVEDatasetSingle(X_train_torch, target_train, task_name='posi_peaktime')
    data_test = BLEVEDatasetSingle(X_test_torch, target_test, task_name ='posi_peaktime')


    train_loader = DataLoader(data_train, batch_size=512, shuffle=True)  
    test_loader = DataLoader(data_test, batch_size=7200, shuffle=False) # PyTorch Dataloader knows how to concatenate to load labels in parallel, 
                                                                      # even as a dict, as long as our batch have indexing


    task_dict = {
                'posi_peaktime': {'metrics':['MAPE'], 
                            'metrics_fn': MAPE(quantile),
                            'loss_fn': HuberLoss(),
                            'weight': [0]}
                }


    num_out_channels = {'posi_peaktime': 1}

    # decoders = nn.ModuleDict({task: nn.Linear(256, 
    #                                             num_out_channels[task]) for task in list(task_dict.keys())})

    decoders = nn.ModuleDict({task: nn.Sequential(
                                                nn.Linear(256,256),
                                                nn.Mish(),
                                                nn.Dropout(0.1),
                                                nn.Linear(256, num_out_channels[task])
                                                ) 
                                                for task in list(task_dict.keys())})


    optim_param = {'optim': 'adam', 'lr': 0.005, 'weight_decay': 1e-5}

   
    kwargs = {'weight_args': {}, 'arch_args': {}}
    
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
    

    BLEVENet.train(train_loader, test_loader, 500)

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # print(params.weighting)
    # set device
    # set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)

    