import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset




def get_data(mode, task_name=None):

    # Read-in Data
    df1 = pd.read_excel("data_new/train.xlsx", sheet_name="positive_peak_time")
    df2 = pd.read_excel("data_new/train.xlsx", sheet_name="negative_peak_time")
    df3 = pd.read_excel("data_new/train.xlsx", sheet_name="arrival_time")
    df4 = pd.read_excel("data_new/train.xlsx", sheet_name="positive_duration")
    df5 = pd.read_excel("data_new/train.xlsx", sheet_name="negative_duration")
    df6 = pd.read_excel("data_new/train.xlsx", sheet_name="positive_pressure")
    df7 = pd.read_excel("data_new/train.xlsx", sheet_name="negative_pressure")
    df8 = pd.read_excel("data_new/train.xlsx", sheet_name="positive_impulse")



    dt1 = pd.read_excel("data_new/test.xlsx", sheet_name="positive_peak_time")
    dt2 = pd.read_excel("data_new/test.xlsx", sheet_name="negative_peak_time")
    dt3 = pd.read_excel("data_new/test.xlsx", sheet_name="arrival_time")
    dt4 = pd.read_excel("data_new/test.xlsx", sheet_name="positive_duration")
    dt5 = pd.read_excel("data_new/test.xlsx", sheet_name="negative_duration")
    dt6 = pd.read_excel("data_new/test.xlsx", sheet_name="positive_pressure")
    dt7 = pd.read_excel("data_new/test.xlsx", sheet_name="negative_pressure")
    dt8 = pd.read_excel("data_new/test.xlsx", sheet_name="positive_impulse")

    
    # Drop 'Status' Column, any df will work
    LE = LabelEncoder()
    df5['Status'] = LE.fit_transform(df5['Status'])
    dt5['Status'] = LE.fit_transform(dt5['Status'])

    X_traindf = df5.drop(['ID','Target'], axis=1)
    X_testdf = dt5.drop(['ID','Target'], axis=1)

    y1_train = df1['Target']
    y2_train = df2['Target']
    y3_train = df3['Target']
    y4_train = df4['Target']
    y5_train = df5['Target']
    y6_train = df6['Target']
    y7_train = df7['Target']
    y8_train = df8['Target']

    y1_test = dt1['Target']
    y2_test = dt2['Target']
    y3_test = dt3['Target']
    y4_test = dt4['Target']
    y5_test = dt5['Target']
    y6_test = dt6['Target']
    y7_test = dt7['Target']
    y8_test = dt8['Target']

    # Standardized the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_traindf)
    X_test = scaler.transform(X_testdf)


    # Quantile Transform The Target
    quantile1 = QuantileTransformer(output_distribution='normal', random_state=42)
    quantile2 = QuantileTransformer(output_distribution='normal', random_state=42)
    quantile3 = QuantileTransformer(output_distribution='normal', random_state=42)
    quantile4 = QuantileTransformer(output_distribution='normal', random_state=42)
    quantile5 = QuantileTransformer(output_distribution='normal', random_state=42)
    quantile6 = QuantileTransformer(output_distribution='normal', random_state=42)
    quantile7 = QuantileTransformer(output_distribution='normal', random_state=42)
    quantile8 = QuantileTransformer(output_distribution='normal', random_state=42)


    y1_train_normal = quantile1.fit_transform(y1_train.values.reshape(-1,1))
    y1_test_normal = quantile1.transform(y1_test.values.reshape(-1,1))


    y2_train_normal = quantile2.fit_transform(y2_train.values.reshape(-1,1))
    y2_test_normal = quantile2.transform(y2_test.values.reshape(-1,1))

    y3_train_normal = quantile3.fit_transform(y3_train.values.reshape(-1,1))
    y3_test_normal = quantile3.transform(y3_test.values.reshape(-1,1))

    y4_train_normal = quantile4.fit_transform(y4_train.values.reshape(-1,1))
    y4_test_normal = quantile4.transform(y4_test.values.reshape(-1,1))

    y5_train_normal = quantile5.fit_transform(y5_train.values.reshape(-1,1))
    y5_test_normal = quantile5.transform(y5_test.values.reshape(-1,1))

    y6_train_normal = quantile6.fit_transform(y6_train.values.reshape(-1,1))
    y6_test_normal = quantile6.transform(y6_test.values.reshape(-1,1))

    y7_train_normal = quantile7.fit_transform(y7_train.values.reshape(-1,1))
    y7_test_normal = quantile7.transform(y7_test.values.reshape(-1,1))

    y8_train_normal = quantile8.fit_transform(y8_train.values.reshape(-1,1))
    y8_test_normal = quantile8.transform(y8_test.values.reshape(-1,1))

    quantiles = [quantile1, quantile2, quantile3, quantile4, quantile5, quantile6, quantile7, quantile8]

    
    # convert data to torch.FloatTensor
    X_train_torch = torch.from_numpy(X_train.astype(np.float32))
    X_test_torch = torch.from_numpy(X_test.astype(np.float32))



    y1_train_torch = torch.from_numpy(y1_train_normal.astype(np.float32))
    y1_test_torch = torch.from_numpy(y1_test_normal.astype(np.float32))

    y2_train_torch = torch.from_numpy(y2_train_normal.astype(np.float32))
    y2_test_torch = torch.from_numpy(y2_test_normal.astype(np.float32))

    y3_train_torch = torch.from_numpy(y3_train_normal.astype(np.float32))
    y3_test_torch = torch.from_numpy(y3_test_normal.astype(np.float32))

    y4_train_torch = torch.from_numpy(y4_train_normal.astype(np.float32))
    y4_test_torch = torch.from_numpy(y4_test_normal.astype(np.float32))

    y5_train_torch = torch.from_numpy(y5_train_normal.astype(np.float32))
    y5_test_torch = torch.from_numpy(y5_test_normal.astype(np.float32))

    y6_train_torch = torch.from_numpy(y6_train_normal.astype(np.float32))
    y6_test_torch = torch.from_numpy(y6_test_normal.astype(np.float32))

    y7_train_torch = torch.from_numpy(y7_train_normal.astype(np.float32))
    y7_test_torch = torch.from_numpy(y7_test_normal.astype(np.float32))

    y8_train_torch = torch.from_numpy(y8_train_normal.astype(np.float32))
    y8_test_torch = torch.from_numpy(y8_test_normal.astype(np.float32))

    
    target_train = torch.cat((y1_train_torch, y2_train_torch, 
                          y3_train_torch, y4_train_torch,
                          y5_train_torch, y6_train_torch,
                          y7_train_torch, y8_train_torch),dim=1)


    target_test = torch.cat((y1_test_torch, y2_test_torch, 
                            y3_test_torch, y4_test_torch,
                            y5_test_torch, y6_test_torch,
                            y7_test_torch, y8_test_torch),dim=1)
    
    if mode == 'multitask':
        return X_train_torch, X_test_torch, target_train, target_test, quantiles
    elif mode == 'single':
        if task_name == 'posi_peaktime':
            return X_train_torch, X_test_torch, y1_train_torch.reshape(21600), y1_test_torch.reshape(7200), quantile1
        elif task_name == 'nega_peaktime':
            return X_train_torch, X_test_torch, y2_train_torch.reshape(21600), y2_test_torch.reshape(7200), quantile2
        elif task_name == 'arri_time':
            return X_train_torch, X_test_torch, y3_train_torch.reshape(21600), y3_test_torch.reshape(7200), quantile3
        elif task_name == 'posi_dur':
            return X_train_torch, X_test_torch, y4_train_torch.reshape(21600), y4_test_torch.reshape(7200), quantile4
        elif task_name == 'nega_dur':
            return X_train_torch, X_test_torch, y5_train_torch.reshape(21600), y5_test_torch.reshape(7200), quantile5
        elif task_name == 'posi_pressure':
            return X_train_torch, X_test_torch, y6_train_torch.reshape(21600), y6_test_torch.reshape(7200), quantile6
        elif task_name == 'nega_pressure':
            return X_train_torch, X_test_torch, y7_train_torch.reshape(21600), y7_test_torch.reshape(7200), quantile7
        else:
            return X_train_torch, X_test_torch, y8_train_torch.reshape(21600), y8_test_torch.reshape(7200), quantile8
        
        





class BLEVEDataset(Dataset):

    def __init__(self, data, target):
        self.data = data
        self.targets = target
        # self.target1 = target[:,0]
        # self.target2 = target[:,1]
        # ...

            
    def __len__(self):
        return len(self.data)  
    

    def get_batch_targets(self, idx):
        r"""Fetch a batch of 8 targets
        """
        return [self.targets[:,i][idx] for i in range(8)]   # A list of 8 targets, can use [:idx] to have more samples in a batch
                                                            # More elegant than returning 8 targets
                                                            # self.targets.shape = (21600,8)
        # return self.target1[idx], self.target2[idx], ....
    

    def get_batch_input(self, idx):
        r"""Fetch a batch of inputs
        """
        return self.data[idx]
    
    
    def __getitem__(self, idx):                 # targets[0].reshape(1) or not does not matter
        inputs = self.get_batch_input(idx)      # in fact, do not reshape here improves the running time, even though we need to reshape the gt in metrics.py
        targets = self.get_batch_targets(idx)
        targets_dict = {'posi_peaktime': targets[0], 
                        'nega_peaktime': targets[1], 
                        'arri_time': targets[2], 
                        'posi_dur': targets[3], 
                        'nega_dur': targets[4], 
                        'posi_pressure': targets[5], 
                        'nega_pressure': targets[6], 
                        'posi_impulse': targets[7]}
        
        return inputs, targets_dict





class BLEVEDatasetSingle(Dataset):

    def __init__(self, data, target, task_name):
        self.data = data
        self.targets = target
        self.name = task_name
      

            
    def __len__(self):
        return len(self.data)  
    

    def get_batch_targets(self, idx):
        r"""Fetch a batch of 1 target
        """
        return self.targets[idx]
    

    def get_batch_input(self, idx):
        r"""Fetch a batch of inputs
        """
        return self.data[idx]
    
    
    def __getitem__(self, idx):
        inputs = self.get_batch_input(idx)
        targets = self.get_batch_targets(idx)
        targets_dict = {self.name: targets}
        
        return inputs, targets_dict