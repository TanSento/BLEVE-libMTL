from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error


class HuberLoss(AbsLoss):
    """The Huber Error loss function.
    """
    def __init__(self):
        super(HuberLoss, self).__init__()
        
        self.loss_fn = nn.HuberLoss()

    def compute_loss(self, pred, gt):
        loss = self.loss_fn(pred, gt.reshape(-1,1))
        return loss
    

    
    
    """AbsMetric is an abstract class for the performance metrics of a task. 
    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """



class MAPE(AbsMetric):
    """Calculate the Mean Absolute Percentage Error (MAPE).
    """
    def __init__(self, quantile):
        super(MAPE, self).__init__()
        self.quantile = quantile


    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration(batch size) and update :attr:`record`.
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """

        pred = self.quantile.inverse_transform(pred.cpu().numpy())
        gt = self.quantile.inverse_transform(gt.cpu().numpy().reshape(-1,1))
        
        score = mean_absolute_percentage_error(pred, gt)  
        self.record.append(score)
        # self.bs.append(pred.shape[0])     # pred.size() == pred.shape
    
    
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).
        Return:
            list: A list of metric scores.
        """ 
        records = np.array(self.record)
        # batch_size = np.array(self.bs)
        return [np.mean(records)*100]

   


class BLEVEMetrics(AbsMetric):
    """Calculate the Mean Absolute Percentage Error (MAPE).
    """
    def __init__(self, quantile):
        super(BLEVEMetrics, self).__init__()
        self.quantile = quantile
        self.r2_list = []
        self.mape_list = []
        self.rmse_list = []


    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration(batch size) and update :attr:`record`.
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """

        pred = self.quantile.inverse_transform(pred.cpu().numpy())
        gt = self.quantile.inverse_transform(gt.cpu().numpy().reshape(-1,1))
        
        # R_squared Calculation
        r2 = r2_score(pred,gt)
        self.r2_list.append(r2)

        # MAPE Calculation
        rl_err = mean_absolute_percentage_error(pred, gt)  
        self.mape_list.append(rl_err)

        #RMSE Calculation
        rmse = np.sqrt(mean_squared_error(pred, gt))
        self.rmse_list.append(rmse)
        
        
    
    
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).
        Return:
            list: A list of metric scores.
        """
        
        r2_records = np.array(self.r2_list)
        mape_records = np.array(self.mape_list)
        rmse_records = np.array(self.rmse_list)
        # batch_size = np.array(self.bs)
        return [np.mean(r2_records)*100, np.mean(mape_records)*100, np.mean(rmse_records)]
    

    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.r2_list = []
        self.mape_list = []
        self.rmse_list = []