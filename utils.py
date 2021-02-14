"""Utility function for INVASE.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
           "INVASE: Instance-wise Variable Selection using Neural Networks," 
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com

---------------------------------------------------

(1) Feature performance metrics
(2) Prediction performance metrics
(3) Bernoulli sampling
"""

import os
import shutil

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import torch

def feature_performance_metric(ground_truth, importance_score):
    """Performance metrics for feature importance (TPR and FDR).
    
    Args:
        - ground_truth: ground truth feature importance
        - importance_score: computed importance scores for each feature
        
    Returns:
        - mean_tpr: mean value of true positive rate
        - std_tpr: standard deviation of true positive rate
        - mean_fdr: mean value of false discovery rate
        - std_fdr: standard deviation of false discovery rate
    """

    n = importance_score.shape[0]
    
    tpr = np.zeros([n, ])
    fdr = np.zeros([n, ])

    # For each sample
    for i in range(n):    
        # tpr   
        tpr_nom = np.sum(importance_score[i, :] * ground_truth[i, :])
        tpr_den = np.sum(ground_truth[i, :])
        tpr[i] = 100 * float(tpr_nom)/float(tpr_den + 1e-8)
            
        # fdr
        fdr_nom = np.sum(importance_score[i, :] * (1-ground_truth[i, :]))
        fdr_den = np.sum(importance_score[i,:])
        fdr[i] = 100 * float(fdr_nom)/float(fdr_den+1e-8)
        
    mean_tpr = np.mean(tpr)
    std_tpr = np.std(tpr)
    mean_fdr = np.mean(fdr)
    std_fdr = np.std(fdr)  
    
    return mean_tpr, std_tpr, mean_fdr, std_fdr


def prediction_performance_metric (y_test, y_hat):
    """Performance metrics for prediction (AUC, APR, Accuracy).
    
    Args:
        - y_test: testing set labels
        - y_hat: prediction on testing set
        
    Returns:
        - auc: area under roc curve
        - apr: average precision score
        - acc: accuracy
    """
    
    auc = roc_auc_score (y_test[:, 1], y_hat[:, 1])
    apr = average_precision_score (y_test[:, 1], y_hat[:, 1])
    acc = accuracy_score (y_test[:, 1], 1.*(y_hat[:, 1] > 0.5))
    
    return auc, apr, acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint="checkpoint.pth.tar", best="best.pth.tar"):
    """Saves a checkpoint and overwrites the best model when is_best = True
    """
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def load_previous_state(path, model, device, optimizer=None):
    """Loads saved state 
    """
    assert os.path.isfile(path), "no checkpoint found at '{}'".format(path)

    checkpoint = torch.load(path, map_location=device)
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]
    saved_args = checkpoint["args"]
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print("Loaded '{}'".format(path))

    return model, optimizer, best_acc, start_epoch, saved_args
