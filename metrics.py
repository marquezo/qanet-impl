# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import f1_score
import sys

def F1_Score(pred, target, context_size=400):
    """
    Compute F1 Score
    
    Parameters
    ----------
    pred   : torch.LongTensor, shape=(batch_size, 2)
        Beginning and end of predicted spans for each element of a batch
    target : torch.LongTensor, shape=(batch_size, 2)
        Beginning and end of target spans for each element of a batch

    Returns
    -------
    f1     : scalar
        F1 score averaged over the batch_size
    """
    
    tmp1 = torch.zeros(context_size)
    tmp2 = torch.zeros(context_size)
    
    tmp1[pred[0]:pred[1]+1] = 1
    tmp2[target[0]:target[1]+1] = 1
        
    f1 = f1_score(tmp1, tmp2)
    
    return f1 * 100

def EM_Score(pred, target):
    """
    Compute EM Score
    
    Parameters
    ----------
    pred   : torch.LongTensor, shape=(2)
        Beginning and end of predicted spans for each element of a batch
    target : torch.LongTensor, shape=(2)
        Beginning and end of target spans for each element of a batch
        
    Returns
    -------
    em     : scalar
        EM score averaged over the batch_size
    """
    
    em = torch.equal(pred, target)

    return em * 100
    
if __name__ == "__main__":
    
    pred = torch.LongTensor([[0,1], [2,4], [1,3], [1,2]])
    target = torch.LongTensor([[0,2], [2,4], [0,2], [3,4]])

    f1 = F1_Score(pred, target)
    
    em = EM_Score(pred, target)
    
    print(f1)
    print(em)