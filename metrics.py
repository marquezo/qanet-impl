# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import f1_score

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
    batch_size = target.shape[0]
    f1 = torch.zeros(batch_size)
    
    for i in range(batch_size):
        tmp1 = torch.zeros(context_size)
        tmp2 = torch.zeros(context_size)
        
        tmp1[pred[i,0]:pred[i,1]+1] = 1
        tmp2[target[i,0]:target[i,1]+1] = 1
        
        f1[i] = f1_score(tmp1, tmp2)
    
    return f1.mean() * 100

def EM_Score(pred, target):
    """
    Compute EM Score
    
    Parameters
    ----------
    pred   : torch.LongTensor, shape=(batch_size, 2)
        Beginning and end of predicted spans for each element of a batch
    target : torch.LongTensor, shape=(batch_size, 2)
        Beginning and end of target spans for each element of a batch
        
    Returns
    -------
    em     : scalar
        EM score averaged over the batch_size
    """
    
    batch_size = target.shape[0]
    em = torch.zeros(batch_size)
    
    for i in range(batch_size):
        em[i] = torch.equal(pred[i], target[i])

    return em.mean() * 100
    
if __name__ == "__main__":
    
    pred = torch.LongTensor([[0,1], [2,4], [1,3], [1,2]])
    target = torch.LongTensor([[0,2], [2,4], [0,2], [3,4]])

    f1 = F1_Score(pred, target)
    
    em = EM_Score(pred, target)
    
    print(f1)
    print(em)