#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from qanet.qanet import QANet
from qanet.squad_dataset import SquadDataset
from metrics import F1_Score, EM_Score
from constants import use_cuda

import json
import numpy as np
import time
import sys

data_prefix = 'data/'
params_file = "params.json"
word_embed_file = data_prefix + 'glove.trimmed.300d.npz'
char_embed_file = data_prefix + 'char2ix.json'

def evaluate(model, dev_loader, batch_size=8):
    
    if use_cuda:
        model = model.cuda()
        
    model.eval()

    start = time.time()
    n_batches = len(dev_loader)
    
    em_total = 0
    f1_total = 0

    for batch_idx, (context_word, question_word, context_char, question_char, spans, ctx_raw, q_raw) in enumerate(dev_loader):

        context_word = Variable(context_word)
        question_word = Variable(question_word)
        context_char = Variable(context_char.long())
        question_char = Variable(question_char.long())
        

        span_begin = Variable(spans[:,0])
        span_end = Variable(spans[:,1])
        
        if use_cuda:
            context_word = context_word.cuda()
            question_word = question_word.cuda()
            context_char = context_char.cuda()
            question_char = question_char.cuda()
            span_begin = span_begin.cuda()
            span_end = span_end.cuda()
        
        p1, p2 = model(context_word, question_word, context_char, question_char)

        p1, p2 = F.softmax(p1, dim=-1), F.softmax(p2, dim=-1)
        
        p1 = p1.cpu()
        p2 = p2.cpu()
        p_matrix = torch.bmm(p1.unsqueeze(2), p2.unsqueeze(1))

        pred_spans = torch.zeros(batch_size, 2).long()
        n_items = p_matrix.shape[0]
        # no support for batch triu in pytorch currently
        for i in range(n_items):
            p_matrix[i] = torch.triu(p_matrix[i])
            
            tmp = np.argmax(p_matrix[i].data.numpy())
            
            pred_spans[i,0] = int(tmp // p_matrix.shape[1])
            pred_spans[i,1] = int(tmp % p_matrix.shape[2])
    
        del p_matrix
        
        for i in range(n_items):
            em_max = 0
            f1_max = 0
            for j in range(len(spans)//2):
                curr = spans[i,2*j:2*(j+1)]
                if curr[0] == -1:
                    continue
                
                em_tmp = EM_Score(pred_spans[i], curr)
                f1_tmp = F1_Score(pred_spans[i], curr)
                
                if em_tmp > em_max:
                    em_max = em_tmp
                if f1_tmp > f1_max:
                    f1_max = f1_tmp
            
            em_total += em_max
            f1_total += f1_max
            
        rem_time = (time.time()-start) * (n_batches-batch_idx + 1) / (batch_idx + 1)
        
        rem_h = int(rem_time // 3600)
        rem_m = int(rem_time // 60 - rem_h * 60)
        rem_s = int(rem_time % 60)
        print("Batch : %d / %d ----- Time remaining : %02d:%02d:%02d" % (batch_idx, n_batches, rem_h, rem_m, rem_s), end="\r")  

        if batch_idx == 1000:
            break

#    em_total /= (len(dev_loader) * batch_size)
#    f1_total /= (len(dev_loader) * batch_size)
            
    em_total /= (1000 * batch_size)
    f1_total /= (1000 * batch_size)        

    print()
    print("EM Score : %f" % em_total)
    print("F1 Score : %f" % f1_total)

if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        pretrained_file = sys.argv[-1]
    else:
        pretrained_file = 'qanet.pt'
    
    # load model parameters
    with open(params_file) as f:
        params = json.load(f)
        
    batch_size = params["batch_size"]
        
    # loading dataset
#    dataset = SquadDataset(file_ids_ctx=data_prefix + 'dev.context.ids', 
#                           file_ids_q=data_prefix + 'dev.question.ids',
#                           file_ctx =data_prefix + 'dev.context', 
#                           file_q=data_prefix + 'dev.question', 
#                           file_span=data_prefix + 'dev.span', 
#                           char2ix_file=char_embed_file)

    dataset = SquadDataset(file_ids_ctx=data_prefix + 'train.context.ids', 
                           file_ids_q=data_prefix + 'train.question.ids',
                           file_ctx =data_prefix + 'train.context', 
                           file_q=data_prefix + 'train.question', 
                           file_span=data_prefix + 'train.span', 
                           char2ix_file=char_embed_file)
    
    dev_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # using a pre-trained model
    model = torch.load(pretrained_file)

    evaluate(model, dev_loader, batch_size=batch_size)    
    
    