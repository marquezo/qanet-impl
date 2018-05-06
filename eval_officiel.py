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
        
        em = EM_Score(pred_spans, spans)
        f1 = F1_Score(pred_spans, spans)
        
        em_total += em
        f1_total += f1
        
        rem_time = (time.time()-start) * (n_batches-batch_idx + 1) / (batch_idx + 1)
        
        rem_h = int(rem_time // 3600)
        rem_m = int(rem_time // 60 - rem_h * 60)
        rem_s = int(rem_time % 60)
        print("Batch : %d / %d ----- Time remaining : %02d:%02d:%02d" % (batch_idx, n_batches, rem_h, rem_m, rem_s), end="\r")  

    em_total /= len(dev_loader)
    f1_total /= len(dev_loader)

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
    dataset = SquadDataset(file_ids_ctx=data_prefix + 'dev1.context.ids', 
                           file_ids_q=data_prefix + 'dev1.question.ids',
                           file_ctx =data_prefix + 'dev1.context', 
                           file_q=data_prefix + 'dev1.question', 
                           file_span=data_prefix + 'dev1.span', 
                           char2ix_file=char_embed_file)
    
    dev_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # using a pre-trained model
    model = torch.load(pretrained_file)

    evaluate(model, dev_loader, batch_size=batch_size)    
    
    