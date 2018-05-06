#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import sys

import torch
from torch import nn        
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from qanet.squad_dataset import SquadDataset
from qanet.qanet import QANet
from evaluate import evaluate
from constants import use_cuda

print("Is CUDA available?", end=' ')
if use_cuda:
    print("Yes!")
    torch.cuda.empty_cache()
else:
    print("N0 :(")
     
data_prefix = 'data/'
params_file = "params.json"
word_embed_file = data_prefix + 'glove.trimmed.300d.npz'
char_embed_file = data_prefix + 'char2ix.json'

def train(model, train_loader,dev_loader, n_epochs=20, learning_rate=1e-3, betas=(0.8, 0.999),
          batch_size=32, save_model=False, print_every=1000):
    if use_cuda:
        model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate, betas=betas)

    loss_tracker = []
    for epoch in range(n_epochs):
        
        total_loss = 0
        n_batches = len(train_loader)
        start = time.time()
        
        for batch_idx, (context_word, question_word, context_char, question_char, spans, ctx_raw, q_raw) in enumerate(train_loader):
            
            optimizer.zero_grad()
        
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
            
            loss = criterion(p1, span_begin)
            loss += criterion(p2, span_end)
            
            total_loss += loss.data[0]
            
            loss.backward()
            optimizer.step()
            
            # time utils
            rem_time = (time.time()-start) * (n_batches-batch_idx + 1) / (batch_idx + 1)
            
            rem_h = int(rem_time // 3600)
            rem_m = int(rem_time // 60 - rem_h * 60)
            rem_s = int(rem_time % 60)
            print("Batch : %d / %d ----- Time remaining : %02d:%02d:%02d" % (batch_idx, n_batches, rem_h, rem_m, rem_s), end="\r")            
            
            if batch_idx % print_every == 0:
                if batch_idx != 0 :
                    loss_tracker.append(total_loss / print_every)
                    print("\nLoss : %.3f" % (loss_tracker[-1]))
                    np.savetxt('loss.txt', np.array(loss_tracker))
                    plt.semilogy(loss_tracker)
                    plt.xlabel('Update')
                    plt.ylabel('Loss')
                    plt.savefig('loss.png')
                    plt.clf()
                elif epoch == 0:
                    loss_tracker.append(total_loss)
                total_loss = 0

        print("Epoch : %d ----- Loss : %.3f" % (epoch, total_loss / len(train_loader)))
        
        if save_model:
            torch.save(model, 'qanet.pt')
        
        print("\nEvaluating model on dev set")
        evaluate(model, dev_loader, batch_size=batch_size)

if __name__ == "__main__":
    
    # load model parameters
    with open(params_file) as f:
        params = json.load(f)
        
    n_epochs = params["n_epochs"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    betas = (params["beta1"], params["beta2"])
    
    embeddings = np.load(word_embed_file)['glove']
    
    with open(char_embed_file) as json_data:
        char2ix = json.load(json_data)
    
    # loading dataset
    train_set = SquadDataset(file_ids_ctx=data_prefix + 'train1.context.ids', 
                             file_ids_q=data_prefix + 'train1.question.ids',
                             file_ctx =data_prefix + 'train1.context', 
                             file_q=data_prefix + 'train1.question', 
                             file_span=data_prefix + 'train1.span', 
                             char2ix_file=char_embed_file)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    dev_set = SquadDataset(file_ids_ctx=data_prefix + 'dev1.context.ids', 
                           file_ids_q=data_prefix + 'dev1.question.ids',
                           file_ctx =data_prefix + 'dev1.context', 
                           file_q=data_prefix + 'dev1.question', 
                           file_span=data_prefix + 'dev1.span', 
                           char2ix_file=char_embed_file)
    
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    if len(sys.argv) == 2:
    	#resuming training of model
    	model = torch.load(sys.argv[-1])
    else:
    	model = QANet(params, embeddings, len(char2ix))
    
    # save a bit of RAM
    del embeddings
    del char2ix
    
    train(model, train_loader, dev_loader, n_epochs=n_epochs, learning_rate=learning_rate, 
          betas=betas, batch_size=batch_size, save_model=True)