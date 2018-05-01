#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test

"""
import time
start = time.time()

import numpy as np
from qanet.squad_dataset import SquadDataset
from torch.utils.data import DataLoader
import json

from qanet.qanet import QANet

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
use_cuda = False
print("Is CUDA available?", end=' ')
if use_cuda:
    print("Yes!")
    torch.cuda.empty_cache()
else:
    print("N0 :(")

data_prefix = 'data/'
save_path = data_prefix + 'glove.trimmed.300d.npz'
char2ix_file = data_prefix + 'char2ix.json'

with open(char2ix_file) as json_data:
    char2ix = json.load(json_data)

embeddings = np.load(save_path)['glove']

# loading dataset
dataset = SquadDataset(file_ids_ctx=data_prefix + 'train.context.ids', file_ids_q=data_prefix + 'train.question.ids',
                      file_ctx =data_prefix + 'train.context', file_q=data_prefix + 'train.question', file_span=data_prefix + 'train.span', char2ix_file=char2ix_file)

train_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

model = QANet(embeddings, len(char2ix))

if use_cuda:
    model = model.cuda()
    
# save a bit of RAM
del embeddings
del char2ix

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for batch_idx, (context_word, question_word, context_char, question_char, spans, ctx_raw, q_raw) in enumerate(train_loader):

    context_word = Variable(context_word)
    question_word = Variable(question_word)
    context_char = Variable(context_char.long())
    question_char = Variable(question_char.long())
    
    if use_cuda:
        context_word = context_word.cuda()
        question_word = question_word.cuda()
        context_char = context_char.cuda()
        question_char = question_char.cuda()
        
    p1, p2 = model(context_word, question_word, context_char, question_char)
    
    print(p1.shape)
    print(p2.shape)

    break


print("Time elapsed : %.3f" % (time.time()-start))
