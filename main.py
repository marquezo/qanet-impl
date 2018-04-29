from __future__ import print_function
import numpy as np
import os
import random
from parse_json import data_from_json, read_write_dataset
from vocab_util import create_vocabulary, initialize_vocabulary, process_glove, data_to_token_ids, create_vocab2charix_dict
import json

random.seed(42)
np.random.seed(42)

root_project = '/home/orlandom/'

train_path = root_project + '/datasets/train-v1.1.json'
dev_path = root_project + '/datasets/dev-v1.1.json'
glove_file_path = root_project + '/Downloads/glove.6B/glove.6B.300d.txt'
data_prefix = 'data/'
save_path = data_prefix + 'glove.trimmed.300d.npz'
vocab_path = data_prefix + 'vocab'
train_context_path = data_prefix + 'train.context'
train_question_path = data_prefix + 'train.question'
train_context_ids_path = train_context_path + ".ids"
train_question_ids_path = train_question_path + ".ids"

######################################################################################################################
# Do vocabulary and word pre-processing
#####################################################################################################################
train_data = data_from_json(train_path)
dev_data = data_from_json(dev_path)

if not os.path.exists(data_prefix):
    os.makedirs(data_prefix)

train_num_questions, train_num_answers = read_write_dataset(train_data, 'train', data_prefix)

create_vocabulary(vocab_path,
                  [train_context_path,
                   train_question_path])

vocab, rev_vocab = initialize_vocabulary(vocab_path)

process_glove(glove_file_path, rev_vocab, save_path)

data_to_token_ids(train_context_path, train_context_ids_path, vocab_path)
data_to_token_ids(train_question_path, train_question_ids_path, vocab_path)

######################################################################################################################
# Do character pre-processing
#####################################################################################################################
char2ix = {'<pad>': '0'}
ix2char = {'0': '<pad>'}
char2ix_file = data_prefix + 'char2ix.json'
ix2char_file = data_prefix + 'ix2char.json'
vocab2charix_file = data_prefix + 'vocab2charix.json'

with open(vocab_path) as f:
    i = len(char2ix)
    for line in f:
        line = line.strip()
        if line in ['<pad>', '<sos>', '<unk>']:
            continue
        for char in line:
            if char not in char2ix:
                char2ix[char] = str(i)
                ix2char[str(i)] = char
                i += 1


with open(char2ix_file, 'w') as f:
    json.dump(char2ix, f)

with open(ix2char_file, 'w') as f:
    json.dump(ix2char, f)

create_vocab2charix_dict(vocab_path, vocab2charix_file, char2ix)