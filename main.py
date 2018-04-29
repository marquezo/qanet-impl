from __future__ import print_function
import numpy as np
import os
import random
from parse_json import data_from_json, read_write_dataset
from vocab_util import create_vocabulary, initialize_vocabulary, process_glove, data_to_token_ids

random.seed(42)
np.random.seed(42)

train_path = '/home/orlandom/datasets/train-v1.1.json'
dev_path = '/home/orlandom/datasets/dev-v1.1.json'
glove_file_path = '/home/orlandom/Downloads/glove.6B/glove.6B.300d.txt'
data_prefix = 'data/'
save_path = data_prefix + 'glove.trimmed.300d.npz'
vocab_path = data_prefix + 'vocab'
train_context_path = data_prefix + 'train.context'
train_question_path = data_prefix + 'train.question'
train_context_ids_path = train_context_path + ".ids"
train_question_ids_path = train_question_path + ".ids"


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