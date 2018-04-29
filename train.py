import numpy as np
from qanet.squad_dataset import SquadDataset
from torch.utils.data import DataLoader
import json
from qanet.input_embedding import InputEmbedding

data_prefix = 'data/'
save_path = data_prefix + 'glove.trimmed.300d.npz'
char2ix_file = data_prefix + 'char2ix.json'

with open(char2ix_file) as json_data:
    char2ix = json.load(json_data)

embeddings = np.load(save_path)['glove']

input_embedding_layer = InputEmbedding(embeddings, len(char2ix))

dataset = SquadDataset(file_ids_ctx=data_prefix + 'train.context.ids', file_ids_q=data_prefix + 'train.question.ids',
                      file_ctx =data_prefix + 'train.context', file_q=data_prefix + 'train.question', char2ix_file=char2ix_file)

train_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)

for batch_idx, (context_word, question_word, context_char, question_char) in enumerate(train_loader):

    input_context, input_question = input_embedding_layer(context_word, question_word, context_char.long(), question_char.long())

    print (input_context.shape)
    print (input_question.shape)

    break
