import numpy as np
from InputEmbeddingLayer import InputEmbeddingLayer
from qanet.squad_dataset import SquadDataset
from torch.utils.data import DataLoader

data_prefix = 'data/'
save_path = data_prefix + 'glove.trimmed.300d.npz'
char2ix_file = data_prefix + 'char2ix.json'

embeddings = np.load(save_path)['glove']
embedding_layer = InputEmbeddingLayer(embeddings)

dataset = SquadDataset(file_ids_ctx='data/train.context.ids', file_ids_q='data/train.question.ids',
                      file_ctx = 'data/train.context', file_q='data/train.question', char2ix_file=char2ix_file)

train_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)


for batch_idx, (context_word, question_word, context_char, question_char) in enumerate(train_loader):

    input_context = embedding_layer.word_embedding(context_word)
    input_question = embedding_layer.word_embedding(question_word)

    print (context_char.shape)
    print (question_char.shape)


    break

#print (input_context)
print (input_question.shape)