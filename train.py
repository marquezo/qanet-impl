import numpy as np
from qanet.squad_dataset import SquadDataset
from torch.utils.data import DataLoader
import json
from qanet.input_embedding import InputEmbedding
from qanet.residual_block import ResidualBlock

data_prefix = 'data/'
save_path = data_prefix + 'glove.trimmed.300d.npz'
char2ix_file = data_prefix + 'char2ix.json'

with open(char2ix_file) as json_data:
    char2ix = json.load(json_data)

embeddings = np.load(save_path)['glove']

dataset = SquadDataset(file_ids_ctx=data_prefix + 'train.context.ids', file_ids_q=data_prefix + 'train.question.ids',
                      file_ctx =data_prefix + 'train.context', file_q=data_prefix + 'train.question', file_span=data_prefix + 'train.span', char2ix_file=char2ix_file)

train_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)


input_embedding_layer = InputEmbedding(embeddings, len(char2ix))

embedding_encoder_layer_context = ResidualBlock(num_blocks=1, num_conv_layers=4, kernel_size=7, mask=None,
                 num_filters=128, input_projection=True)

embedding_encoder_layer_question = ResidualBlock(num_blocks=1, num_conv_layers=4, kernel_size=7, mask=None,
                 num_filters=128, input_projection=True)

for batch_idx, (context_word, question_word, context_char, question_char, spans, ctx_raw, q_raw) in enumerate(train_loader):

    input_context, input_question = input_embedding_layer(context_word, question_word, context_char.long(), question_char.long())

    #Permute the dimensions so that the subsequent layers take the input in the expected shape
    input_context = input_context.permute(0, 2, 1)
    input_question = input_question.permute(0, 2, 1)

    input_context = embedding_encoder_layer_context(input_context)
    input_question = embedding_encoder_layer_question(input_question)

    print (input_context.shape)
    print (input_question.shape)

    break


