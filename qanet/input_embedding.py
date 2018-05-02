import torch
from torch import nn

from qanet.word_embedding import WordEmbedding
from qanet.character_embedding import CharacterEmbedding
from qanet.highway import Highway

class InputEmbedding(nn.Module):
    
    def __init__(self, word_embeddings, n_char_embeddings, word_embed_dim=300,
                 char_embed_dim=32, char_embed_n_filters=200, 
                 char_embed_kernel_size=7, char_embed_pad=3, highway_n_layers=2):   
        
        super(InputEmbedding, self).__init__()
        
        self.wordEmbedding = WordEmbedding(word_embeddings)
        self.characterEmbedding = CharacterEmbedding(n_char_embeddings,
                                                     embedding_dim=char_embed_dim,
                                                     n_filters=char_embed_n_filters,
                                                     kernel_size=char_embed_kernel_size,
                                                     padding=char_embed_pad)
        self.highway = Highway(input_size = word_embed_dim + char_embed_n_filters,
                               n_layers=highway_n_layers)
    
    def forward(self, context_w, question_w, context_char, question_char):
        
        context_word, question_word = self.wordEmbedding(context_w, question_w)
        context_char = self.characterEmbedding(context_char)
        question_char = self.characterEmbedding(question_char)

        context = torch.cat((context_word, context_char), dim=-1 )
        question = torch.cat((question_word, question_char), dim=-1)
        
        context = self.highway(context)
        question = self.highway(question)
        
        return context, question