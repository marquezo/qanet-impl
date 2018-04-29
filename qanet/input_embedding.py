import torch
from torch import nn

from qanet.word_embedding import WordEmbedding
from qanet.character_embedding import CharacterEmbedding
from qanet.highway import Highway

class InputEmbedding(nn.Module):
    
    def __init__(self, embeddings, num_chars):
        super(InputEmbedding, self).__init__()
        
        self.wordEmbedding = WordEmbedding(embeddings)
        self.contextCharacterEmbedding = CharacterEmbedding(num_chars)
        self.questionCharacterEmbedding = CharacterEmbedding(num_chars)
        self.highway = Highway()
    
    def forward(self, context_w, question_w, context_char, question_char):
        
        context_word, query_word = self.wordEmbedding(context_w, question_w)
        context_char = self.contextCharacterEmbedding(context_char)
        query_char = self.questionCharacterEmbedding(question_char)

        context = torch.cat((context_word, context_char), dim=-1 )
        query = torch.cat((query_word, query_char), dim=-1)
        
        context = self.highway(context)
        query = self.highway(query)
        
        return context, query