import torch

from qanet.word_embedding import WordEmbedding
from qanet.character_embedding import CharacterEmbedding
from qanet.highway import Highway

class InputEmbedding(nn.Module):
    
    def __init__(self):
        super(InputEmbedding, self).__init__()
        
        self.ContextWordEmbedding = WordEmbedding()
        self.ContextCharacterEmbedding = CharacterEmbedding()

        # TODO: parameter sharing between context and query layers?
        
        self.QueryWordEmbedding = WordEmbedding()
        self.QueryCharacterEmbedding = CharacterEmbedding()

        self.Highway = Highway()
    
    def forward(self, context, query):
        
        context_word = self.ContextWordEmbedding(context)
        context_char = self.ContextCharacterEmbedding(context)
        
        context = torch.cat((context_word, context_char), dim=-1 )
        
        query_word = self.QueryWordEmbedding(query)
        query_char = self.QueryCharacterEmbedding(query)
        
        query = torch.cat((query_word, query_char), dim=-1)
        
        context = self.Highway(context)
        query = self.Highway(query)
        
        return context, query