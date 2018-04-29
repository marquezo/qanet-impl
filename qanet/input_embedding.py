import torch
from torch import nn

from qanet.word_embedding import WordEmbedding
from qanet.character_embedding import CharacterEmbedding
from qanet.highway import Highway

class InputEmbedding(nn.Module):
    
    def __init__(self, embeddings, num_chars):
        super(InputEmbedding, self).__init__()
        
        self.wordEmbedding = WordEmbedding(embeddings)
        self.characterEmbedding = CharacterEmbedding(num_chars)
        self.highway = Highway()
    
    def forward(self, context_w, question_w, context_char, question_char):
        
        context_word, question_word = self.wordEmbedding(context_w, question_w)
        context_char = self.characterEmbedding(context_char)
        question_char = self.characterEmbedding(question_char)

        context = torch.cat((context_word, context_char), dim=-1 )
        question = torch.cat((question_word, question_char), dim=-1)
        
        context = self.highway(context)
        question = self.highway(question)
        
        return context, question