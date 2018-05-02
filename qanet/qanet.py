import torch
from torch import nn

from qanet.input_embedding import InputEmbedding
from qanet.embedding_encoder import EmbeddingEncoder
from qanet.context_query_attention import ContextQueryAttention
from qanet.model_encoder import ModelEncoder
from qanet.output import Output

class QANet(nn.Module):
    ''' All-in-one wrapper for all modules '''

    def __init__(self, word_embeddings, n_char_embeddings):
        super(QANet, self).__init__()

        self.inputEmbedding = InputEmbedding(word_embeddings, n_char_embeddings)
        self.embeddingEncoder = EmbeddingEncoder()
        self.contextQueryAttention = ContextQueryAttention()
        self.modelEncoder = ModelEncoder()
        self.output = Output()

    def forward(self, context_word, question_word, context_char, question_char):

        context_emb, question_emb = self.inputEmbedding(context_word,
                                                question_word, 
                                                context_char,
                                                question_char)
        
        # permuting to feed to embedding encoder layer
        context_emb = context_emb.permute(0, 2, 1)
        question_emb = question_emb.permute(0, 2, 1)  
        
        context_emb, question_emb = self.embeddingEncoder(context_emb, question_emb)

        c2q_attn, q2c_attn = self.contextQueryAttention(context_emb, question_emb)

        del question_emb
        
        mdl_emb = torch.cat((context_emb, 
                   c2q_attn.permute(0, 2, 1), 
                   context_emb*c2q_attn.permute(0, 2, 1), 
                   context_emb*q2c_attn.permute(0, 2, 1)), 1)
        
    
        M0, M1, M2 = self.modelEncoder(mdl_emb)

        del mdl_emb
        
        # permuting to feed to output layer
        M0 = M0.permute(0, 2, 1)
        M1 = M1.permute(0, 2, 1)
        M2 = M2.permute(0, 2, 1)
        
        p1, p2 = self.output(M0, M1, M2)

        return p1, p2