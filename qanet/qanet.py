from torch import nn

from qanet.input_embedding import InputEmbedding
from qanet.embedding_encoder import EmbeddingEncoder
from qanet.context_query_attention import ContextQueryAttention
from qanet.model_encoder import ModelEncoder
from qanet.output import Output

class QANet(nn.Module):
    ''' All-in-one wrapper for all modules '''
    
    def __init__(self):
        super(QAnet, self).__init__()
        
        self.InputEmbedding = InputEmbedding()
        self.EmbeddingEncoder = EmbeddingEncoder()
        self.ContextQueryAttention = ContextQueryAttention()
        self.ModelEncoder = ModelEncoder()
        self.Output = Output()
        
    def forward(self, context, query):
        
        context, query = self.InputEmbedding(context, query)
        context, query = self.EmbeddingEncoder(context, query)
        
        context2query_attn, query2context_attn = self.ContextQueryAttention(context, query)

        # TODO: concatenate context2query_attn and query2context_attn to feed to ModelEncoder

        # TODO: add ModelEncoder and Output forward statements