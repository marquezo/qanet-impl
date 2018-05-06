import torch
from torch import nn
import torch.nn.functional as F

class CharacterEmbedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim=32, n_filters=200, 
                 kernel_size=5, padding=2):
        super(CharacterEmbedding, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kernel_size = (1, kernel_size)
        self.padding = (0, padding)
        
        self.char_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                           embedding_dim=embedding_dim)
        
        self.char_conv = nn.Conv2d(in_channels=embedding_dim, 
                                   out_channels=n_filters,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding)
            
    def forward(self, x):

        # embedding layer only supports 2D inputs for now. 
        # reshape input tensor first, then pass it through the layer.
        batch_size = x.shape[0]
        word_length = x.shape[-1]
        
        x = x.view(batch_size, -1)
        x = self.char_embedding(x)
        x = x.view(batch_size, -1, word_length, self.embedding_dim)
        
        # embedding dim of characters is number of channels of conv layer
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.char_conv(x))
        x = x.permute(0, 2, 3, 1)
        
        # max pooling over word length to have final tensor
        x, _ = torch.max(x, dim=2)

        x = F.dropout(x, p=0.05, training=self.training)

        return x