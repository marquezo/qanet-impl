import torch
from torch import nn

class CharacterEmbedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim=32,conv_out=200, 
                 norm_word_length=16, max_num_words=400, kernel_size=5,
                 padding=2):
        super(CharacterEmbedding, self).__init__()
        
        self.norm_word_length = norm_word_length
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kernel_size=kernel_size
        
        self.char_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                           embedding_dim=embedding_dim)
        
        #self.char_conv = nn.Conv2d(in_channels=max_num_words, out_channels=100,
        #                           kernel_size=(1, kernel_size))
        
        self.char_conv = nn.Conv1d(in_channels=embedding_dim, 
                                   out_channels=conv_out,
                                   kernel_size=kernel_size,
                                   padding=padding)
        self.max_pool = nn.MaxPool1d(kernel_size=norm_word_length)
    
    def forward(self, x):
        
        # here we assume each word is fed as torch longtensor
        assert x.shape[-1] == self.norm_word_length

        x = self.char_embedding(x).view(-1, self.embedding_dim, self.norm_word_length)
        x = self.char_conv(x)

        #x = self.max_pool(x).squeeze()
        # maybe just use torch.max instead 
        x = torch.max(x, dim=-1)

        return x