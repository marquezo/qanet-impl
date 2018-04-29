class CharacterEmbedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim=32, conv_out=200, 
                 word_length=16, kernel_size=5, padding=2):
        super(CharacterEmbedding, self).__init__()
        
        self.word_length = word_length
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kernel_size = (1, kernel_size)
        self.padding = (0, padding)
        
        self.char_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                           embedding_dim=embedding_dim)
        
        self.char_conv = nn.Conv2d(in_channels=embedding_dim, 
                                   out_channels=conv_out,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding)
            
    def forward(self, x):

        # embedding layer only supports 2D inputs for now. 
        # reshape input tensor first, then pass it through the layer.
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.char_embedding(x)
        x = x.view(batch_size, -1, self.word_length, self.embedding_dim)
        
        # embedding dim of characters is number of channels of conv layer
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.char_conv(x))
        x = x.permute(0, 2, 3, 1)
        
        # max pooling over word length to have final tensor
        x, _ = torch.max(x, dim=2)

        return x