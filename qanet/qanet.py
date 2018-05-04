import torch
from torch import nn

from qanet.input_embedding import InputEmbedding
from qanet.embedding_encoder import EmbeddingEncoder
from qanet.context_query_attention import ContextQueryAttention
from qanet.model_encoder import ModelEncoder
from qanet.output import Output

class QANet(nn.Module):
    ''' All-in-one wrapper for all modules '''

    def __init__(self, params, word_embeddings, n_char_embeddings):
        super(QANet, self).__init__()
        
        self.batch_size = params["batch_size"]
        
        # Defining dimensions using data from the params.json file
        self.word_embed_dim = params["word_embed_dim"]
        
        self.char_word_len = params["char_word_len"]
        
        self.char_embed_dim = params["char_embed_dim"]
        self.char_embed_n_filters = params["char_embed_n_filters"]
        self.char_embed_kernel_size = params["char_embed_kernel_size"]
        self.char_embed_pad = params["char_embed_pad"]
        
        self.highway_n_layers = params["highway_n_layers"]
        
        self.hidden_size = params["hidden_size"]
        
        self.embed_encoder_resize_kernel_size = params["embed_encoder_resize_kernel_size"]
        self.embed_encoder_resize_pad = params["embed_encoder_resize_pad"]
        
        self.embed_encoder_n_blocks = params["embed_encoder_n_blocks"]
        self.embed_encoder_n_conv = params["embed_encoder_n_conv"]
        self.embed_encoder_kernel_size = params["embed_encoder_kernel_size"]
        self.embed_encoder_pad = params["embed_encoder_pad"]
        self.embed_encoder_conv_type = params["embed_encoder_conv_type"]
        self.embed_encoder_n_heads = params["embed_encoder_n_heads"]

        self.model_encoder_n_blocks = params["model_encoder_n_blocks"]
        self.model_encoder_n_conv = params["model_encoder_n_conv"]
        self.model_encoder_kernel_size = params["model_encoder_n_blocks"]
        self.model_encoder_pad = params["model_encoder_pad"]
        self.model_encoder_conv_type = params["model_encoder_conv_type"]
        self.model_encoder_n_heads = params["model_encoder_n_heads"]

        # Initializing model layers

        self.inputEmbedding = InputEmbedding(word_embeddings, 
                                             n_char_embeddings,
                                             word_embed_dim=self.word_embed_dim,
                                             char_embed_dim=self.char_embed_dim,
                                             char_embed_n_filters=self.char_embed_n_filters,
                                             char_embed_kernel_size=self.char_embed_kernel_size,
                                             char_embed_pad=self.char_embed_pad,
                                             highway_n_layers=self.highway_n_layers)
        
        self.embeddingEncoder = EmbeddingEncoder(resize_in=self.word_embed_dim + self.char_embed_n_filters,
                                                 hidden_size=self.hidden_size,
                                                 resize_kernel=self.embed_encoder_resize_kernel_size,
                                                 resize_pad=self.embed_encoder_resize_pad,
                                                 n_blocks=self.embed_encoder_n_blocks,
                                                 n_conv=self.embed_encoder_n_conv,
                                                 kernel_size=self.embed_encoder_kernel_size,
                                                 padding=self.embed_encoder_pad,
                                                 conv_type=self.embed_encoder_conv_type,
                                                 n_heads=self.embed_encoder_n_heads,
                                                 batch_size=self.batch_size)

        self.contextQueryAttention = ContextQueryAttention(hidden_size=self.hidden_size)
        
        self.modelEncoder = ModelEncoder(n_blocks=self.model_encoder_n_blocks,
                                         n_conv=self.model_encoder_n_conv,
                                         kernel_size=self.model_encoder_kernel_size,
                                         padding=self.model_encoder_pad,
                                         hidden_size=4*self.hidden_size,
                                         conv_type=self.model_encoder_conv_type,
                                         n_heads=self.model_encoder_n_heads)
        
        self.output = Output(input_dim=4*self.hidden_size)

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