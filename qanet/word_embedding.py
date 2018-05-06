import torch
import torch.nn as nn
import torch.nn.functional as F
import constants

class WordEmbedding(nn.Module):

    # word_embeddings comes from numpy
    def __init__(self, word_embeddings):
        super(WordEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=word_embeddings.shape[0],
                                           embedding_dim=word_embeddings.shape[1])
        #Cast to float because the character embeding will be returned as a float, and we need to concatenate the two
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(word_embeddings).float())

        # Only the unknown embedding requires grad
        self.word_embedding.weight.requires_grad = False
        #self.word_embedding.weight[constants.UNK_ID].requires_grad = True

        del word_embeddings

    def forward(self, input_context, input_question):
        
        context_word_emb = self.word_embedding(input_context)

        context_word_emb = F.dropout(context_word_emb, p=0.1, training=self.training)

        question_word_emb = self.word_embedding(input_question)

        question_word_emb = F.dropout(question_word_emb, p=0.1, training=self.training)
        
        return context_word_emb, question_word_emb