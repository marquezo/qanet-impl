import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
Les matrices C (#batch x d x n) et Q (#batch x d x m) sont output de layer 2.

n est nombre de mots dans C (contexte) et m, celui dans Q (question).

ContextQueryAttention prend en input les matrices C et Q et les matrices A et B,
respectivement, query-to-contex and context-to-query de dim (#batch x n x d) les 2.

"""
class ContextQueryAttention(nn.Module):

    def __init__(self, embed_size=128):
        super(ContextQueryAttention, self).__init__()

        self.d = embed_size

        self.W0 = nn.Linear(3 * self.d, 1)
        # Initialize W0 with Xavier
        nn.init.xavier_normal(self.W0.weight)

    def forward(self, C, Q):
        self.batch_size = C.shape[0]
        self.n = C.shape[2]
        self.m = Q.shape[2]

        # Evaluate the Similarity matrix, S
        S = self.similarity(C.permute(0, 2, 1), Q.permute(0, 2, 1))

        S_ = F.softmax(S, dim=2)
        S__ = F.softmax(S, dim=1)

        A = torch.bmm(S_, Q.permute(0, 2, 1))
        #   AT = A.permute(0,2,1)
        B = torch.matmul(torch.bmm(S_, S__.permute(0, 2, 1)), C.permute(0, 2, 1))
        #   BT = B.permute(0,2,1)

        # following the paper, this layer should return the context2query attention
        # and the query2context attention
        return A, B

    #         return torch.cat((self.C, AT, self.C*AT, self.C*BT), 0)

    """
    Evaluate Similarity tensor S of size (#batch x n x m).
    Creates a tensor (#batch x n*m x 3*d) to avoid for loops 
    """

    def similarity(self, C, Q):
        # Create QSim (#batch x n*m x d) where each of the m original rows are repeated n times
        QSim = self.repeatRowsTensor(Q, self.n)
        # Create CSim (#batch x n*m x d) where C is reapted m times
        CSim = C.repeat(1, self.m, 1)
        assert QSim.shape == CSim.shape
        QCSim = QSim * CSim

        # The "learned" Similarity in 1 col, put back
        Sim_col = self.W0(torch.cat((QSim, CSim, QCSim), dim=2))
        # Put it back in right dim
        Sim = Sim_col.view(self.batch_size, self.m, self.n).permute(0, 2, 1)

        return Sim

    """
    Inputs a Tensor (#batch x #rows x #cols) and # of time to repeat rows.
    Returns a Tensor (#batch x #lines*#repeat x #cols) with each row repeated consecutively
    """

    def repeatRowsTensor(self, X, rep):
        (depth, _, col) = X.shape
        # Open dim after batch ("depth")
        X = torch.unsqueeze(X, 1)
        # Repeat the matrix in the dim opened ("depth")
        X = X.repeat(1, rep, 1, 1)
        # Permute depth and lines to get the repeat over lines
        X = X.permute(0, 2, 1, 3)
        # Return to input (#batch x #lines*#repeat x #cols)
        X = X.contiguous().view(depth, -1, col)

        return X

#     def similarity(self):

#         # vectorizing equations instead of using for loops
#         self.S = self.W0(torch.cat((Q, C, Q * C), dim=-1))

#         for i in range(self.n):
#             for j in range(self.m):
#                 # concatenate c, q and their element wise product, row wise. SIZE = 3dx1
#                 concat_c_q = torch.cat((self.Q[:,j], self.C[:,i], self.Q[:,j] * self.C[:,i]), 0)
#                 self.S[i,j] = self.W0(concat_c_q)