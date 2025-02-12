import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout

        self.q_in_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.k_in_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.v_in_proj = nn.Linear(embed_dim, embed_dim).cuda()

        # self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        q = self.q_in_proj(q)
        k = self.k_in_proj(k)
        v = self.v_in_proj(v)

        sigma_u = torch.einsum('bi,bj->bij', q, k)
        sigma_u = torch.softmax(sigma_u, dim=2)
        sigma_u = F.dropout(sigma_u, p=self.dropout, training=self.training)
        ru = torch.einsum('bii,bi->bi', sigma_u, v)

        return ru
