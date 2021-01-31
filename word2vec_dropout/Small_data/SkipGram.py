import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


def cf(input):
    us = []
    vs = []
    ns = []
    for u,v,n in input:
        us.append(u)
        vs.append(v)
        ns.append(n)
    return torch.cat(us), torch.cat(vs), torch.cat(ns)

class skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=True, p=0.5):
        super(skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.init_emb()
        self.dropoutQ = dropout
        self.dropout = nn.Dropout(p=p)
    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
    def forward(self, u_pos, v_pos, v_neg, batch_size):

        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)
        #embed_v = self.dropout(embed_v)
        if self.dropoutQ:
            embed_v = self.dropout(embed_v)
        #print(embed_u)
        #print(embed_v)

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

        neg_embed_v = self.v_embeddings(v_neg)
        #print(neg_embed_v)

        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        #print(neg_score)
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target + sum_log_sampled

        return -1*loss.sum()/batch_size
    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()
    def save_embedding(self, file_name, id2word):
        embeds = self.u_embeddings.weight.data
        fo = open(file_name, 'w')
        for idx in range(len(embeds)):
            word = id2word(idx)
            embed = ' '.join(embeds[idx])
            fo.write(word+' '+embed+'\n')
