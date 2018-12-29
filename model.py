import torch
import torch.nn as nn
import torch.nn.functional as F

class DAuM(nn.Module):

    def __init__(self):
        super(DAuM, self).__init__()
        pass

    def forward(self):
        pass

class Layer(nn.Module):

    def __init__(self):
        super(Layer, self).__init__()
        pass

    def forward(self):
        pass

class AdditiveAttention(nn.Module):

    def __init__(self, query_size, key_size):
        super(AdditiveAttention, self).__init__()
        self.project = nn.Linear(query_size + key_size, 1)

    def forward(self, query, key, mask):
        # query: (batch_size, query_size)
        # key: (batch_size, time_step, key_size)
        # mask: (batch_size, time_step)
        time_step = key.size(1)
        query = query.repeat(time_step, 1, 1).transpose(0, 1)   # (batch_size, time_step, query_size)
        scores = self.project(torch.cat([key, query], dim=2)).squeeze()     # (batch_size, time_step)
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        scores = torch.exp(scores) * mask
        attn_weights = scores / scores.sum(dim=1, keepdim=True)
        return attn_weights

class MultiplicativeAttention(nn.Module):

    def __init__(self, query_size, key_size):
        super(MultiplicativeAttention, self).__init__()
        self.weights = nn.Parameter(torch.rand(key_size, query_size) * 0.2 - 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, query, key, mask):
        # query: (batch_size, query_size)
        # key: (batch_size, time_step, key_size)
        # mask: (batch_size, time_step)
        batch_size = key.size(0)
        time_step = key.size(1)
        weights = self.weights.repeat(batch_size, 1, 1)  # (batch_size, key_size, query_size)
        query = query.unsqueeze(-1)  # (batch_size, query_size, 1)
        mids = weights.matmul(query)  # (batch_size, key_size, 1)
        mids = mids.repeat(time_step, 1, 1, 1).transpose(0, 1)  # (batch_size, time_step, key_size, 1)
        key = key.unsqueeze(-2)  # (batch_size, time_step, 1, key_size)
        scores = torch.tanh(key.matmul(mids).squeeze() + self.bias).squeeze()  # (batch_size, time_step)
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        scores = torch.exp(scores) * mask
        attn_weights = scores / scores.sum(dim=1, keepdim=True)
        return attn_weights