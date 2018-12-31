import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DAuM(nn.Module):

    def __init__(self, config):
        super(DAuM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.model_size)
        self.aspect_memory = nn.Parameter(torch.rand(config.aspect_kinds, config.model_size) * 0.02 - 0.01)
        self.layers = clones(Layer(config.model_size, config.lamda))
        self.fc = nn.Linear(config.model_size, config.n_class)
        self.weight = nn.Parameter(torch.rand(config.model_size, config.model_size) * 0.02 - 0.01)
        self.aspect_kinds = config.aspect_kinds

    def forward(self, sentence, aspect, sentence_mask, aspect_mask, aspect_positions):
        sentiment_memory = self.embedding(sentence)
        aspect = self.embedding(aspect) * aspect_mask.unsqueeze(-1)
        raw_aspect = aspect.sum(dim=1, keepdim=False) / aspect_mask.sum(dim=1, keepdim=True)
        aspect = raw_aspect
        sentiment = raw_aspect
        for layer in self.layers:
            sentiment, aspect = layer(sentiment, aspect, sentiment_memory, self.aspect_memory, sentence_mask)
        output = self.fc(sentiment)
        sentence_scores = self.score(aspect, sentiment_memory)  # (batch_size, time_step)
        aspect_scores = self.aspect_score(aspect, raw_aspect)   # (batch_size)
        time_step = sentence_scores.size(1)
        aspect_scores = aspect_scores.repeat(time_step, 1).transpose(0, 1)
        scores = torch.max(0, 1 - aspect_scores + sentence_scores)
        scores = scores * (1 - aspect_positions) * sentence_mask
        pre_loss = scores.mean()
        reg = self.aspect_memory.matmul(self.aspect_memory.transpose()) - torch.eye(self.aspect_kinds)
        reg_loss = (reg * reg).sum()
        return output, pre_loss, reg_loss

    def score(self, aspect, sentence):
        batch_size = sentence.size(0)
        time_step = sentence.size(1)
        weight = self.weight.repeat(batch_size, 1, 1)
        aspect = aspect.unsqueeze(1)
        mid = aspect.matmul(weight).squeeze()
        mid = mid.repeat(time_step, 1, 1).transpose(0, 1).unsqueeze(-2)
        sentence = sentence.unsqueeze(-1)
        scores = mid.matmul(sentence).squeeze()
        return scores

    def aspect_score(self, aspect, raw_aspect):
        batch_size = aspect.size(0)
        weight = self.weight.repeat(batch_size, 1, 1)
        aspect = aspect.unqueeze(1)
        mid = aspect.matmul(weight)
        raw_aspect = raw_aspect.unsqueeze(-1)
        scores = mid.matmul(raw_aspect).squeeze()
        return scores

class Layer(nn.Module):

    def __init__(self, model_size, lamda):
        super(Layer, self).__init__()
        self.aspect_attn = MultiplicativeAttention(model_size, model_size)
        self.sentiment_sentiment_attn = AdditiveAttention(model_size, model_size)
        self.sentiment_aspect_attn = AdditiveAttention(model_size, model_size)
        self.lamda = lamda

    def forward(self, sentiment, aspect, sentiment_memory, aspect_memory, mask):
        aspect_attn = self.aspect_attn(aspect, aspect_memory).unsqueeze(1)
        new_aspect = aspect_attn.matmul(aspect_memory).squeeze()
        sentiment_sentiment_attn = self.sentiment_sentiment_attn(sentiment, sentiment_memory, mask).unsqueeze(1)
        sentiment_aspect_attn = self.sentiment_aspect_attn(new_aspect, sentiment_memory, mask).unsqueeze(1)
        sentiment_attn = (1 - self.lamda) * sentiment_sentiment_attn + self.lamda * sentiment_aspect_attn
        sentiment = sentiment_attn.matmul(sentiment_memory)
        aspect = aspect + new_aspect
        return sentiment, aspect

class AdditiveAttention(nn.Module):

    def __init__(self, query_size, key_size):
        super(AdditiveAttention, self).__init__()
        self.proj = nn.Linear(query_size + key_size, 1)

    def forward(self, query, key, mask=None):
        # query: (batch_size, query_size)
        # key: (batch_size, time_step, key_size)
        # mask: (batch_size, time_step)
        time_step = key.size(1)
        query = query.repeat(time_step, 1, 1).transpose(0, 1)   # (batch_size, time_step, query_size)
        scores = self.proj(torch.cat([key, query], dim=2)).squeeze()     # (batch_size, time_step)
        if mask is not None:
            scores = scores - scores.max(dim=1, keepdim=True)[0]
            scores = torch.exp(scores) * mask
            attn_weights = scores / scores.sum(dim=1, keepdim=True)
        else:
            attn_weights = F.softmax(scores, dim=1)
        return attn_weights

class MultiplicativeAttention(nn.Module):

    def __init__(self, query_size, key_size):
        super(MultiplicativeAttention, self).__init__()
        self.weights = nn.Parameter(torch.rand(key_size, query_size) * 0.2 - 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, query, key, mask=None):
        # query: (batch_size, query_size)
        # key: (time_step, key_size)
        # mask: (batch_size, time_step)
        batch_size = key.size(0)
        time_step = key.size(1)
        weights = self.weights.repeat(batch_size, 1, 1)  # (batch_size, key_size, query_size)
        query = query.unsqueeze(-1)  # (batch_size, query_size, 1)
        mids = weights.matmul(query)  # (batch_size, key_size, 1)
        mids = mids.repeat(time_step, 1, 1, 1).transpose(0, 1)  # (batch_size, time_step, key_size, 1)
        key = key.repeat(batch_size, 1, 1)
        key = key.unsqueeze(-2)  # (batch_size, time_step, 1, key_size)
        scores = torch.tanh(key.matmul(mids).squeeze() + self.bias).squeeze()  # (batch_size, time_step)
        if mask is not None:
            scores = scores - scores.max(dim=1, keepdim=True)[0]
            scores = torch.exp(scores) * mask
            attn_weights = scores / scores.sum(dim=1, keepdim=True)
        else:
            attn_weights = F.softmax(scores, dim=1)
        return attn_weights

class DAuMDataset(Dataset):

    def __init__(self, path):
        data = np.load(path)
        self.aspects = torch.from_numpy(data['aspects']).long()
        self.sentences = torch.from_numpy(data['sentences']).long()
        self.labels = torch.from_numpy(data['labels']).long()
        self.aspect_lens = torch.from_numpy(data['aspect_lens']).long()
        self.sentence_lens = torch.from_numpy(data['sentence_lens']).long()
        self.aspect_positions = torch.from_numpy(data['aspect_positions']).long()
        self.len = self.labels.shape[0]
        aspect_max_len = self.aspects.size(1)
        sentence_max_len = self.sentences.size(1)
        self.aspect_mask = torch.zeros(aspect_max_len, aspect_max_len)
        self.sentence_mask = torch.zeros(sentence_max_len, sentence_max_len)
        for i in range(aspect_max_len):
            self.aspect_mask[i, 0:i + 1] = 1
        for i in range(sentence_max_len):
            self.context_mask[i, 0:i + 1] = 1

    def __getitem__(self, index):
        return self.aspects[index], self.sentences[index], self.labels[index], \
               self.aspect_mask[self.aspect_lens[index] - 1], \
               self.sentence_mask[self.sentence_lens[index] - 1], self.aspect_positions[index]

    def __len__(self):
        return self.len