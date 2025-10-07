import math
import torch
import torch.nn as nn

class Embedding(nn.Module): 
    def __init__(self, vocab_len, max_seq_len, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1)  # Shape: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)  
        pe[:, 1::2] = torch.cos(pos * div_term)  
        self.register_buffer('pe', pe)

    def get_embedding(self, tokens):
        tokens = tokens.long().to(self.embedding.weight.device)
        word_embedding = self.embedding(tokens) * math.sqrt(self.d_model)
        word_embedding = word_embedding + self.pe
        return word_embedding
    