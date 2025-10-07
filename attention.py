import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__() 
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(0.1)
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_out = torch.nn.Linear(d_model, d_model) 
 
    def forward(self, x, padding_mask, lookahead_mask):
        batch_size, max_seq_len, d_model = x.shape
        d_k = d_model // self.n_heads

        q = self.w_q(x).view(batch_size, max_seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
        k = self.w_k(x).view(batch_size, max_seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
        v = self.w_v(x).view(batch_size, max_seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = attention_scores + lookahead_mask.to(attention_scores.device)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, max_seq_len, d_model)
        output = self.w_out(attention_output)
        return self.dropout(output)