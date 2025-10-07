import torch.nn as nn
from attention import Attention

class Layer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, eps): 
        super(Layer, self).__init__()
        self.eps = eps
        self.n_heads = n_heads
        self.masked_attention = Attention(d_model, n_heads)
        self.fnn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, embedding_output, padding_mask, lookahead_mask):
        x = self.layernorm1(embedding_output)
        attention_output = self.masked_attention(x, padding_mask, lookahead_mask)
        attention_output = attention_output + x

        fnn_output = self.fnn(self.layernorm2(attention_output))
        output = fnn_output + attention_output
        return output 