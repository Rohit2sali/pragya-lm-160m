import torch
import torch.nn as nn
from embedding import Embedding
from layer import Layer
from tokenizer_pretraining import Tokenization
import torch.nn.functional as F
import random

tokenizer_obj = Tokenization(max_seq_len = 127)

class Transformer(nn.Module):
    def __init__(self, vocab_len, max_seq_len, n_heads, d_model, d_ff, n_layers, eps):
        super(Transformer, self).__init__()
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.embedding = Embedding(vocab_len, max_seq_len, d_model)
        self.layer = nn.ModuleList([Layer(d_model, d_ff, n_heads, eps) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_len)
        self.final_layernorm = nn.LayerNorm(d_model)
        self.tokenizer = tokenizer_obj.get_tokenizer()

    def lookahead_mask(self, batch_size, max_seq_len):
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask[mask == 1] = float("-inf")
        mask = mask.view(1, 1, max_seq_len, max_seq_len)
        mask = mask.expand(batch_size, -1, -1, -1)
        return mask

    def padding_mask(self, tokens):
        batch_size, max_seq_len = tokens.shape
        padding_mask = (tokens != self.tokenizer.pad_token_id)
        padding_mask = padding_mask.view(batch_size, 1, 1, max_seq_len)
        padding_mask = padding_mask.expand(-1, -1, max_seq_len, -1).float()
        mask = padding_mask.masked_fill(padding_mask == 0, float('-inf')).masked_fill(padding_mask == 1, 0.0)
        return mask
    
    def mask_answer_tokens(self, input_tokens):
        for i in range(len(input_tokens)):
            if(input_tokens[i] == self.tokenizer.sep_token_id):
                break
            input_tokens[i] = self.tokenizer.pad_token_id
        return input_tokens
    
    def apply_temperature(self, output, temperature: float = 1.0):
        scaled_output = output / temperature  
        
        output = F.softmax(scaled_output, dim=-1)  
        return output
    
    def repetition_penalty(self, output, predicted_tokens, col, penalty = 1.5):
        for token_id in (set(predicted_tokens)):
            if(output[0][col][token_id] < 0):
                output[0][col][token_id] *= penalty
            else:
                output[0][col][token_id] /= penalty
        return output
    
    def generate_greedy(self, input_tokens):# input_tokens must be single sentence 
        batch_size, max_seq_len = input_tokens.shape
        lookahead_mask = self.lookahead_mask(batch_size, max_seq_len)
        output_tokens = input_tokens
        predicted_tokens = []
        for i in range(self.max_seq_len-1):
            if(input_tokens[0][i+1] == self.tokenizer.pad_token_id):
                padding_mask = self.padding_mask(input_tokens)
                decoder_input = self.embedding.get_embedding(input_tokens)
                decoder_output = self.layer[0].forward(decoder_input, padding_mask, lookahead_mask)
                for layer in self.layer[1:]:
                    decoder_output = layer(decoder_output, padding_mask, lookahead_mask)
                output = self.linear(self.final_layernorm(decoder_output)) # (batch_size, max_seq_len, vocab_size)
                output = self.repetition_penalty(output,predicted_tokens, col = i)
                next_token = torch.argmax(output[:, i, :], dim=-1)
                predicted_tokens.append(next_token)
                output_tokens[0][i+1] = next_token
                input_tokens = output_tokens
                if (next_token == self.tokenizer.eos_token_id).any():
                    return self.tokenizer.decode(self.mask_answer_tokens(input_tokens[0]), skip_special_tokens=True)
                
        return self.tokenizer.decode(self.mask_answer_tokens(input_tokens[0]), skip_special_tokens=True)
    
    def generate_top_p(self, input_tokens):# input_tokens must be single sentence 
        batch_size, max_seq_len = input_tokens.shape
        lookahead_mask = self.lookahead_mask(batch_size, max_seq_len)
        output_tokens = input_tokens
        predicted_tokens = []
        for i in range(self.max_seq_len-1):
            if(input_tokens[0][i+1] == self.tokenizer.pad_token_id):
                padding_mask = self.padding_mask(input_tokens)
                decoder_input = self.embedding.get_embedding(input_tokens)
                decoder_output = self.layer[0].forward(decoder_input, padding_mask, lookahead_mask)
                for layer in self.layer[1:]:
                    decoder_output = layer(decoder_output, padding_mask, lookahead_mask)
                output = self.linear(self.final_layernorm(decoder_output)) # (batch_size, max_seq_len, vocab_size)
                p = 0.95
                x = 0
                k = 0
                y = []
                output = self.repetition_penalty(output,predicted_tokens, col = i)
                output = self.apply_temperature(output, temperature = 0.2)
                temp_output = output.squeeze(0)
                while(x < p):
                    b = torch.argsort(temp_output, dim=-1, descending=True)
                    j = b[i][k].item()
                    y.append(j)
                    x += output[0][i][j].item()
                    k += 1
                next_token = random.choice(y)
                predicted_tokens.append(next_token)
                output_tokens[0][i+1] = next_token
                input_tokens = output_tokens
                if (next_token == self.tokenizer.eos_token_id):
                    return self.tokenizer.decode(self.mask_answer_tokens(input_tokens[0]), skip_special_tokens=True)
        return self.tokenizer.decode(self.mask_answer_tokens(input_tokens[0]), skip_special_tokens=True)

    def forward(self, input_tokens):       
        embedding_output = self.embedding.get_embedding(input_tokens)
        lookahead_mask = self.lookahead_mask(len(input_tokens), len(input_tokens[0]))
        layer_output = self.layer[0](embedding_output, padding_mask=None, lookahead_mask=lookahead_mask) # in pre-training data there is no pad token that's padding_mask is set to None
        for layer in self.layer[1:]:
            layer_output = layer(layer_output, padding_mask=None, lookahead_mask=lookahead_mask)
        output = self.linear(self.final_layernorm(layer_output))

        return output 
