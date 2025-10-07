from transformers import AutoTokenizer

import torch 
import torch.nn as nn
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token' : '<pad>',
                              'bos_token' : '<bos>',
                              'eos_token' : '<eos>',
                              'sep_token' : '<sep>',
                             })


class Tokenization(nn.Module):
    def __init__(self, max_seq_len):
        super(Tokenization, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def get_vocab_len(self):
        return len(self.tokenizer)
    
    def get_tokenizer(self):
        return self.tokenizer

    def process_sentence(self, input_sentences):
        token_ids = self.tokenizer.encode(input_sentences, return_tensors=None)
        if(len(token_ids) > self.max_seq_len):
            token_ids = token_ids[:self.max_seq_len]
        
        token_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(token_ids)) 
        return token_ids
    
    def tokenize(self, input_text):
        tokens = []
        if (type(input_text) == str or type(input_text) == np.str_):
            tokens.append(self.process_sentence(input_text))
        else:
            for sentence in input_text:
                if(self.process_sentence(sentence)):
                    tokens.append(self.process_sentence(sentence))
        tokens = torch.tensor(tokens)
        return tokens
    
    def decode(self, token_ids):
        output_tokens = token_ids.tolist()
        tokens = [token for token in output_tokens if token != self.tokenizer.pad_token_id]
        text = []
        if((type(tokens) == list) and (type(tokens[0]) != list)):
            text.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        else:
            for batch in tokens:
                token = [self.tokenizer.decode(batch, skip_special_tokens=True)]
                text.append(token) 
        return text