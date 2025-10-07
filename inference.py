import torch
from transformer import Transformer
from tokenizer_finetune import Tokenization

max_seq_len = 127
toke = Tokenization(max_seq_len)
vocab_len = toke.get_vocab_len()

n_heads = 12
d_model = 768
d_ff = 3072
n_layers = 12
eps = 1e-9 
n_epoch = 1
batch_size = 64


model = Transformer(vocab_len, max_seq_len, n_heads, d_model, d_ff, n_layers, eps)

model_ck = torch.load(r"\fine_tuned_transformer.pth")

new_state_dict = {}
for key, value in model_ck.items():
    new_key = key.replace("module.", "")  
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model = model.to(device)

model.eval()
print(model.generate_greedy(toke.tokenize("what is the world?")))

print(model.generate_top_p(toke.tokenize("what is the world?")))

