import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LambdaLR
scaler = torch.cuda.amp.GradScaler()

from transformer import Transformer
from tokenizer_finetune import Tokenization

tokenizer_obj = Tokenization(max_seq_len = 127)
tokenizer = tokenizer_obj.get_tokenizer()

def mask_answer_tokens(target_tokens):
    for i in range(target_tokens.shape[0]):
        for j in range(target_tokens.shape[1]):
            if(target_tokens[i][j] == tokenizer.sep_token_id):
                target_tokens[i][j] = tokenizer.pad_token_id
                break
            target_tokens[i][j] = tokenizer.pad_token_id
    return target_tokens

def accuracy_fn(predictions, target):
    _, prediction = torch.max(predictions, dim=-1)
    m_target = mask_answer_tokens(target.detach().clone())
    mask = m_target != tokenizer.pad_token_id
    correct = (prediction == m_target) & mask 
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy

def calculate_loss(prediction, target_tokens):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=tokenizer.pad_token_id)
    prediction = prediction.reshape(-1, prediction.size(-1)) # (batch_size * seq_len, vocab_size)
    m_target_tokens = mask_answer_tokens(target_tokens.detach().clone())
    m_target_tokens = m_target_tokens.reshape(-1)
    return criterion(prediction, m_target_tokens)

def eval(input_tokens):
    model.eval()
    with torch.no_grad():
        prediction = model(input_tokens[:, :-1])
        loss = calculate_loss(prediction, input_tokens[:, 1:])
        acc = accuracy_fn(prediction, input_tokens[:, 1:])
    return prediction, loss, acc

def get_scheduler(optimizer, warmup_steps, total_steps, base_lr=5e-4, min_lr=5e-6):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))  # Linear warmup to 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # in [0,1]
        scaled = cosine * (1.0 - min_lr / base_lr) + (min_lr / base_lr)  # in [min_lr/base_lr, 1.0]
        return scaled
    return LambdaLR(optimizer, lr_lambda)

def train(input_tokens):
    model.train()
    with torch.cuda.amp.autocast():
        prediction = model(input_tokens[:, :-1])
        loss = calculate_loss(prediction, input_tokens[:, 1:])
        acc = accuracy_fn(prediction, input_tokens[:, 1:])
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    scaler.step(optimizer)  # Update model weights
    scaler.update()
    scheduler.step()
    return loss, acc

def test(input_tokens):
    model.eval()
    with torch.no_grad():
        prediction = model(input_tokens[:, :-1])
        loss = calculate_loss(prediction, input_tokens[:, 1:])
        acc = accuracy_fn(prediction, input_tokens[:, 1:])
    return prediction, loss, acc

if __name__ == "__main__":
    n_heads = 12
    d_model = 768
    d_ff = 3072
    n_layers = 12
    eps = 1e-9 
    n_epoch = 1
    batch_size = 64

    max_seq_len = 127
    toke = Tokenization(max_seq_len)
    vocab_len = toke.get_vocab_len()

    train_data = torch.load("ELI5.pt") 
   
    model = Transformer(vocab_len, max_seq_len, n_heads, d_model, d_ff, n_layers, eps)

    model_ck = torch.load("/pretrained/transformer.pth")
    new_state_dict = {}
    for key, value in model_ck.items():
        new_key = key.replace("module.", "")  # Remove 'module.' from the keys
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    scheduler = get_scheduler(optimizer, warmup_steps=500, total_steps=3125, base_lr=5e-4, min_lr=1e-5)
   

    train_losses = []
    train_accuracies = []
    for epoch in range(n_epoch):
        train_loss = 0
        train_acc = 0
        for i in range(0, len(train_data), batch_size):
            loss, acc = train(train_data[i:i+batch_size].to(device))
            train_loss += loss.item()
            train_acc += acc
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
        torch.save(model.state_dict(), "transformer.pth")
        torch.save(optimizer.state_dict(), "optimizer.pth")
        torch.save(scheduler.state_dict(), "scheduler.pth")
        torch.save(train_losses, "train_losses.pt")
        torch.save(train_accuracies, "train_accuracies.pt")
        print(f"the train loss is : {train_loss/(len(train_data)/batch_size)}, the train acc is : {train_acc/(len(train_data)/batch_size)}")