import torch
import torch.nn as nn
from torch.nn import functional as F
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

# Access hyperparameters
batch_size = int(config.getint('HYPERPARAMETERS', 'batch_size'))
block_size = int(config.getint('HYPERPARAMETERS', 'block_size'))
max_iters = int(config.getint('HYPERPARAMETERS', 'max_iters'))
eval_interval = int(config.getint('HYPERPARAMETERS', 'eval_interval'))
learning_rate = int(config.getfloat('HYPERPARAMETERS', 'learning_rate'))
device = int(config.get('HYPERPARAMETERS', 'device'))
eval_iters = int(config.getint('HYPERPARAMETERS', 'eval_iters'))
n_embed = int(config.getint('HYPERPARAMETERS', 'n_embed'))
n_head = int(config.getint('HYPERPARAMETERS', 'n_head'))
n_layer = int(config.getint('HYPERPARAMETERS', 'n_layer'))
dropout = int(config.getfloat('HYPERPARAMETERS', 'dropout'))
random_seed = int(config.getint('HYPERPARAMETERS', 'random_seed'))

torch.manual_seed(random_seed)

input_file = config.get('FILE', 'input_file')
with open(input_file, 'r', encoding="utf-8") as f:
    text = f.read()


# unique characters in the text
chars = sorted(set(text))
vocab_size = len(chars)

# create mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Train and Test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

# create dataloaders
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        v = self.value(x) 
        out = wei @ v 
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output =  torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.dropout(self.proj(output))
        return output
    

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DecoderLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x =  token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = DecoderLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter}, Train loss: {losses["train"]}, Test loss: {losses["test"]}')
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
idx = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(idx, max_new_tokens=500)
decoded = decode(generated[0].tolist())
print(decoded)

output_file = config.get('FILE', 'output_file')
with open(output_file, 'w', encoding="utf-8") as f:
    f.write(decoded)