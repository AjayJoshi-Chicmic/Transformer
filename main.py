import requests
import torch
import torch.nn as nn
from torch.nn import functional as f
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 300
max_iters =3000
learning_rate = 1e-2

torch.manual_seed(21434)


url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

if response.status_code == 200:
    with open("input.txt", "wb") as file:
        file.write(response.content)
    print("Download complete!")
else:
    print("Failed to download file, status code:", response.status_code)


with open('input.txt','r',encoding='utf-8') as F:
    text = F.read()

print(len(text))

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi ={ch:i for i,ch in enumerate(chars)}

itos ={i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])

data = torch.tensor(encode(text),dtype=torch.long)
n= int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"target = {target} context = {context}")


batch_size = 4
block_size = 8
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y
    
xb, yb = get_batch('train')
print(xb)
print(yb)

@torch.no_grad()
def estimate_loss():
    out ={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k]=loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            target = target.view(B*T)
            logits = logits.view(B*T,C)
            loss = f.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx 


model = BigramLanguageModel(vocab_size)
m =model.to(device)
logits , loss = m.forward(xb,yb)

print(logits.shape)
print(loss)
 
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long),max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step : {iter} , trainloss : {losses['train']} , val loss : {losses['val']}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long),max_new_tokens=500)[0].tolist()))
