import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Hyperparameters ----- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
batch_size = 64
block_size = 256
train_val_split = 0.9
embed_size = 192
learning_rate = 3e-4
head_count = 4
layer_count = 4
dropout = 0.2
# --------------------------- #

torch.manual_seed(1337)

with open('inputs.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

strings_to_int = {char: i for i, char in enumerate(chars)}
int_to_strings = {i: char for i, char in enumerate(chars)}

encode = lambda string: [strings_to_int[char] for char in string]
decode = lambda tokens: ''.join([int_to_strings[token] for token in tokens])
data = torch.tensor(encode(text), dtype=torch.long)

split_index = int(train_val_split*len(data))
train_data = data[:split_index]
val_data = data[split_index:]

def generate_batch(split):
    data = train_data if split == 'train' else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    
    inputs = torch.stack([data[i:i+block_size] for i in indices])
    targets = torch.stack([data[i+1:i+block_size+1] for i in indices])
    
    inputs, targets = inputs.to(device), targets.to(device)
    
    return inputs, targets

class Head(nn.Module):
    """ Single Head for Self-Attention """
    
    def __init__(self, head_size):
        super().__init__()
        
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, inputs):
        batch_size, block_size, embed_size = inputs.shape # (B,T,C)
        
        queries = self.query(inputs) # what you want (B,T,C)
        keys = self.key(inputs) # what you have (B,T,C)
        values = self.value(inputs) # what fits best (B,T,C)
        
        weights = queries @ keys.transpose(-2, -1) * embed_size**-0.5 # sum of dot products of queries and keys for each token (B,T,T)
        weights = weights.masked_fill(self.tril[:block_size, :block_size] == 0, float('-inf')) # mask out future tokens (B,T,T)
        weights = F.softmax(weights, dim=-1) # normalize weights (B,T,T)
        
        weights = self.dropout(weights) # dropout
        
        output = weights @ values # weighted sum of values (B,T,C)
        
        return output # (B,T,C)

class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, inputs):
        output = torch.cat([head(inputs) for head in self.heads], dim=-1)
        output = self.projection(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, inputs):
        return self.net(inputs)
    
class Block(nn.Module):
    def __init__(self, embed_size, head_count):
        super().__init__()
        self.self_attention = MultiHeadAttention(head_count, embed_size//head_count)
        self.feed_forward = FeedForward(embed_size)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
        
    def forward(self, inputs):
        inputs = inputs + self.self_attention(self.layernorm1(inputs))
        inputs = inputs + self.feed_forward(self.layernorm2(inputs))
        return inputs

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(block_size, embed_size)
        
        self.blocks = nn.Sequential(*[Block(embed_size, head_count) for _ in range(layer_count)])
        self.layer_norm = nn.LayerNorm(embed_size)
        
        self.linear_output = nn.Linear(embed_size, vocab_size)
        
    def forward(self, inputs, targets=None):
        batch_size, block_size = inputs.shape
        
        embedded_tokens = self.token_embeddings(inputs)
        embedded_position = self.position_embeddings(torch.arange(block_size).to(device))
        
        x = embedded_tokens + embedded_position
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear_output(x)
        
        if targets is not None:
            batch, block_size, embed_size = logits.shape
            logits = logits.view(batch*block_size, embed_size)
            targets = targets.view(batch*block_size)
            
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        
        return logits, loss
    
    def generate(self, inputs, max_tokens):
        for _ in range(max_tokens):
            inputs_cropped = inputs[:, -block_size:]
            logits, loss = self(inputs_cropped)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            inputs = torch.cat([inputs, next_index], dim=1)
        return inputs



model = BigramModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for steps in range(5000):
    inputs, targets = generate_batch('train')
    
    logits, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 100 == 0:
        print(f'Loss at step {steps}: {loss.item()}')

input_val = torch.zeros(1, 1, dtype=torch.long).to(device)

print(decode(model.generate(input_val, 1000)[0].tolist()))