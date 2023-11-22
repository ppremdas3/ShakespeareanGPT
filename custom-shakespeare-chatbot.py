import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')
# Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()

# Now we can use the argument value in our program.
print(f'batch size: {args.batch_size}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
block_size = 128
batch_size = int(args.batch_size)
max_iters = 100
learning_rate = 3e-4
eval_iters = 100
n_embed = 384  # Reduce oif you have limitation in your PC
dropout = 0.2
n_layer = 8
n_head = 8

chars = ""
with open('vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

# Function for encoding each character in the book to an integer and decoding the same integer to the character
str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [str_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_str[i] for i in l])


# Splitting into train and validation
# memory map for using small snippets of text from a single file of any size
def get_random_split(split):
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split):
    data = get_random_split(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ Each head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Register the no look ahead mask so that it doesn't have to be initialized each and every time
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size: (batch, time-step, channels)
        # output size: (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # Computing the attention scored (B,T,hs) @ (B,hs,T) --> (B,T,T) Sqrt is done to scale down the value of the
        # key @ query so that a single head doesn't overpower the total weight
        wi = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # All zeros are converted to -inf so that when it is exponential in softmax it results in zero
        wi = wi.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wi = F.softmax(wi, dim=-1)
        wi = self.dropout(wi)
        v = self.value(x)
        out = wi @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create and run num_heads in parallel
        # ModuleList is used for parallelism and simultaneous computation
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Just projecting onto n_embed to keep the dimensionality constant also adds another learnable parameter
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the head along the channel dimension or last dimension
        out = torch.cat([h(x) for h in self.heads],
                        dim=-1)  # (B,T,F) --> (B,T,[h1 h1 h1 h1 h2 h2 h2 h2 .... hn hn hn hn])
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ Linear feed forward network """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer Block: this is the Decoder block of the GPT model """

    def __init__(self, n_embed, n_head):
        super().__init__()
        # n_embed: Embedding dimension, n_head: Number of heads
        # number of features in each head
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.f_fwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Post norm architecture used here
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.f_fwd(x)
        x = self.ln2(x + y)
        return x


class ShakespeareanGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding and position encoding Make it vocab_size*vocab_size as it will give the probability of each
        # character appearing after a particular character These are learnable (Sin & Cos embedding are
        # non-learnable and are used for base transformers)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])

        # Add final layer normalization for better convergence, this can be changed to different normalizations to
        # see the effect of the normalization on th emodel performance
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Initializing weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        # B is batch, T is time dimension, C is channel/all the characters
        B, T = index.shape
        tkn_embd = self.token_embedding_table(index)  # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tkn_embd + pos_embd  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # cross_entropy expects the dimension to be N, C so we reshape it
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current contex
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # gte the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=-1)  # (B, T+1)
        return index


model = ShakespeareanGPT(vocab_size)
print('Loading model ...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load()
print('Loaded model parameters successfully...')
m = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
    print(f'Completion:\n{generated_chars}')