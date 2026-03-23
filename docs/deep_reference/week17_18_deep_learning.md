# Weeks 17-18: Deep Learning Foundations (Deep Dive)

> **2026 Note:** LSTMs and GRUs are largely **legacy** for text tasks. Focus the majority of your neural architecture time on Transformers and the Attention mechanism. That is what runs every modern LLM you will interact with.

---

## 1. What Is a Neural Network? (From Scratch)
A neural network is a series of mathematical operations stacked in layers. Each layer takes numbers in, transforms them, and passes results to the next layer.

- **Neuron:** Takes a weighted sum of inputs and applies an activation function: `output = activation(w1*x1 + w2*x2 + ... + b)`
- **Layer:** Collection of neurons operating in parallel.
- **Activation function:** Adds non-linearity so the network can learn complex patterns:
  - `ReLU(x) = max(0, x)` — most common hidden layer activation.
  - `Sigmoid(x) = 1 / (1 + e^-x)` — binary output (0 to 1).
  - `Softmax` — multi-class output (probabilities summing to 1).
- **Loss function:** Measures how wrong the model is. Goal: minimize it.
  - `CrossEntropyLoss` for classification.
  - `MSELoss` for regression.
- **Backpropagation:** Computing gradients (how much to adjust each weight) by applying the chain rule backwards through the network.
- **Optimizer:** Uses gradients to update weights:
  - SGD: `weight = weight - lr * gradient`
  - Adam: Adaptive learning rate per parameter (default choice).

## 2. CNN — Convolutional Neural Networks
CNNs are built for spatial data like images. The key idea: instead of connecting every pixel to every neuron (expensive), apply small filters that slide across the image.

- **Convolution:** Apply a small kernel (e.g., 3x3) across the image. Each kernel detects a feature (edge, color, shape).
- **Pooling:** Downsamples feature maps (max pooling takes the largest value in a region).
- **Flatten → Fully connected:** After convolution layers, flatten into a vector and feed to standard dense layers.
- **Residual blocks:** Add skip connections (`output = F(x) + x`) to allow gradients to flow without vanishing. Used in ResNet.

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)          # halves spatial size
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc    = nn.Linear(64 * 8 * 8, 10) # 10 classes

    def forward(self, x):                      # x: [batch, 3, 32, 32]
        x = self.pool(torch.relu(self.conv1(x)))  # [batch, 32, 16, 16]
        x = self.pool(torch.relu(self.conv2(x)))  # [batch, 64, 8, 8]
        x = x.view(x.size(0), -1)                 # flatten
        return self.fc(x)
```

## 3. LSTM/GRU — Brief Context (Legacy for Text)
LSTMs and GRUs were the standard for sequences (text, time series) before Transformers.

**The core problem they solve:** Standard RNNs forget early context because gradients vanish over long sequences.

**LSTM fix:** Four gates control what to remember, forget, input, and output.

**Why they are now legacy for text:** They process tokens sequentially (slow, can't parallelize). Transformers replaced them for language by using Attention, which processes all tokens simultaneously.

**Still useful for:** Multivariate time series forecasting where temporal local patterns matter and sequence length is short (<200 steps).

```python
# Brief LSTM example — time series forecasting
lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(x)  # x: [batch, seq_len, features]
```

## 4. Attention Mechanism — The Core of Modern AI ⭐
Attention is the key innovation behind every modern LLM (GPT, Claude, Gemini). Understanding it is the difference between an engineer and an architect.

### The core idea
When processing a word, don't treat all other words equally. Learn to "attend" (pay attention) to the most relevant ones.

Given word "bank" in the sentence "I went to the river bank to fish" — the model should attend more to "river" and "fish" than to "the".

### Scaled Dot-Product Attention
The building block. Given three matrices:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What does each token contain?"
- **V (Value):** "What should I return?"

Formula: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`

Step by step:
1. `QK^T` — dot product of Query with all Keys → raw attention scores (how relevant each token is)
2. `/ sqrt(d_k)` — scale down so softmax doesn't saturate (d_k = key dimension)
3. `softmax(...)` — convert scores to probabilities (attention weights summing to 1)
4. `* V` — weighted sum of Values — the output

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, heads, seq_len, d_k]
    Returns weighted value vectors.
    """
    d_k = Q.size(-1)
    # Step 1+2: scaled similarity scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Step 3: optional mask (for causal/padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Step 4: softmax weights * values
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

### Multi-Head Attention
Run attention multiple times in parallel with different learned projections. Each "head" can attend to different types of relationships.

```python
# Conceptually:
# head_i = Attention(Q*W_q_i, K*W_k_i, V*W_v_i)
# MultiHead = Concat(head_1, ..., head_h) * W_o

multihead = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
output, attn_weights = multihead(query=x, key=x, value=x)  # self-attention
```

### Quadratic Complexity — Critical for Cost Understanding
Attention computes similarity between **every pair** of tokens: `O(n^2 * d)` where n = sequence length.

- 1K tokens → 1M operations
- 10K tokens → 100M operations (100x more expensive)
- 100K tokens → 10B operations (10,000x more expensive)

This is why long-context models are expensive to run. Understanding this is essential before trying to optimize token costs in Phase 2.

## 5. KV Cache — How Transformers Manage Memory ⭐
KV Cache is the most important production optimization concept in 2026 for LLM deployments.

### The problem
When generating text token-by-token, the transformer recomputes K and V matrices for **all previous tokens** at every step. Extremely wasteful.

### The solution
Cache (save) the computed K and V for each layer as tokens are generated. At each new step, only compute K/V for the **new token** and append to the cache.

```
Without KV Cache:
  Step 1: compute K,V for [token1]            → 1 computation
  Step 2: recompute K,V for [token1, token2]  → 2 computations
  Step 3: recompute for [token1,token2,token3]→ 3 computations
  Total for 100 tokens: 1+2+...+100 = 5,050 computations

With KV Cache:
  Step 1: compute K,V for [token1]   → save to cache
  Step 2: compute K,V for [token2]   → append to cache
  Step 3: compute K,V for [token3]   → append to cache
  Total for 100 tokens: 100 computations  (50x fewer!)
```

### Memory cost of KV Cache
`KV Cache size = 2 * layers * heads * d_head * seq_len * bytes_per_value`

For a 7B model with 32 layers, 32 heads, d_head=128, FP16:
- 1K context: ~2 * 32 * 32 * 128 * 1024 * 2 bytes ≈ **536 MB**
- 32K context: ~17 GB — this is why long context is GPU-memory-constrained

### Why this matters for you
- When designing RAG systems: keep retrieved context chunks small and targeted
- When choosing models: context window × batch size × KV cache = your VRAM budget
- Optimizations: Grouped Query Attention (GQA), Multi-Query Attention (MQA) reduce KV heads

## 6. Practical PyTorch Training Loop
```python
from torch.utils.data import DataLoader
import torch.optim as optim

# Full training loop pattern
model = SimpleCNN().to('mps')  # mps = Apple Silicon GPU
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()               # enable dropout, batchnorm training mode
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to('mps'), labels.to('mps')
        optimizer.zero_grad()   # clear gradients from last step
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()         # backprop: compute gradients
        optimizer.step()        # update weights
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()                # disable dropout
    correct = 0
    with torch.no_grad():       # no gradients needed for inference
        for images, labels in loader:
            images, labels = images.to('mps'), labels.to('mps')
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

# Training with checkpointing and early stopping
best_val_acc = 0
patience, patience_counter = 5, 0
for epoch in range(50):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_acc = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch}: loss={train_loss:.4f}, val_acc={val_acc:.4f}')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')  # checkpoint
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print('Early stopping triggered')
        break
```

## 7. Transfer Learning
Instead of training from scratch (slow, needs lots of data), use a model already trained on millions of examples and adapt to your task.

- **Feature extraction:** Freeze all pre-trained weights. Only train your new classification head.
- **Fine-tuning:** Unfreeze some or all layers and train them on your data with a very small learning rate.

```python
import torchvision.models as models

# Load ResNet50 pre-trained on ImageNet
model = models.resnet50(weights='IMAGENET1K_V2')

# Option 1: Feature extraction - freeze all, replace final layer
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)  # only this layer trains

# Option 2: Fine-tune - unfreeze last 2 blocks
for param in model.layer4.parameters():
    param.requires_grad = True
# Use small lr for pre-trained layers, larger for new head
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])
```

## 8. References
- "Attention Is All You Need" — Vaswani et al. (2017). The original Transformer paper.
- "The Illustrated Transformer" — Jay Alammar (jalammar.github.io). Best visual explanation.
- "The KV Cache" — Explain it Like I'm 5. (Hugging Face blog).
- Fast.ai course — practical deep learning top-down.
- PyTorch official tutorials — pytorch.org/tutorials.

## 9. Challenge
1. Implement `scaled_dot_product_attention` from scratch using only NumPy. Verify that `softmax(QK^T / sqrt(d_k)) * V` produces the correct output shape.
2. Build a minimal 1-layer Transformer block in PyTorch (MultiheadAttention + LayerNorm + FeedForward).
3. Train a ResNet50 (transfer learning) on CIFAR-10. Compare accuracy and training time vs your CNN from scratch.
4. Measure how far KV cache grows (in MB) as you increase sequence length from 128 to 4096 tokens for a 2-layer, 4-head model.
