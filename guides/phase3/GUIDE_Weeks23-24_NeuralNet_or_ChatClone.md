# Guide: Neural Net from Scratch or Mini ChatGPT Clone (Weeks 23-24)

## Beginner Start Here
This optional project is designed for deep understanding.
You choose one of two tracks:
- Track A: build a neural net from scratch.
- Track B: build a tiny ChatGPT-like decoder model.

### What this project teaches
- How gradients and updates actually work.
- How tokenization, masking, and generation work.
- How training loops fail, and how to debug them.

### Key terms
- `Forward pass`: computing predictions from inputs.
- `Backpropagation`: computing gradients of loss wrt parameters.
- `Optimizer`: update rule (SGD/Adam).
- `Causal mask`: prevents model from seeing future tokens.
- `Perplexity`: language model quality proxy (lower is better).

### How to use this guide
1. Pick one track first.
2. Build the simplest end-to-end baseline.
3. Log losses and sample outputs every epoch.
4. Improve one variable at a time.

---

## Track A: Neural Net from Scratch (NumPy MLP)

### Goal
Implement an MLP classifier manually (no autograd).

### Suggested dataset
- MNIST (flattened) or sklearn digits dataset.

### What to implement
1. Parameter initialization
2. Linear layers
3. Activation functions (ReLU)
4. Softmax + cross-entropy
5. Manual backprop gradients
6. SGD/Adam-style updates
7. Validation loop
8. Confusion matrix and error analysis

### Minimal architecture
- Input -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(num_classes)

### Success criteria
- Stable training loss decline
- Non-trivial validation accuracy
- Correct gradient shapes and no exploding values

---

## Track B: Mini ChatGPT Clone (Tiny Decoder-Only Transformer)

### Goal
Train a small causal language model and generate text.

### Suggested corpus
- A small clean text corpus: docs, stories, code comments, or synthetic data.

### What to implement
1. Tokenization (character-level or BPE-lite)
2. Dataset windows (context length N)
3. Decoder block with causal self-attention
4. Positional encoding/embeddings
5. Next-token loss
6. Training and validation loops
7. Top-k or temperature sampling generation
8. Tiny chat-style prompt wrapper

### Minimal architecture
- Token + positional embeddings
- 2-4 decoder blocks
- Final linear projection to vocab

### Success criteria
- Training/validation loss trend improves
- Generated text is syntactically plausible
- Model obeys short prompt context reasonably

---

## Shared Engineering Checklist

- [ ] Reproducible seed and config block
- [ ] Clean train/val split
- [ ] Logged metrics each epoch
- [ ] Gradient/loss stability checks
- [ ] Saved best checkpoint
- [ ] Inference demo function
- [ ] Error analysis section
- [ ] Final technical summary

---

## Debugging Playbook

### If loss is NaN
- lower learning rate,
- check normalization,
- clip gradients,
- inspect logit range.

### If model does not learn
- verify label alignment,
- overfit tiny subset first,
- check activation and loss wiring,
- print gradient norms.

### If ChatClone output is gibberish
- train longer,
- reduce vocab complexity,
- reduce model size then scale up,
- verify causal mask and token shift.

---

## Evaluation Targets

### Track A
- Accuracy
- Macro-F1
- Per-class error slices

### Track B
- Validation loss/perplexity
- Prompt-following quality samples
- Repetition and degeneration checks

---

## Reflection Questions

1. Which part was hardest: architecture, optimization, or debugging?
2. What changed quality the most: data, model size, or learning rate?
3. What bug taught you the most about how neural networks work?
4. If you had one more week, what would you improve first?

---

## Deliverables

- Training notebook or script
- Saved model checkpoint
- Inference demo
- Metrics and plots
- One-page architecture + lessons summary

---

*Guide for `phases/phase3/starters/STARTER_Weeks23-24_NeuralNet_or_ChatClone.ipynb` | Phase 3 Optional Project*
