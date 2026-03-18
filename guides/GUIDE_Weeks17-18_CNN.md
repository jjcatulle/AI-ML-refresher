# Guide: CNN Image Classification (Weeks 17-18)

## Big Picture
Build a Convolutional Neural Network to classify images into categories.

**Why?** Deep learning = hierarchical feature learning. CNNs are foundational for vision tasks.

**Key Skills:**
- Convolutional layers (filters, kernels)
- Pooling and dimensionality reduction
- Data augmentation
- Transfer learning (pretrained models)
- Training optimization (learning rate, batch size)

---

## Concept 1: From Pixels to Classes

**What:** Images as 3D arrays transformed to class predictions.

```python
# Image: 28x28 pixels, 3 channels (RGB)
image = np.random.rand(28, 28, 3)  # Shape (28, 28, 3)

# Pass through CNN →
# Extract features →
# Classify →

# Output: [0.02, 0.96, 0.01, 0.01]  (class probabilities)
# → "Class 1" (96% confidence)
```

**Why CNN vs Dense Layers?**
- Dense: Each pixel → each neuron = 28*28*3 = 2,352 connections per neuron
- CNN: Local filters = reuse weights = fewer parameters = better for images

---

## Concept 2: Convolution Operation

**What:** Sliding filter that detects patterns.

```python
# Input image
[1, 2, 3, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
[13, 14, 15, 16]

# Filter (3x3 kernel)
[1, 0, -1]
[1, 0, -1]
[1, 0, -1]

# Convolution at position (0,0):
(1*1 + 2*0 + 3*-1) + (5*1 + 6*0 + 7*-1) + (9*1 + 10*0 + 11*-1)
= 1 + 0 - 3 + 5 + 0 - 7 + 9 + 0 - 11
= -6

# Slide filter across entire image
# Result: smaller feature map
```

**PyTorch Example:**
```python
import torch.nn as nn

conv_layer = nn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=16,    # 16 different filters
    kernel_size=3,      # 3x3 filters
    stride=1,           # Move 1 pixel at a time
    padding=1           # Keep spatial dimensions
)

output = conv_layer(image)  # (1, 16, 28, 28)
```

---

## Concept 3: Pooling

**What:** Reduce spatial dimensions by summarizing regions.

```python
# Max pooling: take maximum in each region
Input:            Max Pooling (2x2):
[1, 2, 3, 4]     [2, 4]
[5, 6, 7, 8]     [14, 16]
[9, 10, 11, 12]
[13, 14, 15, 16]

# Benefits: Fewer parameters, detect important features
```

```python
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
# Reduces 28x28 → 14x14
```

---

## Concept 4: CNN Architecture

**What:** Stacking conv + pooling layers.

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Flatten: (64, 7, 7) → 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))      # (32, 14, 14)
        x = self.pool2(F.relu(self.conv2(x)))      # (64, 7, 7)
        x = x.view(x.size(0), -1)                   # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## Concept 5: Data Augmentation

**What:** Artificially expand dataset using transformations.

```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomRotation(15),        # Rotate 0-15°
    transforms.RandomHorizontalFlip(),    # Flip horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary colors
    transforms.RandomCrop(32, padding=4), # Crop and pad
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 1 original image → many variations
# Improves generalization
```

---

## Concept 6: Transfer Learning

**What:** Using pretrained models on large datasets.

```python
from torchvision.models import resnet50

# Model trained on ImageNet (1M images, 1000 classes)
model = resnet50(pretrained=True)

# Freeze early layers (learned good features)
for param in model.parameters():
    param.requires_grad = False

# Replace last layer for your task
model.fc = nn.Linear(2048, 10)  # 10 classes instead of 1000

# Only train new layer (fast!)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**Why?** ImageNet features useful for any image task.

---

## Concept 7: Training Loop

**What:** Iteratively improve model.

```python
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()  # Disable dropout/batch norm changes
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
```

---

## Concept 8: Hyperparameter Tuning

**What:** Finding optimal learning rate, batch size, etc.

```python
# Too high learning rate: loss diverges
# Too low learning rate: training too slow
# Goldilocks: 0.001 is often good start

Learning rates to try: [0.0001, 0.001, 0.01, 0.1]
Batch sizes to try: [32, 64, 128]
Epochs: 10-50 typical for transfer learning

# Plot training/test loss to find best
```

---

## Concept 9: Monitoring Training

**What:** Detecting overfitting.

```python
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss = ... # average loss on train
    test_loss = ...  # average loss on test
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Plot
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')

# If test loss increases while train decreases = overfitting
# Solutions: More data, regularization, simpler model
```

---

## Concept 10: Evaluation Metrics

**What:** Beyond accuracy for multi-class.

```python
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)

# Confusion matrix shows which classes get confused
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm)
```

---

## Challenge Approach

### Challenge 1-3: Dataset & Preprocessing
- Load CIFAR-10 or MNIST
- Understand shape, visualize samples
- Create train/test/val splits

### Challenge 4-6: Build CNN
- Create simple CNN model
- Train on small subset (quick iteration)
- Visualize learned filters

### Challenge 7-9: Transfer Learning
- Load pretrained model (ResNet)
- Fine-tune on your data
- Compare with simple CNN

### Challenge 10-12: Evaluation & Optimization
- Measure accuracy per class
- Create confusion matrix
- Visualize what model learned
- Document architecture and results

---

## Key Takeaways

✅ **Convolution = local pattern detection** (more parameters efficient than dense)

✅ **Pooling = dimensionality reduction** (keeps important info, discards spatial noise)

✅ **Transfer learning = start with pretrained weights** (faster training, better results)

✅ **Data augmentation = virtual dataset expansion** (improves generalization)

✅ **Monitor train/test loss** (overfitting = test loss increases while train decreases)
