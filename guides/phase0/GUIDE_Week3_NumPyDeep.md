# Guide: Week 3 — NumPy Deep Dive
> **Phase 0 | Foundation** — The math engine under every ML library.

---

## Beginner Start Here

NumPy is to Python what a calculator is to pencil-and-paper arithmetic. Every time `sklearn`, `TensorFlow`, or `PyTorch` does a computation, it's using NumPy arrays internally. To truly understand ML — especially to debug model behavior or implement algorithms from scratch — you must know NumPy.

### What This Guide Covers
- Why NumPy arrays replace Python lists for ML
- Shape, dimensions, and axes
- Broadcasting — NumPy's "magic" rule
- Vectorized operations vs loops
- Dot product and matrix multiplication
- Key math operations for ML
- The Normal Equation (first matrix formula you'll derive yourself)

### Key Terms
| Term | Plain English |
|------|---------------|
| **ndarray** | NumPy's array type. N-dimensional array of numbers |
| **shape** | The size in each dimension, as a tuple. e.g., (3, 4) means 3 rows, 4 cols |
| **dtype** | The numeric type: float64, int32, bool, etc. |
| **axis** | Which direction to operate. axis=0 = down rows, axis=1 = across columns |
| **broadcasting** | How NumPy handles operations between arrays of different shapes |
| **vectorized** | Operation applied to every element at once — no Python loops |
| **dot product** | Multiply paired elements, then sum. Foundation of neural networks |
| **transpose** | Flip a matrix: rows become columns |
| **scalar** | A single number (vs an array) |

---

## How to Study This Guide

1. Open a fresh Python / Jupyter cell.
2. For each section: read the explanation, then type (don't paste) the code examples.
3. After each code block, **modify the shape or values** and observe how the output changes.
4. Complete the companion notebook: `STARTER_Week3_NumPyDeep.ipynb`
5. Attempt the Normal Equation exercise at the end before reading the solution.

---

## Section 1: Why NumPy Instead of Python Lists?

### Speed

Python loops are slow because Python is an interpreted language. NumPy operations are implemented in C and operate on contiguous memory blocks.

```python
import numpy as np
import time

n = 1_000_000
lst = list(range(n))
arr = np.arange(n, dtype=float)

# Python list double
t0 = time.time()
lst2 = [x * 2 for x in lst]
print(f"Python loop: {time.time() - t0:.4f}s")

# NumPy double
t0 = time.time()
arr2 = arr * 2
print(f"NumPy:       {time.time() - t0:.4f}s")
# NumPy is typically 50–100x faster
```

### Homogeneous Type

Python lists can hold any type: `[1, "hello", 3.14, True]`. NumPy arrays hold one type only. This makes memory layout efficient and operations faster.

```python
a = np.array([1, 2, 3])       # int64
b = np.array([1.0, 2.0, 3.0]) # float64
```

### When to Use Each

| Situation | Use |
|-----------|-----|
| Small list of mixed types | Python list |
| Batch of numbers to compute on | NumPy array |
| Row/column selections on a table | pandas (uses NumPy underneath) |
| Training a model, doing matrix math | NumPy array |
| Tabular data with column names | pandas DataFrame |

---

## Section 2: Creating Arrays

```python
import numpy as np

# From a list
arr = np.array([10, 20, 30])

# Special arrays
np.zeros((3, 4))           # 3x4 matrix of zeros
np.ones((2, 5))            # 2x5 matrix of ones
np.eye(4)                  # 4x4 identity matrix
np.full((3, 3), 7)         # 3x3 matrix filled with 7

# Sequential
np.arange(0, 10, 2)        # [0, 2, 4, 6, 8] — like range()
np.linspace(0, 1, 5)       # [0.0, 0.25, 0.5, 0.75, 1.0] — 5 evenly spaced

# Random (set seed for reproducibility!)
np.random.seed(42)
np.random.rand(3, 4)       # uniform [0, 1)
np.random.randn(3, 4)      # standard normal (~mean 0, std 1)
np.random.randint(0, 10, size=(3, 4))  # integers in [0, 10)
```

---

## Section 3: Shape, Dimensions, and Axes

### Shape and Dimension Vocabulary

```
1D array: [10, 20, 30]             shape = (3,)      ndim = 1
2D array: [[1, 2], [3, 4]]         shape = (2, 2)    ndim = 2
3D array: stack of 2D arrays       shape = (B, H, W) ndim = 3
```

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.shape   # (2, 3) — 2 rows, 3 cols
arr.ndim    # 2
arr.size    # 6 (total elements = 2 × 3)
arr.dtype   # dtype('int64')
```

### Understanding Axes — The Key to Getting NumPy Right

The **axis** parameter controls which direction a function collapses:

```
A 2D array (shape 3×4):

          col 0  col 1  col 2  col 3    (axis=1 →)
row 0  [  1,     2,     3,     4  ]
row 1  [  5,     6,     7,     8  ]
row 2  [  9,    10,    11,    12  ]
         ↓      ↓      ↓      ↓
       (axis=0 ↓)

axis=0  → collapses rows → result has 1 row  → e.g., sum of each column
axis=1  → collapses cols → result has 1 col  → e.g., sum of each row
```

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

A.sum(axis=0)   # [15, 18, 21, 24] — sum down each column (length 4)
A.sum(axis=1)   # [10, 26, 42]      — sum across each row  (length 3)
A.sum()         # 78                 — grand total

A.mean(axis=0)  # mean of each column
A.max(axis=1)   # max of each row
```

**Memory trick:** axis=0 gives you **c**olumn statistics (think: **c** for **c**ollapse rows). axis=1 gives you row statistics.

### Reshape — Re-slicing Without Copying Data

```python
arr = np.arange(12)         # [ 0,  1,  2, ..., 11]
arr.reshape(3, 4)           # 3 rows × 4 cols
arr.reshape(4, 3)           # 4 rows × 3 cols
arr.reshape(2, 2, 3)        # 3D: 2 "pages" of 2×3

# The -1 trick: let NumPy infer one dimension
arr.reshape(-1, 4)          # NumPy figures out rows: 12/4 = 3
arr.reshape(3, -1)          # NumPy figures out cols: 12/3 = 4
arr.reshape(-1)             # flatten to 1D
```

The `-1` trick is everywhere in real ML code. When you reshape a batch of images, you don't always know the batch size in advance, so you write `.reshape(-1, 784)` to flatten each image.

---

## Section 4: Vectorized Operations

### Element-wise Math

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b         # [5, 7, 9]
a * b         # [4, 10, 18]
a ** 2        # [1, 4, 9]
np.sqrt(a)    # [1.0, 1.41, 1.73]

# Scalar operations
a * 10        # [10, 20, 30]
a + 100       # [101, 102, 103]
```

### Boolean Operations

```python
a = np.array([12, 7, 19, 4, 25])
a > 10                     # [True, False, True, False, True]
a[a > 10]                  # [12, 19, 25]  — fancy indexing
(a > 5) & (a < 20)         # [True, True, True, False, False]
np.where(a > 10, 1, 0)     # [1, 0, 1, 0, 1]  — conditional like Excel IF
```

### Useful Functions

```python
np.abs(arr)            # absolute value
np.clip(arr, 0, 1)     # cap values: anything < 0 → 0, > 1 → 1 (used in neural nets)
np.argmax(arr)         # index of the maximum value (use in classification!)
np.argmin(arr)         # index of the minimum value
np.argsort(arr)        # sorted indices
np.unique(arr)         # unique values
np.concatenate([a, b]) # join arrays
np.vstack([a, b])      # stack vertically (new rows)
np.hstack([a, b])      # stack horizontally (new cols)
```

`np.argmax` is used in neural network classification: `predicted_class = np.argmax(output_probabilities)`.

---

## Section 5: Broadcasting — NumPy's Superpower

Broadcasting is how NumPy handles operations between arrays of **different shapes**. It's the reason NumPy code can look like it's doing "impossible" math.

### The Three Rules

When operating on two arrays:
1. If arrays don't have the same number of dimensions, **prepend 1s** to the smaller shape.
2. Arrays with size 1 along a dimension are **stretched** to match the other.
3. If shapes don't match in any dimension and neither is 1 → **error**.

### Example: Normalizing a Dataset

```python
# Dataset: 100 samples × 4 features
X = np.random.randn(100, 4)   # shape (100, 4)

# Compute statistics per column (across rows)
means = X.mean(axis=0)        # shape (4,)
stds  = X.std(axis=0)         # shape (4,)

# Normalize (subtract mean, divide by std)
X_norm = (X - means) / stds   # shape (100, 4) — broadcasting!
```

**What happened?** `means` has shape `(4,)`. NumPy silently prepends a 1 → `(1, 4)`, then stretches it to `(100, 4)`. The operation applies to every row correctly.

This is **standardization** (also called "mean normalization" or "z-score normalization"). You'll use it in nearly every ML project.

### Without Broadcasting (How You'd Do It Without NumPy)

```python
# The hard way (slow, verbose)
for i in range(100):
    for j in range(4):
        X[i, j] = (X[i, j] - means[j]) / stds[j]
```

Broadcasting eliminates this entirely.

---

## Section 6: Dot Product and Matrix Multiplication

### Dot Product — Two Vectors

The dot product of two vectors multiplies paired elements and sums:

```
a = [2, 3, 5]
b = [1, 4, 2]
a · b = (2×1) + (3×4) + (5×2) = 2 + 12 + 10 = 24
```

```python
a = np.array([2, 3, 5])
b = np.array([1, 4, 2])
np.dot(a, b)    # 24
a @ b           # same thing — @ is the matrix multiply operator (use this)
```

**ML interpretation:** In linear regression, the prediction is `y_hat = X @ w` — multiply each feature by its weight and sum them up. The dot product is this operation for one sample. `X @ w` does it for all samples at once.

### Matrix Multiplication — Batch Predictions

```python
# 5 samples, 3 features each
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [2, 0, 1],
              [3, 3, 3]])    # shape (5, 3)

# Weights for 3 features
w = np.array([0.5, 1.0, -0.5])  # shape (3,)

# Predict for all 5 samples at once
predictions = X @ w              # shape (5,)
```

### Shape Rule for Matrix Multiply

```
(m × n) @ (n × p) = (m × p)
     ↑ these must match ↑
```

If the inner dimensions don't match, you'll get a `ValueError`. When debugging matrix math, **print all shapes before the multiply**.

### Transpose

Flip rows and columns. Often needed to make shapes align for matrix multiply.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
A.T                                    # shape (3, 2)

# Common pattern: A.T @ A gives a square matrix (used in Normal Equation)
```

---

## Section 7: The Normal Equation — ML in Closed Form

The **Normal Equation** is how you solve linear regression analytically (without gradient descent). It's the formula:

$$\hat{\theta} = (X^T X)^{-1} X^T y$$

This gives you the exact weights $\hat{\theta}$ that minimize the Mean Squared Error.

### Why Learn This If sklearn Does It Automatically?

1. You'll see this formula in research papers. You need to recognize it.
2. Implementing it proves you understand matrix math.
3. sklearn's `LinearRegression` actually uses a very similar algorithm internally.
4. This exact pattern appears in more advanced algorithms (Ridge regression, PCA, etc.).

### Implementing It

```python
import numpy as np

# Generate data: y = 2x + 1 + noise
np.random.seed(42)
n = 50
x = np.random.uniform(0, 10, n)      # feature: size (0 to 10)
y = 2 * x + 1 + np.random.randn(n)   # target: price

# Add bias column (column of 1s so we can learn intercept)
X_b = np.column_stack([np.ones(n), x])   # shape (50, 2)

# Normal Equation
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Intercept: {theta[0]:.2f}")   # should be ≈ 1.0
print(f"Slope:     {theta[1]:.2f}")   # should be ≈ 2.0
```

### Why It Doesn't Scale

The Normal Equation requires computing $(X^T X)^{-1}$. For $n$ features, this is an $n \times n$ matrix inversion — cubic time complexity $O(n^3)$. For a dataset with 1,000 features, this is fine. For a dataset with 1,000,000 features (e.g., pixel data), gradient descent is used instead.

---

## Section 8: Indexing and Slicing

```python
arr = np.array([10, 20, 30, 40, 50])
arr[0]       # 10  — first element
arr[-1]      # 50  — last element
arr[1:4]     # [20, 30, 40]
arr[::-1]    # [50, 40, 30, 20, 10]  — reverse

# 2D
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

mat[0, 0]    # 1  — top-left
mat[1, :]    # [4, 5, 6]  — second row
mat[:, 2]    # [3, 6, 9]  — third column
mat[0:2, 1:] # [[2, 3], [5, 6]]  — sub-matrix

# Boolean indexing
arr = np.array([1, -2, 3, -4, 5])
arr[arr > 0]           # [1, 3, 5]

# Fancy indexing (select by list of indices)
arr[[0, 2, 4]]         # [1, 3, 5]
```

---

## NumPy Cheat Sheet for ML

| Operation | Code | Result Shape |
|-----------|------|-------------|
| Create dataset | `X = np.random.randn(1000, 20)` | (1000, 20) |
| Standardize | `(X - X.mean(axis=0)) / X.std(axis=0)` | (1000, 20) |
| Add bias | `np.column_stack([np.ones(n), X])` | (n, 21) |
| Predictions | `X @ w` | (1000,) |
| MSE | `np.mean((y_pred - y) ** 2)` | scalar |
| Max probability class | `np.argmax(probs, axis=1)` | (1000,) |
| Flatten images | `X.reshape(n, -1)` | (n, H×W) |
| Correlation matrix | `np.corrcoef(X.T)` | (20, 20) |

---

## Reflection Questions

1. What does `axis=0` mean when calling `np.sum()`? What about `axis=1`?
2. Why can you subtract a shape `(4,)` array from a shape `(100, 4)` array?
3. What two shapes can you matrix-multiply? What's the result shape?
4. When should you use `.dot()` vs. `@`?
5. Why does the Normal Equation become impractical for millions of features?
6. Write from memory: the code to standardize a dataset stored in `X`.

---

## Checklist for This Week

- [ ] I can create arrays with zeros, ones, arange, linspace, and random
- [ ] I can check `.shape`, `.ndim`, `.size`, `.dtype` and know what they mean
- [ ] I can reshape with both numeric dimensions and the `-1` trick
- [ ] I understand axis=0 vs axis=1 and can predict which direction a function collapses
- [ ] I wrote the broadcasting normalization example by memory
- [ ] I understand the dot product and can write `X @ w` for batch predictions
- [ ] I implemented and ran the Normal Equation on generated data
- [ ] I can index a 2D array with both slices and boolean masks

---

*Guide for `STARTER_Week3_NumPyDeep.ipynb` | Phase 0 | ML-AI-learning roadmap*
