# Guide: Week 1 — Python Basics for ML
> **Phase 0 | Foundation** — The language of machine learning.

---

## Beginner Start Here

Python is the universal language of ML/AI. Every library you'll use — pandas, NumPy, scikit-learn, PyTorch, HuggingFace — is Python. The goal of Week 1 is to get comfortable with the Python features you'll actually use every day in data science, not to learn all of Python.

### What This Guide Covers
- Variables and data types (and why they matter for ML)
- Lists and dictionaries (the two most important containers)
- Conditions and loops
- Functions — how to write and call them
- Imports — loading libraries
- How to read and debug Python errors

### Key Terms
| Term | Plain English |
|------|---------------|
| **Variable** | A named container for a value. `x = 5` binds the name `x` to the value `5` |
| **Data type** | The kind of value stored: `int`, `float`, `str`, `bool`, `list`, `dict` |
| **List** | An ordered, mutable sequence: `[1, 2, 3]`. Access by index |
| **Dictionary** | An unordered mapping of key → value: `{"a": 1, "b": 2}`. Access by key |
| **Function** | A named, reusable block of code. Takes input (parameters), returns output |
| **Parameter** | A variable name in a function definition. `def f(x):` → x is the parameter |
| **Argument** | The actual value passed when calling a function. `f(5)` → 5 is the argument |
| **Return** | The value a function sends back. `return result` |
| **Import** | Load a library/module into your session: `import numpy as np` |
| **Module** | A `.py` file containing reusable code |
| **Method** | A function that belongs to an object: `list.append()`, `str.lower()` |

---

## How to Study This Guide

1. Read each section to understand the concept first
2. Then open `STARTER_Week1_PythonBasics.ipynb` and run the matching code
3. Complete every `YOUR TURN` exercise — don't skip them
4. After finishing the notebook, come back and answer the reflection questions below

---

## Section 1: Variables and Data Types

### The 4 Basic Types

```python
age = 25            # int   — whole numbers: ..., -2, -1, 0, 1, 2, ...
price = 9.99        # float — decimal numbers: 9.99, -0.5, 3.14159
name = "Alice"      # str   — text, in quotes (single or double)
is_valid = True     # bool  — True or False (capital T/F)
```

### Why Types Matter in ML

ML models require **numbers**. Everything that's not a number must be converted:
- Text category → integer encoding (`"cat" → 0`, `"dog" → 1`)
- Boolean → integer (`True → 1`, `False → 0`)
- Price as string `"$9.99"` → float `9.99`

When you encounter an error like `TypeError: unsupported operand type(s) for +: 'int' and 'str'` — Python is telling you that you mixed types where it expected the same type.

### Type Conversion

```python
float("79.99")    # → 79.99  (string to float)
int(9.99)         # → 9      (float to int — truncates, does NOT round)
str(42)           # → "42"   (int to string)
bool(0)           # → False  (0 = False, everything else = True)
```

---

## Section 2: Lists

### Core Concept
A list is a **sequence** of items in a fixed order. Items can be any type but in ML they're usually all the same type (all floats, all strings, etc.).

```python
items  = []          # empty list
nums   = [1, 2, 3]   # list of ints
prices = [10.5, 9.0] # list of floats
names  = ["Alice"]   # list of strings
```

### The Most Important Operations

| Operation | Code | Result |
|-----------|------|--------|
| Access by index | `lst[0]` | First element |
| Access last element | `lst[-1]` | Last element |
| Slice | `lst[1:3]` | Elements at index 1, 2 (not 3) |
| Length | `len(lst)` | Number of elements |
| Add to end | `lst.append(x)` | Modifies list in-place |
| Remove value | `lst.remove(x)` | Removes first occurrence |
| Sort (in-place) | `lst.sort()` | Modifies original |
| Sorted (new list) | `sorted(lst)` | Returns new sorted list |

### List Comprehensions — Used Constantly in ML

Python's list comprehension is a compact, readable way to build lists. You'll see these in production ML code every day.

```python
# General form:
[expression for item in iterable if condition]

# Examples:
squares = [x**2 for x in range(5)]          # [0, 1, 4, 9, 16]
evens   = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
scaled  = [x / 100 for x in [50, 75, 100]]      # [0.5, 0.75, 1.0]
```

Why not just use a for loop? Comprehensions are shorter and slightly faster. More importantly, they communicate intent — "I'm building a list by transforming another list."

---

## Section 3: Dictionaries

### Core Concept

A dictionary maps **keys** to **values**. Think of a Python `dict` as a hash map or a JSON object.

```python
d = {
    "name": "Alice",      # "name" is the KEY, "Alice" is the VALUE
    "age": 30,
    "active": True
}
```

Keys must be unique and immutable (usually strings or ints). Values can be anything.

### The Most Important Operations

| Operation | Code | Result |
|-----------|------|--------|
| Access value | `d["key"]` | Raises `KeyError` if missing |
| Safe access | `d.get("key", default)` | Returns `default` if missing |
| Set value | `d["key"] = value` | Adds or overwrites |
| Delete key | `del d["key"]` | Removes key |
| All keys | `d.keys()` | `dict_keys` view |
| All values | `d.values()` | `dict_values` view |
| All pairs | `d.items()` | `dict_items` — use in for loops |

### ML Use Cases for Dictionaries

1. **A dataset row** — `{"age": 30, "income": 50000, "churned": 1}`
2. **Hyperparameter config** — `{"n_estimators": 100, "max_depth": 5}`
3. **Metric tracking** — `{"accuracy": 0.87, "f1": 0.84}`
4. **Label encoding** — `{"cat": 0, "dog": 1, "bird": 2}`
5. **API request payload** (JSON ≈ dict)

---

## Section 4: Conditions

### Syntax
```python
if condition:
    # True branch
elif other_condition:
    # Only reached if first condition was False
else:
    # Only reached if all conditions above were False
```

### Comparison Operators
| Operator | Meaning |
|----------|---------|
| `==` | Equal to |
| `!=` | Not equal to |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal |
| `<=` | Less than or equal |
| `in` | Membership: `x in [1, 2, 3]` |
| `not in` | Not membership |

### Logical Operators
- `and` — both must be True
- `or` — at least one must be True
- `not` — flips True to False

### ML Use Case: Decision Rules
Before you have an ML model, you might write rule-based logic:
```python
if tenure < 6 and monthly_charges > 90:
    risk = "high"
elif tenure < 6 or monthly_charges > 90:
    risk = "medium"
else:
    risk = "low"
```

---

## Section 5: Loops

### `for` loop — iterate over items
```python
for item in collection:
    do_something(item)
```

`enumerate()` gives both index and value — use when you need both:
```python
for i, name in enumerate(["a", "b", "c"]):
    print(i, name)  # 0 a / 1 b / 2 c
```

`zip()` lets you iterate over multiple lists in parallel:
```python
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")
```

`range(start, stop, step)` — numeric ranges:
```python
for epoch in range(1, 11):      # 1 to 10
    train_one_epoch()
```

### `while` loop — repeat until condition False
```python
while not converged:
    update_params()
    check_convergence()
```

### `break` and `continue`
- `break` — exit the loop immediately
- `continue` — skip the rest of the current iteration and move to the next

---

## Section 6: Functions

### Why Functions Are Essential in ML

Every step in an ML pipeline is (or should be) a function:
- `load_data()` → reads CSV, returns DataFrame
- `clean_data(df)` → handles missing values, returns cleaned DataFrame
- `extract_features(df)` → returns X array
- `train_model(X_train, y_train)` → returns fitted model
- `evaluate(model, X_test, y_test)` → returns metrics dict

This makes code testable, reusable, and readable.

### Function Anatomy

```python
def function_name(param1, param2, param3=default_value):
    """Docstring: what this function does, what it returns."""
    # body
    result = param1 + param2
    return result
```

- `def` — keyword to define a function
- `param3=default_value` — optional parameter with a default
- `"""docstring"""` — optional description (always add one for complex functions)
- `return` — sends a value back. Functions without `return` return `None`

### Multiple Return Values (very common in ML)
```python
def train_and_evaluate(X, y):
    model = RandomForestClassifier().fit(X[:800], y[:800])
    accuracy = model.score(X[800:], y[800:])
    return model, accuracy   # returns a tuple

model, acc = train_and_evaluate(X, y)  # unpack the tuple
```

---

## Section 7: Imports

### The Standard ML Imports (memorize these)

```python
# Foundation — always imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML — imported when doing ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Deep Learning (weeks 17+)
import torch
import torch.nn as nn

# LLMs (weeks 13+)
from transformers import AutoTokenizer, AutoModel
```

### Import Styles
```python
import numpy              # use as: numpy.array([1, 2, 3])
import numpy as np        # standard alias — everyone does this
from numpy import array   # import one thing — use as: array([1, 2, 3])
from numpy import *       # BAD — pollutes namespace, avoid this
```

---

## Section 8: Reading Error Messages

### How to Read a Python Traceback
```
Traceback (most recent call last):
  File "notebook.py", line 15, in <module>      ← where it happened
    result = my_function(x)
  File "notebook.py", line 8, in my_function     ← deeper call
    return a + b
TypeError: unsupported operand type(s) for +: 'int' and 'str'
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           THIS IS THE ACTUAL ERROR MESSAGE
```

**Read from the bottom up.** The last line is always the most useful.

### The 5 Most Common Errors in ML Code

| Error | Typical Cause | Fix |
|-------|---------------|-----|
| `NameError: name 'x' is not defined` | Used variable before creating it, or typo | Check spelling, check you ran the cell that creates `x` |
| `TypeError: can't convert X to Y` | Wrong type in operation | Use `float()`, `int()`, `str()` to convert |
| `IndexError: list index out of range` | Accessed past the end of a list | Check `len(lst)`, fix your index |
| `KeyError: 'column_name'` | pandas column doesn't exist | Run `df.columns` to see what columns actually exist |
| `ValueError: could not convert string to float` | Non-numeric text in a column you expected to be numeric | Clean the column with `pd.to_numeric(col, errors='coerce')` |

---

## Reflection Questions

After finishing the notebook, write answers from memory:

1. What is the difference between a list and a dictionary?
2. When would you use `.get()` instead of `["key"]` to access a dictionary?
3. What is the difference between `lst.sort()` and `sorted(lst)`?
4. Write from memory: a function `def compute_f1(precision, recall)` that returns `2 * p * r / (p + r)`.
5. What does `enumerate()` give you that a regular `for x in lst` doesn't?
6. What `import` statement gives you access to `np.array()`?

---

## Checklist for This Week

- [x] I know all 4 basic data types and how to convert between them
- [x] I can create, index, slice, and append to a list
- [x] I can write a list comprehension with and without a filter condition
- [x] I can create, access, and iterate over a dictionary
- [x] I can write an if/elif/else block
- [x] I can write a for loop with `enumerate()` and `zip()`
- [x] I can define a function with multiple parameters and a return value
- [x] I can write standard ML imports from memory
- [x] I completed every `YOUR TURN` exercise in the notebook

---

*Guide for `STARTER_Week1_PythonBasics.ipynb` | Phase 0 | ML-AI-learning roadmap*
