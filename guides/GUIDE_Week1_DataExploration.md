# Week 1: Data Exploration - Learning Guide

Before tackling the challenges, understand these concepts!

## 💼 Real-World Use Cases
- **Marketing analytics:** Analyze customer behavior to improve conversion rates.
- **Finance:** Validate and clean trading/transaction data before modeling.
- **Operations:** Monitor system logs to detect anomalies early.

---

## 📊 Recommended Datasets for Week 1

Choose ONE dataset below to complete all 10 challenges:

### Option 1: Iris Dataset ✅ **Easiest (Start here!)**
- **What:** Flower measurements (sepal/petal length & width)
- **Size:** 150 flowers, 4 features
- **Where:** Built-in to scikit-learn
- **How to load:**
  ```python
  from sklearn.datasets import load_iris
  iris = load_iris()
  df = pd.DataFrame(iris.data, columns=iris.feature_names)
  df['species'] = iris.target_names[iris.target]
  ```
- **Why:** Small, clean, perfect for learning. No missing data.

### Option 2: Kaggle - Titanic Dataset 🚢
- **What:** Passenger data (survived or not, age, class, fare, etc.)
- **Size:** 891 passengers, 11 features
- **Where:** https://www.kaggle.com/datasets/titanic
- **How to load:**
  ```python
  df = pd.read_csv('titanic.csv')
  ```
- **Why:** Classic, real data with missing values and categorical features.
- **Missing data:** Teaches you real-world data cleaning.

### Option 3: Kaggle - Housing Dataset 🏡
- **What:** California housing prices (median age, rooms, households, etc.)
- **Size:** 20,640 homes, 8 features
- **Where:** https://www.kaggle.com/datasets/camnugent/california-housing-prices
- **How to load:**
  ```python
  from sklearn.datasets import fetch_california_housing
  df = fetch_california_housing(as_frame=True).frame
  ```
- **Why:** Larger dataset, better for visualization practice.

### Option 4: UCI ML Repository - Iris Alternative 🌺
- **What:** Abalone data (physical measurements, age in rings)
- **Size:** 4,177 abalones, 8 features
- **Where:** https://archive.ics.uci.edu/dataset/1/abalone
- **How to load:**
  ```python
  df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data')
  ```
- **Why:** Larger than Iris, teaches grouping/aggregation.

---

## 🎯 Learning Objectives

By the end of Week 1, you should understand:
1. How to load and inspect data
2. What data types and structures exist
3. How to compute statistics
4. How to visualize relationships
5. How to filter and group data

---

## 📚 Concept 1: Importing Libraries

### What are libraries?
Libraries are pre-written code that extends Python's capabilities. Instead of writing everything from scratch, we use tested, optimized libraries.

### The 4 Libraries You Need

```python
import numpy as np        # For numerical operations
import pandas as pd       # For data tables (DataFrames)
import matplotlib.pyplot as plt    # For plotting graphs
import seaborn as sns     # For prettier statistical plots
```

### Why Each?
- **NumPy**: Fast math operations on arrays
- **Pandas**: Organize data into rows/columns (like Excel)
- **Matplotlib**: Create any plot (line, scatter, histogram)
- **Seaborn**: Statistical plots with pretty defaults

### Real-world example
In real projects, you often start by importing these tools to quickly inspect data from CSV exports, log files, or database queries. For example, a data analyst may use `pandas` to load a customer transaction report and `seaborn` to visualize spending trends by month.

### Best practice
In professional code, import libraries once at the top of your script or notebook. Keep imports organized and only include what you actually use.

### Challenge 1 Approach
- Import all 4 libraries at the top
- Print the version to verify (e.g., `print(np.__version__)`)

**Syntax Reference**: `import X as Y` means "import library X, but I'll call it Y in my code"

---

## 📚 Concept 2: Loading Data

### Where does data come from?
- CSV files (spreadsheets)
- Databases (SQL)
- APIs (web services)
- Built-in datasets (like Iris)

### Real-world example
A marketing analyst might receive a monthly CSV export of customer purchases. An operations team might query a SQL database for system logs. In both cases, the first step is to load the data into a DataFrame so you can explore it.

### How to Load?
```python
# From CSV file
df = pd.read_csv('myfile.csv')

# From built-in dataset
from sklearn.datasets import load_iris
iris = load_iris()
```

### What's a DataFrame?
A DataFrame is a table with rows and columns, like Excel:

| sepal_length | sepal_width | species |
|---|---|---|
| 5.1 | 3.5 | setosa |
| 4.9 | 3.0 | setosa |

### Challenge 2 Approach
1. Import `load_iris` from sklearn
2. Load the iris dataset: `iris = load_iris()`
3. Create DataFrame from the data
4. Add species column using target names

**Key insight**: `iris.data` has the features, `iris.target` has numeric labels, `iris.target_names` has actual names

---

## 📚 Concept 3: Exploring Data Structure

### Questions to Ask
1. **Shape**: How many rows and columns? (`.shape`)
2. **Columns**: What's the data called? (`.columns`)
3. **Types**: Numbers? Text? (`.dtypes`)
4. **Content**: What do first rows look like? (`.head()`)

### Key Methods

```python
# Dimensions
df.shape              # Returns (rows, columns)

# Column info
df.columns           # List of column names
df.info()            # Detailed info about each column

# First/last rows
df.head()            # First 5 rows
df.tail(3)           # Last 3 rows

# Data types
df.dtypes            # What type is each column?
```

### Challenge 3 Approach
- Use all 5 methods above
- Interpret what each tells you
- Answer: "What is the shape?", "How many numeric columns?"

---

## 📚 Concept 4: Summary Statistics

### Why Statistics?
- **Mean**: Average value (is data centered around 0 or 100?)
- **Median**: Middle value (robust to outliers)
- **Std**: Spread (is data tightly grouped or scattered?)
- **Min/Max**: Boundaries (what's the range?)

### Real-world example
In finance, you might compute mean and standard deviation of daily returns to understand volatility. In manufacturing, min/max checks can detect sensors producing impossible values (e.g., temperature below -100°C).

### Key Methods

```python
# All statistics at once
df.describe()        # Count, mean, std, min, max, percentiles

# Individual statistics
df.mean()            # Average of each column
df.median()          # Middle value
df.std()             # Spread/variation
df.max()             # Largest value
df.min()             # Smallest value
```

### Reading `.describe()` Output
```
         sepal_length  sepal_width
count      150.0       150.0         <- 150 data points
mean         5.84        3.05         <- Average values
std          0.83        0.43         <- How spread out
min          4.30        2.00         <- Smallest values
25%          5.10        2.80         <- Bottom quarter
50%          5.80        3.00         <- Middle (median)
75%          6.50        3.30         <- Top quarter
max          7.90        4.40         <- Largest values
```

### Challenge 4 Approach
- Call `.describe()` and interpret output
- Calculate `.mean()`, `.median()` individually
- For species (text), use `.value_counts()` instead

---

## 📚 Concept 5: Missing Data

### What's Missing Data?
Sometimes datasets have gaps - no value recorded. In pandas, missing = `NaN` (Not a Number).

### Why Care?
Most ML models can't handle NaN. You need to either:
1. Remove rows with NaN
2. Fill NaN with a value (mean, median, etc.)
3. Keep if very rare

### Real-world example
In healthcare, missing vitals could mean a patient was not checked (important signal), not just a data error. In e-commerce, missing shipping address might mean a warning sign about order quality. Understanding why data is missing helps you decide how to treat it.

### How to Check

```python
# Which columns have NaN?
df.isnull()          # True/False for each cell
df.isnull().sum()    # Count of NaN per column

# Total missing percentage
missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
```

### Challenge 5 Approach
- Count missing per column
- Calculate % of missing
- Interpret: "Is data clean or do we need to clean it?"

---

## 📚 Concept 6: Filtering Data

### What's Filtering?
Selecting only rows that meet criteria. Like an Excel filter.

### Real-world example
A data scientist might filter transaction records to only look at refunds, or select customers from a specific region to analyze regional sales behavior. Filtering lets you focus on the subset that matters for your question.

### How to Filter

```python
# Filter for specific value
df[df['species'] == 'setosa']    # Only setosa flowers

# Filter for numeric comparison
df[df['sepal_length'] > 6.5]     # Only big sepals

# Multiple conditions
df[(df['species'] == 'setosa') & (df['sepal_length'] > 5)]

# Find max/min
max_idx = df['petal_length'].idxmax()
df.loc[max_idx]                  # Get that row
```

### Challenge 6 Approach
1. Filter for one species, count results
2. Filter for large sepal_length, count results
3. Find flower with max petal_length using `.idxmax()`
4. Find flower with min sepal_width

---

## 📚 Concept 7: Grouping & Aggregation

### What's Grouping?
Split data into groups, then calculate stats for each group.

### Real-world example
In sales analytics, you might group by region or product category to compare revenue per group. In customer analytics, grouping by subscription type reveals which plans have higher churn rates.

### How to Group

```python
# Group by species, calculate mean of each column
df.groupby('species').mean()

# Group and get specific aggregation
df.groupby('species')['sepal_length'].max()

# Multiple aggregations
df.groupby('species').agg({
    'sepal_length': 'mean',
    'petal_length': 'max'
})
```

### Challenge 7 Approach
1. Group by species
2. Calculate mean of all numeric columns
3. Find max sepal_length per species
4. Find min petal_width per species
5. Count flowers in each species group

---

## 📚 Concept 8: Correlation Analysis

### What's Correlation?
How two variables move together:
- **+1.0**: Perfect positive (both increase together)
- **0.0**: No relationship
- **-1.0**: Perfect negative (one increases, other decreases)

### Real-world example
Correlation is used in finance to measure how two stocks move together. In product analytics, you might correlate time spent on a page with conversion rate to see which pages are most effective.

### How to Correlate

```python
# Get numeric columns only
numeric_df = df.select_dtypes(include=[np.number])

# Correlation matrix (all pairs)
corr_matrix = numeric_df.corr()

# Correlation with one column
df['sepal_length'].corr(df['petal_length'])
```

### Reading Correlations
```
             sepal_length  sepal_width  petal_length
sepal_length       1.00      -0.12          0.87   <- Strong with petal!
sepal_width       -0.12       1.00         -0.43
petal_length       0.87      -0.43          1.00
```

### Challenge 8 Approach
1. Calculate correlation matrix
2. Find highest correlation (besides 1.0 diagonal)
3. Visualize with heatmap: `sns.heatmap(corr_matrix, annot=True)`

---

## 📚 Concept 9: Visualizations

### Why Visualizations?
- Numbers are hard to understand
- Plots show patterns instantly
- Communicate findings clearly

### Real-world example
In marketing, a time series chart can reveal seasonal purchase trends. In operations, a heatmap of server latency can quickly highlight outages or slow zones.

### Plot Types for Week 1

#### Histogram
Shows distribution of one variable.
```python
plt.hist(df['sepal_length'], bins=20)  # 20 bars
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Count')
plt.title('Distribution of Sepal Length')
plt.show()
```

#### Scatter Plot
Shows relationship between two variables.
```python
plt.scatter(df['sepal_length'], df['petal_length'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()
```

#### Box Plot
Shows spread and outliers by group.
```python
df.boxplot(column='sepal_length', by='species')
```

#### Heatmap
Shows correlation matrix visually.
```python
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
```

### Challenge 9 Approach
1. Create histogram of one feature
2. Create scatter plot comparing two features
3. Color points by species group
4. Create box plot showing distribution by species
5. Add labels and titles to each

---

## 📚 Concept 10: Drawing Insights

### What's an Insight?
A meaningful pattern or finding from data.

### Real-world example
A product team might discover that customers who use a feature more than 3 times per week are 4× more likely to renew. That insight can drive product prioritization and marketing focus.

### Good Insights vs Bad Insights
❌ **Bad**: "The mean sepal length is 5.84"  
✅ **Good**: "Virginica flowers have 20% longer sepals than Setosa, suggesting species differ by size"

❌ **Bad**: "There are 0 missing values"  
✅ **Good**: "The dataset is clean with no missing data, ready for modeling"

### Challenge 10 Approach
1. Look at your plots and statistics
2. Ask: "What patterns do I see?"
3. Compare species - are they different?
4. Compare variables - are they related?
5. Write 3 meaningful insights in sentences

---

## 🎯 Challenge Sequence

**Do them in order:**

1. **Import** → Setup environment
2. **Load Data** → Get your dataset
3. **Explore Structure** → Understand what you have
4. **Summary Statistics** → Know the numbers
5. **Missing Values** → Ensure data quality
6. **Filtering** → Extract subsets
7. **Grouping** → Compare groups
8. **Correlation** → Understand relationships
9. **Visualizations** → See the patterns
10. **Insights** → Interpret findings

---

## 💡 Pro Tips

### Tip 1: Print After Each Step
```python
df.head()           # See what happened
print(df.shape)     # Verify dimensions
print(df.dtypes)    # Check types
```

### Tip 2: Read Error Messages Carefully
- "KeyError: 'column_name'" → Column doesn't exist
- "AttributeError: 'DataFrame' object has no attribute 'X'" → Method doesn't exist
- "TypeError" → Wrong data type for operation

### Tip 3: Use QUICK_REFERENCE.md
Stuck on syntax? Check the cheat sheet!

### Tip 4: Comment Your Code
```python
# Load iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]  # Add species labels
```

---

## 🚀 Ready?

You now understand every concept. Time to code!

Open `STARTER_Week1_DataExploration.ipynb` and tackle Challenge 1.

**Remember**: This is YOUR learning. Type every line. Understand every output.

Good luck! 💪
