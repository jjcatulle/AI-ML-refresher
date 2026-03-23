# Guide: Week 4 — Visualization Deep Dive
> **Phase 0 | Foundation** — See your data before you model it.

---

## Beginner Start Here

"Plot the data first" is the most important habit in data science. A visualization tells you in seconds what takes a hundred lines of summary statistics to describe. This guide teaches you to think visually: which chart to choose, how to read it, and how to make it professional.

### What This Guide Covers
- The matplotlib object model (Figure, Axes)
- When to use each chart type
- Seaborn vs matplotlib — when to use which
- How to read a correlation heatmap
- Professional chart standards
- The full EDA visualization workflow

### Key Terms
| Term | Plain English |
|------|---------------|
| **Figure** | The entire canvas — the window or image file |
| **Axes** | A single chart area within the Figure (confusingly, not the axis labels) |
| **Artist** | Any drawn object: line, bar, text, marker, patch |
| **KDE** | Kernel Density Estimate — a smooth curve showing the distribution shape |
| **Histogram** | Divides values into bins, shows count per bin (bar chart of distributions) |
| **Box plot** | Shows quartiles, median, and outliers compactly |
| **Violin plot** | Box plot + KDE rotated — shows the full distribution shape |
| **Heatmap** | Color-encodes a 2D matrix of values (e.g., correlation matrix) |
| **Pairplot** | Grid of scatterplots for every pair of numeric features |
| **Correlation** | How strongly and in which direction two features move together (range: -1 to 1) |

---

## How to Study This Guide

1. Open `STARTER_Week4_VisualizationDeep.ipynb`.
2. Before running each code cell — read this guide section for that chart type.
3. For each chart: run the notebook cell, then change ONE thing (color, bins, title) and re-run.
4. Work through the final YOUR TURN dashboard exercise without looking at any examples.
5. Save one of your charts to `outputs/` as a PNG.

---

## Section 1: The matplotlib Architecture — Figure and Axes

This is the concept most beginners skip, and it causes endless confusion. Learn it once.

```
┌──────────────────────────────────────────┐
│  Figure  (the whole canvas)              │
│                                          │
│  ┌──────────────┐  ┌──────────────┐     │
│  │  Axes [0,0]  │  │  Axes [0,1]  │     │
│  │  (chart 1)   │  │  (chart 2)   │     │
│  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐     │
│  │  Axes [1,0]  │  │  Axes [1,1]  │     │
│  │  (chart 3)   │  │  (chart 4)   │     │
│  └──────────────┘  └──────────────┘     │
└──────────────────────────────────────────┘
```

### Two Ways to Use matplotlib

**Style 1 — pyplot (quick and dirty):** One chart, one call

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("My Chart")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

**Style 2 — Object-Oriented (correct way for real projects):**

```python
fig, ax = plt.subplots()          # creates Figure and one Axes
ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title("My Chart")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.tight_layout()
plt.show()
```

**Style 3 — Multiple subplots:**

```python
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axes.flat

ax1.plot(x, y)
ax2.hist(data)
ax3.scatter(x, y)
ax4.barh(categories, values)

fig.suptitle("EDA Dashboard", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("outputs/dashboard.png", dpi=150, bbox_inches='tight')
plt.show()
```

**The key rule:** When you have multiple subplots, always use the **object-oriented** style. Call methods on `ax` (like `ax.set_title()`), not on `plt`.

---

## Section 2: Chart Type Selection Guide

The most important skill is choosing the **right chart for the question you're asking**.

### Is this a distribution question?

| Question | Chart | Function |
|----------|-------|----------|
| What does this variable's distribution look like? | Histogram or KDE | `ax.hist()` / `sns.histplot()` |
| Where is the center and spread? Any outliers? | Box plot | `sns.boxplot()` |
| How does each group's distribution compare? | Violin or Box | `sns.violinplot()` |
| Is it normal / skewed / bimodal? | KDE + rug | `sns.kdeplot()` |

### Is this a relationship question?

| Question | Chart | Function |
|----------|-------|----------|
| Do two variables move together? | Scatter plot | `ax.scatter()` / `sns.scatterplot()` |
| How correlated are all feature pairs? | Correlation heatmap | `sns.heatmap(df.corr())` |
| Which features are related to the target? | Bar chart of correlations | sorted `df.corr()['target'].plot.barh()` |
| Relationships between all pairs at once | Pairplot | `sns.pairplot()` |

### Is this a composition / count question?

| Question | Chart | Function |
|----------|-------|----------|
| How many in each category? | Bar chart (vertical) | `sns.countplot()` |
| Category proportions? | Horizontal bar or pie | `ax.barh()` |
| How does a number vary by category? | Box / bar with error bars | `sns.boxplot()` |

### Is this a trend / time question?

| Question | Chart | Function |
|----------|-------|----------|
| How does a value change over time? | Line chart | `ax.plot()` |
| Multiple lines (e.g., train vs val loss) | Multi-line chart | `ax.plot()` twice |

---

## Section 3: matplotlib — The Core Charts

### Line Chart

Use for: time series, training curves, any ordered sequence.

```python
epochs = range(1, 21)
train_loss = [0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.31, 0.28, 0.25, 0.23,
              0.21, 0.20, 0.19, 0.18, 0.17, 0.17, 0.16, 0.16, 0.15, 0.15]
val_loss   = [0.95, 0.74, 0.65, 0.57, 0.48, 0.43, 0.40, 0.38, 0.37, 0.36,
              0.36, 0.36, 0.37, 0.38, 0.39, 0.41, 0.42, 0.44, 0.46, 0.48]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_loss, label='Train', color='#2196F3', linewidth=2)
ax.plot(epochs, val_loss, label='Val', color='#FF5722', linewidth=2,
        linestyle='--')
ax.axvline(x=11, color='gray', linestyle=':', label='Start of overfit')
ax.set_title("Training Loss Curve", fontsize=14)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.tight_layout()
```

**What to look for:** The gap between train and validation loss. When val_loss starts rising while train_loss keeps falling → overfitting.

### Histogram

Use for: exploring the distribution of one numeric feature.

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['monthly_charges'], bins=30, color='#2196F3', edgecolor='white',
        alpha=0.8)
ax.axvline(df['monthly_charges'].mean(), color='red', linestyle='--',
           label=f"Mean: {df['monthly_charges'].mean():.1f}")
ax.axvline(df['monthly_charges'].median(), color='orange', linestyle='--',
           label=f"Median: {df['monthly_charges'].median():.1f}")
ax.set_title("Monthly Charges Distribution")
ax.set_xlabel("Monthly Charges ($)")
ax.set_ylabel("Count")
ax.legend()
```

**What to look for:** Skew (mean ≠ median means it's skewed). Bimodal peaks (two humps usually suggest two subpopulations). Outliers (long tail on one side).

### Scatter Plot

Use for: relationship between two numeric features.

```python
fig, ax = plt.subplots(figsize=(8, 6))
colors = df['churned'].map({0: '#2196F3', 1: '#FF5722'})
ax.scatter(df['tenure_months'], df['monthly_charges'], c=colors, alpha=0.5,
           s=20)
ax.set_title("Tenure vs Monthly Charges by Churn")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Monthly Charges ($)")
# Add manual legend
import matplotlib.patches as mpatches
legend = [mpatches.Patch(color='#2196F3', label='Active'),
          mpatches.Patch(color='#FF5722', label='Churned')]
ax.legend(handles=legend)
```

---

## Section 4: seaborn — Statistical Visualization Made Easy

seaborn builds on matplotlib. Use it when you want:
- Built-in statistical calculations (confidence intervals, regression lines)
- Automatic color grouping by a category column
- Less code for common patterns

```python
import seaborn as sns
sns.set_theme(style='whitegrid')  # always set the theme at the top
```

### Histplot with KDE

```python
sns.histplot(data=df, x='monthly_charges', hue='churned',
             kde=True, stat='density', alpha=0.4)
```

`hue='churned'` automatically splits by group and colors them differently — you'd write 10+ lines to do this in pure matplotlib.

### Box Plot — Compact Distribution Summary

```
                 ┌─────────────────────┐
 ─────────────── │    Q1    │   Q3     │ ─────────────────── ● outlier
                 └─────────────────────┘
                      ↑median↑
             whisker                  whisker
            (Q1 - 1.5*IQR)          (Q3 + 1.5*IQR)
```

- **Box**: Q1 to Q3 (middle 50% of data)
- **Line in box**: median
- **Whiskers**: extend to 1.5× the interquartile range (IQR)
- **Dots beyond whiskers**: outliers

```python
sns.boxplot(data=df, x='contract', y='monthly_charges', hue='churned',
            palette='Set2')
```

### Violin Plot

Same as a box plot but the shape shows the full distribution (like a mirrored KDE). Better when you want to see bimodal distributions.

```python
sns.violinplot(data=df, x='contract', y='monthly_charges', hue='churned',
               split=True, palette='Set2')
```

### Correlation Heatmap

Shows the pairwise correlation between every numeric column. This is one of the most important charts in Feature Analysis.

```python
corr = df.select_dtypes(include='number').corr()

mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle (mirror)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix")
```

---

## Section 5: Reading a Correlation Heatmap

Correlation values range from **-1.0 to +1.0**:

| Value | Meaning |
|-------|---------|
| **1.0** | Perfect positive — both go up together |
| **0.7 to 0.99** | Strong positive |
| **0.4 to 0.69** | Moderate positive |
| **0.1 to 0.39** | Weak positive |
| **~0.0** | No linear relationship |
| **-0.1 to -0.39** | Weak negative |
| **-0.4 to -0.69** | Moderate negative |
| **-1.0** | Perfect negative — as one goes up, other goes down |

### What to Look For in a Heatmap

1. **Target column**: Find the row or column for your target variable (`churned`). Which features have the highest absolute correlation with it? Those are your best starting features.

2. **Multicollinearity**: If two *input* features are strongly correlated with each other (e.g., `monthly_charges` and `total_charges` both above 0.95), they carry redundant information. Consider dropping one.

3. **Surprising correlations**: If two unrelated-seeming features are strongly correlated, that might indicate data leakage (one feature was derived from the other, or from the target).

4. **Correlation ≠ causation**: High correlation means "they move together in this dataset", not "one causes the other".

---

## Section 6: Pairplot — The Overview Chart

Use once per project on a new dataset. Shows scatter of every numeric pair, histograms/KDE on diagonal.

```python
cols = ['monthly_charges', 'tenure_months', 'total_charges', 'support_tickets']
sns.pairplot(df[cols + ['churned']], hue='churned', diag_kind='kde',
             plot_kws={'alpha': 0.5, 's': 15}, palette='Set1')
plt.suptitle("Pairplot of Key Features", y=1.02)
```

**Limitation:** For more than ~6 features, it becomes an unreadable grid. Use it as a first look, then focus on specific pairs.

---

## Section 7: The Professional Chart Checklist

Every chart you include in a report or notebook should pass this checklist:

- [ ] **Title**: Descriptive (says what the chart shows). Under 10 words.
- [ ] **Axis labels**: Both axes labeled with units where applicable.
- [ ] **Legend**: Present if multiple groups/lines are shown.
- [ ] **Font size**: Large enough to read in a presentation (≥ 12pt for labels, ≥ 14pt for title).
- [ ] **Color**: Accessible (avoid red/green for the main contrast — use blue/orange for colorblind accessibility).
- [ ] **Tight layout**: Call `plt.tight_layout()` so labels don't overlap.
- [ ] **Caption / annotation**: Add a note explaining the key takeaway when non-obvious.
- [ ] **No chart junk**: Remove 3D effects, excessive gridlines, borders, redundant text.

```python
# Minimal professional template
fig, ax = plt.subplots(figsize=(9, 5))
# ... your chart code here ...
ax.set_title(title, fontsize=14, pad=12)
ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)
ax.tick_params(labelsize=11)
if legend: ax.legend(fontsize=11)
sns.despine()              # removes top and right border lines (looks cleaner)
plt.tight_layout()
fig.savefig(f"outputs/{filename}.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

## Section 8: Saving Figures

```python
# PNG (for presentations, reports, web)
fig.savefig("outputs/my_chart.png", dpi=150, bbox_inches='tight')

# PDF (for papers — vector, infinitely scalable)
fig.savefig("outputs/my_chart.pdf", bbox_inches='tight')

# SVG (for web — vector, editable in design tools)
fig.savefig("outputs/my_chart.svg", bbox_inches='tight')
```

`bbox_inches='tight'` prevents labels and titles from being cropped at the edges. Always include it.

`dpi=150` balances file size and quality for most use cases. Use `dpi=300` for print.

---

## Section 9: seaborn vs matplotlib — When to Use Which

| Use Case | Recommended |
|----------|-------------|
| Quick EDA, grouped distributions, heatmaps | seaborn |
| Training loss curves, custom multi-panel dashboards | matplotlib (OOP) |
| Complex subplot assembly with custom spacing | matplotlib |
| Histograms with KDE + hue grouping | seaborn |
| Reproducible publication figures | Both (sns for styling, fig.savefig) |
| Adding arrows, annotations, patches | matplotlib artists |

**Rule**: Start with seaborn. If you need to customize something it doesn't support, drop to matplotlib artists via the `ax` object.

---

## Section 10: The EDA Visualization Workflow

When you receive a new dataset, run through these 5 stages in order. Don't skip any.

### Stage 1: Target Distribution

```python
# For classification
sns.countplot(data=df, x='target')
df['target'].value_counts(normalize=True)
# → are the classes balanced? If one class is < 20%, you have a class imbalance problem

# For regression
sns.histplot(df['target'], kde=True)
# → is it normally distributed? Skewed? Are there outliers?
```

### Stage 2: Feature Distributions

```python
numeric = df.select_dtypes(include='number').columns
n = len(numeric)
fig, axes = plt.subplots(nrows=(n+2)//3, ncols=3, figsize=(15, 4*((n+2)//3)))
for i, col in enumerate(numeric):
    sns.histplot(df[col], kde=True, ax=axes.flat[i])
    axes.flat[i].set_title(col)
plt.tight_layout()
```

### Stage 3: Feature vs. Target

```python
# For numeric features vs classification target
for col in numeric:
    sns.boxplot(data=df, x='target', y=col)
    plt.title(f"{col} by Target Class")
    plt.show()

# Correlation bar chart (fastest)
corr_with_target = df.corr()['target'].drop('target').sort_values()
corr_with_target.plot.barh(figsize=(8, 6), title="Feature Correlation with Target")
```

### Stage 4: Feature Relationships (Multicollinearity)

```python
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='RdBu_r',
            mask=np.triu(np.ones(...)))
```

### Stage 5: Categorical Features

```python
for col in df.select_dtypes(include='object').columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x=col, ax=axes[0])
    axes[0].set_title(f"{col} Frequency")
    sns.boxplot(data=df, x=col, y='monthly_charges', ax=axes[1])
    axes[1].set_title(f"Charges by {col}")
    plt.tight_layout()
    plt.show()
```

---

## Reflection Questions

1. What is the difference between a Figure and an Axes in matplotlib?
2. You have a distribution with a mean much larger than its median. What does this tell you?
3. A correlation heatmap shows that `total_charges` has a 0.95 correlation with `monthly_charges`. Should you use both features? Why or why not?
4. When would you use a violin plot over a box plot?
5. Your model's validation loss starts increasing at epoch 15. What matplotlib chart would you draw to see this, and what would it look like?
6. Write from memory: the code to save a figure with no label clipping at 150 DPI.

---

## Checklist for This Week

- [ ] I created at least one of each: line, histogram, scatter, bar chart using matplotlib OOP style
- [ ] I created a 2×2 subplot dashboard
- [ ] I used seaborn for at least: histplot+kde, boxplot, heatmap
- [ ] I read and interpreted a correlation heatmap (found a feature correlated with the target)
- [ ] I used the professional chart checklist for at least one chart (title, labels, legend, tight_layout, save)
- [ ] I ran the complete `run_eda()` workflow function in the notebook
- [ ] I completed the YOUR TURN dashboard exercise without help
- [ ] I saved at least one chart to the `outputs/` folder

---

## 🎉 Phase 0 Complete!

You've built the foundation. Here's what you've earned:

| Week | Skill |
|------|-------|
| Week 0 | Understand what ML is, what it can and can't do |
| Week 1 | Write Python code that does computational work |
| Week 2 | Clean, filter, group, and transform tabular data with pandas |
| Week 3 | Understand and write NumPy matrix math — the language of ML models |
| Week 4 | Visualize distributions, relationships, and model behavior |

**Next:** Move to Phase 1 — your first end-to-end ML project with scikit-learn.

---

*Guide for `STARTER_Week4_VisualizationDeep.ipynb` | Phase 0 Complete | ML-AI-learning roadmap*
