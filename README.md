# Advanced AI Labs

> Advanced artificial intelligence labs covering the full data science lifecycle, deep learning with PyTorch, and hands-on applied ML on real-world datasets. Completed as part of the Advanced AI course at Effat University.

---

## Labs Overview

### Lab 1 — Foundations of Data Science: End-to-End Scenario
**File:** `AdvancedAILab1.ipynb`

A comprehensive guided walkthrough of the complete data science project lifecycle, covering theory and practice from raw data to insight generation.

**Topics covered:**
- Data sources: internal, external, APIs, RSS feeds, web scraping
- Data types: numeric, boolean, strings, datetime, lists, dictionaries
- Data formats: tabular (CSV), structured (JSON/XML), semi-structured, textual, temporal, geolocation
- Variable types: quantitative (discrete/continuous) vs. categorical
- Full pipeline: ask questions → collect → explore → model → analyze → visualize

**Tools:** Python · Pandas · Matplotlib

---

### Lab 3 (Part 1) — Weather Prediction Pipeline
**File:** `AdvancedAILab.ipynb`

Applied the full data science process to the weatherHistory dataset to predict weather conditions from atmospheric features.

**Dataset:** `weatherHistory.csv` — historical weather records with temperature, humidity, wind speed, visibility, pressure, and precipitation type

**Tasks completed:**
- **Data Collection:** Loaded dataset using Pandas, displayed first 5 rows, inspected shape and dtypes
- **Data Preprocessing:**
  - Identified missing values in `Precip Type` column
  - Filled missing values with `"none"`
  - Encoded `Precip Type` into numeric labels using `LabelEncoder`
  - Dropped irrelevant columns (`Formatted Date`, `Daily Summary`)
  - Applied `StandardScaler` for feature normalization
- **EDA:** Checked unique precipitation types, data types, shape, and descriptive statistics
- **Modeling:** Linear Regression with train/test split, MSE and R² evaluation

**Tools:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · SciPy

---

### Lab 3 (Part 2) — Weather Prediction Pipeline (Extended)
**File:** `AdvancedAILab3.ipynb`

Extended version of the weather prediction lab applying additional preprocessing steps and a more structured task-based approach to the same weatherHistory dataset.

**Extended tasks:**
- Repeated full preprocessing pipeline with additional data validation steps
- Verified unique precipitation types: `['rain', 'snow', nan]`
- Applied Haversine formula for geospatial distance feature engineering
- More detailed EDA: correlation analysis, distribution plots, outlier inspection
- Model comparison and performance reporting

**Tools:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · SciPy

---

### Lab 5 — Deep Learning with PyTorch: Neural Network Classification
**File:** `AdvancedAILab5.py`

Built, trained, and evaluated multiple neural network architectures from scratch using PyTorch on a synthetic binary classification dataset (1,000 samples, 20 features).

**Neural network architectures implemented:**

| Model | Architecture | Notes |
|---|---|---|
| `SimpleNeuralNetwork` | Input → Hidden → Output | Basic 2-layer network with ReLU |
| `ImprovedNetwork` | Input → [64 → 32 → 16] → Output | Multi-layer with Dropout regularization |
| `BinaryClassifier` | Input → 64 → 32 → 2 | Full training pipeline with Adam optimizer |

**Key techniques:**
- Tensor operations: creation, arithmetic, matrix multiplication, reshaping, transpose
- Custom `EarlyStopping` class with configurable patience and min_delta
- Learning rate scheduling: `StepLR` (step_size=30, gamma=0.1) and `ReduceLROnPlateau`
- Mini-batch training loop with gradient zeroing, backpropagation, and optimizer step
- Full evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix (TP/TN/FP/FN)
- Probability outputs using `F.softmax` for new sample prediction
- Model persistence: saving/loading weights only, full model, and training checkpoints

**Tools:** Python · PyTorch · Scikit-learn · NumPy

---

## Skills Demonstrated

| Area | Skills |
|------|--------|
| Data Science Lifecycle | Problem definition, data collection, preprocessing, EDA, modeling, evaluation |
| Data Preprocessing | LabelEncoder, StandardScaler, missing value imputation, feature dropping |
| Deep Learning | PyTorch, custom nn.Module classes, Dropout, ReLU, CrossEntropyLoss |
| Training Techniques | Mini-batch gradient descent, early stopping, LR scheduling, checkpointing |
| Model Evaluation | Accuracy, F1, Precision, Recall, Confusion Matrix, R², MSE |
| Feature Engineering | Geospatial features (Haversine), categorical encoding, normalization |

---

## Course
**Advanced Artificial Intelligence**  
Effat University · College of Engineering · Department of Computer Science

## Author
**Jouri Aldaghma**  
Computer Science (AI Concentration) · Effat University  
[LinkedIn](https://linkedin.com/in/jouri-aldaghma-b7a8b8307) · [GitHub](https://github.com/Jourialdagh)
