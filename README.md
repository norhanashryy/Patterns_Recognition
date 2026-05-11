# Movie Popularity Classification — Machine Learning Pipeline
## Overview

This project predicts movie popularity levels using a large-scale machine learning pipeline built on structured movie metadata, engineered text signals, temporal features, production metadata, and imbalance-aware classification methods.

The project evolved across two milestones:

* **Milestone 1:** baseline preprocessing, feature engineering, and initial model experimentation.
* **Milestone 2:** advanced feature engineering, feature selection, imbalance handling experiments, and comparative model evaluation.

The final pipeline combines:

* feature engineering
* missingness-aware modeling
* text-derived semantic signals
* imbalance handling
* ensemble feature selection
* multiple tree-based and linear classifiers
* hyperparameter studies

---

# Problem Statement

Movie datasets are highly heterogeneous and contain:

* missing metadata
* noisy text descriptions
* sparse categorical information
* heavily imbalanced popularity classes

The goal of this project is to classify movies into popularity levels:

* Very Low
* Low
* Medium
* High

while investigating:

* the effect of feature engineering
* missing-data behavior
* imbalance handling techniques
* model sensitivity to synthetic oversampling

---

# Dataset Characteristics

The dataset contains:

* movie metadata
* release information
* production information
* financial attributes
* text overviews
* genres
* language information
* popularity-related attributes

Challenges included:

* high missingness in text fields
* skewed numerical distributions
* categorical sparsity
* class imbalance (~44:1)
* mixed numeric + categorical + text-derived data

---

# Key Contributions

## 1. Missingness-Aware Feature Engineering

Instead of treating null values as noise only, missingness was modeled as a predictive signal.

Examples:

* `has_overview`
* `text_null_count`
* `meta_null_count`
* `_was_null` indicators

This approach captures Missing-Not-At-Random (MNAR) behavior.

---

## 2. Text-Derived Semantic Signals

Movie overviews were transformed into structured signals:

* overview length
* word count
* sentiment analysis (VADER)
* sequel indicators
* true-story indicators
* theme detection:

  * action
  * romance
  * horror
  * sci-fi
  * family

These features convert unstructured text into model-friendly semantic representations.

---

## 3. Temporal Feature Engineering

Release dates were transformed into:

* release year
* release month
* release season
* release decade
* movie age
* missing-date indicators

This allows models to learn temporal popularity patterns.

---

## 4. Financial Feature Engineering

Financial variables were heavily skewed and required transformation.

Created features include:

* budget_log
* revenue_log
* ROI
* blockbuster indicators
* known/missing financial flags

Log transforms reduced skewness and improved model stability.

---

## 5. Advanced Feature Selection

A consensus-based feature selection framework was implemented using:

* Mutual Information
* ANOVA F-score
* LightGBM feature importance
* Spearman correlation

Features were ranked using normalized consensus scores.

---

## 6. Imbalance Handling Experiments

The dataset exhibited severe class imbalance.

Three strategies were compared:

1. ADASYN + undersampling
2. SMOTE + undersampling
3. Undersampling only (no synthetic data)

Results showed that:

* synthetic oversampling improved minority recall
* but often introduced noisy synthetic boundaries
* undersampling-only generalized best overall

This highlighted the sensitivity of high-dimensional engineered feature spaces to synthetic oversampling.

---

# Models Evaluated

## Linear Models

* Logistic Regression (SGDClassifier)

## Bagging Methods

* Random Forest
* Extra Trees

## Gradient Boosting

* XGBoost
* LightGBM
* CatBoost

---

# Evaluation Metrics

Because the dataset is imbalanced and ordinal, multiple evaluation metrics were used:

* Accuracy
* Macro F1-score
* Balanced Accuracy
* Cohen Kappa
* Weighted Cohen Kappa

Weighted Kappa was particularly useful because it penalizes larger class-distance errors more heavily.

---

# Key Findings

## Synthetic Oversampling Was Not Always Beneficial

ADASYN and SMOTE occasionally caused:

* overfitting
* unstable decision boundaries
* reduced generalization

The best-performing setup used:

* controlled undersampling
* class-weighted learning
* no synthetic minority generation

---

## Feature Engineering Had Major Impact

The largest performance improvements came from:

* semantic overview features
* missingness-aware signals
* temporal engineering
* grouped categorical representations

---

## Tree-Based Ensembles Performed Best

Boosted tree methods consistently outperformed linear models because they:

* captured nonlinear interactions
* handled mixed feature types effectively
* exploited sparse binary indicators efficiently

LightGBM and XGBoost showed the strongest overall performance.

---

# Technologies Used

## Core Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy

## Visualization

* Matplotlib
* Seaborn

## NLP

* NLTK (VADER sentiment)

## Imbalance Handling

* imbalanced-learn

## Models

* XGBoost
* LightGBM
* CatBoost

---

# Project Structure

```text
project/
│
├── milestone1.ipynb
├── milestone2.ipynb
├── data/
├── outputs/
├── figures/
├── requirements.txt
└── README.md
```

---

# Future Improvements

Potential future work includes:

* transformer-based text embeddings
* Optuna/Bayesian hyperparameter optimization
* probability calibration
* target encoding for high-cardinality features
* stacked ensembles
* explainability tools (SHAP)

---

# Why This Project Matters

This project goes beyond basic classification tutorials by exploring:

* real-world noisy data
* imbalance-aware learning
* feature engineering strategy
* synthetic oversampling failure modes
* ensemble feature selection
* comparative model behavior

The pipeline reflects practical machine learning experimentation rather than only textbook implementation.

---

## Contributors
Farah, Mariam, Sama, Norhan, Yomna & Hana 
