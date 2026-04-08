# FH-DA: A Density-Aware Adaptive Fuzzy Hyperbox Dynamic Ensemble for Predicting Complications in Cirrhosis

## Project Overview

FH-DA (Fuzzy Hyperbox with Density-Aware Adaptation) is a novel dynamic ensemble selection framework designed for early risk prediction of cirrhosis complications using multi-center electronic health records (EHRs). The framework integrates clustering and local density information into fuzzy hyperbox-based competence modeling to address data heterogeneity and severe class imbalance in clinical prediction tasks.

This repository contains the implementation of FH-DA, along with all baseline models, data preprocessing pipelines, and evaluation scripts used in the study. The code is built on Python and extends the DESlib library with fuzzy hyperbox-based dynamic selection algorithms.

## Key Innovations

### 1. Density-Aware Fuzzy Hyperbox Competence Modeling
- Hyperboxes are constructed to represent each base classifier's competence and non-competence regions.
- Hyperbox boundaries are adaptively adjusted using clustering structure and local density estimates.
- A scaling factor derived from intra-cluster density shrinks hyperboxes in dense regions (refining decision boundaries) and expands them in sparse regions (improving coverage).

### 2. Four Hyperbox Geometry Variants
- **Rectangular hyperbox** (axis-aligned)
- **Spherical hyperbox** (Euclidean distance + Gaussian membership)
- **Ellipsoidal hyperbox** (Mahalanobis distance + Gaussian decay)
- **Bi-slope rectangular hyperbox** (linear decay inside, exponential decay outside)

### 3. Hybrid Data Augmentation for Class Imbalance
- Majority class undersampling combined with multiple minority augmentation techniques: SMOTE, ADASYN, Borderline-SMOTE, SVM-SMOTE, Gaussian Copula, CTGAN, TVAE, CopulaGAN.
- Generates 40 augmented subsets + original training set → 41 candidate training sets.
- 11 base classifier types (NB, KNN, LR, LDA, QDA, RF, ET, ADA, GBC, LGBM, XGB) trained on each → large heterogeneous pool.

### 4. Cluster- and Density-Aware Enhancement Factor
- Query sample is projected into cluster space; distance and local density to each cluster are combined to form a cluster membership vector.
- Hyperbox–cluster correlations are computed based on distances from hyperbox centers to cluster centers.
- An enhancement factor (inner product of query’s membership vector and classifier’s cluster relevance) is linearly combined with base competence to produce final competence score.

### 5. Interpretability
- SHAP analysis identifies top predictive features (e.g., readmission days, albumin, platelet count, coagulation markers).
- Hyperbox structures can be visualized in low-dimensional principal component space, providing intuitive explanation for classifier selection.

## Repository Structure

```
FH_DA/
├── my_deslib/                # Main library directory
│   ├── des/                 # Dynamic Ensemble Selection methods
│   │   ├── des_fh.py                    # Core fuzzy hyperbox DES implementation
│   │   ├── adaptive_fhdes.py            # Adaptive fuzzy hyperbox DES
│   │   ├── adaptive_fhdes_he.py         # Hyperellipsoid-based adaptive fuzzy hyperbox DES
│   │   ├── fh_des_*_vector.py           # Various fuzzy hyperbox vector implementations
│   │   ├── fh_des_clustering/           # Clustering-based fuzzy hyperbox methods
│   │   ├── hyperellipsoid_versions/     # Hyperellipsoid-based versions
│   │   ├── rectangle_versions/          # Rectangle-based versions
│   │   └── sphere_versions/             # Sphere-based versions
│   └── util/
│       └── fuzzy_hyperbox.py            # Fuzzy hyperbox core implementation
├── datasets/                             # Dataset directory
├── data_preprocessing.py                 # Data preprocessing utilities
├── feature_selection.py                  # Feature selection methods
├── feature_importance_analysis.py        # Feature importance analysis
├── baseline_models.py                    # Baseline model implementations
├── dynamic_selection.py                  # Dynamic selection methods
├── enhanced_dynamic_selection.py         # Enhanced dynamic selection with additional features
├── external_dynamic_selection.py         # External dynamic selection methods
├── enhancement_features.py               # Enhancement features implementation
├── best_fh_params_advance.py            # Optimized parameters for fuzzy hyperbox methods
├── best_clustering_fh_params_advance.py # Optimized parameters for clustering-based fuzzy hyperbox methods
├── best_baslines_params.py              # Optimized parameters for baseline methods
├── explainability_analysis.py           # Explainability tools for fuzzy hyperbox methods
├── visualize_hyperboxes.py              # Fuzzy hyperbox visualization tools
└── README.md                            # Project description
```

## Usage Examples

### Basic Fuzzy Hyperbox DES Usage

```python
from my_deslib.des import DESFH
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Create classifier pool
base_clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
pool_classifiers = BaggingClassifier(base_clf, n_estimators=10, random_state=42)
pool_classifiers.fit(X_train, y_train)

# Initialize DESFH with fuzzy hyperbox parameters
desfh = DESFH(pool_classifiers, k=7, theta=0.1, mu=0.9, mis_sample_based=True)

# Fit the method
desfh.fit(X_dsel, y_dsel)

# Evaluate performance
print("DES-FH Accuracy:", desfh.score(X_test, y_test))