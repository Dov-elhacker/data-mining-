#  Data Mining Course Project – Credit Risk Assessment

**Team Project** – Faculty of computer and data science / Data Mining Course  

This project implements a complete data mining pipeline on a real‑world **credit risk dataset**. The work includes exploratory analysis, preprocessing, K‑Medoids & hierarchical clustering, a fuzzy logic inference system, and genetic algorithm optimization.

---

##  Team Members

| # | Name |
|---|------|
| 1 | David Wageh | 
| 2 | Ahmed Ehab | 
| 3 | Peter Fadel |
| 4 | Ahmed Mohamed Saied |
| 5 | ElHassan Aly |
| 6 | Zeyad Mohamed |
| 7 | Omar Helal |
| 8 | Mohamed Hamed |
| 9 | Mazen Mohamed |
| 10 | Ahmed Hassan |



---

##  Dataset

- **Source:**[(https://www.kaggle.com/datasets/laotse/credit-risk-dataset)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Description:** Loan application data containing demographic, financial, and credit history attributes.
- **Size:** >1000 instances, 6+ features (numeric + categorical)
- **Domain:** Finance / Banking – predicting credit risk and loan default probability.

---

# 🏦 Credit Risk Assessment System

> An intelligent credit risk evaluation pipeline combining **Fuzzy Logic**, **Genetic Algorithm-based Feature Selection**, **K-Medoids Clustering**, and **Hierarchical Clustering** to assess and classify loan applicants by risk profile.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Fuzzy Logic Inference](#3-fuzzy-logic-inference)
  - [4. Genetic Algorithm – Feature Selection](#4-genetic-algorithm--feature-selection)
  - [5. K-Medoids Clustering](#5-k-medoids-clustering)
  - [6. Hierarchical Clustering](#6-hierarchical-clustering)
  - [7. System Integration](#7-system-integration)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional credit scoring models often rely on rigid thresholds that fail to capture the nuanced nature of financial risk. This project introduces a **hybrid intelligent system** that:

- Applies **Fuzzy Logic** to model the inherent uncertainty in creditworthiness evaluation.
- Uses a **Genetic Algorithm (GA)** to evolve an optimal feature subset for clustering.
- Segments applicants into distinct risk profiles via **K-Medoids** and **Agglomerative Hierarchical Clustering**.
- Combines all components into a unified decision-support system that outputs a risk score, risk category, cluster assignment, and actionable recommendation.

---

## Architecture

```text
┌──────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Raw Dataset │────▶│  Preprocessing &   │────▶│  Cleaned Dataset │
│  (CSV)       │     │  Feature Encoding  │     │  (CSV)           │
└──────────────┘     └────────────────────┘     └────────┬─────────┘
                                                         │
                     ┌───────────────────────────────────┐│
                     │                                    ▼│
              ┌──────┴───────┐                   ┌────────┴────────┐
              │ Fuzzy Logic  │                   │ Genetic Algorithm│
              │ Inference    │                   │ Feature Selection│
              │ Engine       │                   └────────┬─────────┘
              └──────┬───────┘                            │
                     │                            ┌───────▼────────┐
                     │                            │   K-Medoids &  │
                     │                            │  Hierarchical  │
                     │                            │   Clustering   │
                     │                            └───────┬────────┘
                     │                                    │
                     └──────────┬──────────────────┬──────┘
                                ▼                  ▼
                     ┌──────────────────────────────────┐
                     │   Final System Implementation    │
                     │  (Risk Score + Cluster + Action) │
                     └──────────────────────────────────┘

credit-risk-assessment/
│
├── 📊 Data Files
│   ├── credit_risk_dataset.csv            # Original raw dataset
│   ├── cleaned_data.csv                   # Preprocessed & encoded dataset
│   └── ga_selected_features2.csv          # GA-optimized feature subset for clustering
│
├── 📓 Individual Stage Notebooks
│   ├── Preprocessing.ipynb                # Data cleaning, outlier removal, encoding
│   ├── visualiztion.ipynb                 # Exploratory data analysis & visualizations
│   ├── Fuzzy.ipynb                        # Fuzzy logic membership functions & rules
│   ├── Genetic2.ipynb                     # Genetic algorithm for feature selection
│   ├── K-medoid.ipynb                     # K-Medoids clustering & evaluation
│   └── hierarchal_clustering.ipynb        # Agglomerative clustering & dendrogram analysis
│
├── 📓 Integrated Pipeline
│   └── Final_Implementation.ipynb         # End-to-end system combining all components
│
└── README.md

```


## Dataset
The project uses a public Credit Risk Dataset containing 32,581 records and features including:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `person_age` | Numeric | Applicant's age |
| `person_income` | Numeric | Annual income |
| `person_emp_length` | Numeric | Employment length (years) |
| `loan_amnt` | Numeric | Requested loan amount |
| `loan_int_rate` | Numeric | Loan interest rate (%) |
| `loan_percent_income` | Numeric | Loan-to-income ratio |
| `cb_person_cred_hist_length` | Numeric | Length of credit history (years) |
| `loan_intent` | Categorical | Purpose of the loan |
| `loan_grade` | Categorical | Assigned loan grade (A–G) |
| `person_home_ownership` | Categorical | Home ownership status |
| `cb_person_default_on_file` | Binary | Historical default flag |
| `loan_status` | Binary | **Target** — 0 = Non-default, 1 = Default |

---

## Methodology

### 1. Data Preprocessing
**Notebook:** `Preprocessing.ipynb`

* Imputation of missing values in `person_emp_length` and `loan_int_rate`.
* Outlier removal (e.g., unrealistic ages > 80, employment lengths > 60).
* One-hot encoding of categorical variables (`loan_intent`, `loan_grade`, `person_home_ownership`, `cb_person_default_on_file`).
* Standard scaling of numeric features.
* Exports `cleaned_data.csv`.

### 2. Exploratory Data Analysis
**Notebook:** `visualiztion.ipynb`

* Age, income, and loan-amount distributions.
* Default-rate analysis across loan intents, grades, and home-ownership categories.
* Correlation heatmaps and pair plots.

### 3. Fuzzy Logic Inference
**Notebook:** `Fuzzy.ipynb`

Defines a Mamdani-type fuzzy inference system using `scikit-fuzzy`:

| Variable | Fuzzy Sets |
| :--- | :--- |
| **Age** | Young · Middle-Aged · Senior |
| **Income** | Low · Medium · High |
| **Loan Amount** | Small · Medium · Large |
| **Credit History** | Short · Medium · Long |
| **Risk (output)** | Safe · Medium · Risky |

11 expert rules map input combinations to risk levels (e.g., low income AND large loan → risky). Defuzzification produces a continuous risk score in [0, 1].

### 4. Genetic Algorithm – Feature Selection
**Notebook:** `Genetic2.ipynb`

| GA Parameter | Value |
| :--- | :--- |
| **Population size** | 10 |
| **Generations** | 10 |
| **Selection** | Tournament (k = 3) |
| **Crossover** | Single-point |
| **Mutation** | Bit-flip (p = 0.2) |
| **Fitness** | 0.6 × Silhouette + 0.4 × Risk Separation |

The GA evolves binary masks over all features to find the subset that maximises clustering quality and inter-cluster default-rate separation. The best subset is persisted in `ga_selected_features2.csv`.

**Selected Features:**
`loan_percent_income`, `loan_intent_HOMEIMPROVEMENT`, `loan_grade_B`, `loan_grade_C`, `loan_grade_E`, `cb_person_default_on_file_Y`

### 5. K-Medoids Clustering
**Notebook:** `K-medoid.ipynb`

* **Algorithm:** K-Medoids (`sklearn_extra`) with Manhattan distance.
* **k = 3** clusters (validated via silhouette analysis).
* Model persisted as `kmedoids_model.pkl`.
* Accepts real-time user input, encodes & scales it, and predicts cluster membership.

### 6. Hierarchical Clustering
**Notebook:** `hierarchal_clustering.ipynb`

* Ward-linkage agglomerative clustering on the same GA-selected features.
* Dendrogram analysis to validate the 3-cluster structure.
* Cluster profiling yields three interpretable segments:

| Cluster | Label | Profile |
| :--- | :--- | :--- |
| **0** | The Stable Majority | Moderate income, can afford larger loans, financially strong |
| **1** | The High-Leverage Borrowers | Lower interest rates, trusted by banks, conservative loan sizes |
| **2** | The Elite Low-Risk Niche | Longest credit histories, smallest loans, highest approval likelihood |


### 7. System Integration
**Notebook:** `Final_Implementation.ipynb`
*The function system_implementation(record) combines all components:

Input Record → Fuzzy Score → Risk Category → Cluster Assignment → Recommendation

**Output Example:** 

{
  "fuzzy_risk_score": 0.452,
  "risk_category": "Medium Risk",
  "cluster": 0,
  "recommendation": "Review manually"
}

| Risk Score Range | Category | Action |
| :--- | :--- | :--- |
| < 0.40 | Low Risk |  Approve |
| 0.40 – 0.70 | Medium Risk |  Review manually |
| > 0.70 | High Risk |  Reject / require collateral |



---


## Installation

### Prerequisites
* Python 3.9+
* Jupyter Notebook or JupyterLab


# Clone the repository
```bash
git clone https://github.com/<your-username>/credit-risk-assessment.git
cd credit-risk-assessment
```

# Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

```

# Install dependencies
```bash
pip install -r requirements.txt

```
# Requirements:
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scikit-fuzzy
scikit-learn-extra
scipy
joblib
```


## Usage

### Run the Full Pipeline

```bash
jupyter notebook Final_Implementation.ipynb
```

---

### The notebook will execute all cells sequentially:
```text
Load and preprocess the raw data.
Build and evaluate the fuzzy inference engine.
Run the genetic algorithm for feature selection.
Train and persist the K-Medoids model.
Perform hierarchical clustering and profiling.
Expose system_implementation() for single-record inference.
```
---

### Run Individual Stages

---

## Open any of the standalone notebooks to inspect or modify a specific stage:

```bash
jupyter notebook Preprocessing.ipynb
jupyter notebook visualiztion.ipynb
jupyter notebook Fuzzy.ipynb
jupyter notebook Genetic2.ipynb
jupyter notebook K-medoid.ipynb
jupyter notebook hierarchal_clustering.ipynb
```
---

### Predict for a New Applicant

```python

record = {
    'person_age': 28,
    'person_income': 55000,
    'loan_amnt': 12000,
    'person_credit_history': 5,
    'loan_intent': 'PERSONAL',
    'loan_grade': 'C',
    'home_ownership': 'RENT',
    'default_on_file': 'N'
}

result = system_implementation(record)
print(result)

```
---

## Results

| Metric | Value |
|--------|-------|
| K-Medoids Silhouette Score | Evaluated in `K-medoid.ipynb` |
| GA Best Fitness | 0.6 × Silhouette + 0.4 × Risk Sep |
| Number of Clusters | 3 (validated by dendrogram & silhouette) |
| Fuzzy Rules | 11 expert-defined |

The system successfully segments applicants into three distinct risk profiles and provides actionable, interpretable recommendations for each loan application.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m "Add my feature"`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with ❤️ for smarter credit decisions.</i>
</p>
