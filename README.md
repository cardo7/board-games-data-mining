# Board Games Data Mining

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0?style=for-the-badge)

Final project for the **Data Mining 1** course (University of Pisa, A.Y. 2025/2026).

**Authors:** Pietro Gagliardi · Riccardo Ranieri · Giambattista Bernardo

The project analyses a dataset of **21,925 board games** from BoardGameGeek, covering the full data mining pipeline: data understanding and preparation, clustering, classification, regression and pattern mining.

📄 **Full report:** [`report/board_games_data_mining_report.pdf`](report/board_games_data_mining_report.pdf)

---

## Repository structure

```
.
├── data/
│   ├── raw/board_games.csv                  # original dataset (21,925 games, 46 attributes)
│   ├── processed/board_games_clean.csv      # cleaned dataset (20,773 games, 38 attributes)
│   └── feature_description.txt              # description of the attributes
├── notebooks/
│   ├── 01_data_understanding_preparation.ipynb
│   ├── 02_clustering_kmeans.ipynb
│   ├── 03_clustering_dbscan.ipynb           # see note below
│   ├── 04_clustering_hierarchical.ipynb     # see note below
│   ├── 05_regression.ipynb
│   ├── 06_pattern_mining.ipynb
│   ├── 07_classification_knn_naive_bayes.ipynb
│   └── 08_classification_decision_tree.ipynb
└── report/                                  # final report (PDF)
```

The notebooks are commented in Italian; the report is in English.

## Data understanding and preparation

- **Placeholders:** the `Rank:` attributes use the sentinel value 21,926 for games outside the category, and `GameWeight`/`ComWeight` use 0 for missing ratings (the real scale goes from 1 to 5).
- **Distributions:** popularity and playtime attributes are strongly right-skewed with long tails; most outliers reflect the domain (very popular or very long games) rather than errors.
- **Removed attributes:** text and image fields, variables with more than 92% missing values (`GoodPlayers`, `BestPlayers`), a constant column (`NumComments`) and redundant ones (correlation ≥ 0.92 with another attribute, e.g. `GameWeight` vs `ComWeight`).
- **Cleaning and transformations:** games published before 1950 (406 rows) and inconsistent playtimes were removed; skewed counts were log-transformed; `Family` became the boolean `belong_to_family`; `Rating` (Low/Medium/High) was encoded as an ordinal 1–3 variable.

## Clustering

Three algorithms were compared on the same subset of features.

| Algorithm | Configuration | Silhouette | Outcome |
|---|---|---:|---|
| **K-Means** | k = 6 (elbow + silhouette on k ∈ [2, 50]) | **0.179** | six balanced and interpretable clusters |
| DBSCAN | ε = 1.5, min_samples = 40 | 0.12 | two clusters and 29% noise: no clear dense regions |
| Hierarchical (Ward) | dendrogram on 2,000 K-Means micro-clusters, cut at 6 | 0.13 | overlapping clusters, one very small |

> **Notes on the notebooks.** The tables in this README report the final configurations described in the report. The DBSCAN notebook uses ε = 1.4 and min_samples = 100 (plus a second attempt on a different feature set), while the report describes a grid search leading to ε = 1.5 and min_samples = 40. The hierarchical notebook follows the same procedure as the report (K-Means with k = 2,000 to pick representative points, then Ward linkage), but with its own feature selection and a cut at 3 clusters instead of 6.

**K-Means** gave the best result. The six clusters correspond to recognisable market segments:

- **Cluster 1 – Family & party games:** lowest complexity, more players, recommended age around 8.
- **Cluster 3 – Established games:** low complexity but very high ownership and wishlist counts.
- **Cluster 4 – Vintage games:** older games with medium-low complexity, owned but rarely wished for.
- **Cluster 2 – Modern hobby games:** recent (around 2014), more complex and longer.
- **Cluster 5 – Strategy blockbusters:** high complexity and the highest popularity.
- **Cluster 0 – Hardcore games:** maximum complexity and playtime, dominated by war games.

## Classification

The target is the game `Rating` (Low / Medium / High), with a stratified 60/40 train-test split; hyperparameters are tuned with grid search and 5-fold cross-validation on the training set only. K-NN and Naive Bayes use `PassedYears`, `ComWeight`, `belong_to_family`, `NumOwned` and `NumWish`; the Decision Tree uses `PassedYears`, `ComWeight`, `ComAgeRec`, `NumOwned` and `NumWish`.

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| **K-NN** (k = 100, Manhattan distance) | **0.68** | **0.68** |
| Decision Tree (Gini, max_depth = 12, min_samples_leaf = 50) | 0.67 | 0.67 |
| Gaussian Naive Bayes | 0.61 | 0.60 |

K-NN gives the best result, with the Decision Tree very close; Naive Bayes is weaker because the features are correlated, while it assumes they are independent. The *Medium* class is the hardest to predict because it overlaps with the other two, which reflects how subjective ratings are.

Code: [`07_classification_knn_naive_bayes.ipynb`](notebooks/07_classification_knn_naive_bayes.ipynb), [`08_classification_decision_tree.ipynb`](notebooks/08_classification_decision_tree.ipynb).

## Regression

Target: `NumWish` (log-transformed).

| Model | Features | R² (test) |
|---|---|---:|
| Linear / Ridge | `NumOwned` | 0.647 |
| Linear / Ridge | `NumOwned`, `ComWeight`, `YearPublished` | 0.745 |
| Decision Tree | same three features | 0.765 |
| **K-NN** (k = 50, Manhattan distance) | same three features | **0.782** |

Ownership alone explains about 65% of the variance in wishlist counts. Adding complexity and year of publication brings it to 75% with linear models and 78% with K-NN, which suggests non-linear effects. Lasso with its default penalty underfits (R² ≈ 0.49), but recovers the linear result once the penalty is tuned with cross-validation.

## Pattern mining

After discretising all attributes, frequent, closed and maximal itemsets were extracted with Apriori (minimum support 4%, at least 4 items), and FP-Growth gave the same itemsets. The most frequent patterns describe easy, short games for children and groups. Among the association rules (confidence > 70%, ranked by lift), very long games for 1–2 players aged 14+ are war games 93% of the time, with a lift of about 6.

Code: [`06_pattern_mining.ipynb`](notebooks/06_pattern_mining.ipynb) (uses the [`pyfim`](https://borgelt.net/pyfim.html) library).

Run `01_data_understanding_preparation.ipynb` first: it creates `data/processed/board_games_clean.csv`, which the other notebooks read. The cleaned file is already included, so the other notebooks can also be run on their own.
