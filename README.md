# Recipe for Rating: Predict Food Ratings using ML

A machine learning project that predicts food recipe ratings (0–5 scale) from user interaction data and review text, built as part of an ML competition on Kaggle.

---

## Problem Statement

Given a dataset of 13,636 food recipe reviews with features like user reputation, thumbs up/down counts, reply counts, best scores, and free-text recipe reviews, the goal is to predict the rating a user would assign to a recipe.

---

## Dataset

| Split | Rows | Features |
|-------|------|----------|
| Train | 13,636 | 14 features + 1 label (`Rating`) |
| Test | 4,546 | 14 features |

**Key features used:**
- `UserReputation`, `ReplyCount`, `ThumbsUpCount`, `ThumbsDownCount`, `BestScore`
- `Recipe_Review` (text feature)
- `CreationTimestamp` (engineered into `HourOfRating`, `DayOfRating`)

---

## Approach

### 1. Exploratory Data Analysis
- Descriptive statistics and distribution analysis across all numerical features
- Correlation heatmap to assess feature-label relationships
- Scatter plots to identify linear dependencies with `Rating`

**Key finding:** Rating is heavily skewed toward 5 (majority class), with weak direct correlations to numerical features — pointing toward the importance of text features.

### 2. Data Preprocessing
- Dropped identifier columns (`ID`, `RecipeNumber`, `RecipeCode`, `UserID`, `UserName`, etc.)
- Imputed 2 missing values in `Recipe_Review` with empty strings
- Confirmed zero duplicate rows
- Engineered `HourOfRating` and `DayOfRating` from Unix timestamp

### 3. Feature Engineering & Encoding
- Applied **TF-IDF Vectorization** on `Recipe_Review` text
- Scaled numerical features using **StandardScaler**
- Combined sparse TF-IDF matrix with dense numerical features via `scipy.sparse.hstack`

### 4. Model Training
Three classifiers trained and evaluated:

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| Logistic Regression | 76.94% | **77.54%** |
| K-Nearest Neighbors | 75.92% | 76.31% |
| Random Forest | 76.32% | 76.99% |

### 5. Hyperparameter Tuning
Used **5-fold GridSearchCV** for each model:

- **Logistic Regression:** `C ∈ {0.1, 1, 10}`, `penalty = l2`
- **KNN:** `n_neighbors ∈ {3, 5, 7, 9, 11}`
- **Random Forest:** `n_estimators = 100`, `max_depth ∈ {None, 10, 20}`, `min_samples_split ∈ {2, 5, 10}`

**Best model after tuning: Logistic Regression at 77.54% test accuracy**

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn (LogisticRegression, KNeighborsClassifier, RandomForestClassifier) |
| Text Features | TF-IDF Vectorizer |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Confusion Matrix, Classification Report, GridSearchCV |

---

## Project Structure

```
recipe-for-rating/
│
├── notebook.ipynb         # Full EDA, preprocessing, modelling, and evaluation
├── submission.csv         # Final predictions on test set
└── README.md
```

---

## Results Summary

- Logistic Regression outperformed both KNN and Random Forest on this dataset
- Text features from `Recipe_Review` via TF-IDF were critical for classification performance
- Hyperparameter tuning produced marginal gains, suggesting the feature set is the primary performance lever
- Class imbalance (majority of ratings = 5) remains the main challenge for further improvement

---

## Future Improvements

- Apply SMOTE or class-weight balancing to address rating skew
- Experiment with sentence embeddings (e.g. `sentence-transformers`) instead of TF-IDF
- Try gradient boosting models (XGBoost, LightGBM) for potential accuracy gains
- Explore ensemble stacking across the three trained models
