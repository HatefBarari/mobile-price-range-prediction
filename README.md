# 📱 Mobile Price Range Prediction

A multi-class classification project that predicts a mobile phone's price
range (Low, Medium, High, Very High) from its technical specifications,
using a full ML pipeline: exploratory data analysis, data-quality auditing,
cleaning, model comparison, and hyperparameter tuning.

---

## 🎯 Objective

Build a machine learning model that accurately classifies a mobile phone into
one of four price categories based on specs such as RAM, battery power,
camera resolution, screen size, and connectivity features. This is a
multi-class classification problem with a perfectly balanced target variable.

## 📊 Dataset

- **Source:** Mobile Price Classification dataset (`train.csv`, `test.csv`)
- **Target:** `price_range` — 4 balanced classes (Low, Medium, High, Very High Cost)
- **Features:** 20 technical specifications per device (RAM, battery, camera
  megapixels, screen dimensions, connectivity flags, CPU cores/speed, etc.)

> Note: add a link to the original dataset source (e.g. Kaggle) in this
> section instead of committing the raw CSV files, unless you've confirmed
> the license allows redistribution.

## 🔍 Exploratory Data Analysis — Key Findings

- Identified that several features contain **physically implausible values**,
  most notably a large number of phones listed with a screen width (`sc_w`)
  of 0–3 cm — treated as data-entry errors and corrected rather than dropped.
- Investigated zero-value front/primary camera megapixels and confirmed they
  represent genuine "no camera" devices rather than missing data.
- Found that binary features (Wi-Fi, dual SIM, touchscreen, etc.) are roughly
  evenly split and largely uncorrelated with price — while clock speed shows
  a distinct peak for older, low-end devices.
- Built a full correlation heatmap to confirm most features are close to
  independent, which explains why simple tree-based splits alone weren't
  enough to reach top performance.

## 🧹 Data Cleaning & Preprocessing

- Corrected implausible `sc_w` values using distribution-informed imputation.
- Applied consistent transformations across train and test sets to avoid
  data leakage.
- Standardized features for scale-sensitive models (SVM).

## 🤖 Modeling & Results

Three algorithms were trained and evaluated, each with a tuned counterpart
via `GridSearchCV`:

| Model                          | Baseline Accuracy | Tuned Accuracy        |
| ------------------------------ | ----------------- | --------------------- |
| Decision Tree                  | 82%               | improved after tuning |
| Random Forest                  | —                 | 89%                   |
| **SVM (Linear kernel, C=100)** | 88%               | **98%** 🏆            |

**Champion model:** Tuned **Linear SVM (C=100)**, reaching **98% accuracy**
on the held-out test set — the strong result confirms the data becomes
near linearly separable once properly scaled.

## ✅ Conclusion

Starting from raw, imperfect data, the project moved through systematic EDA,
targeted data cleaning, and iterative model tuning to land on a
high-performance linear SVM classifier, which was then retrained on the full
cleaned dataset and used to generate the final predictions for the unseen
test set.

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn`

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook mobile_price_prediction.ipynb
```

## 📄 License

MIT (or your preferred license) — add a `LICENSE` file to the repo.
