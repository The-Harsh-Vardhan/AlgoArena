# 🚀 Tabular Data Algorithms - Model Zoo for the Adult Income Dataset

Welcome to the jungle, where algorithms compete to predict income brackets 🧠💼. This notebook (`1_Tabular_Data_Algorithms.ipynb`) showcases a lineup of popular machine learning models applied to the **Adult Income dataset**, a classic in the tabular ML world.

---

## 📊 Dataset: Adult Income

- **Source**: UCI Machine Learning Repository
- **Goal**: Predict whether a person earns `>50K` or `<=50K` per year.
- **Features** include: age, workclass, education, occupation, relationship status, hours-per-week, and more.

---

## 🛠️ Steps Performed

### 1. **Imports Galore**
All essential libraries for preprocessing, modeling, evaluation, and plotting are imported. Includes:
- `scikit-learn` for classic models
- `xgboost`, `lightgbm`, `catboost` for gradient boosting
- `tensorflow.keras` for building a simple ANN
- `json` to export results for dashboarding

---

### 2. **Loading the Dataset**
The dataset is loaded from the path:
```python
df = pd.read_csv("tabular/adult.csv")
```

---

### 3. **Data Cleaning & Preprocessing**
- Replaces `"?"` with `NaN` and drops those rows.
- Strips whitespace from string values (because `" Private"` ≠ `"Private"`).
- Uses **Label Encoding** on categorical features.
- Scales features using **StandardScaler** for model compatibility.

---

### 4. **Splitting Data**
The data is split into:
- **Training set**: 80%
- **Test set**: 20%

Scaling ensures better performance for models like KNN, SVM, and ANN.

---

### 5. **Model Evaluation Function**
A helper function `evaluate_model()`:
- Predicts on test data
- Calculates: Accuracy, Precision, Recall, F1-score
- Stores everything in a global `results` dictionary

---

### 6. **Training Models**
A total of **10 models** are trained:

| Model              | Notes                         |
|-------------------|-------------------------------|
| Logistic Regression | Baseline linear model        |
| Decision Tree       | Simple tree-based classifier |
| Random Forest       | Ensemble of trees            |
| K-Nearest Neighbors | Lazy learner                 |
| Naive Bayes         | Probabilistic classifier     |
| Support Vector Machine | With default RBF kernel   |
| XGBoost             | Extreme gradient boosting    |
| LightGBM            | Gradient boosting with speed |
| CatBoost            | Handles categorical features |
| Artificial Neural Network (ANN) | Built with Keras |

Each one is trained using:
```python
model.fit(X_train, y_train)
evaluate_model("Model Name", model, X_test, y_test)
```

---

### 7. **ANN Model (Keras)**
A shallow feed-forward neural network:
```python
Input → Dense(64) → Dense(32) → Output(sigmoid)
```
- Binary Crossentropy loss
- Adam optimizer
- Accuracy as metric
- Trained for 10 epochs silently

Prediction is thresholded at 0.5:
```python
y_pred_ann = (ann.predict(X_test) > 0.5).astype("int32")
```

---

### 8. **Saving Results**
All model scores are exported to:
```
tabular/tabular_results.json
```
This is used later in the `2_Tabular_Streamlit_Dashboard.py` file to visualize and compare model performance.

---

## 📦 Sample JSON Output
```json
{
  "Random Forest": {
    "accuracy": 0.86,
    "precision": 0.78,
    "recall": 0.75,
    "f1_score": 0.76
  },
  ...
}
```

---

## 📈 Final Output
All models are ranked by accuracy and their metrics are displayed in a clean table:
```python
pd.DataFrame(results).T.sort_values("accuracy", ascending=False)
```

---

## 🤖 Why This Matters
This notebook:
- Teaches practical end-to-end tabular ML workflow
- Provides model benchmarking on real-world data
- Generates reusable JSON for dashboarding
- Bridges classical ML, boosting, and deep learning in one script

---

## 📂 Folder Structure

```
project-root/
│
├── tabular/
│   ├── adult.csv
│   └── tabular_results.json
│
├── 1_Tabular_Data_Algorithms.ipynb
├── 2_Tabular_Streamlit_Dashboard.py
└── README.md ← You’re here
```

---

## 🧠 Bonus Tips
- You can easily plug in other tabular datasets by changing the CSV path and tweaking preprocessing.
- Want even more models? Try `ExtraTreesClassifier`, `AdaBoost`, or `VotingClassifier`.

---

## ✨ Author's Note
This notebook is the beating heart of your model benchmarking pipeline. It's clean, modular, and designed to feed your Streamlit dashboard with juicy metrics.

Now go tweak hyperparameters and break some records! 💪🚀