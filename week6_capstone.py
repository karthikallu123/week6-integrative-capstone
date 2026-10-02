"""
Week 6 - Integrative Capstone Project and Evaluation
Titanic End-to-End Data Science Pipeline

Stages:
1. Data acquisition/input and preprocessing
2. Exploratory data analysis
3. Unsupervised learning: K-Means clustering
4. Supervised learning: Logistic Regression + Random Forest
5. Deep learning: PyTorch neural network
6. Evaluation, comparison, insights, limitations and next steps

Install:
    pip install -r requirements.txt
Run:
    python week6_capstone.py
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, silhouette_score, confusion_matrix, roc_curve
)

# -------------------- 1. Load and preprocess --------------------
df = pd.read_csv("titanic_cleaned.csv")
df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)
df["Survived"] = df["Survived"].astype(int)

# -------------------- 2. EDA --------------------
print(df.describe(include="all").transpose())
print("\nSurvival rate:", df["Survived"].mean())

# -------------------- 3. Unsupervised learning --------------------
cluster_features = [
    "Age", "Fare", "Pclass", "SibSp", "Parch",
    "FamilySize", "IsAlone", "CabinKnown"
]
Z = df[cluster_features].copy()
Z = Z.fillna(Z.median())
cluster_scaler = StandardScaler()
Z_scaled = cluster_scaler.fit_transform(Z)

silhouette_scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Z_scaled)
    silhouette_scores[k] = silhouette_score(Z_scaled, labels)

# k=5 is retained for continuity and interpretability from Week 3.
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(Z_scaled)

print("\nCluster silhouette scores:", silhouette_scores)
print("\nCluster profiles:")
print(df.groupby("Cluster")[cluster_features + ["Survived"]].mean())

pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(Z_scaled)

# -------------------- 4. Supervised learning --------------------
features = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "CabinKnown", "Title", "FamilySize",
    "IsAlone", "FarePerPerson"
]
X = df[features]
y = df["Survived"]

numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "CabinKnown", "FamilySize", "IsAlone", "FarePerPerson"
]
categorical_features = ["Sex", "Embarked", "Title"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

logistic = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

random_forest = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    ))
])

models = {"Logistic Regression": logistic, "Random Forest": random_forest}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1:", f1_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, prob))

# -------------------- 5. Deep learning --------------------
# The PyTorch implementation used in Week 5 is included separately
# in this capstone repository as week5_deep_learning.py.
# Its reported held-out test metrics are included in the final comparison.

print("\nDeep learning reference: see week5_deep_learning.py and training_history.csv")

# -------------------- 6. Save capstone data --------------------
df.to_csv("titanic_capstone_analysis.csv", index=False)
