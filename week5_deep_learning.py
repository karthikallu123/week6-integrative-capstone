"""
Week 5 - Deep Learning Application in Data Science
Titanic Survival Classification using PyTorch.

Install:
    pip install -r requirements.txt

Run:
    python week5_deep_learning.py
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df = pd.read_csv("titanic_cleaned.csv")
df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)

features = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "CabinKnown", "Title", "FamilySize",
    "IsAlone", "FarePerPerson"
]

X = df[features].copy()
y = df["Survived"].astype(np.float32)

X = pd.get_dummies(
    X, columns=["Sex", "Embarked", "Title"], drop_first=False, dtype=float
)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20,
    random_state=42, stratify=y_train_full
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_np = y_test.values.astype(int)

class TitanicNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

model = TitanicNet(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

best_loss = float("inf")
best_state = None
patience = 0

for epoch in range(1, 151):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()

    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    if patience >= 15:
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    probabilities = torch.sigmoid(model(X_test)).numpy().ravel()

predictions = (probabilities >= 0.5).astype(int)

print("Test Accuracy:", accuracy_score(y_test_np, predictions))
print("Precision:", precision_score(y_test_np, predictions))
print("Recall:", recall_score(y_test_np, predictions))
print("F1-score:", f1_score(y_test_np, predictions))
print("ROC-AUC:", roc_auc_score(y_test_np, probabilities))
