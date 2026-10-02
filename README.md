# Week 6 - Integrative Capstone Project and Evaluation

## Project
**End-to-End Titanic Data Science Pipeline using Python**

This capstone integrates the major skills developed during Weeks 1-5:
- Data acquisition and preprocessing
- Exploratory data analysis and visualization
- Unsupervised learning with K-Means
- Supervised learning with Logistic Regression and Random Forest
- Deep learning with a PyTorch neural network
- Model evaluation, interpretation, limitations and next steps

## Dataset
Titanic: Machine Learning from Disaster. The cleaned dataset used throughout the internship contains 891 passenger records and 13 original columns.

## Main question
Can passenger characteristics be used to identify patterns and build predictive models for the historical Titanic survival outcome, while also discovering data-driven passenger segments?

## Key results
- Overall observed survival rate: 38.38%
- K-Means selected solution: k=5, silhouette score 0.449
- Logistic Regression: 83.24% accuracy, 0.872 ROC-AUC
- Random Forest: 80.45% accuracy, 0.847 ROC-AUC
- PyTorch neural network: 84.36% accuracy, 0.857 ROC-AUC

## Reproduce
```bash
pip install -r requirements.txt
python week6_capstone.py
python week5_deep_learning.py
```

## Important interpretation note
These models describe patterns in the historical Titanic dataset. They do not establish causal relationships, and their performance should not be assumed to generalize to other populations or modern data without validation.
