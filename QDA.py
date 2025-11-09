from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from time import time

# Fetch Wine dataset
wine = fetch_ucirepo(id=109)

# Data preparation
X = wine.data.features
y = wine.data.targets.iloc[:, 0].to_numpy()

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)

# QDA implementation
start = time()
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
end = time()
qda_time = end - start

# LDA for comparison
start = time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
end = time()
lda_time = end - start

# Evaluation metrics
qda_acc = accuracy_score(y_test, qda_pred)
lda_acc = accuracy_score(y_test, lda_pred)

print("=== Quadratic Discriminant Analysis ===")
print(f"QDA training time: {qda_time:.4f} seconds")
print(f"QDA accuracy: {qda_acc:.4f}")

print("\n=== Linear Discriminant Analysis ===")
print(f"LDA training time: {lda_time:.4f} seconds")
print(f"LDA accuracy: {lda_acc:.4f}")

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

y_index, class_names = pd.factorize(y_test)

plt.figure(figsize=(12, 4))

# QDA predictions plot
plt.subplot(1, 3, 1)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=qda_pred, cmap="viridis", s=40)
plt.title("QDA Predictions on Wine (PCA projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# LDA predictions plot
plt.subplot(1, 3, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=lda_pred, cmap="viridis", s=40)
plt.title("LDA Predictions on Wine (PCA projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# True labels plot
plt.subplot(1, 3, 3)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_index, cmap="viridis", s=40)
plt.title("True Labels of Wine (PCA projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

# Save results
results = pd.DataFrame({
    "true_label": y_index,
    "wine_class": y_test,
    "qda_prediction": qda_pred,
    "lda_prediction": lda_pred,
    "qda_correct": (y_test == qda_pred).astype(int),
    "lda_correct": (y_test == lda_pred).astype(int)
})

results.to_csv("wine_qda_results.csv", index=False)
print('\nSaved "wine_qda_results.csv" in the current directory.')

# Print classification reports
print("\n=== QDA Classification Report ===")
print(classification_report(y_test, qda_pred))

print("\n=== LDA Classification Report ===")
print(classification_report(y_test, lda_pred))
