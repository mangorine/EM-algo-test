import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import seaborn as sns

# Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=== Dataset Iris ===")
print(f"Nombre d'échantillons: {X.shape[0]}")
print(f"Nombre de features: {X.shape[1]}")
print(f"Classes: {target_names}")

# Standardisation des données
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# QDA avec validation Leave-One-Out (LOOCV)
print("\n=== QDA avec Leave-One-Out Cross-Validation ===")
loo = LeaveOneOut()
qda = QuadraticDiscriminantAnalysis()

# Stockage des prédictions LOOCV
loocv_predictions = []
loocv_true_labels = []

# Validation LOOCV
for train_idx, test_idx in loo.split(X_std):
    X_train_loo, X_test_loo = X_std[train_idx], X_std[test_idx]
    y_train_loo, y_test_loo = y[train_idx], y[test_idx]
    
    # Entraînement sur n-1 échantillons
    qda.fit(X_train_loo, y_train_loo)
    
    # Prédiction sur l'échantillon restant
    pred = qda.predict(X_test_loo)
    
    loocv_predictions.append(pred[0])
    loocv_true_labels.append(y_test_loo[0])

# Conversion en arrays numpy
loocv_predictions = np.array(loocv_predictions)
loocv_true_labels = np.array(loocv_true_labels)

# Calcul de l'accuracy LOOCV
loocv_accuracy = accuracy_score(loocv_true_labels, loocv_predictions)
print(f"Accuracy LOOCV: {loocv_accuracy:.4f}")

# Rapport de classification
print("\n=== Rapport de Classification LOOCV ===")
print(classification_report(loocv_true_labels, loocv_predictions, target_names=target_names))

# PCA pour la visualisation (réduction à 2D)
print("\n=== Analyse en Composantes Principales (PCA) ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Variance expliquée
explained_variance = pca.explained_variance_ratio_
print(f"Variance expliquée par PC1: {explained_variance[0]:.3f}")
print(f"Variance expliquée par PC2: {explained_variance[1]:.3f}")
print(f"Variance totale expliquée: {explained_variance.sum():.3f}")

# Entraîner QDA sur toutes les données pour la visualisation
qda_full = QuadraticDiscriminantAnalysis()
qda_full.fit(X_std, y)
full_predictions = qda_full.predict(X_std)

# Visualisation
plt.figure(figsize=(15, 5))

# Graphique 1: Données originales (projection PCA)
plt.subplot(1, 3, 1)
colors = ['red', 'green', 'blue']
for i, (target, color) in enumerate(zip(target_names, colors)):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=color, label=target, alpha=0.7, s=50)
plt.title('Dataset Iris - Vraies Classes\n(Projection PCA)')
plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
plt.legend()
plt.grid(True, alpha=0.3)

# Graphique 2: Prédictions QDA (toutes les données)
plt.subplot(1, 3, 2)
for i, (target, color) in enumerate(zip(target_names, colors)):
    mask = full_predictions == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=color, label=f'Prédit {target}', alpha=0.7, s=50)
plt.title('QDA - Prédictions\n(Projection PCA)')
plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
plt.legend()
plt.grid(True, alpha=0.3)

# Graphique 3: Matrice de confusion
plt.subplot(1, 3, 3)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(loocv_true_labels, loocv_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matrice de Confusion\n(LOOCV)')
plt.xlabel('Prédictions')
plt.ylabel('Vraies Classes')

plt.tight_layout()
plt.savefig('iris_qda_loocv_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Sauvegarde des résultats
results_df = pd.DataFrame({
    'sepal_length': X[:, 0],
    'sepal_width': X[:, 1], 
    'petal_length': X[:, 2],
    'petal_width': X[:, 3],
    'true_class': y,
    'true_class_name': [target_names[i] for i in y],
    'loocv_prediction': loocv_predictions,
    'loocv_prediction_name': [target_names[i] for i in loocv_predictions],
    'loocv_correct': (loocv_true_labels == loocv_predictions).astype(int),
    'pc1': X_pca[:, 0],
    'pc2': X_pca[:, 1]
})

results_df.to_csv('iris_qda_loocv_results.csv', index=False)
print(f'\nRésultats sauvegardés dans "iris_qda_loocv_results.csv"')

# Analyse des erreurs LOOCV
errors = results_df[results_df['loocv_correct'] == 0]
if len(errors) > 0:
    print(f"\n=== Analyse des Erreurs LOOCV ({len(errors)} erreurs) ===")
    print(errors[['true_class_name', 'loocv_prediction_name', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
else:
    print("\n=== Aucune erreur en LOOCV ! ===")
