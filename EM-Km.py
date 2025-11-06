from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets.iloc[:, 0].to_numpy()

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans_labels = kmeans.fit_predict(X_std)

# EM using Gaussian Mixture Models
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
gmm_labels = gmm.fit_predict(X_std)

# Evaluation metrics
km_ari = adjusted_rand_score(y, kmeans_labels)
em_ari = adjusted_rand_score(y, gmm_labels)
km_sil = silhouette_score(X_std, kmeans_labels, metric="euclidean")
em_sil = silhouette_score(X_std, gmm_labels, metric="euclidean")

print("=== K-means ===")
print(f"K-means Silhouette: {km_sil:.4f}")
print(f"ARI: {km_ari:.4f}")

print("\n=== Gaussian Mixture (EM) ===")
print(f"GMM Silhouette   : {em_sil:.4f}")
print(f"ARI: {em_ari:.4f}")

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(12, 5))

# K-means plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=40)
plt.title("K-means Clustering on Iris (PCA projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# GMM plot
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="viridis", s=40)
plt.title("Gaussian Mixture (EM) on Iris (PCA projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

# Saving results to CSV
y_index, class_names = pd.factorize(iris.data.targets.iloc[:, 0])

results = pd.DataFrame(
    {
        "true_label": y_index,
        "flower_name": y,
        "kmeans_label": kmeans_labels,
        "gmm_label": gmm_labels,
    }
)
results.to_csv("iris_clusters_sklearn.csv", index=False)
print('\nSaved "iris_clusters_sklearn.csv" in the current directory.')
