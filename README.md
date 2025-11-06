# EM-algo-test

The [Code](EM-km.py) file compares The Expectation-Maximization for Mixture fo Gaussian and the K-means algorithm, on convergence speed and accuracy.

You can find information on the dataset here [Iris Dataset Information](iris).

## EM GMM and K-means comparaisons

We use the scikit-learn librairy to implement both K-means and GMM. And use ARI and SS to compare them. Why ?

- We are not mesuring how close we are to a known value but wether our clusters are the same as the known one with no attention to actual label (we don't care if label 1 is predicted with label 0, as long as the cluster 1 is almost the same as cluster 0). So loss function like RMSE are unsuable.
- We use label permutation invariant indicators such as ARI ( Adjusted Rand Index) and SS ( Silhouette Score )
  - SS mesures compactness (-1,1)
  - ARI checks if the clusters are the same as the true one. (1 = perfect, 0 = random, <0 worse than random)

## How to run

If you have issues running the code, it may be because your python doesn't trust the HTTPS certificate from the archive's website, to fix run this command in your terminal :

/Applications/Python\ 3.13/Install\ Certificates.command

You'll also have to install the ucimlrepo, to do this run this command in your terminal:

pip install ucimlrepo

## Results

**K-means**

K-means Silhouette: 0.4590

ARI: 0.6201

**Gaussian Mixture (EM)**

GMM Silhouette : 0.3728

ARI: 0.9039

You will find the results of both algorithm here : [Result](iris_clusters_sklearn.csv)

You will also find the graph of clusters of both algorithm plotted by reducing dimensionality with PCA here :

![Alt Result Graph](Figure_1.png)

## Analysis

Looking at ARI:

- GMM has clearly better predictions of the clusters acheiving almost 90%, K-means only achieves 62%. That's because while K-means finds geometrical cleanest clusters (often results to spherical clusters) so it's limited to prediction with that kind of logic, GMM can fit and predict clusters accurately even if they are not equal variance and spherical.

- However the clusters are less dense with GMM than K-means This result is no surprise, K-means does clusters using euclidian distance, just like silhouette. On the other hand, GMM uses probalistic log-likelihood.
