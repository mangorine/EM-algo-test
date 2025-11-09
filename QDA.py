import numpy as np
from scipy.stats import multivariate_normal

class CustomQDA:
    """Custom implementation of Quadratic Discriminant Analysis"""
    
    def __init__(self):
        self.classes_ = None
        self.means_ = {}
        self.covariances_ = {}
        self.priors_ = {}
    
    def fit(self, X, y):
        """Fit QDA model"""
        self.classes_ = np.unique(y)
        n_samples = len(y)
        
        for cls in self.classes_:
            X_cls = X[y == cls]
            
            # Calculate class statistics
            self.means_[cls] = np.mean(X_cls, axis=0)
            self.covariances_[cls] = np.cov(X_cls.T)
            self.priors_[cls] = len(X_cls) / n_samples
    
    def predict(self, X):
        """Predict class labels"""
        predictions = []
        
        for x in X:
            posteriors = {}
            
            for cls in self.classes_:
                # Calculate log posterior probability
                log_prior = np.log(self.priors_[cls])
                log_likelihood = multivariate_normal.logpdf(
                    x, self.means_[cls], self.covariances_[cls]
                )
                posteriors[cls] = log_prior + log_likelihood
            
            # Predict class with highest posterior
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        probabilities = []
        
        for x in X:
            log_posteriors = {}
            
            for cls in self.classes_:
                log_prior = np.log(self.priors_[cls])
                log_likelihood = multivariate_normal.logpdf(
                    x, self.means_[cls], self.covariances_[cls]
                )
                log_posteriors[cls] = log_prior + log_likelihood
            
            # Convert to probabilities
            max_log_post = max(log_posteriors.values())
            posteriors = {cls: np.exp(log_post - max_log_post) 
                         for cls, log_post in log_posteriors.items()}
            
            total = sum(posteriors.values())
            posteriors = {cls: prob/total for cls, prob in posteriors.items()}
            
            probabilities.append([posteriors[cls] for cls in self.classes_])
        
        return np.array(probabilities)
