from sklearn.cluster import KMeans

class ClusteringModel:
    """
    A class for a base clustering model.

    Attributes:
    model (object): The clustering model to use. If None, a default model will be used.
    """
     
    def __init__(self, model=None, **kwargs):
        """
        Initialize the clustering model.

        Parameters:
        model (object): The clustering model to use. If None, KMeans will be used.
        **kwargs: Additional keyword arguments to be passed to the clustering model.
        """
        if model is None:
            self.model = KMeans(**kwargs)
        else:
            self.model = model(**kwargs)

    def fit(self, X):
        """
        Fit the clustering model to the data.

        Parameters:
        X (array-like): Input data.
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Predict cluster labels for the given data.

        Parameters:
        X (array-like): Input data.

        Returns:
        labels (array): Predicted cluster labels.
        """
        return self.model.predict(X)