from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """
    A class for handling text embedding models.

    Attributes:
    model (object): The embedding model to use. If None, a default model will be used.
    """

    def __init__(self, model=None, **kwargs):
        """
        Initialize the EmbeddingModel.

        Parameters:
        model (object): The embedding model to use. If None, a default model will be used.
        **kwargs: Additional keyword arguments to be passed to the embedding model.
        """
        if model is None:
            self.model = SentenceTransformer('bert-base-nli-mean-tokens', **kwargs)
        else:
            self.model = model

    def encode(self, corpus):
        """
        Encode the given corpus using the embedding model.

        Parameters:
        corpus (list): A list of strings representing the corpus to be encoded.

        Returns:
        corpus_embeddings (numpy array): An array of embeddings for the input corpus.
        """
        corpus_embeddings = self.model.encode(corpus)
        return corpus_embeddings
