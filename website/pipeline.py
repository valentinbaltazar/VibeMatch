"""
ML Pipeline Module for VibeMatch
Orchestrates: Data Loading -> BERT Encoding -> KMeans Clustering -> WordCloud Generation
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import base64
from io import BytesIO
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import Counter
import re
import threading


@dataclass
class PipelineProgress:
    """Tracks progress of the ML pipeline"""
    step: str
    step_number: int
    total_steps: int
    progress: float  # 0-100
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Final result of the ML pipeline"""
    success: bool
    clusters: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    wordclouds: Dict[int, str]  # cluster_id -> base64 image
    songs_by_cluster: Dict[int, List[Dict[str, str]]]
    error: Optional[str] = None


class MLPipeline:
    """
    Orchestrates the complete ML pipeline for song clustering.

    Pipeline Steps:
    1. Data Loading - Sample lyrics from CSV
    2. BERT Encoding - Generate embeddings for lyrics
    3. KMeans Clustering - Group similar lyrics
    4. WordCloud Generation - Create visual representation of clusters
    """

    # Available models with their properties
    AVAILABLE_MODELS = {
        'all-MiniLM-L6-v2': {
            'name': 'all-MiniLM-L6-v2',
            'display_name': 'MiniLM (Fast)',
            'dimensions': 384,
            'description': 'Fast and efficient, good for demos'
        },
        'all-mpnet-base-v2': {
            'name': 'all-mpnet-base-v2',
            'display_name': 'MPNet (Balanced)',
            'dimensions': 768,
            'description': 'Best quality-speed tradeoff'
        },
        'bert-base-nli-mean-tokens': {
            'name': 'bert-base-nli-mean-tokens',
            'display_name': 'BERT NLI (Original)',
            'dimensions': 768,
            'description': 'Original BERT model used in project'
        }
    }

    def __init__(self, n_samples: int = 500, n_clusters: int = 4,
                 model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize the ML Pipeline.

        Args:
            n_samples: Number of songs to sample from dataset
            n_clusters: Number of clusters (vibes) to create
            model_name: SentenceTransformer model to use
            device: Device for model inference ('cpu' or 'cuda')
        """
        self.n_samples = min(n_samples, 5000)  # Cap at 5000 for safety
        self.n_clusters = min(max(n_clusters, 2), 15)  # Between 2-15
        self.model_name = model_name
        self.device = device

        # Pipeline state
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None

        # Progress tracking
        self.current_progress: Optional[PipelineProgress] = None
        self._progress_callback = None

        # Timing metrics
        self.timings: Dict[str, float] = {}

    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self._progress_callback = callback

    def _update_progress(self, step: str, step_number: int, progress: float,
                         message: str, data: Dict = None, completed: bool = False):
        """Update and broadcast progress"""
        self.current_progress = PipelineProgress(
            step=step,
            step_number=step_number,
            total_steps=4,
            progress=progress,
            message=message,
            data=data or {},
            completed=completed
        )
        if self._progress_callback:
            self._progress_callback(self.current_progress)

    def load_data(self) -> bool:
        """
        Step 1: Load and sample data from all_lyrics.csv

        Returns:
            bool: Success status
        """
        start_time = time.time()
        self._update_progress(
            step="Data Loading",
            step_number=1,
            progress=0,
            message="Reading dataset..."
        )

        try:
            # Read the CSV file
            data_path = './website/static/data/all_lyrics.csv'

            # First, get total row count for progress
            self._update_progress(
                step="Data Loading",
                step_number=1,
                progress=20,
                message="Counting available songs..."
            )

            # Read only necessary columns to save memory
            columns_needed = ['artist_name', 'song_title', 'corpus', 'song_url']
            df_full = pd.read_csv(data_path, usecols=columns_needed, on_bad_lines='skip')

            self._update_progress(
                step="Data Loading",
                step_number=1,
                progress=50,
                message=f"Found {len(df_full)} songs, sampling {self.n_samples}..."
            )

            # Remove rows with empty corpus
            df_full = df_full.dropna(subset=['corpus'])
            df_full = df_full[df_full['corpus'].str.len() > 50]  # Min 50 chars

            # Sample the data
            if len(df_full) > self.n_samples:
                self.df = df_full.sample(n=self.n_samples, random_state=42).reset_index(drop=True)
            else:
                self.df = df_full.reset_index(drop=True)

            self.timings['data_loading'] = time.time() - start_time

            self._update_progress(
                step="Data Loading",
                step_number=1,
                progress=100,
                message=f"Loaded {len(self.df)} songs",
                data={
                    'songs_loaded': len(self.df),
                    'time_seconds': round(self.timings['data_loading'], 2)
                },
                completed=True
            )

            return True

        except Exception as e:
            self._update_progress(
                step="Data Loading",
                step_number=1,
                progress=0,
                message=f"Error: {str(e)}",
                data={'error': str(e)}
            )
            return False

    def encode_corpus(self) -> bool:
        """
        Step 2: Encode lyrics using BERT/SentenceTransformer

        Returns:
            bool: Success status
        """
        if self.df is None:
            return False

        start_time = time.time()
        self._update_progress(
            step="BERT Encoding",
            step_number=2,
            progress=0,
            message=f"Loading {self.model_name} model..."
        )

        try:
            # Load the model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            model_info = self.AVAILABLE_MODELS.get(self.model_name, {})

            self._update_progress(
                step="BERT Encoding",
                step_number=2,
                progress=10,
                message="Model loaded, starting encoding..."
            )

            # Get corpus as list
            corpus = self.df['corpus'].tolist()

            # Encode in batches with progress updates
            batch_size = 32
            all_embeddings = []
            total_batches = (len(corpus) + batch_size - 1) // batch_size

            for i in range(0, len(corpus), batch_size):
                batch = corpus[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)

                # Calculate progress (10-90% range for encoding)
                batch_num = i // batch_size + 1
                progress = 10 + (batch_num / total_batches) * 80

                self._update_progress(
                    step="BERT Encoding",
                    step_number=2,
                    progress=progress,
                    message=f"Encoding batch {batch_num}/{total_batches}..."
                )

            self.embeddings = np.vstack(all_embeddings)
            self.timings['bert_encoding'] = time.time() - start_time

            self._update_progress(
                step="BERT Encoding",
                step_number=2,
                progress=100,
                message=f"Encoded {len(corpus)} lyrics",
                data={
                    'embeddings_shape': list(self.embeddings.shape),
                    'dimensions': self.embeddings.shape[1],
                    'model_name': self.model_name,
                    'time_seconds': round(self.timings['bert_encoding'], 2)
                },
                completed=True
            )

            return True

        except Exception as e:
            self._update_progress(
                step="BERT Encoding",
                step_number=2,
                progress=0,
                message=f"Error: {str(e)}",
                data={'error': str(e)}
            )
            return False

    def cluster(self) -> bool:
        """
        Step 3: Cluster embeddings using KMeans

        Returns:
            bool: Success status
        """
        if self.embeddings is None:
            return False

        start_time = time.time()
        self._update_progress(
            step="KMeans Clustering",
            step_number=3,
            progress=0,
            message=f"Initializing KMeans with k={self.n_clusters}..."
        )

        try:
            # Run KMeans
            self._update_progress(
                step="KMeans Clustering",
                step_number=3,
                progress=20,
                message="Running KMeans algorithm..."
            )

            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )

            self.cluster_labels = kmeans.fit_predict(self.embeddings)

            self._update_progress(
                step="KMeans Clustering",
                step_number=3,
                progress=70,
                message="Calculating cluster metrics..."
            )

            # Calculate silhouette score
            silhouette = silhouette_score(self.embeddings, self.cluster_labels)

            # Get cluster distribution
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            distribution = {int(k): int(v) for k, v in zip(unique, counts)}

            # Add cluster labels to dataframe
            self.df['cluster'] = self.cluster_labels

            self.timings['clustering'] = time.time() - start_time

            self._update_progress(
                step="KMeans Clustering",
                step_number=3,
                progress=100,
                message=f"Created {self.n_clusters} clusters",
                data={
                    'n_clusters': self.n_clusters,
                    'silhouette_score': round(silhouette, 4),
                    'distribution': distribution,
                    'inertia': round(kmeans.inertia_, 2),
                    'time_seconds': round(self.timings['clustering'], 2)
                },
                completed=True
            )

            return True

        except Exception as e:
            self._update_progress(
                step="KMeans Clustering",
                step_number=3,
                progress=0,
                message=f"Error: {str(e)}",
                data={'error': str(e)}
            )
            return False

    def generate_wordclouds(self) -> Dict[int, str]:
        """
        Step 4: Generate WordCloud for each cluster

        Returns:
            Dict mapping cluster_id to base64 encoded image
        """
        if self.df is None or 'cluster' not in self.df.columns:
            return {}

        start_time = time.time()
        self._update_progress(
            step="WordCloud Generation",
            step_number=4,
            progress=0,
            message="Generating word clouds..."
        )

        wordclouds = {}

        try:
            for i, cluster_id in enumerate(range(self.n_clusters)):
                progress = (i / self.n_clusters) * 90
                self._update_progress(
                    step="WordCloud Generation",
                    step_number=4,
                    progress=progress,
                    message=f"Generating cloud for cluster {cluster_id + 1}/{self.n_clusters}..."
                )

                # Get lyrics for this cluster
                cluster_lyrics = self.df[self.df['cluster'] == cluster_id]['corpus'].tolist()

                # Combine all lyrics
                combined_text = ' '.join(cluster_lyrics)

                # Clean text for wordcloud
                combined_text = self._clean_for_wordcloud(combined_text)

                if len(combined_text.strip()) < 10:
                    continue

                # Generate wordcloud
                wc = WordCloud(
                    width=400,
                    height=300,
                    background_color='black',
                    colormap='viridis',
                    max_words=100,
                    min_font_size=10,
                    max_font_size=80,
                    random_state=42
                ).generate(combined_text)

                # Convert to base64
                buffer = BytesIO()
                wc.to_image().save(buffer, format='PNG')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                wordclouds[cluster_id] = img_base64

            self.timings['wordcloud'] = time.time() - start_time

            self._update_progress(
                step="WordCloud Generation",
                step_number=4,
                progress=100,
                message=f"Generated {len(wordclouds)} word clouds",
                data={
                    'clouds_generated': len(wordclouds),
                    'time_seconds': round(self.timings['wordcloud'], 2)
                },
                completed=True
            )

            return wordclouds

        except Exception as e:
            self._update_progress(
                step="WordCloud Generation",
                step_number=4,
                progress=0,
                message=f"Error: {str(e)}",
                data={'error': str(e)}
            )
            return {}

    def _clean_for_wordcloud(self, text: str) -> str:
        """Clean text for better wordcloud generation"""
        # Remove common stopwords and filler words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'dare', 'ought', 'used', 'it', 'its', "it's", 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'im', "i'm", 'dont', "don't",
            'got', 'get', 'like', 'know', 'yeah', 'oh', 'uh', 'ah', 'na', 'la',
            'da', 'gonna', 'wanna', 'gotta', 'cause', "'cause", 'cuz', 'let',
            'say', 'see', 'come', 'go', 'make', 'take', 'one', 'two', 'now',
            'up', 'out', 'if', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'any', 'as', 'back', 'way', 'right', 'still', 'well', 'also',
            'verse', 'chorus', 'bridge', 'intro', 'outro', 'hook', 'pre'
        }

        # Convert to lowercase and split
        words = text.lower().split()

        # Filter words
        filtered = [
            word for word in words
            if word not in stopwords
            and len(word) > 2
            and word.isalpha()
        ]

        return ' '.join(filtered)

    def get_songs_by_cluster(self) -> Dict[int, List[Dict[str, str]]]:
        """Get songs organized by cluster"""
        if self.df is None or 'cluster' not in self.df.columns:
            return {}

        songs_by_cluster = {}
        for cluster_id in range(self.n_clusters):
            cluster_df = self.df[self.df['cluster'] == cluster_id]
            songs = []
            for _, row in cluster_df.iterrows():
                songs.append({
                    'song_title': row.get('song_title', 'Unknown'),
                    'artist_name': row.get('artist_name', 'Unknown'),
                    'song_url': row.get('song_url', '')
                })
            songs_by_cluster[cluster_id] = songs
        return songs_by_cluster

    def run(self) -> PipelineResult:
        """
        Run the complete ML pipeline

        Returns:
            PipelineResult with all outputs
        """
        # Step 1: Load data
        if not self.load_data():
            return PipelineResult(
                success=False,
                clusters=[],
                metrics={},
                wordclouds={},
                songs_by_cluster={},
                error="Failed to load data"
            )

        # Step 2: Encode corpus
        if not self.encode_corpus():
            return PipelineResult(
                success=False,
                clusters=[],
                metrics={},
                wordclouds={},
                songs_by_cluster={},
                error="Failed to encode corpus"
            )

        # Step 3: Cluster
        if not self.cluster():
            return PipelineResult(
                success=False,
                clusters=[],
                metrics={},
                wordclouds={},
                songs_by_cluster={},
                error="Failed to cluster"
            )

        # Step 4: Generate wordclouds
        wordclouds = self.generate_wordclouds()

        # Compile results
        songs_by_cluster = self.get_songs_by_cluster()

        # Build cluster info
        clusters = []
        for cluster_id in range(self.n_clusters):
            cluster_songs = songs_by_cluster.get(cluster_id, [])
            clusters.append({
                'id': cluster_id,
                'song_count': len(cluster_songs),
                'wordcloud': wordclouds.get(cluster_id, ''),
                'sample_songs': cluster_songs[:5]  # First 5 songs as sample
            })

        # Compile metrics
        metrics = {
            'total_songs': len(self.df) if self.df is not None else 0,
            'n_clusters': self.n_clusters,
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_used': self.model_name,
            'timings': self.timings,
            'total_time': sum(self.timings.values())
        }

        # Add clustering metrics from progress
        if self.current_progress and self.current_progress.data:
            if 'silhouette_score' in self.current_progress.data:
                metrics['silhouette_score'] = self.current_progress.data['silhouette_score']
            if 'inertia' in self.current_progress.data:
                metrics['inertia'] = self.current_progress.data['inertia']

        return PipelineResult(
            success=True,
            clusters=clusters,
            metrics=metrics,
            wordclouds=wordclouds,
            songs_by_cluster=songs_by_cluster
        )


# Pipeline manager for handling concurrent requests
class PipelineManager:
    """Manages pipeline instances and their state"""

    def __init__(self):
        self.pipelines: Dict[str, MLPipeline] = {}
        self.results: Dict[str, PipelineResult] = {}
        self.progress: Dict[str, PipelineProgress] = {}
        self._lock = threading.Lock()

    def create_pipeline(self, session_id: str, n_samples: int,
                       n_clusters: int, model_name: str) -> MLPipeline:
        """Create a new pipeline for a session"""
        with self._lock:
            pipeline = MLPipeline(
                n_samples=n_samples,
                n_clusters=n_clusters,
                model_name=model_name
            )

            def progress_callback(progress: PipelineProgress):
                self.progress[session_id] = progress

            pipeline.set_progress_callback(progress_callback)
            self.pipelines[session_id] = pipeline
            return pipeline

    def get_progress(self, session_id: str) -> Optional[PipelineProgress]:
        """Get current progress for a session"""
        return self.progress.get(session_id)

    def store_result(self, session_id: str, result: PipelineResult):
        """Store pipeline result"""
        with self._lock:
            self.results[session_id] = result

    def get_result(self, session_id: str) -> Optional[PipelineResult]:
        """Get stored result for a session"""
        return self.results.get(session_id)

    def cleanup(self, session_id: str):
        """Clean up resources for a session"""
        with self._lock:
            self.pipelines.pop(session_id, None)
            self.progress.pop(session_id, None)
            # Keep results for a while for retrieval


# Global pipeline manager instance
pipeline_manager = PipelineManager()
