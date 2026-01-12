# VibeMatch: AI-Powered Music Discovery Through Lyrical Analysis

A full-stack machine learning web application that discovers music through semantic lyrical analysis and unsupervised clustering. Instead of traditional genre-based categorization, VibeMatch groups songs by their lyrical similarity, creating "vibes" - clusters of songs with similar themes, emotions, and meaning.

**Capstone Project** | General Assembly Data Science Immersive | Adobe Digital Academy Scholar 2024

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [What I Learned](#what-i-learned)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

VibeMatch reimagines music discovery by analyzing the semantic meaning of lyrics rather than relying on traditional metadata like genre or artist. The application:

1. **Encodes lyrics into vector embeddings** using BERT-based transformer models
2. **Clusters songs** using KMeans based on lyrical similarity
3. **Visualizes clusters** with WordClouds showing dominant themes
4. **Recommends songs** based on cosine similarity between user input and corpus

### The Problem

Traditional music recommendation systems rely on:
- Genre labels (subjective and often inaccurate)
- Collaborative filtering (requires user history)
- Audio features (misses lyrical content)

### The Solution

VibeMatch uses Natural Language Processing to understand *what songs are about*, enabling:
- Discovery of thematically similar songs across different genres
- Recommendations based on mood, theme, or lyrical style
- Exploration of lyrical "vibes" through interactive clustering

---

## Key Features

### 1. Dynamic ML Pipeline
Run the complete machine learning pipeline in real-time with configurable parameters:
- **Sample size**: 100-5,000 songs
- **Number of clusters**: 2-15 vibes
- **BERT model selection**: Choose between speed and accuracy

### 2. Real-Time Progress Tracking
Watch the ML pipeline execute with live updates using Server-Sent Events (SSE):
- Step-by-step progress visualization
- Percentage completion for each stage
- Dynamic status messages

### 3. Semantic Playlist Generator
Get song recommendations by:
- Pasting lyrics, poetry, or mood descriptions
- Searching for a song (automatically fetches lyrics via Genius API)
- Viewing similarity scores and exploring recommendations on Spotify

### 4. Interactive Vibe Exploration
- Click on any cluster to discover songs within that "vibe"
- View WordClouds representing dominant themes
- Play random songs from each cluster with Spotify integration

---

## Machine Learning Pipeline

The core ML pipeline consists of four stages:

| Stage | Process | Technology | Output |
|-------|---------|------------|--------|
| **1. Data Loading** | Load and filter lyrics dataset | Pandas | Clean DataFrame (16K+ songs) |
| **2. BERT Encoding** | Convert lyrics to semantic vectors | Sentence-Transformers | N×D embedding matrix |
| **3. KMeans Clustering** | Group songs by lyrical similarity | scikit-learn | Cluster assignments + metrics |
| **4. Visualization** | Generate cluster representations | WordCloud, Matplotlib | PNG images |

### Embedding Models

Three pre-trained BERT models are available:

| Model | Dimensions | Speed | Best For |
|-------|------------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Quick demos, prototyping |
| `all-mpnet-base-v2` | 768 | Balanced | Production use |
| `bert-base-nli-mean-tokens` | 768 | Standard | Original project model |

### Similarity Measurement

Recommendations use **cosine similarity** to find semantically similar songs:

```
similarity(a, b) = (a · b) / (||a|| × ||b||)
```

This measures the angle between embedding vectors, ranging from 0 (unrelated) to 1 (identical meaning).

### Clustering Evaluation

The pipeline calculates:
- **Silhouette Score**: Measures cluster separation quality (-1 to 1, higher is better)
- **Inertia**: Within-cluster sum of squares (lower indicates tighter clusters)
- **Distribution**: Song count per cluster for balance analysis

---

## Technologies Used

### Machine Learning & NLP
- **Sentence-Transformers**: Pre-trained BERT models for semantic embeddings
- **scikit-learn**: KMeans clustering, silhouette scoring
- **NumPy**: Vector operations and similarity calculations

### Backend
- **Flask**: Python web framework with RESTful API design
- **Spotipy**: Spotify Web API integration
- **LyricsGenius**: Genius API for lyric fetching

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Bootstrap 4**: Responsive UI components
- **Server-Sent Events**: Real-time progress streaming

### Data & Visualization
- **Pandas**: Data manipulation and preprocessing
- **WordCloud**: Cluster theme visualization
- **Matplotlib**: Image generation

---

## Installation

### Prerequisites
- Python 3.8+
- Spotify Developer Account (for API credentials)
- Genius API Token

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/VibeMatch.git
cd VibeMatch
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r VibeMatch/requirements.txt
```

4. **Configure API credentials**

Create a `.env` file or set environment variables:
```bash
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
GENIUS_API_TOKEN=your_genius_token
```

5. **Run the application**
```bash
python VibeMatch/main.py
```

6. **Open in browser**
```
http://localhost:5000
```

---

## Usage

### Running the ML Pipeline

1. Navigate to the **Home** page
2. Configure parameters:
   - Adjust sample size (more songs = more diverse results)
   - Set number of clusters (vibes to discover)
   - Select BERT model
3. Click **"Run Pipeline"**
4. Watch real-time progress as each stage completes
5. Explore the resulting vibes and their WordClouds

### Generating Playlists

1. Go to the **Playlist** page
2. Choose input method:
   - **Paste lyrics**: Enter any text (lyrics, poems, mood descriptions)
   - **Search song**: Enter song name and artist
3. Click **"Generate Playlist"**
4. Browse 20 recommended songs ranked by similarity
5. Click any song to open in Spotify

### Exploring Vibes

- Click on any vibe card to see a random song from that cluster
- View the WordCloud to understand the cluster's dominant themes
- Expand to see all songs in a vibe

---

## Project Architecture

```
VibeMatch/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── website/
    ├── __init__.py            # Flask app factory
    ├── routes.py              # API endpoints & page routes
    ├── pipeline.py            # ML pipeline orchestration
    ├── embedings.py           # BERT embedding wrapper
    ├── models.py              # KMeans clustering wrapper
    ├── static/
    │   └── data/
    │       ├── all_lyrics.csv # Primary dataset (16K+ songs)
    │       └── rec_data.csv   # Recommendation corpus
    └── templates/
        ├── home.html          # Dynamic pipeline UI
        ├── playlist.html      # Recommendation generator
        └── about.html         # Project documentation
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/run_pipeline` | POST | Start ML pipeline with configuration |
| `/api/pipeline_stream/<session_id>` | GET | SSE stream for real-time updates |
| `/api/playlist` | POST | Generate song recommendations |
| `/api/models` | GET | List available BERT models |
| `/api/cluster_songs/<session_id>/<cluster_id>` | GET | Get all songs in a cluster |

---

## What I Learned

### Machine Learning
- **Transformer architectures**: Understanding how BERT creates contextual embeddings
- **Sentence embeddings**: Converting variable-length text to fixed-dimension vectors
- **Unsupervised learning**: KMeans clustering for pattern discovery without labels
- **Evaluation metrics**: Silhouette scores, inertia, and cluster quality assessment

### Software Engineering
- **Full-stack development**: Building a complete web application from scratch
- **API design**: RESTful endpoints with proper error handling
- **Real-time systems**: Server-Sent Events for live progress updates
- **Thread management**: Background task execution without blocking the UI
- **Session handling**: Managing concurrent pipeline runs with unique IDs

### Data Engineering
- **Batch processing**: Handling large datasets efficiently with progress callbacks
- **Data pipelines**: Orchestrating multi-stage ML workflows
- **API integration**: Working with rate limits and error handling (Spotify, Genius)

### Problem Solving
- **Meaningful visualization**: Creating WordClouds with custom stopword filtering
- **User experience**: Real-time feedback for long-running ML operations
- **Graceful degradation**: Handling API failures without breaking the application

---

## Future Improvements

- [ ] Add database persistence for caching embeddings
- [ ] Implement user accounts for saving playlists
- [ ] Add more clustering algorithms (DBSCAN, hierarchical)
- [ ] Enable audio feature analysis alongside lyrics
- [ ] Build a Chrome extension for Spotify integration
- [ ] Deploy to cloud platform (AWS/GCP) with GPU support

---

## Author

**Valentin Urena Baltazar**

Adobe Digital Academy Scholar | General Assembly Data Science Immersive 2024

- GitHub: [github.com/yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

## License

This project is for educational purposes as part of the General Assembly Data Science Immersive program.

---

## Acknowledgments

- General Assembly Data Science Immersive program
- Adobe Digital Academy Scholarship
- Spotify and Genius for their APIs
- Hugging Face for Sentence-Transformers library
