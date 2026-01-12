from flask import Blueprint, render_template, request, redirect, url_for, jsonify, Response, session
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer
import lyricsgenius as genius
import uuid
import json
import threading
import time
from dataclasses import asdict

from .pipeline import MLPipeline, pipeline_manager, PipelineProgress

DEVICE = "cpu"

# Store for dynamic pipeline results (session_id -> result)
pipeline_results = {}

routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    return render_template('home.html')

@routes.route('/search')
def search():
    return render_template('search.html')


@routes.route('/about')
def about():
    return render_template('about.html')


@routes.route('/test')
def test():
    return render_template('test.html')


@routes.route('/demo')
def demo():
    return render_template('demo.html')


@routes.route('/playlist')
def playlist():
    return render_template('playlist.html')


@routes.route('/vibe', methods=['POST'])
def random_song():
    file_name = request.form['file_name']
    print('file name is:', file_name)

    cluster_number = int(file_name.split('.')[0].split('_')[-1])
    print("cluster is number:",cluster_number)
    

    df = pd.read_csv('./website/static/data/data.csv')
    print(df.head())

    df_cluster = df[df['cluster']==cluster_number]
    
    random_row = df_cluster.sample(n=1)
    
    song_name = random_row['song_title'].values[0]
    artist_name = random_row['artist_name'].values[0]

    print(str(song_name))
    print(str(artist_name))
    
    CLIENT_ID = '9e037a45a23c4f22a1df30d1820bae65'
    CLIENT_SECRET = '88412132bc70474e8ade57eb6c080039'

    # Initialize Spotipy
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID,
                                                        client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    current_directory = os.getcwd()
    print("Current directory:", current_directory)


    # Search for the song
    res = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
    # print(res)
    if res['tracks']['items']:
        # Get the first track found
        track_id = res['tracks']['items'][0]['id']
        print(track_id)
        # return redirect(url_for('play', track_id=track_id))
        return redirect(url_for('routes.play_song', track_id=track_id))
    else:
        return "Song not found."

@routes.route('/search', methods=['POST'])
def search_song():
        
    CLIENT_ID = '9e037a45a23c4f22a1df30d1820bae65'
    CLIENT_SECRET = '88412132bc70474e8ade57eb6c080039'

    # Initialize Spotipy
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID,
                                                        client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    song_name = request.form['song_name']
    artist_name = request.form['artist_name']

    # Search for the song
    results = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
    print(results)
    if results['tracks']['items']:
        # Get the first track found
        track_id = results['tracks']['items'][0]['id']
        print(track_id)
        # return redirect(url_for('play', track_id=track_id))
        return redirect(url_for('routes.play_song', track_id=track_id))
    else:
        return "Song not found."


def generate_playlist(input_text, df, corpus_embeddings, model, return_scores=False):
    """
    Generate playlist recommendations using cosine similarity in embedding space.

    Args:
        input_text: User's lyrics or text input
        df: DataFrame with song metadata
        corpus_embeddings: Pre-computed BERT embeddings for all songs
        model: SentenceTransformer model for encoding
        return_scores: If True, include similarity scores in results

    Returns:
        List of song dictionaries with optional similarity scores
    """
    # Encode the input text into embedding space
    input_embedding = model.encode([input_text])

    # Compute cosine similarity between input vector and all corpus vectors
    similarities = cosine_similarity(input_embedding, corpus_embeddings)

    # Get indices sorted by similarity (highest first)
    most_similar_idx = np.argsort(similarities)[0][::-1]

    playlist_songs = []

    for i in most_similar_idx[:20]:
        song_data = {
            'song_name': df.iloc[i]['song_title'],
            'artist_name': df.iloc[i]['artist_name']
        }
        if return_scores:
            song_data['similarity'] = float(similarities[0][i])
        playlist_songs.append(song_data)

    return playlist_songs, input_embedding if return_scores else (playlist_songs, None)

@routes.route('/playlist', methods=['POST'])
def make_playlist():
    user_input = request.form['user_input']

    df = pd.read_csv('./website/static/data/rec_data.csv')
    model = SentenceTransformer('bert-base-nli-mean-tokens', device=DEVICE)
    corpus_embeddings = np.load('./website/static/data/corpus_embeddings.npy')

    if user_input != "None":
        print(user_input)
        songs, _ = generate_playlist(user_input, df, corpus_embeddings, model)
        return render_template('rec_songs.html', songs=songs)
    else:
        song_name = request.form['song_name']
        artist_name = request.form['artist_name']

        GENIUS_ACCESS_TOKEN = "3TVMsMFnRiJGJnZ7r4Wl2pgmy2_hPdMiAoXED6Jofnp2AnAHKgY97q1J9b6RMBkz"
        api = genius.Genius(GENIUS_ACCESS_TOKEN, timeout=60, retries=2)
        song = api.search_song(title=song_name, artist=artist_name)
        songs, _ = generate_playlist(song.lyrics, df, corpus_embeddings, model)
        return render_template('rec_songs.html', songs=songs)


@routes.route('/api/playlist', methods=['POST'])
def api_playlist():
    """
    API endpoint for playlist generation with similarity scores.
    Returns JSON with song recommendations and ML metadata.

    Expected JSON body:
    {
        "mode": "lyrics" | "song",
        "lyrics": "user lyrics text...",  // if mode is "lyrics"
        "song_name": "...",               // if mode is "song"
        "artist_name": "..."              // if mode is "song"
    }
    """
    data = request.get_json()
    mode = data.get('mode', 'lyrics')

    # Load model and embeddings
    df = pd.read_csv('./website/static/data/rec_data.csv')
    model = SentenceTransformer('bert-base-nli-mean-tokens', device=DEVICE)
    corpus_embeddings = np.load('./website/static/data/corpus_embeddings.npy')

    input_text = ""

    try:
        if mode == 'lyrics':
            input_text = data.get('lyrics', '')
            if not input_text or len(input_text.strip()) < 10:
                return jsonify({
                    'success': False,
                    'error': 'Please enter at least 10 characters of lyrics'
                }), 400
        else:
            # Song mode - fetch lyrics from Genius
            song_name = data.get('song_name', '')
            artist_name = data.get('artist_name', '')

            if not song_name or not artist_name:
                return jsonify({
                    'success': False,
                    'error': 'Please provide both song name and artist name'
                }), 400

            GENIUS_ACCESS_TOKEN = "3TVMsMFnRiJGJnZ7r4Wl2pgmy2_hPdMiAoXED6Jofnp2AnAHKgY97q1J9b6RMBkz"
            api = genius.Genius(GENIUS_ACCESS_TOKEN, timeout=60, retries=2)

            song = api.search_song(title=song_name, artist=artist_name)
            if not song:
                return jsonify({
                    'success': False,
                    'error': f'Could not find lyrics for "{song_name}" by {artist_name}'
                }), 404

            input_text = song.lyrics

        # Generate playlist with similarity scores
        songs, input_embedding = generate_playlist(
            input_text, df, corpus_embeddings, model, return_scores=True
        )

        # Get Spotify info for top results
        CLIENT_ID = '9e037a45a23c4f22a1df30d1820bae65'
        CLIENT_SECRET = '88412132bc70474e8ade57eb6c080039'

        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET
            )
            sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

            # Enrich results with Spotify data
            for song in songs[:10]:  # Only first 10 to avoid rate limits
                try:
                    results = sp.search(
                        q=f"track:{song['song_name']} artist:{song['artist_name']}",
                        type='track',
                        limit=1
                    )
                    if results['tracks']['items']:
                        track = results['tracks']['items'][0]
                        song['spotify_url'] = f"https://open.spotify.com/track/{track['id']}"
                        song['album_art'] = track['album']['images'][0]['url'] if track['album']['images'] else None
                        song['preview_url'] = track.get('preview_url')
                except Exception:
                    pass  # Skip Spotify enrichment on error
        except Exception:
            pass  # Continue without Spotify data if API fails

        return jsonify({
            'success': True,
            'songs': songs,
            'metadata': {
                'input_length': len(input_text),
                'embedding_dimensions': 768,
                'corpus_size': len(corpus_embeddings),
                'model': 'bert-base-nli-mean-tokens',
                'top_similarity': songs[0]['similarity'] if songs else 0,
                'avg_similarity': sum(s['similarity'] for s in songs) / len(songs) if songs else 0
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500





@routes.route('/play/<track_id>')
def play_song(track_id):
    return redirect(f"https://open.spotify.com/track/{track_id}")

@routes.route('/vibe_demo', methods=['POST'])
def random_song_demo():

    file_name = request.form['file_name']
    print('file name is:', file_name)

    cluster_number = int(file_name.split('.')[0].split('_')[-1])
    print("cluster is number:",cluster_number)
    
    df = pd.read_csv('./website/static/data/demo.csv')
    # print(df.head())

    df_cluster = df[df['cluster']==cluster_number]
    
    random_row = df_cluster.sample(n=3)
    
    # song_name = random_row['song_title'].values[0]
    # artist_name = random_row['artist_name'].values[0]

    # print(str(song_name))
    # print(str(artist_name))
    
    CLIENT_ID = '9e037a45a23c4f22a1df30d1820bae65'
    CLIENT_SECRET = '88412132bc70474e8ade57eb6c080039'

    # Initialize Spotipy
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID,
                                                        client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    current_directory = os.getcwd()
    print("Current directory:", current_directory)

    for i in range(len(random_row)):
        song_name = random_row['song_title'].values[i]
        artist_name = random_row['artist_name'].values[i]
        try:
            # Search for the song
            res = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
            # print(res)
            if res['tracks']['items']:
                # Get the first track found
                track_id = res['tracks']['items'][0]['id']
                print(track_id)
                # return redirect(url_for('play', track_id=track_id))
                return redirect(url_for('routes.play_song', track_id=track_id))
            # else:
                # return "Song not found."
                # ret random_song_demo()
                # return redirect(url_for('routes.random_song_demo', file_name_try=file_name))
        except Exception as e:
            print("An error occurred:", e)
            continue


# =============================================================================
# ML Pipeline API Routes - Dynamic Clustering
# =============================================================================

@routes.route('/api/models')
def get_available_models():
    """Get list of available BERT models for embedding"""
    from .pipeline import MLPipeline
    models = []
    for key, info in MLPipeline.AVAILABLE_MODELS.items():
        models.append({
            'id': key,
            'name': info['display_name'],
            'dimensions': info['dimensions'],
            'description': info['description']
        })
    return jsonify({'models': models})


@routes.route('/api/run_pipeline', methods=['POST'])
def run_pipeline():
    """
    Run the ML pipeline with user-specified parameters.

    Expected JSON body:
    {
        "n_samples": 500,
        "n_clusters": 4,
        "model_name": "all-MiniLM-L6-v2"
    }
    """
    data = request.get_json()

    n_samples = int(data.get('n_samples', 500))
    n_clusters = int(data.get('n_clusters', 4))
    model_name = data.get('model_name', 'all-MiniLM-L6-v2')

    # Validate parameters
    n_samples = max(100, min(n_samples, 5000))
    n_clusters = max(2, min(n_clusters, 15))

    # Create session ID for this pipeline run
    session_id = str(uuid.uuid4())

    # Create and run pipeline
    pipeline = pipeline_manager.create_pipeline(
        session_id=session_id,
        n_samples=n_samples,
        n_clusters=n_clusters,
        model_name=model_name
    )

    # Run pipeline in background thread
    def run_in_background():
        result = pipeline.run()
        pipeline_manager.store_result(session_id, result)
        # Store in local dict for vibe route access
        if result.success:
            pipeline_results[session_id] = {
                'songs_by_cluster': result.songs_by_cluster,
                'n_clusters': n_clusters
            }

    thread = threading.Thread(target=run_in_background)
    thread.start()

    return jsonify({
        'session_id': session_id,
        'message': 'Pipeline started',
        'params': {
            'n_samples': n_samples,
            'n_clusters': n_clusters,
            'model_name': model_name
        }
    })


@routes.route('/api/pipeline_status/<session_id>')
def get_pipeline_status(session_id):
    """Get current status/progress of a pipeline run"""
    progress = pipeline_manager.get_progress(session_id)
    result = pipeline_manager.get_result(session_id)

    if result is not None:
        # Pipeline completed
        return jsonify({
            'status': 'completed',
            'success': result.success,
            'error': result.error,
            'clusters': result.clusters,
            'metrics': result.metrics,
            'wordclouds': result.wordclouds
        })
    elif progress is not None:
        # Pipeline in progress
        return jsonify({
            'status': 'running',
            'step': progress.step,
            'step_number': progress.step_number,
            'total_steps': progress.total_steps,
            'progress': progress.progress,
            'message': progress.message,
            'data': progress.data,
            'completed': progress.completed
        })
    else:
        return jsonify({
            'status': 'not_found',
            'message': 'Pipeline session not found'
        }), 404


@routes.route('/api/pipeline_stream/<session_id>')
def pipeline_stream(session_id):
    """
    Server-Sent Events endpoint for real-time pipeline progress updates.
    """
    def generate():
        last_step = None
        last_progress = -1
        check_count = 0
        max_checks = 600  # 10 minutes max (at 1 check per second)

        while check_count < max_checks:
            progress = pipeline_manager.get_progress(session_id)
            result = pipeline_manager.get_result(session_id)

            if result is not None:
                # Pipeline completed
                data = json.dumps({
                    'status': 'completed',
                    'success': result.success,
                    'error': result.error,
                    'clusters': result.clusters,
                    'metrics': result.metrics
                })
                yield f"data: {data}\n\n"
                break
            elif progress is not None:
                # Only send update if something changed
                if progress.step != last_step or abs(progress.progress - last_progress) >= 5:
                    last_step = progress.step
                    last_progress = progress.progress
                    data = json.dumps({
                        'status': 'running',
                        'step': progress.step,
                        'step_number': progress.step_number,
                        'total_steps': progress.total_steps,
                        'progress': progress.progress,
                        'message': progress.message,
                        'data': progress.data,
                        'completed': progress.completed
                    })
                    yield f"data: {data}\n\n"

            time.sleep(0.5)
            check_count += 1

        # Timeout
        if check_count >= max_checks:
            yield f"data: {json.dumps({'status': 'timeout'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@routes.route('/api/vibe_dynamic', methods=['POST'])
def vibe_dynamic():
    """
    Get a random song from a dynamically generated cluster.
    Uses Spotify API to find and redirect to song.
    """
    data = request.get_json()
    session_id = data.get('session_id')
    cluster_id = int(data.get('cluster_id', 0))

    # Get stored pipeline results
    if session_id not in pipeline_results:
        return jsonify({'error': 'Session not found'}), 404

    session_data = pipeline_results[session_id]
    songs_by_cluster = session_data.get('songs_by_cluster', {})

    if cluster_id not in songs_by_cluster:
        return jsonify({'error': 'Cluster not found'}), 404

    cluster_songs = songs_by_cluster[cluster_id]
    if not cluster_songs:
        return jsonify({'error': 'No songs in cluster'}), 404

    # Pick a random song
    import random
    song = random.choice(cluster_songs)
    song_name = song.get('song_title', '')
    artist_name = song.get('artist_name', '')

    # Search on Spotify
    CLIENT_ID = '9e037a45a23c4f22a1df30d1820bae65'
    CLIENT_SECRET = '88412132bc70474e8ade57eb6c080039'

    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        results = sp.search(
            q=f"track:{song_name} artist:{artist_name}",
            type='track',
            limit=1
        )

        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            track_id = track['id']
            return jsonify({
                'success': True,
                'song_name': song_name,
                'artist_name': artist_name,
                'spotify_url': f"https://open.spotify.com/track/{track_id}",
                'track_id': track_id,
                'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Song not found on Spotify',
                'song_name': song_name,
                'artist_name': artist_name
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@routes.route('/api/cluster_songs/<session_id>/<int:cluster_id>')
def get_cluster_songs(session_id, cluster_id):
    """Get all songs in a specific cluster"""
    if session_id not in pipeline_results:
        return jsonify({'error': 'Session not found'}), 404

    session_data = pipeline_results[session_id]
    songs_by_cluster = session_data.get('songs_by_cluster', {})

    if cluster_id not in songs_by_cluster:
        return jsonify({'error': 'Cluster not found'}), 404

    return jsonify({
        'cluster_id': cluster_id,
        'songs': songs_by_cluster[cluster_id]
    })

