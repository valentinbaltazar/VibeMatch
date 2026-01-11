from flask import Blueprint, render_template, request, redirect, url_for
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer
import lyricsgenius as genius

DEVICE = "cpu"

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


def generate_playlist(input_text, df, corpus_embeddings, model):
    print("this is make playlist def")
    # Encode the input text
    input_embedding = model.encode([input_text])
    
    # Compute cosine similarity between input text and corpus
    similarities = cosine_similarity(input_embedding, corpus_embeddings)
    
    # Find the index of the most similar text
    n = 5
    most_similar_idx = np.argsort(similarities)[0][::-1]
   
    playlist_songs = []
    
    for i in most_similar_idx[:20]:
        name  = df.iloc[i]['artist_name']
        song  = df.iloc[i]['song_title']
        lyrics = df.iloc[i]['lyrics']

        playlist_songs.append({'song_name': song, 'artist_name': name})
        
    return playlist_songs

@routes.route('/playlist', methods=['POST'])
def make_playlist():
    user_input = request.form['user_input']

    df = pd.read_csv('./website/static/data/rec_data.csv')
    model = SentenceTransformer('bert-base-nli-mean-tokens',device=DEVICE)
    corpus_embeddings = np.load('./website/static/data/corpus_embeddings.npy')

    if user_input != "None":

        print(user_input)

        songs = generate_playlist(user_input,df, corpus_embeddings, model)

        return render_template('rec_songs.html', songs=songs)
    else:
        song_name = request.form['song_name']
        artist_name = request.form['artist_name']

        GENIUS_ACCESS_TOKEN = "3TVMsMFnRiJGJnZ7r4Wl2pgmy2_hPdMiAoXED6Jofnp2AnAHKgY97q1J9b6RMBkz"

        api = genius.Genius(GENIUS_ACCESS_TOKEN,timeout=60,retries=2)

        song = api.search_song(title=song_name, artist=artist_name)

        songs = generate_playlist(song.lyrics,df, corpus_embeddings, model)

        return render_template('rec_songs.html', songs=songs)





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

