from flask import Blueprint, render_template, request, redirect, url_for
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os

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

    
    # song_name = 'Monster'
    # artist_name = 'Imagine Dragons'

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


@routes.route('/play/<track_id>')
def play_song(track_id):
    return redirect(f"https://open.spotify.com/track/{track_id}")

@routes.route('/fetch_images', methods=['GET'])
def fetch_images():
    # Get the URL requested by the client
    url = request.args.get('url')

    # Log the URL to the terminal
    print("URL requested:", url)

    # You can also log other details like user-agent, IP, etc.
    print("User-Agent:", request.headers.get('User-Agent'))
    print("Client IP:", request.remote_addr)

    # Here you can perform additional processing if needed, then return a response
    # For now, just return a placeholder response
    return "Fetching images..."