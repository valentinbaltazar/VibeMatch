import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

CLIENT_ID = '9e037a45a23c4f22a1df30d1820bae65'
CLIENT_SECRET = '88412132bc70474e8ade57eb6c080039'

# Initialize Spotipy
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID,
                                                      client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

@app.route('/')
def index():
    return render_template('vibe.html')

    
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_song():
    song_name = request.form['song_name']
    artist_name = request.form['artist_name']

    # Search for the song
    results = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)

    if results['tracks']['items']:
        # Get the first track found
        track_id = results['tracks']['items'][0]['id']
        return redirect(url_for('play_song', track_id=track_id))
    else:
        return "Song not found."

@app.route('/play/<track_id>')
def play_song(track_id):
    return redirect(f"https://open.spotify.com/track/{track_id}")

if __name__ == '__main__':
    app.run(debug=True)
