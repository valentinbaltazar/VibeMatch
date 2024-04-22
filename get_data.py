import requests
import pandas as pd

# Define your Musixmatch API key
api_key = "a086593ac977427660aa0d8481bebd69"

def get_albums(artist_name):
    album_dict = []

    # API endpoint for searching an artist
    search_artist_url = f"https://api.musixmatch.com/ws/1.1/artist.search?q_artist={artist_name}&apikey={api_key}"

    # Make a GET request to search for the artist
    response = requests.get(search_artist_url)
    data = response.json()

    # Check if the request was successful
    if data["message"]["header"]["status_code"] == 200:
        # Extract artist ID
        artist_id = data["message"]["body"]["artist_list"][0]["artist"]["artist_id"]
        
        # API endpoint for getting the albums of the artist
        artist_albums_url = f"https://api.musixmatch.com/ws/1.1/artist.albums.get?artist_id={artist_id}&apikey={api_key}"

        # Make a GET request to get the albums of the artist
        response = requests.get(artist_albums_url)
        data = response.json()

        # Check if the request was successful
        if data["message"]["header"]["status_code"] == 200:
            # Extract album list
            albums = data["message"]["body"]["album_list"]
            
            # Print the album names
            for album in albums:
                album_name = album["album"]["album_name"]
                album_id = album["album"]["album_id"]
                # print(album_name, album_id)

                album_dict.append({album_name:album_id,artist_name:artist_id})
            
            # print(album_dict)
            return(album_dict)
        
        else:
            print("Error occurred while fetching albums:", data["message"]["header"]["status_code"])
    else:
        print("Error occurred while searching for the artist:", data["message"]["header"]["status_code"])



def get_songs(album_id):
    song_dict = []

    # Define the album ID you want to search for
    album_id = album_id  # Change this to the album ID you want to search for

    # API endpoint for getting the tracks of the album
    album_tracks_url = f"https://api.musixmatch.com/ws/1.1/album.tracks.get?album_id={album_id}&apikey={api_key}"

    # Make a GET request to get the tracks of the album
    response = requests.get(album_tracks_url)
    data = response.json()

    # Check if the request was successful
    if data["message"]["header"]["status_code"] == 200:
        # Extract track list
        tracks = data["message"]["body"]["track_list"]
        
        # Print the track names
        for track in tracks:
            track_name = track["track"]["track_name"]
            track_id = track["track"]["track_id"]
            # print(track_name,track_id)
            song_dict.append({track_name:track_id})

        # print(song_dict)
        return(song_dict)
       
    else:
        print("Error occurred while fetching tracks:", data["message"]["header"]["status_code"])


def get_lyrics(track_id):

    # API endpoint for getting the lyrics of the track
    track_lyrics_url = f"https://api.musixmatch.com/ws/1.1/track.lyrics.get?track_id={track_id}&apikey={api_key}"

    # Make a GET request to get the lyrics of the track
    response = requests.get(track_lyrics_url)
    data = response.json()

    # Check if the request was successful
    if data["message"]["header"]["status_code"] == 200:
        # Extract the lyrics
        lyrics = data["message"]["body"]["lyrics"]["lyrics_body"]
        
        # Clean the lyrics (removing metadata)
        # print(lyrics)
        lyrics = lyrics.split("*******")[0].strip()
        
        # print(lyrics)
        return(lyrics)
    else:
        print("Error occurred while fetching lyrics:", data["message"]["header"]["status_code"])


def get_all_lyrics(artist):
    # From a given Artist, this will fetch the albumns, finds album the songs, and then lyrics for each song
    albums = get_albums(artist)

    data = []

    for alb in albums:
        # print(alb)
        # print()
        # data = []

        album_name = list(alb.keys())[0]
        album_id = list(alb.values())[0]

        artist_name = list(alb.keys())[1]
        artist_id = list(alb.values())[1]

        songs = get_songs(album_id)
        for song in songs:
            song_name = list(song.keys())[0]
            song_id = list(song.values())[0]

            lyrics = get_lyrics(song_id)

            data.append({"artist":artist_name,"artist_id":artist_id,
                         "album":album_name,"album_id":album_id,
                         "song":song_name,"song_id":song_id,
                         "lyrics":lyrics})


            # get_lyrics(list(song.values())[0])
    
    df = pd.DataFrame(data)

    print(data)
    print(df.head())

    file_name = "_".join(artist_name.split(" "))

    df.to_csv(f"./data/{file_name}.csv")


# get_albums("Kanye West")
# get_songs(46691158)
# get_lyrics(get_album("Kanye West"))

get_all_lyrics("Kanye West")