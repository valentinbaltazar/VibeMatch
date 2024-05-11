import lyricsgenius as genius
import pandas as pd
import numpy as np

# Your genius access token here
GENIUS_ACCESS_TOKEN = "3TVMsMFnRiJGJnZ7r4Wl2pgmy2_hPdMiAoXED6Jofnp2AnAHKgY97q1J9b6RMBkz"

# Modify if you want to save in different order, or remove paramater to not save
song_attr = ['id' ,'artist_names', 'title', 
            'annotation_count', 
            'api_path', 'full_title', 'header_image_thumbnail_url',
            'header_image_url', 'lyrics_owner_id', 'lyrics_state', 'path',
            'pyongs_count', 'song_art_image_thumbnail_url', 'song_art_image_url',
            'title_with_featured', 'url']

album_attr = ['api_path', 'cover_art_url', 'full_title',
            'id', 'name', 'release_date_for_display', 'url']

api = genius.Genius(GENIUS_ACCESS_TOKEN,timeout=60,retries=2)

def get_artist(artist_name):
    artist = api.search_artist(artist_name,max_songs=1)
    # print(artist_name,artist.id)
    return {'search_name':artist_name,'artist_id':artist.id,'artist_name':artist.name}


def get_album(artist_attr):
    all_albums=[]
    albums = api.artist_albums(artist_attr['artist_id'])

    for i in range(len(albums['albums'])):
        # print(albums['albums'][n]['name'], albums['albums'][n]['id'])
        # all_albums.append(albums['albums'][n]['id'])
        album_dict={}
        for attr in album_attr:
            value = albums['albums'][i][attr]
            # print(attr,":",value)
            album_dict['album_'+attr] = value
        
        all_albums.append({**artist_attr,**album_dict})

    # print("sample albums:",all_albums[:5])
    return all_albums

def get_songs(albums):
    all_songs = []

    for album in albums:
        songs = api.album_tracks(album['album_id'])
        for song in songs['tracks']:
            # print(songs['tracks'][i]['song']['title'],songs['tracks'][i]['song']['id'])
            song_dict={}
            for attr in song_attr:
                value = song['song'][attr]
                # print(attr,":",value)
                song_dict['song_'+attr] = value
            
            all_songs.append({**album,**song_dict})
    
    # print("sample song",all_songs[:5])
    return all_songs



def get_lyrics(songs):
    all_lyrics=[]

    for song in songs:
        lyrics_dict={}
        try:
            print("Now searching lyrics for...",song['album_name'],":",song['song_title'])
            value = api.lyrics(song['song_id'])
            lyrics_dict['lyrics'] = value
            all_lyrics.append({**song,**lyrics_dict})
        except Exception as e:
            print("An error occurred:", e)
            print("Could Not Get Lyrics")
            lyrics_dict['lyrics'] = np.nan
            all_lyrics.append({**song,**lyrics_dict})
            continue

    return all_lyrics

def main(artist_name):
        
    artist_info = get_artist(artist_name)
    print(artist_info)

    album_info = get_album(artist_info)

    # print(album_info[0])

    songs_info = get_songs(album_info)

    # print(songs_info[0])

    lyrics_info = get_lyrics(songs_info)

    df = pd.DataFrame(lyrics_info)

    artist_name_file = '_'.join(artist_name.split(' ')) + '.csv'
    data_path = './data/'

    file_path = data_path + artist_name_file

    df.to_csv(file_path, index=False)


# main("HAIM")

with open('./artist.txt','r') as f:
    lines = f.readlines()
    artist_names = [line.rstrip('\n') for line in lines]
    print(f"Foud {len(artist_names)} artist in file...")
    print(artist_names)
    for name in artist_names:
        main(name)