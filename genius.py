import lyricsgenius as genius
import pandas as pd
import numpy as np

# Your genius access token here
GENIUS_ACCESS_TOKEN = "3TVMsMFnRiJGJnZ7r4Wl2pgmy2_hPdMiAoXED6Jofnp2AnAHKgY97q1J9b6RMBkz"

# Modify if you want to save in different order, or remove paramater to not save
song_attr = ['id' ,'artist', 'title', 'lyrics', 
            'annotation_count', 
            'api_path', 'full_title', 'header_image_thumbnail_url',
            'header_image_url', 'lyrics_owner_id', 'lyrics_state', 'path',
            'pyongs_count', 'song_art_image_thumbnail_url', 'song_art_image_url',
            'title_with_featured', 'url']

album_attr = ['api_path', 'cover_art_url', 'full_title',
            'id', 'name', 'release_date_for_display', 'url']

# fetch artist data and save to csv
def get_artist_lyrics(artist_name,data_path):
        
    api = genius.Genius(GENIUS_ACCESS_TOKEN,timeout=20)

    artist = api.search_artist(artist_name,max_songs=10)

    lyrics_df = pd.DataFrame()


    for song in artist.songs:
        song_dict = {}
        album_dict = {}

        for attr in song_attr:
            try:
                value = getattr(song, attr)
                # print(attr, ":", value)

                song_dict[attr] = value
            except Exception as e:
                print("An error occurred:", e)
                song_dict[attr] = np.nan
                continue

                
            
        for key in album_attr:
            val = song._body['album'][key]
            # print(key, ":", val)

            album_dict['album_'+key] = val

        attr_df = pd.DataFrame(song_dict,index=[0])
        album_df = pd.DataFrame(album_dict,index=[0])

        song_df = pd.concat([attr_df,album_df],axis=1)

        lyrics_df = pd.concat([lyrics_df,song_df],ignore_index=True)

        # print(song_df.head())

    # print(lyrics_df.head())

    artist_name_file = '_'.join(artist_name.split(' ')) + '.csv'

    file_path = data_path + artist_name_file

    lyrics_df.to_csv(file_path, index=False)



test_path = './data/'

get_artist_lyrics('Andy Shauf',test_path)
