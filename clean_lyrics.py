import os

import pandas as pd
import re
import string



def drop_pattern(df,pattern):
    # Filter rows where the column string matches the pattern
    data = df[~df['song_title'].str.contains(pattern, regex=True)]

    return data


def sub_pattern(pattern,lyrics):
    # Sub pattern for space character
    lyrics = re.sub(pattern,' ',lyrics)

    return lyrics


def clean_lyrics(lyrics):
    #no need for unicode char
    lyrics = lyrics.encode('ascii', 'ignore').decode()
    
    # lyrics = re.sub(r'\u2005', ' ', lyrics)
    # lyrics = lyrics.replace('\u2005', ' ')
    # Some have other strange punctuations
    # lyrics = lyrics.replace('’', '')
    # lyrics = lyrics.replace('‘', '')

    #differnt dashes...?
    lyrics = lyrics.replace('—', '')
    lyrics = lyrics.replace('—', '')
    # lyrics = lyrics.replace('–', '')

    # This may cuase some lyrics to be removed as well...
    lyrics = lyrics.replace('You might also likeEmbed', ' ')
    lyrics = lyrics.replace('You might also like', ' ')
   

    # Substitute pattern that are not lyrics with space
    lyrics = sub_pattern(r'\[.*?(Verse|Chorus|Interlude|Outro|Refrain|Bridge|Hook).*?\]',lyrics)
    
    # Remove punctuation
    lyrics = ''.join(s if s not in string.punctuation else '' for s in lyrics)

    lyrics = sub_pattern(r'x\d+',lyrics)
    lyrics = sub_pattern(r'See .*? LiveGet tickets as low as .*? might also like',lyrics)
    lyrics = sub_pattern(r'\d+Embed',lyrics)
    lyrics = sub_pattern(r'See .*? LiveGet tickets as low as \d+',lyrics)

    # lyrics = re.sub(pattern6,' ',lyrics)

    # Have to do this after regex for non int matches
    lyrics = lyrics.replace('Embed', ' ')

    #first row is always non lyrics
    lyrics = lyrics.split('\n')[1:]

    final_lyrics = ' '.join(lyrics)

    return final_lyrics.lower()


def df_lyrics(data_dir):
    dfs = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir,filename)
        dfn = pd.read_csv(filepath)
        dfs.append(dfn)
    return  pd.concat(dfs, ignore_index=True)


def main(directory):
    df = df_lyrics(directory)
    print(df.shape)
    df = df.dropna(subset=['lyrics'])
    print(df.shape)
    # First remove duplicates, different title but same lyrics, also (remix), (live)...etc
    df = drop_pattern(df,r"\(.*\)$")
    df = drop_pattern(df,r"\[Live.*\]$")

    print(df.shape)
    m_out = map(clean_lyrics,df['lyrics'])
    df['corpus'] = list(m_out)

    df = df.drop_duplicates(subset=['corpus'])
    print(df.shape)

    df.to_csv('./temp/clean_lyrics.csv')


main('./data/')
