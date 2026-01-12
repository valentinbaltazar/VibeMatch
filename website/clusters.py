from models import ClusteringModel
from embedings import EmbeddingModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
import os

def make_clusters(corpus):

    embed = EmbeddingModel(device='cpu')
    corpus_embeddings = embed.encode(corpus)


    model = ClusteringModel(n_clusters = 4, random_state = 42)
    model.fit(corpus_embeddings)

    clusters = model.predict(corpus_embeddings)

    return clusters

def make_cloud(df,clusters,tag):
    # Create a dictionary to store texts for each cluster
    cluster_texts = {i: [] for i in range(len(np.unique(clusters)))}
    
    # Populate the dictionary with texts belonging to each cluster
    for i, text in enumerate(df[tag]):
        cluster_texts[clusters[i]].append(text)
    
    # Generate WordCloud for each cluster
    for cluster_id, texts in cluster_texts.items():
        # Combine all texts in the cluster into a single string
        cluster_text = " ".join(texts)
        
        # Generate the WordCloud
        wordcloud = WordCloud(width=800, height=400).generate(cluster_text)
        
        # Display the WordCloud
        plt.figure(figsize=(10, 6))
        # plt.imshow(wordcloud)
        plt.title(f"Cluster {cluster_id} WordCloud")
        plt.axis('off')
        plt.imshow(wordcloud)

        output_path = os.path.join('./static/images/', f'cluster_{str(cluster_id)}.png')
        plt.savefig(output_path)

        plt.show()

def main(df):
    # Make clusters and save word clouds for each
    
    clusters = make_clusters(df['corpus_clean'])
    df['cluster'] = clusters

    output_path = os.path.join('./static/data/','data.csv')

    df.to_csv(output_path)

    groups = df.groupby('cluster')

    print(groups['artist_name'].describe())

    make_cloud(df,clusters,'corpus_clean')



if __name__ == '__main__':
    
    nltk.download('stopwords')

    custom_stopwords = {'like', 'got', 'dont', 'aint','im','want','youre','oh','let','ill',
                    'get','cant','make','know','come','cant','go','said','could','wan','na',
                    'gon','see','yeah','doo','ooh','la','da','back','time','cause'}

    stop_words = set(stopwords.words('english')) | custom_stopwords

    file = '../temp/clean_lyrics.csv'
    df0 = pd.read_csv(file)
    df = df0.dropna(subset=['corpus'],ignore_index=True)
    df['corpus_clean'] = df['corpus'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    main(df)


