"""
This code downloads the transcript from a YouTube video and saves the output
    in a format that can be used by our streamlit app
"""

# This part of the code brings in my OpenAI key , which I have stored as an
#   environment variable as OPENAI_API_KEY.  We could choose to use a fully
#   open source model such as one in Huggingface and/or SBERT
from dotenv import load_dotenv
load_dotenv() 

# Chunk the transcript by minute and store embeddings directly with NumPy.
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi as yta
from pytube import YouTube

import numpy as np
import pandas as pd
import pickle

from pyprojroot import here


def vector_storize_youtube_vid(video_id):
    """Download a transcript and save local embeddings for the Streamlit app."""

    # Get the title and description for the given YouTube video    
    yt = YouTube(f'http://youtube.com/watch?v={video_id}')
    title = yt.title
    description0 = yt.description
    description_list = []
    if description0 is not None:
        for tx0 in description0.split('\n'):
            if 'Want to hear more lecture like this?' in tx0:
                break
            if len(tx0) > 0:
                description_list.append(tx0)
        description1 = '|||'.join(description_list)
    else:
        description0 = ''
        
    # Download the automated transcript
    transcript = yta().fetch(video_id, preserve_formatting=True).to_raw_data()

    # Reformat as dataframe for manipulation
    df_tx = pd.DataFrame(transcript)
    df_tx['text0'] = df_tx['text']
    
    # Identify which minute each section of the text takes place in
    df_tx['minute1'] = (df_tx['start']/60).astype(int)

    # Allow ovelap of 15 seconds
    df_tx['minute2'] = ((df_tx['start']-15)/60).astype(int)
    
    # Each element of the text will be one minute 15 seconds long
    dict_texts = []
    for i in df_tx['minute1'].unique():
        # just get the rows with in our minute
        df_minute = df_tx[(df_tx['minute1']==i)|(df_tx['minute2']==i)].copy() 
        # combine the texts into one long string
        my_text = ' '.join(df_minute['text'])
        # get exact moment start of minute
        start_seconds = df_minute['start'].min()
        # get exact moment end of minute
        last_min = df_minute.iloc[-1]
        end_seconds = last_min['start'] + last_min['duration']
        out_dict = {'minute_id':i,'text0':my_text,'start_seconds':start_seconds, 'end_seconds':end_seconds}
        dict_texts.append(out_dict)
    df_texts = pd.DataFrame(dict_texts)

    # Use the NumPy retrieval format already consumed by streamlit.py. ChromaDB
    # has no patched release for its current security advisories.
    response = OpenAI().embeddings.create(
        model='text-embedding-ada-002', input=df_texts['text0'].tolist()
    )
    embeddings = np.stack([item.embedding for item in sorted(response.data, key=lambda item: item.index)])
    fp_storing = here()/'video_data'
    fp_storing.mkdir(exist_ok=True)
    with open(fp_storing/f'{video_id}.pkl', 'wb') as output:
        pickle.dump((df_tx, df_texts, embeddings), output)
    
    fp_texts = here()/'text_dfs'
    if not fp_texts.exists():
        fp_texts.mkdir()

    # Save raw transcript by minute, too
    df_tx.to_pickle(fp_texts/f'{video_id}_raw.pkl')
    
    # Save metadat for streamlit
    fp_meta = here()/'metadata.csv'
    if fp_meta.exists():
        df_meta = pd.read_csv(fp_meta)
    else:
        df_meta = pd.DataFrame(columns = ['video_id', 'title', 'description'])
    
    df_meta = df_meta.loc[df_meta['video_id'] != video_id].copy()
    row_info = pd.DataFrame([{'video_id':video_id, 'title':title, 'description':description0}])
    df_meta = pd.concat([df_meta, row_info], ignore_index=True)
    df_meta.to_csv(fp_meta, index=False)


if __name__ == "__main__":
    # list of Youtube video ids (downloaded manually)
    video_id_list = ["mscgmmgMs3M", "BUCIu4KwcAw", "ZbTbvNjFpMM", 'dx4VKF9DT5Y']
    for video_id in video_id_list:
        vector_storize_youtube_vid(video_id)
