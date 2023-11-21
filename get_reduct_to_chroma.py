"""
This code downloads the transcript from a YouTube video and saves the output
    in a format that can be used by our streamlit app
"""


# This part of the code brings in my OpenAI key , which I have stored as an
#   environment variable as OPENAI_API_KEY.  We could choose to use a fully
#   open source model such as one in Huggingface and/or SBERT
# from dotenv import load_dotenv
# load_dotenv() 

# There are loaders in langchain for Youtube videos, but I didn't like the way
#   it did its default chunking, so I am doing it manually
# from langchain.document_loaders import DataFrameLoader
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from youtube_transcript_api import YouTubeTranscriptApi as yta
# from pytube import YouTube

import pandas as pd

from pyprojroot import here
import json
import pickle

import tkinter as tk
from tkinter import messagebox

from sentence_transformers import SentenceTransformer

def ask_user():
    # Create a root window but keep it hidden
    root = tk.Tk()
    root.withdraw()

    # Popup message box
    response = messagebox.askyesno("Video already exists", "This video has previously been processed. Do you want to overwrite what is there?")

    # Destroy the root window after response
    root.destroy()

    return response

def vector_storize_youtube_vid(fp1):
    """ This is the function that puts the data into a vector store.  If I was 
        creating an app with data this small, I'd probably just store the data
        in a pickle file, but I am keeping this way for the example
    """
    fp_storing = here()/'video_data'
    if not fp_storing.exists():
        fp_storing.mkdir()

    title = fp1.stem[:-18]
    video_id = fp1.stem[-16:-5]

    my_video_path = fp_storing/video_id

    if not my_video_path.exists():
        my_video_path.mkdir()
    else:
        user_response = ask_user()
        if not user_response:
            return
    # Continue with the rest of your script

    
    # import reduct output
    with open(fp1, 'r') as f1:
        transcript_words = json.load(f1)

    # agg to segment
    transcript_segs = []
    for segment0 in transcript_words['segments']:
        out_dict = {}
        out_dict['start'] = segment0['start']
        out_dict['end'] = segment0['end']
        text_output = ''
        for wd0 in segment0['wdlist']:
            text_output += wd0['word']
        out_dict['text0'] = text_output
        transcript_segs.append(out_dict)

    # Reformat as dataframe for manipulation
    df_tx = pd.DataFrame(transcript_segs)

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
        my_text = ' '.join(df_minute['text0'])
        # get exact moment start / end of minute
        start_seconds = df_minute['start'].min()
        end_seconds = df_minute['end'].max()
        out_dict = {'minute_id':i,'text0':my_text,'start_seconds':start_seconds, 'end_seconds':end_seconds}
        dict_texts.append(out_dict)
    df_texts = pd.DataFrame(dict_texts)

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    embeddings = model.encode(df_texts['text0'].values)

    # # Move it into langchain
    # loader = DataFrameLoader(df_texts, page_content_column='text0')
    # docs = loader.load()
    
    # # save ChromaDB locally
    # vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(), persist_directory=str(here()/"chroma_db"/video_id))
    
    # fp_texts = here()/'text_dfs'
    # fp_storing = here()/'video_data'
    # if not fp_storing.exists():
    #     fp_storing.mkdir()

    # my

    # Save raw transcript by minute, too
    df_tx.to_pickle(my_video_path/f'{video_id}_raw.pkl')
    df_texts.to_pickle(my_video_path/f'{video_id}_minute.pkl')
    with open(my_video_path/f'{video_id}_embed.pkl', 'wb') as f1:
        pickle.dump(embeddings, f1)
    
    # Save metadata for streamlit
    fp_meta = here()/'metadata.csv'
    if fp_meta.exists():
        df_meta = pd.read_csv(fp_meta)
    else:
        df_meta = pd.DataFrame(columns = ['video_id', 'title'])
    
    if video_id in df_meta['video_id']:
        df_meta = df_meta.loc[df_meta['video_id'] != video_id].copy().reset_index(drop=True)

    row_info = {'video_id':video_id, 'title':title}
    df_size = df_meta.shape[0]
    df_meta.loc[df_size] = row_info
    df_meta.to_csv(fp_meta, index=False)


if __name__ == "__main__":
    reduct_folder = here()/'reduct_videos'
    for fp1 in reduct_folder.iterdir():
        print(fp1.stem)
        vector_storize_youtube_vid(fp1)
