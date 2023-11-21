"""
This code downloads the transcript from a YouTube video and saves the output
    in a format that can be used by our streamlit app
"""



import pandas as pd

from pyprojroot import here
import json
import pickle

import tkinter as tk
from tkinter import messagebox

from sentence_transformers import SentenceTransformer

def should_overwrite():
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

    # make an output folder if it doesn't exist
    fp_storing = here()/'video_data'
    if not fp_storing.exists():
        fp_storing.mkdir()

    # get title and video id from file name
    title = fp1.stem[:-18]
    video_id = fp1.stem[-16:-5]

    # make a subfolder for data
    file_out = f'{video_id}.pkl'
    my_video_path = fp_storing/file_out

    if my_video_path.exists():
        to_overwrite = should_overwrite()
        if not to_overwrite:
            return
    
    # import reduct output
    with open(fp1, 'r') as f1:
        transcript_words = json.load(f1)

    # aggregate to segment
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

    # get data embeddings
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(df_texts['text0'].values)


    # Save all data in pickle files

    # df_tx.to_pickle(my_video_path/f'{video_id}_raw.pkl')
    # df_texts.to_pickle(my_video_path/f'{video_id}_minute.pkl')
    # with open(my_video_path/f'{video_id}_embed.pkl', 'wb') as f1:
    #     pickle.dump(embeddings, f1)

    with open(my_video_path, 'wb') as f1:
        pickle.dump((df_tx, df_texts, embeddings), f1)
    
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
