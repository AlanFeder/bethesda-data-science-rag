import streamlit as st
import numpy as np
import pandas as pd
from pyprojroot import here
from langchain.vectorstores import Chroma
# from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from pyprojroot import here
import openai
from openai import OpenAI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import tiktoken
# load_dotenv() 
st.set_page_config(page_title='Q&A of Previous Meetups',
                    layout="wide",
                    initial_sidebar_state="expanded"
                    )


with st.sidebar:
    api_key = st.text_input(
        label="Input your OpenAI API Key (don't worry, this isn't stored anywhere)",
        type='password'
    )


def split_into_consecutive(arr):
    # List to store the result
    result = []
    # Temporary list to store current sequence
    temp = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            # Continue the current sequence
            temp.append(arr[i])
        else:
            # Start a new sequence
            result.append(np.array(temp))
            temp = [arr[i]]

    # Append the last sequence
    result.append(np.array(temp))

    return result



st.title("Ask Questions of Previous Bethesda Data Science Speakers with ChatGPT!")


fp_meta = here()/'metadata.csv'
df_meta = pd.read_csv(fp_meta)

video_format_funct = lambda x: df_meta.set_index('video_id')['title'].to_dict()[x]


video_id = st.selectbox(
    "Which talk do you want to ask questions about?",
    options=df_meta['video_id'],
    format_func=video_format_funct
)
model_format_funct1 = lambda x: 'GPT 4 (more accurate)' if 'gpt-4' in x else 'GPT 3.5 (cheaper)'
model_format_funct2 = lambda x: 'GPT 4' if x =='gpt-4' else 'GPT 3.5'
model_choice = st.radio(
    'Which ChatGPT version do you want to use?',
    options=['gpt-3.5-turbo-1106', 'gpt-4-1106-preview'],
    index=1,
    format_func = model_format_funct1)

embed_model = 'ada'
question = st.text_area(
    label = 'What question do you want to ask of the meetup talk?'
)
if api_key == '':
    st.error('Please input an Openai API Key')
else:
    openai.api_key = api_key# os.getenv('OPENAI_API_KEY')
    title0 = df_meta.loc[df_meta['video_id'] == video_id, 'title'].iloc[0]
    vectorstore = Chroma(persist_directory=str(here()/'chroma_db'/video_id), embedding_function=OpenAIEmbeddings(openai_api_key=api_key))
    cost = 0

    if st.button('Submit Question'):
        st.subheader('Your question')
        st.markdown(question)
        st.markdown(f'Video chosen: {video_format_funct(video_id)}')
        st.divider()
        with st.spinner('Asking the AI...'):

            if embed_model == 'ada':
                encoding = tiktoken.get_encoding("cl100k_base")
                n_tok = len(encoding.encode(question))
                cost += 0.0001 * n_tok / 1000
            out_docs = vectorstore.similarity_search(question)



            minute_ids = np.array([doc0.metadata['minute_id'] for doc0 in out_docs])
            main_minute_id = minute_ids[0]
            minute_list = np.unique(np.concatenate([minute_ids, minute_ids + 1, minute_ids-1]))

            minute_list_list = split_into_consecutive(minute_list)
            fp_texts = here()/'text_dfs'
            df_tx_raw = pd.read_pickle(fp_texts/f'{video_id}_raw.pkl')
            # df_tx_minute = pd.read_pickle(fp_texts/f'{video_id}_minute.pkl')

            df_main_minute = df_tx_raw.loc[(df_tx_raw['minute1']==main_minute_id)|(df_tx_raw['minute2']==main_minute_id)]
            start_time = df_main_minute['start'].iloc[0]
            # end_time = df_main_minute['start'].iloc[-1] + df_main_minute['duration'].iloc[-1]


            context_text_list = []
            for i, minute_list0 in enumerate(minute_list_list):
                content0 = f'Context Transcript Section #{i+1}:  '
                first_min = minute_list0.min()
                last_min = minute_list0.max()
                content0 += df_tx_raw.loc[(df_tx_raw['minute1']>= first_min)&(df_tx_raw['minute2']<= last_min), 'text'].str.cat(sep=' ')
                context_text_list.append(content0)


            # context_text_list = []
            # for i, doc0 in enumerate(out_docs):
            #     context_text0 = f'Context Transcript Section #{i+1}:   '
            #     context_text0 += doc0.page_content
            #     context_text_list.append(context_text0)
            context_text = '\n\n'.join(context_text_list)

            system_prompt = f'''
            You are an assistant for question-answering tasks. You need to answer questions about the YouTube video with the title "{title0}".  

            Use the following selections from the transcript as context to answer the question.  Only answer based on the information included in the context below. If you don't know the answer, just say that you don't know, don't make anything up. Use three sentences maximum and keep the answer concise.
            '''

            user_prompt = f'''
            ---
            QUESTION: {question}
            ---
            CONTEXT:
            {context_text}
            ---
            '''

            my_prompt = f'''
            {system_prompt}

            {user_prompt}
            '''
            client = OpenAI(api_key=api_key)
            if 'gpt-4' in model_choice:
                completion = client.chat.completions.create(
                    model=model_choice, 
                    temperature=0,
                    messages=[{'role':'system', 'content':system_prompt},
                            {'role':'user', 'content': user_prompt}]
                )
            else:
                completion = client.chat.completions.create(
                    model=model_choice, 
                    temperature=0,
                    messages=[{'role':'user', 'content':my_prompt}]
                )

            model_used = completion.model
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            out_message = completion.choices[0].message.content


            # cost_4 = cost35 = cost
            if 'gpt-4' in model_choice:
                cost += .01 * prompt_tokens / 1000
                cost += .03 * completion_tokens / 1000
            elif 'gpt-3.5-turbo' in model_choice:
                cost += .001 * prompt_tokens / 1000
                cost += .002 * completion_tokens / 1000

        st.subheader('Answer')
        st.markdown(out_message)
        st.caption(f'This query cost you ${cost:.4f} since you used {model_format_funct2(model_choice)}')

        st.markdown('This is the most likely relevant location in the video (although the answer above might have come from other sections as well)')

        width = 40
        side = max((100 - width) / 2, 0.01)

        _, container, _ = st.columns([side, width, side])
        container.video(f'http://youtube.com/watch?v={video_id}', start_time=int(start_time))
