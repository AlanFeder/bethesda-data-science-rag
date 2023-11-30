""" This is the live streamlit app I showed during the meetup
"""


# Packages
import streamlit as st
import numpy as np
import pandas as pd
from pyprojroot import here
import pickle

from pyprojroot import here
import openai
from openai import OpenAI

# I don't need this part when I run it locally, it has something to do with the
#   streamlit cloud environment for chroma 
#   https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import tiktoken

# This needs to be at the top of most streamlit apps
st.set_page_config(page_title='Q&A of Previous Meetups',
                    layout="wide",
                    initial_sidebar_state="expanded"
                    )


# If you run it locally, you can just do a dotenv to store your OpenAI API Key
# as an environment variable 
with st.sidebar:
    api_key = st.text_input(
        label="Input your OpenAI API Key (don't worry, this isn't stored anywhere)",
        type='password'
    )

    st.markdown('''If you don't have an OpenAI API key, you can sign up [here](https://platform.openai.com/account/api-keys).''')


def split_into_consecutive(arr):
    """ I got the following function from the following GPT prompt
    > I have a numpy array of numbers.  some are consecutive, some are not (e.g. 3, 4, 5, 6, 10, 11, 12, 13, 19, 20, 21).  How can I split it into a list of arrays, where each sub-element is just the consecutive parts (e.g. [[3,4,5,6],[10,11,12,13],[19,20,21]])?

    The goal is so that if there are consecutive minutes, and I look at the before-and-after, they are combined as needed
    """
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

# Get list of all available videos
fp_meta = here()/'metadata.csv'
df_meta = pd.read_csv(fp_meta)

# Selectbox for choosing which video to query
video_format_funct = lambda x: df_meta.set_index('video_id')['title'].to_dict()[x]
video_id = st.selectbox(
    "Which talk do you want to ask questions about?",
    options=df_meta['video_id'],
    index=2,
    format_func=video_format_funct
)

# radio button for choosing between gpt-4 and gpt-3.5 . I'd love to build in 
#   ability to use an open-source model, or claude 2.1
model_format_funct1 = lambda x: 'GPT 4 (more accurate)' if 'gpt-4' in x else 'GPT 3.5 (cheaper)'
model_format_funct2 = lambda x: 'GPT 4' if 'gpt-4' in x else 'GPT 3.5'
model_choice = st.radio(
    'Which ChatGPT version do you want to use?',
    options=['gpt-3.5-turbo-1106', 'gpt-4-1106-preview'],
    index=1,
    format_func = model_format_funct1)

# st.write(model_choice)
embed_model = 'ada'#'sentence-transformers/all-mpnet-base-v2'
question = st.text_area(
    label = 'What question do you want to ask of the meetup talk?'
)
if api_key == '':
    st.error('Please input an Openai API Key in the sidebar area')
else:
    openai.api_key = api_key# os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key = api_key)
    title0 = df_meta.loc[df_meta['video_id'] == video_id, 'title'].iloc[0]

    fp_storing = here()/'video_data'
    file_out = f'{video_id}.pkl'
    my_video_path = fp_storing/file_out
    with open(my_video_path, 'rb') as f1:
        df_tx_raw, df_tx_minute, doc_embeddings = pickle.load(f1)


    # vectorstore = Chroma(persist_directory=str(here()/'chroma_db'/video_id), embedding_function=OpenAIEmbeddings(openai_api_key=api_key))
    cost = 0

    if st.button('Submit Question'):
        st.subheader('Your question')
        st.markdown(question)
        st.markdown(f'Video chosen: {video_format_funct(video_id)}')
        st.divider()
        with st.spinner('Asking the AI...'):


            encoding = tiktoken.get_encoding("cl100k_base")
            n_tok = len(encoding.encode(question))
            if embed_model in ('ada', 'text-embedding-ada-002'):
                # cost += 0.0001 * n_tok / 1000
                q_embedding = client.embeddings.create(model='text-embedding-ada-002', input=question)
                q_embedding = np.array(q_embedding.data[0].embedding)
            else:
                model = SentenceTransformer(embed_model)
                q_embedding = model.encode(question)

            cos_sims = np.dot(doc_embeddings, q_embedding)

            df_tx_minute['cos_sims'] = cos_sims
            top_texts = df_tx_minute.sort_values('cos_sims', ascending=False, ignore_index=True).head(4).copy()

            minute_ids = top_texts['minute_id'].values

            main_minute_id = minute_ids[0]
            minute_list = np.unique(np.concatenate([minute_ids, minute_ids + 1, minute_ids-1]))

            minute_list_list = split_into_consecutive(minute_list)
            # fp_texts = here()/'text_dfs'
            # df_tx_raw = pd.read_pickle(fp_texts/f'{video_id}_raw.pkl')
            # df_tx_minute = pd.read_pickle(fp_texts/f'{video_id}_minute.pkl')

            df_main_minute = df_tx_raw.loc[(df_tx_raw['minute1']==main_minute_id)|(df_tx_raw['minute2']==main_minute_id)]
            start_time = df_main_minute['start'].iloc[0]
            # end_time = df_main_minute['start'].iloc[-1] + df_main_minute['duration'].iloc[-1]


            context_text_list = []
            for i, minute_list0 in enumerate(minute_list_list):
                content0 = f'Context Transcript Section #{i+1}:  '
                first_min = minute_list0.min()
                last_min = minute_list0.max()
                content0 += df_tx_raw.loc[(df_tx_raw['minute1']>= first_min)&(df_tx_raw['minute2']<= last_min), 'text0'].str.cat(sep=' ')
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


st.divider()

st.markdown('''This streamlit app was created for Alan Feder's [talk at the Bethesda Data Science Meetup](https://www.meetup.com/bethesda-data-science-networking-meetup/events/297264697/).  The slides he used are [here](https://bit.ly/bethesda-ds-presentation).  The Github repository that houses all the code is [here](https://github.com/AlanFeder/bethesda-data-science-rag) -- feel free to fork it and use it on your own!''')
st.markdown("""I will add a link to the YouTube when available.""")


