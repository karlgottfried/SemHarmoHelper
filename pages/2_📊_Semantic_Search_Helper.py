import io
import os

import pandas as pd
import streamlit as st
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_condition_tree import condition_tree, config_from_dataframe

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

#from scikit-learn.metrics.pairwise import cosine_similarity

#from openai.embeddings_utils import cosine_similarity, get_embedding

st.set_page_config(page_title="Semantic Search Helper", page_icon="📈")


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/karlgottfried/SemHarmoHelper/blob/main/app.py)"
    #"[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)]()"

def get_data():
    """
    Upload data via a file.

    Returns:
    - df: DataFrame containing the uploaded data or None if no data was uploaded
    """

    # File uploader for data file
    file_types = ["csv", "xlsx", "xls"]
    data_upload = st.file_uploader("Upload a data file", type=file_types)
    wait_msg = 'Loading data...'

    with st.spinner(wait_msg):

        if data_upload:
            # Check the type of file uploaded and read accordingly
            if data_upload.name.endswith('.csv'):
                df = pd.read_csv(data_upload)
            elif data_upload.name.endswith('.xlsx') or data_upload.name.endswith('.xls'):
                df = pd.read_excel(data_upload)
            else:
                df = None
            return df

        return None


def get_similarity_dataframe(df, cosine_sim, item_column, questionaire_column):
    results = []
    with st.spinner("Building pairs"):
        for i, row_i in enumerate(df.itertuples()):
            questionaire_value_i = getattr(row_i, questionaire_column)
            for j, row_j in enumerate(df.itertuples()):
                questionaire_value_j = getattr(row_j, questionaire_column)
                if j <= i or questionaire_value_i == questionaire_value_j:
                    continue

                similarity_score = cosine_sim[i][j]
                # if similarity_score >= threshold:
                results.append({
                    'Questionnaire 1': questionaire_value_i,
                    'Item 1': getattr(row_i, item_column),
                    'Questionnaire 2': questionaire_value_j,
                    'Item 2': getattr(row_j, item_column),
                    'Similarity score': similarity_score
                })

                #parent_obj.info(f"Questionnaire '{row_i.questionnaire}', Item '{row_i.item_text}' and Questionnaire '{row_j.questionnaire}', Item '{row_j.item_text}' have similarity score: {similarity_score}\n")
                print(len(results))
                print(f"Questionnaire '{questionaire_value_i}', Item '{getattr(row_i, item_column)}' and Questionnaire '{questionaire_value_j}', Item '{getattr(row_i, item_column)}' have similarity score: {similarity_score}\n")
        results.sort(key=lambda x: x['Similarity score'], reverse=True)
    return pd.DataFrame(results)



def calculate_embeddings(data, model_SBERT_name, sentences):
    with st.spinner(f"calculate embeddings for column {st.session_state.selected_item_column}"):
        model = SentenceTransformer(model_SBERT_name)
        output = model.encode(sentences=sentences,
                              show_progress_bar=True,
                              normalize_embeddings=True)
        data["embedding"] = list(output)
        return data

def calculate_similarity(data, selected_column):
    with st.spinner(f"calculate {selected_column}"):
        if len(data["embedding"]) > 0:
            cos_sim_1 = cosine_similarity(data["embedding"].tolist())
            return cos_sim_1

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

                # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
                    df_with_selections,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                    disabled=df.columns,
                )

                # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

def add_to_selection(df):
    st.session_state.selected_data = pd.concat([st.session_state.selected_data, df]).drop_duplicates()

@st.cache_data
def convert_df_to_excel(df):
        if df.empty:
            return None

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)
        return output.read()


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

####################################

if 'data' not in st.session_state:
    st.session_state.data = None
if 'embedding' not in st.session_state:
    st.session_state.embedding = None
if 'similarity' not in st.session_state:
    st.session_state.similarity = None

if 'selected_item_column' not in st.session_state:
    st.session_state.selected_item_column = None

if 'selected_questionaire_column' not in st.session_state:
    st.session_state.selected_questionaire_column = None

if 'selected_data' not in st.session_state:
    st.session_state.selected_data = pd.DataFrame()

#############

st.title("📒 Semantic Search Helper")
#select_tab, view_tab, store_tab = st.tabs(['Load Sentence Data', 'Build Embeddings', 'View Similarity'])

status_container = st.container()
if st.session_state.data is None:
    meta_status = status_container.info('Step 1 not done yet')
else:
    meta_status = status_container.success('Step 1 done! Data selected containing '
                                                       + str(len(st.session_state.data)) + ' rows')

emb_status_container = st.container()
if st.session_state.embedding is None:
    emb_status = emb_status_container.info('Step 2 not done yet')
else:
    emb_status = emb_status_container.success('Step 2 done! Embeddings build containing '
                                                       + str(len(st.session_state.embedding)) + ' rows')

sim_status_container = st.container()
if st.session_state.similarity is None:
    sim_status = sim_status_container.info('Step 3 not done yet')
else:
    sim_status = sim_status_container.success('Step 3 done! Similarity scores build between '
                                                       + str(len(st.session_state.similarity)) + ' pairs')

load_tab, embedding_tab, pair_tab, similarity_tab = st.tabs(['Step 1: Load Sentence Data', 'Step 2: Build Embeddings', 'Step 3: Build Similarity Pairs', "Step 4: View Pairs"])

with load_tab:
    loaded_file = get_data()
    if loaded_file is not None:
        with st.expander("Show your uploaded data"):
            st.write(loaded_file)
        selected_item_column = st.selectbox("Select the columns with your sentences.", loaded_file.columns)
        selected_questionaire_column = st.selectbox("Select the columns with your questionnaire names.", loaded_file.columns)


        with st.expander("Show column data"):
            st.write(loaded_file[selected_item_column].tolist())
            st.write(f'Selected  column {selected_item_column} cotains '
                     + str(len(loaded_file[selected_item_column])) + ' rows')

        if st.button('use data', type='primary', key="save_data"):
            st.session_state.data = loaded_file
            st.session_state.selected_item_column = selected_item_column
            st.session_state.selected_questionaire_column = selected_questionaire_column
            meta_status.empty()
            meta_status = status_container.success('Step 1 done! Data selected containing '
                                                       + str(len(st.session_state.data)) + ' rows')
with embedding_tab:
    if st.session_state.data is None:
        st.info('You have to select data for viewing and editing. Switch to Step 1 tab first.')
    else:
        embedding_container = st.container()
        model_options = embedding_container.radio(
                "Data Usage", options=["ADA", "SBERT"], horizontal=True
            )
        if model_options == "ADA":
            # Get data uploaded by the user
            if not openai_api_key:
                embedding_container.info("Please add your OpenAI API key in the sidebar to continue.")

            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = OpenAI.API
                df = st.session_state.data
                df['embedding'] = df[selected_item_column].combined.apply(
                    lambda x: get_embedding(x, model='text-embedding-ada-002'))

        if model_options == "SBERT":
            model = embedding_container.selectbox("Select a model to process",
                                 ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'])

            # model_SBERT_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            selected_item_column = st.session_state.selected_item_column
            df = st.session_state.data
            sentences = df[selected_item_column].tolist()

            if embedding_container.button(f"Calculate {model_options} embeddings", type='primary'):
                df_embedding = calculate_embeddings(df, model, sentences)

                if df_embedding is not None:
                    st.write(df_embedding[[selected_item_column, "embedding"]])
                    #if embedding_container.button("Save", type='primary',key='save_emb'):
                    st.session_state.embedding = df_embedding
                    #embedding_container.info(f"Your data was stored move to tab 'View Similarity'")
                    emb_status.empty()
                    emb_status = emb_status_container.success('Step 2 done! Embeddings build containing '
                                                       + str(len(st.session_state.embedding)) + " rows with a vector size of " + str(len(df_embedding.embedding[0])))
with pair_tab:
    if st.session_state.data is None and st.session_state.embedding is None:
        st.info('You have to select data and build embeddings for viewing similarity')
    else:
        pair_container = st.container()
        data = st.session_state.embedding
        selected_item_column = st.session_state.selected_item_column
        selected_questionaire_column = st.session_state.selected_questionaire_column

        if st.button(f'Calculate Similarity for all pairs of column {selected_item_column}', type='primary', key="calc_similarity"):
            cosine_similarity = calculate_similarity(data, selected_item_column)
            df_sim = get_similarity_dataframe(data, cosine_similarity, selected_item_column, selected_questionaire_column)
            st.session_state.similarity = df_sim

            sim_status.empty()
            sim_status = sim_status_container.success('Step 3 done! Similarity scores build between '
                                                       + str(len(st.session_state.similarity)) + ' pairs')
with similarity_tab:
    if st.session_state.similarity is None:
        st.info('You have to select data and build embeddings for viewing similarity')
    else:
        sim_container = st.container()
        df_sim = st.session_state.similarity

        with sim_container.expander("Filtering"):
            st.subheader('Initial DataFrame')
            st.dataframe(df_sim, use_container_width=True)

            st.subheader('Filter tree')
                #st.subheader
            st.markdown("Here you see a text")
            config = config_from_dataframe(df_sim)
            query_string = condition_tree(config)

            st.code(query_string)

            st.subheader('Filtered DataFrame')
            df_filtered = df_sim.query(query_string)
            size = len(df_filtered)
            st.info(f"Filtered {size} elements")

            selection = dataframe_with_selections(df_filtered)
            #st.subheader("Your selection:")
            #st.dataframe(selection, use_container_width=True)

            #AgGrid(df_filtered, fit_columns_on_grid_load=True)

            if st.button('➕ Add selected candidates to final selection'):
                add_to_selection(selection)

            st.divider()
            if 'selected_data' in st.session_state and not st.session_state.selected_data.empty:
                st.subheader("Final Harmonisation Candidates")
                st.dataframe(st.session_state.selected_data,  use_container_width=True)





            csv = convert_df(st.session_state.selected_data)
            st.download_button(
                label="Download Final Selection as CSV",
                data=csv,
                file_name='final_selection.csv',
                mime='text/csv'
            )

            excel = convert_df_to_excel(st.session_state.selected_data)
            if excel is not None:
                st.download_button(
                    label="Download Final Selection as Excel",
                    data=excel,
                    file_name='final_selection.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

