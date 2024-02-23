import io
import time
import colorsys
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.dataframe_explorer import dataframe_explorer

from pyvis.network import Network

st.set_page_config(page_title="Semantic Search Helper", page_icon="2️⃣")

with st.sidebar:

    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/karlgottfried/SemHarmoHelper/blob/main/app.py)"


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
                df_data = pd.read_csv(data_upload)
            elif data_upload.name.endswith('.xlsx') or data_upload.name.endswith('.xls'):
                df_data = pd.read_excel(data_upload)
            else:
                df_data = None
            return df_data

        return None


def get_similarity_dataframe(df_in, cosine_sim, item_column, questionnaire_column):
    results = []
    with st.spinner("Building pairs"):
        for i, row_i in enumerate(df_in.itertuples()):
            questionnaire_value_i = getattr(row_i, questionnaire_column)
            item1 = df_in[item_column].iloc[i]
            for j, row_j in enumerate(df_in.itertuples()):
                questionnaire_value_j = getattr(row_j, questionnaire_column)
                item2 = df_in[item_column].iloc[j]
                if j <= i or questionnaire_value_i == questionnaire_value_j:
                    continue

                similarity_score = cosine_sim[i][j]
                # if similarity_score >= threshold:
                results.append({
                    'Questionnaire 1': questionnaire_value_i,
                    'Item 1': item1,
                    'Questionnaire 2': questionnaire_value_j,
                    'Item 2': item2,
                    'Similarity score': similarity_score
                })

                # parent_obj.info(f"Questionnaire '{row_i.questionnaire}', Item '{row_i.item_text}' and Questionnaire
                # '{row_j.questionnaire}', Item '{row_j.item_text}' have similarity score: {similarity_score}\n")
                print(f"{len(results)} pairs build!")

                print(
                    f"Questionnaire '{questionnaire_value_i}', Item '{item1}' "
                    f"and Questionnaire '{questionnaire_value_j}', Item '{item2}' "
                    f"have similarity score: {similarity_score}\n")
        results.sort(key=lambda x: x['Similarity score'], reverse=True)
    return pd.DataFrame(results)


def calculate_embeddings(data_in, model_name, sentences_in, model_selected):
    with st.spinner(f"calculate embeddings for column {st.session_state.selected_item_column}"):
        if model_selected == "SBERT":

            model_in = SentenceTransformer(model_name)
            output = model_in.encode(sentences=sentences_in.tolist(),
                                     show_progress_bar=True,
                                     normalize_embeddings=True)
            data_in["embedding"] = list(output)

        if model_selected == "ADA":
            data_in["embedding"] = sentences_in.apply(lambda x: get_embedding(x, model_name))
        return data_in


def calculate_similarity(data_in, selected_column):
    with st.spinner(f"calculate {selected_column}"):
        if len(data_in["embedding"]) > 0:
            cos_sim_1 = cosine_similarity(data_in["embedding"].tolist())
            return cos_sim_1


def get_embedding(text, model_in):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model_in).data[0].embedding


def dataframe_with_selections(df_in):
    df_with_selections = df_in.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        use_container_width=True,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df_in.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


def add_to_selection(df_in):
    st.session_state.selected_data = pd.concat([st.session_state.selected_data, df_in]).drop_duplicates()


@st.cache_data
def convert_df_to_excel(df_in):
    if df_in.empty:
        return None

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_in.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output.read()


@st.cache_data
def convert_df(df_in):
    return df_in.to_csv(index=False).encode('utf-8')


# Function zum Berechnen der Kantenbreite basierend auf dem Gewicht
def calculate_edge_width(weight, max_width=20, min_width=1):
    return max_width if weight == 1 else max(min_width, weight * max_width)


def get_graph(df_in, treshold_in):
    got_net = Network(height="600px", width="100%", font_color="black")
    got_net.toggle_physics(False)

    # Set the physics layout of the network
    got_net.barnes_hut()
    print(treshold_in)

    for index, row in df_in.iterrows():
        src = row['Item 1']
        s_q_label = row["Questionnaire 1"]
        dst = row['Item 2']
        t_q_label = row["Questionnaire 2"]
        w = row['Similarity score']

        # Calculate the width of the edge based on the similarity score
        width = calculate_edge_width(round(w,2))

        # Add an edge only if the similarity score is above the threshold
        # and the source and target questions are from different questionnaires
        if round(w,2) >= treshold_in:
            got_net.add_node(f"{src} ({s_q_label})", f"{src} ({s_q_label})", title=s_q_label)
            got_net.add_node(f"{dst} ({t_q_label})",f"{dst} ({t_q_label})", title=t_q_label)
            got_net.add_edge(f"{src} ({s_q_label})", f"{dst} ({t_q_label})", value=round(w,2), width=width)
            print(src, " ", dst ," ",index, " ", w)
    # Generate neighbor map for hover data
    neighbor_map = got_net.get_adj_list()

    # Set options for nodes and interactions
    got_net.set_options('''
    {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4
        },
        "interaction": {
            "dragNodes": true
        }
    }
    ''')

    # Add neighbor data to node hover data
    for node in got_net.nodes:
        node["value"] = len(neighbor_map[node["id"]])

    # Show the network
    got_net.show("graph.html", notebook=False)


####################################

if 'data' not in st.session_state:
    st.session_state.data = None
if 'embedding' not in st.session_state:
    st.session_state.embedding = None
if 'similarity' not in st.session_state:
    st.session_state.similarity = None

if 'selected_item_column' not in st.session_state:
    st.session_state.selected_item_column = None

if 'selected_questionnaire_column' not in st.session_state:
    st.session_state.selected_questionnaire_column = None

if 'selected_data' not in st.session_state:
    st.session_state.selected_data = pd.DataFrame()

if "df_filtered" not in st.session_state:
    st.session_state.df_filtered = None

#############

st.title("📒 Semantic Search Helper")
# select_tab, view_tab, store_tab = st.tabs(['Load Sentence Data', 'Build Embeddings', 'View Similarity'])


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

load_tab, embedding_tab, pair_tab, similarity_tab = st.tabs(
    ['Step 1: Load Sentence Data', 'Step 2: Build Embeddings', 'Step 3: Build Similarity Pairs',
     "Step 4: Select and Explore Pairs"])

with load_tab:
    input_in = st.radio("Use:", ["LOINC-Selection", "New Upload"])

    if input_in == "LOINC-Selection" and st.session_state["loincdf"] is not None:
        loaded_file = st.session_state["loincdf"]

    if input_in == "New Upload" or st.session_state["loincdf"] is None:
        loaded_file = get_data()

    if loaded_file is not None:
        with st.expander("Show your uploaded data"):
            st.write(loaded_file)
        selected_item_column = st.selectbox("Select the columns with your sentences.", loaded_file.columns)
        selected_questionnaire_column = st.selectbox("Select the columns with your questionnaire names.",
                                                     loaded_file.columns)

        with st.expander("Show column data"):
            st.write(loaded_file[selected_item_column].tolist())
            st.write(f'Selected  column {selected_item_column} cotains '
                     + str(len(loaded_file[selected_item_column])) + ' rows')

        if st.button('use data', type='primary', key="save_data"):
            st.session_state.data = loaded_file
            st.session_state.selected_item_column = selected_item_column
            st.session_state.selected_questionnaire_column = selected_questionnaire_column
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
            model = embedding_container.selectbox("Select a model to process",
                                                  ["text-embedding-ada-002", "text-embedding-3-small"])
            openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

            if not openai_api_key:
                embedding_container.info("Please add your OpenAI API key to continue.")

            if openai_api_key:
                client = OpenAI(api_key=openai_api_key)
                df = st.session_state.data
                sentences = df[st.session_state.selected_item_column].tolist()

                if embedding_container.button(f"Calculate {model_options} embeddings",
                                              type='primary', key=f"{model_options}_button"):
                    start_time_em = time.time()
                    df_embedding = calculate_embeddings(df, model, df[selected_item_column], model_options)

                    if df_embedding is not None:
                        st.write(df_embedding[[selected_item_column, "embedding"]])
                        # if embedding_container.button("Save", type='primary',key='save_emb'):
                        st.session_state.embedding = df_embedding
                        # embedding_container.info(f"Your data was stored move to tab 'View Similarity'")
                        end_time_em = time.time()
                        duration_em = end_time_em - start_time_em
                        duration_minutes_em = duration_em / 60

                        emb_status.empty()
                        emb_status = emb_status_container.success('Step 2 done! Embeddings build containing '
                                                                  + str(len(st.session_state.embedding)) +
                                                                  " rows with a vector size of " +
                                                                  str(len(df_embedding.embedding[0]))
                                                                  + f' in {duration_minutes_em:.2f} minutes')

        if model_options == "SBERT":
            model = embedding_container.selectbox("Select a model to process",
                                                  ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'])

            # model_SBERT_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            selected_item_column = st.session_state.selected_item_column
            df = st.session_state.data
            sentences = df[selected_item_column].tolist()

            if embedding_container.button(f"Calculate {model_options} embeddings",
                                          type='primary', key=f"{model_options}_button"):
                start_time_em = time.time()
                df_embedding = calculate_embeddings(df, model, df[selected_item_column], model_options)

                if df_embedding is not None:
                    st.write(df_embedding[[selected_item_column, "embedding"]])
                    # if embedding_container.button("Save", type='primary',key='save_emb'):
                    st.session_state.embedding = df_embedding
                    # embedding_container.info(f"Your data was stored move to tab 'View Similarity'")
                    end_time_em = time.time()
                    duration_em = end_time_em - start_time_em
                    duration_minutes_em = duration_em / 60

                    emb_status.empty()
                    emb_status = emb_status_container.success('Step 2 done! Embeddings build containing '
                                                              + str(len(st.session_state.embedding)) +
                                                              " rows with a vector size of " +
                                                              str(len(df_embedding.embedding[0]))
                                                              + f' in {duration_minutes_em:.2f} minutes')

with pair_tab:
    if st.session_state.data is None and st.session_state.embedding is None:
        st.info('You have to select data and build embeddings in order to view similarity')
    else:
        pair_container = st.container()
        data = st.session_state.embedding
        selected_item_column = st.session_state.selected_item_column
        selected_questionnaire_column = st.session_state.selected_questionnaire_column

        if st.button(f'Calculate Similarity for all pairs of column {selected_item_column}', type='primary',
                     key="calc_similarity"):
            start_time_sim = time.time()
            cosine_similarity = calculate_similarity(data, selected_item_column)
            df_sim = get_similarity_dataframe(data, cosine_similarity, selected_item_column,
                                              selected_questionnaire_column)
            st.session_state.similarity = df_sim
            end_time_sim = time.time()
            duration_sim = end_time_sim - start_time_sim
            duration_minutes = duration_sim / 60

            sim_status.empty()
            sim_status = sim_status_container.success('Step 3 done! Similarity scores build between '
                                                      + str(len(st.session_state.similarity))
                                                      + f' pairs in {duration_minutes:.2f} minutes')
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
            st.session_state.df_filtered = dataframe_explorer(df_sim, case=False)
            st.dataframe(st.session_state.df_filtered, use_container_width=True)

            filter_excel = convert_df_to_excel(st.session_state.df_filtered)
            if filter_excel is not None:
                st.download_button(
                    label=f"Download whole filtered table with "
                          f"{len(st.session_state.df_filtered)} elements as Excel file",
                    data=filter_excel,
                    file_name='ssh_download_filtered_selection_table.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            st.subheader('Filtered DataFrame')
            # df_filtered = df_sim.query(query_string)
            size = len(st.session_state.df_filtered)
            st.info(f"Filtered {size} elements")

            selection = dataframe_with_selections(st.session_state.df_filtered)

            if st.button('➕ Add selected candidates to final selection'):
                add_to_selection(selection)

            st.divider()
            if 'selected_data' in st.session_state and not st.session_state.selected_data.empty:
                st.subheader("Final Harmonisation Candidates")
                st.dataframe(st.session_state.selected_data, use_container_width=True)

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

        with sim_container.expander("Analyse"):
            physics = st.selectbox('Choose a Graph', ["Filter", "All"])
            treshold = st.slider("value", 0.5, 1.0, 0.1)

            if st.button(f'View Graph', type='primary',
                         key="calc_graph"):
                with st.spinner("View Graph"):

                    if physics == "All":
                        get_graph(df_sim, treshold)
                    if physics == "Filter":
                        get_graph(st.session_state.df_filtered, treshold)

                    HtmlFile = open("graph.html", 'r', encoding='utf-8')
                    source_code = HtmlFile.read()
                    components.html(source_code, height=600)
