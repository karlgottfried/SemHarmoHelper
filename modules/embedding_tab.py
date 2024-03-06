# Import necessary libraries
import streamlit as st  # Streamlit for building web apps
from openai import OpenAI  # OpenAI for GPT models
from config import *  # Import configurations
import time  # Time library for timing operations
from sentence_transformers import SentenceTransformer  # For sentence embeddings
from umap.umap_ import UMAP  # UMAP for dimensionality reduction
import plotly.express as px  # Plotly Express for interactive plots
# from sklearn.decomposition import PCA  # PCA for alternative dimensionality reduction (commented out)
from bertopic import BERTopic  # BERTopic for topic modeling (commented out)
import pandas as pd  # Pandas for data manipulation
import hdbscan
import numpy as np

# import   # HDBSCAN for clustering (commented out)
# initialize_session_state()  # Initialize Streamlit session state (commented out)

def show_explore_embedding_tab():
    with st.expander("Explore Embedding Building Step", expanded=True):
        # Annehmen, dass EMBEDDING und SELECTED_ITEM_COLUMN bereits definiert sind
        sentence = st.session_state[EMBEDDING][st.session_state[SELECTED_ITEM_COLUMN]].tolist()

        # Trainieren von BERTopic
        topic_model = BERTopic(min_topic_size=10)
        topics, probs = topic_model.fit_transform(sentence, np.array(list(st.session_state[EMBEDDING][EMBEDDING])))
        #hierarchical_topics = topic_model.hierarchical_topics(sentence)

        st.data_editor(topic_model.get_topic_info()[["Topic","Representation"]], use_container_width=True, hide_index=True)

        # Dimensionalitätsreduktion und Clustering
        reduced_embeddings = UMAP(n_neighbors=10, n_components=3, min_dist=0.0, metric='cosine').fit_transform(np.array(list(st.session_state[EMBEDDING][EMBEDDING])))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
        clusterer.fit(reduced_embeddings)

        # Displaying the filtered table

        # Vorbereitung der Daten für die Visualisierung
        df_reduced = pd.DataFrame(reduced_embeddings, columns=['UMAP 1', 'UMAP 2', "UMAP 3"])
        df_reduced['Sentence'] = sentence  # Fügen Sie Sätze als Spalte hinzu
        df_reduced['Topic'] = [f"Topic {topic}" for topic in topics]  # Fügen Sie Themen als Spalte hinzu

        #st.write(topic_model.get_topic_info().columns())


        # Visualisierung der reduzierten Einbettungen mit Plotly
        fig_3d = px.scatter_3d(
            df_reduced, x="UMAP 1", y="UMAP 2", z="UMAP 3",
            color='Topic', hover_data=["Sentence","Topic"],  # Fügen Sie hier die Sätze und Themen zum Hover hinzu
            color_continuous_scale=px.colors.qualitative.Bold,
            labels={"color": "Topic"}  # Benennen Sie die Farblegende um
        )
        fig_3d.update_traces(hovertemplate='Sentence: %{customdata[0]}<br>Topic: %{customdata[1]}')
        st.plotly_chart(fig_3d, use_container_width=True)




def show_explore_embedding_tab_2():
    with st.expander("Explore Embedding Building Step", expanded=True):
        st.data_editor(st.session_state[EMBEDDING][[st.session_state[SELECTED_ITEM_COLUMN], EMBEDDING]], use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("Dimensionality Reduction to 3D with [UMAP](https://arxiv.org/abs/1802.03426)")

        umap_model = UMAP(n_neighbors=10, n_components=3, min_dist=0.0, metric='cosine')
        reduced_embeddings = umap_model.fit_transform(list(st.session_state[EMBEDDING][EMBEDDING]))

        df_reduced = pd.DataFrame(reduced_embeddings, columns=['UMAP 1', 'UMAP 2', "UMAP 3"])
        df_reduced['Sentence'] = st.session_state[EMBEDDING][st.session_state[SELECTED_ITEM_COLUMN]]

        # Clustering mit HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
        clusterer.fit(reduced_embeddings)
        df_reduced['cluster'] = clusterer.labels_

        # Visualisierung der Cluster
        fig_3d = px.scatter_3d(
            df_reduced, x="UMAP 1", y="UMAP 2", z="UMAP 3",
            color="cluster", hover_data={'UMAP 1': False, 'UMAP 2': False, 'UMAP 3': False, 'Sentence': True},
            color_continuous_scale=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_3d, use_container_width=True)


def get_embedding(text, model):
    """
    Fetches embeddings for a given text using a specified model.

    Args:
        text (str): Text to be encoded.
        model (OpenAI Model): Model used for encoding.

    Returns:
        list: The embedding vector.
    """
    text = text.replace("\n", " ")  # Normalize text by replacing newlines
    # Retrieve embedding from OpenAI API
    return OpenAI.api_key.embeddings.create(input=[text], model=model).data[0]['embedding']


def calculate_embeddings(data_in, model_name, sentences_in, model_selected):
    # Display spinner during embeddings calculation
    with st.spinner(f"calculate embeddings for column {st.session_state[SELECTED_ITEM_COLUMN]}"):
        # If Sentence Transformer model is selected
        if model_selected == MODEL_SBERT:
            model_in = SentenceTransformer(model_name)  # Load model
            # Calculate embeddings and normalize them
            output = model_in.encode(sentences=sentences_in.tolist(), show_progress_bar=True, normalize_embeddings=True)
            data_in[EMBEDDING] = list(output)  # Store embeddings in DataFrame
            st.session_state["step2_completed"] = True

        # If OpenAI's ADA model is selected
        if model_selected == MODEL_ADA:
            # Calculate embeddings using the OpenAI model
            data_in[EMBEDDING] = sentences_in.apply(lambda x: get_embedding(x, model_name))
            st.session_state["step2_completed"] = True
        return data_in  # Return DataFrame with embeddings


def show_embedding_tab():
    # Check if metadata is selected, prompt user otherwise
    if st.session_state.metadata is None:
        st.info('You have to select data for viewing and editing. Switch to Step 1 tab first.')
    else:
        # Container for embeddings section
        embedding_container = st.container()
        # Radio buttons for model selection
        model_options = embedding_container.radio("Data Usage", options=[MODEL_ADA, MODEL_SBERT], horizontal=True)
        st.session_state["model_used"] = model_options  # Store selected model

        # If ADA model is selected
        if model_options == MODEL_ADA:
            # Dropdown for selecting ADA model version
            model = embedding_container.selectbox("Select a model to process",
                                                  ["text-embedding-ada-002", "text-embedding-3-small"])
            # Input for OpenAI API key
            openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

            if openai_api_key:  # If API key is provided
                client = OpenAI(api_key=openai_api_key)  # Initialize OpenAI client
                df = st.session_state.metadata  # Load metadata
                sentences = df[st.session_state.selected_item_column].tolist()  # Extract sentences

                # Button to trigger embeddings calculation
                if embedding_container.button(f"Calculate {model_options} embeddings", type='primary',
                                              key=f"{model_options}_button"):
                    start_time_em = time.time()  # Start timing
                    # Calculate embeddings and update session state
                    df_embedding = calculate_embeddings(df, model, df[st.session_state.selected_item_column],
                                                        model_options)
                    st.session_state[EMBEDDING] = df_embedding

                    if df_embedding is not None:  # If embeddings are calculated
                        # Display editable table of embeddings
                        st.data_editor(df_embedding[[st.session_state.selected_item_column, EMBEDDING]],
                                       use_container_width=True)
                        end_time_em = time.time()  # End timing
                        # Calculate and store duration
                        st.session_state.duration_em = end_time_em - start_time_em
                        st.session_state.duration_minutes_em = st.session_state.duration_em / 60
                        st.session_state['step2_completed'] = True  # Mark step as completed

        # If Sentence Transformer model is selected
        if model_options == MODEL_SBERT:
            # Dropdown for selecting Sentence Transformer model
            model = embedding_container.selectbox("Select a model to process",
                                                  ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'])
            selected_item_column = st.session_state.selected_item_column  # Column with text
            df = st.session_state.metadata  # Load metadata
#            sentences = df[selected_item_column].tolist()  # Extract sentences

            # Button to trigger embeddings calculation
            if embedding_container.button(f"Calculate {model_options} embeddings", type='primary',
                                          key=f"{model_options}_button"):
                start_time_em = time.time()  # Start timing
                # Calculate embeddings and update session state
                df_embedding = calculate_embeddings(df, model, df[selected_item_column], model_options)
                st.session_state[EMBEDDING] = df_embedding
                end_time_em = time.time()  # End timing
                # Calculate and store duration
                duration_em = end_time_em - start_time_em
                duration_minutes_em = duration_em / 60
                st.session_state['step2_completed'] = True  # Mark step as completed

                show_explore_embedding_tab()  # Show the embedding exploration tab
