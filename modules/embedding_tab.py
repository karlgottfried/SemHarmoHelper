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
    # Expander UI component for exploring embeddings
    with st.expander("Explore Embedding Building Step", expanded=True):
        # Assumes EMBEDDING and SELECTED_ITEM_COLUMN are predefined in the Streamlit session state
        sentences = st.session_state[EMBEDDING][st.session_state[SELECTED_ITEM_COLUMN]].tolist()

        # Topic modeling with BERTopic
        topic_model = BERTopic(min_topic_size=10)
        topics, probs = topic_model.fit_transform(sentences, np.array(list(st.session_state[EMBEDDING][EMBEDDING])))
        #hierarchical_topics = topic_model.hierarchical_topics(sentence)

        # Display topic information in a data editor widget
        st.data_editor(topic_model.get_topic_info()[["Topic", "Representation"]], use_container_width=True, hide_index=True)

        # Dimensionalitätsreduktion und Clustering
        reduced_embeddings = UMAP(n_neighbors=10, n_components=3, min_dist=0.0, metric='cosine').fit_transform(np.array(list(st.session_state[EMBEDDING][EMBEDDING])))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)

        # Preparing data for visualization
        df_reduced = pd.DataFrame(reduced_embeddings, columns=['UMAP 1', 'UMAP 2', "UMAP 3"])
        df_reduced['Sentence'] = sentences  # Adding sentences as a column
        df_reduced['Topic'] = [f"Topic {topic}" for topic in topics]  # Adding topics as a column

        st.data_editor(df_reduced[["Sentence","Topic"]], use_container_width=True, hide_index=True)

        # Visualize reduced embeddings with Plotly in a 3D scatter plot
        fig_3d = px.scatter_3d(
            df_reduced, x="UMAP 1", y="UMAP 2", z="UMAP 3",
            color='Topic', hover_data=["Sentence", "Topic"],  # Add sentences and topics to hover data
            color_continuous_scale=px.colors.qualitative.Bold,
            labels={"color": "Topic"}  # Rename color legend for clarity
        ).update_traces(hovertemplate='Sentence: %{customdata[0]}<br>Topic: %{customdata[1]}')

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
    OpenAI().api_key = st.text_input(placeholder="Openai key")
    return OpenAI().api_key.embeddings.create(input=[text], model=model).data[0]['embedding']


def calculate_embeddings(data_in, model_name, sentences_in, model_selected):
    # Display spinner during embeddings calculation
    with st.spinner(f"Calculating embeddings for column {st.session_state[SELECTED_ITEM_COLUMN]}"):
        # If Sentence Transformer model is selected
        if model_selected == MODEL_SBERT:
            model_in = SentenceTransformer(model_name)  # Load model
            # Calculate embeddings and normalize them
            output = model_in.encode(sentences=sentences_in, show_progress_bar=True, normalize_embeddings=True)
            data_in[EMBEDDING] = list(output)  # Store embeddings in DataFrame
            st.session_state["step2_completed"] = True

        # If OpenAI's ADA model is selected
        if model_selected == MODEL_ADA:
            # Calculate embeddings using the OpenAI model
            data_in[EMBEDDING] = sentences_in.apply(lambda x: get_embedding(x, model_name))
            st.session_state["step2_completed"] = True
        return data_in  # Return DataFrame with embeddings


def show_embedding_tab():
    # Check if metadata is selected, otherwise prompt the user
    if st.session_state.get('metadata') is None:
        st.info('You have to select data for viewing and editing. Switch to Step 1 tab first.')
        return  # Exit the function early if no metadata is selected

    embedding_container = st.container()

    # Model selection
    model_options = embedding_container.radio("Data Usage", options=[MODEL_ADA, MODEL_SBERT], horizontal=True)
    st.session_state['model_used'] = model_options

    model = None
    if model_options == MODEL_ADA:
        model = embedding_container.selectbox("Select a model to process",
                                              ["text-embedding-ada-002", "text-embedding-3-small"])
        openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    elif model_options == MODEL_SBERT:
        model = embedding_container.selectbox("Select a model to process",
                                              ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'])

    if not model:
        st.warning("Please select a model to proceed.")
        return

    # Trigger embeddings calculation
    if embedding_container.button(f"Calculate {model_options} embeddings", type='primary'):
        with st.spinner(f"Calculating embeddings using {model}..."):
            calculate_and_display_embeddings(model_options, model,
                                             openai_api_key if model_options == MODEL_ADA else None)


def calculate_and_display_embeddings(model_options, model, api_key=None):
    start_time = time.time()
    df = st.session_state['metadata']
    sentences = df[st.session_state[SELECTED_ITEM_COLUMN]].tolist()

    # Calculate embeddings based on the selected model
    if model_options == MODEL_ADA and api_key:
        df = calculate_embeddings(df, model, sentences, model_options, api_key)
    elif model_options == MODEL_SBERT:
        df = calculate_embeddings(df, model, sentences, model_options)

    if df is not None:
        st.session_state[EMBEDDING] = df  # Update session state with embeddings
        display_embeddings_table(df)
        mark_step_completed(start_time)
        show_explore_embedding_tab()


def display_embeddings_table(df):
    st.data_editor(df[[st.session_state['selected_item_column'], EMBEDDING]], use_container_width=True)


def mark_step_completed(start_time):
    end_time = time.time()
    duration = (end_time - start_time) / 60  # Convert to minutes
    st.session_state['step2_completed'] = True
    st.success(f"Embeddings calculated successfully in {duration:.2f} minutes.")
