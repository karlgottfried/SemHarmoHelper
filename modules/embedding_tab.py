# Import necessary libraries
import streamlit as st  # Streamlit for building web apps
from openai import OpenAI  # OpenAI for GPT models
from config import *  # Import configurations
import time  # Time library for timing operations
from sentence_transformers import SentenceTransformer  # For sentence embeddings
from umap.umap_ import UMAP  # UMAP for dimensionality reduction
import plotly.express as px  # Plotly Express for interactive plots
from bertopic import BERTopic  # BERTopic for topic modeling (commented out due to potential disuse in this context)
import pandas as pd  # Pandas for data manipulation
import hdbscan  # HDBSCAN for clustering
import numpy as np  # NumPy for numerical operations


# Function to show the tab for exploring embeddings
def show_explore_embedding_tab():
    with st.expander("Explore Embedding Building Step", expanded=True):
        # Retrieve sentences to embed from the session state
        sentences = st.session_state[EMBEDDING][st.session_state[SELECTED_ITEM_COLUMN]].tolist()
        embeddings = np.array(list(st.session_state[EMBEDDING][EMBEDDING]))
        # Topic modeling with BERTopic
        topic_model = BERTopic(min_topic_size=10)
        topics, probs = topic_model.fit_transform(sentences, embeddings)

        # Save topics and probabilities in the session state
        st.session_state['topics'] = topics
        st.session_state['probs'] = probs

        # Display topics and their representations using data editor
        st.data_editor(topic_model.get_topic_info()[["Topic", "Representation"]], use_container_width=True, hide_index=True)
        st.caption("""**Topic Representation Table:** Provides a quick overview of the topics identified by the BERTopic model with a list of keywords that best represent the content of each topic.""")
        st.divider()

        # Dimensionality reduction on embeddings using UMAP
        reduced_embeddings = UMAP(n_neighbors=10, n_components=3, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        # Cluster reduced embeddings using HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)

        # Save reduced embeddings and cluster labels in the session state
        st.session_state['reduced_embeddings'] = reduced_embeddings
        st.session_state['cluster_labels'] = cluster_labels

        # Prepare DataFrame for visualization and save it in session state
        df_reduced = pd.DataFrame(reduced_embeddings, columns=['UMAP 1', 'UMAP 2', "UMAP 3"])
        df_reduced['Sentence'] = sentences
        df_reduced['Topic'] = [f"Topic {topic}" for topic in topics]
        st.session_state['df_reduced'] = df_reduced

        # Display sentences and topics using data editor
        st.data_editor(df_reduced[["Sentence", "Topic"]], use_container_width=True, hide_index=True)

        st.caption("""**Sentences and Topics Assignment:** Displays the question sentences alongside the topics they're associated with, allowing to see how the model categorizes the text data into different topics.
        """)
        st.divider()

        # Create and display a 3D scatter plot of reduced embeddings
        fig_3d = px.scatter_3d(df_reduced, x="UMAP 1", y="UMAP 2", z="UMAP 3", color='Topic', hover_data=["Sentence", "Topic"], color_continuous_scale=px.colors.qualitative.Bold, labels={"color": "Topic"}).update_traces(hovertemplate='Sentence: %{customdata[0]}<br>Topic: %{customdata[1]}')
        st.plotly_chart(fig_3d, use_container_width=True)

        st.info("""**3D Scatter Plot Visualization:** Offers an interactive 3D visualization of the calculated sentence embeddings, grouped by topic. It’s useful for exploring the data structure and understanding the relationships and distances between different groups of sentences.
        """)


# Function to fetch embeddings for a given text using a specified model
def get_embedding(text, model):
    text = text.replace("\n", " ")  # Normalize text by replacing newlines with spaces
    OpenAI().api_key = st.text_input(placeholder="OpenAI key")  # Input field for OpenAI API key
    return OpenAI().api_key.embeddings.create(input=[text], model=model).data[0]['embedding']


# Function to calculate embeddings for the given data using the selected model
def calculate_embeddings(data_in, model_name, sentences_in, model_selected):
    with st.spinner(f"Calculating embeddings for column {st.session_state[SELECTED_ITEM_COLUMN]}"):
        if model_selected == MODEL_SBERT:
            model_in = SentenceTransformer(model_name)  # Load Sentence Transformer model
            output = model_in.encode(sentences=sentences_in, show_progress_bar=True, normalize_embeddings=True)
            data_in[EMBEDDING] = list(output)  # Store embeddings in DataFrame

        if model_selected == MODEL_ADA:
            data_in[EMBEDDING] = sentences_in.apply(lambda x: get_embedding(x, model_name))
        return data_in  # Return DataFrame with embeddings


# Function to display the embeddings tab in the Streamlit UI
def main_embedding_tab():

    embedding_container = st.container()  # Container for the embeddings tab

    # Radio button for model selection between Sentence Transformers and OpenAI models
    model_options = choose_model(embedding_container)
    st.session_state['model_used'] = model_options  # Store selected model in session state

    st.divider()

    # Conditional rendering based on selected model
    model = None
    if model_options == MODEL_ADA:
        # Dropdown for selecting an OpenAI model if ADA model is selected
        # embeddings_data = handle_ada_model()
        model = embedding_container.selectbox("Select a model to process", ["text-embedding-ada-002", "text-embedding-3-small"])
        openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")  # API key input for OpenAI
    elif model_options == MODEL_SBERT:
        # embeddings_data = handle_sbert_model()
        # Dropdown for selecting a Sentence Transformer model if SBERT is selected
        model = embedding_container.selectbox("Select a model to process", ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'])

    if not model:
        st.warning("Please select a model to proceed.")
        return

    # Button to initiate embeddings calculation
    if embedding_container.button(f"Calculate {model_options} embeddings", type='primary'):
        with st.spinner(f"Calculating embeddings using {model}..."):
            # Function call to calculate and display embeddings
            calculate_and_display_embeddings(model_options, model, openai_api_key if model_options == MODEL_ADA else None)

    if st.session_state[EMBEDDING] is not None:
        display_embeddings_table(st.session_state[EMBEDDING])
        show_explore_embedding_tab()  # Show the tab for exploring embeddings


def choose_model(embedding_container):
    return embedding_container.radio("Data Usage", options=[MODEL_SBERT, MODEL_ADA], horizontal=True)


# Function to calculate and display embeddings for the selected model
def calculate_and_display_embeddings(model_options, model, api_key=None):
    start_time = time.time()  # Start timing the operation
    df = st.session_state['metadata']  # Retrieve metadata from session state
    sentences = df[st.session_state[SELECTED_ITEM_COLUMN]].tolist()  # List of sentences for embedding

    # Calculate embeddings based on the model selected and whether an API key is provided
    if model_options == MODEL_ADA and api_key:
        st.session_state[EMBEDDING] = calculate_embeddings(df, model, sentences, model_options, api_key)
    elif model_options == MODEL_SBERT:
        st.session_state[EMBEDDING] = calculate_embeddings(df, model, sentences, model_options)

    mark_step_completed(start_time)  # Mark operation as completed and display execution time


# Function to display the embeddings table in the Streamlit UI
def display_embeddings_table(df):
    st.data_editor(df[[st.session_state[SELECTED_ITEM_COLUMN], EMBEDDING]], use_container_width=True)


# Function to mark the completion of an operation and calculate its duration
def mark_step_completed(start_time):
    end_time = time.time()  # End timing the operation
    duration = (end_time - start_time) / 60  # Calculate duration in minutes
    st.session_state.update({'step2_completed': True,}) # Mark step as completed in session state
    st.success(f"Embeddings calculated successfully in {duration:.2f} minutes.")
    st.rerun()# Display success message with duration
