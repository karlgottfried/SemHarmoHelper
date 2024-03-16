import streamlit as st
from config import *
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go


def calculate_cosine_similarity(embeddings):
    """Calculate cosine similarity matrix from embeddings."""
    if embeddings.size > 0:
        return cosine_similarity(embeddings)
    return np.array([])


def build_similarity_dataframe(df, cosine_sim_matrix, item_column, questionnaire_column):
    """Construct a DataFrame of similarity pairs."""
    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df[questionnaire_column].iloc[i] != df[questionnaire_column].iloc[j]:
                pairs.append({
                    QUESTIONNAIRE_1: df[questionnaire_column].iloc[i],
                    ITEM_1: df[item_column].iloc[i],
                    QUESTIONNAIRE_2: df[questionnaire_column].iloc[j],
                    ITEM_2: df[item_column].iloc[j],
                    SIMILARITY_SCORE: cosine_sim_matrix[i, j]
                })
    return pd.DataFrame(pairs).sort_values(by=SIMILARITY_SCORE, ascending=False)


def show_explore_sim_tab():
    """Display UI elements for exploring similarity."""
    if 'similarity' not in st.session_state or st.session_state.similarity is None:
        st.info('No similarity data available. Please calculate similarity first.')
        return
    pd.set_option("styler.render.max_elements", 1231980)
    st.dataframe(st.session_state.similarity.style.format({SIMILARITY_SCORE: "{:.2f}"}), use_container_width=True)
    mean_ada = st.session_state.similarity[SIMILARITY_SCORE].mean()

    # Histogram of similarity scores
    fig = go.Figure(
        data=go.Histogram(x=st.session_state.similarity[SIMILARITY_SCORE], nbinsx=int((st.session_state.similarity[SIMILARITY_SCORE].max() - st.session_state.similarity[
                SIMILARITY_SCORE].min()) / 0.01), opacity=0.7, marker=dict(color='blue')))
    fig.update_layout(title='Similarity Score Distribution', xaxis=dict(title='Similarity Score'),
                      yaxis=dict(title='Count'))
    fig.add_shape(type='line',
                  x0=mean_ada, y0=0,
                  x1=mean_ada, y1=1,
                  xref='x', yref='paper',
                  line=dict(color='Yellow', width=3)
                  )

    fig.update_layout(
        title_text=f'Cosine Similarity Distribution for {st.session_state["model_used"]} Model',
        xaxis_title='Cosine Similarity Score',
        yaxis_title='Frequency',
        bargap=0.2,
        width=600,
        height=400
    )

    fig.update_traces(hoverinfo='x+y', hovertemplate="Score: %{x}<br>Frequency: %{y}")

    st.plotly_chart(fig, use_container_width=True)


def show_similarity_calculation_section(data, item_column, questionnaire_column):
    """UI section for triggering similarity calculation."""
    if st.button('Calculate Similarity'):
        with st.spinner('Calculating similarity...'):
            embeddings = np.array(data[EMBEDDING].tolist())  # Assuming data[EMBEDDING] is a list of lists
            cosine_sim_matrix = calculate_cosine_similarity(embeddings)
            st.session_state.similarity = build_similarity_dataframe(data, cosine_sim_matrix, item_column,
                                                                     questionnaire_column)
            st.session_state['step3_completed'] = True


def show_similarity_pair_tab():
    """Main function to manage the similarity pairs tab."""
    if st.session_state.metadata is None or 'selected_item_column' not in st.session_state:
        st.warning('You need to load data and select columns before proceeding.')
        return

    show_similarity_calculation_section(
        st.session_state[EMBEDDING],
        st.session_state.selected_item_column,
        st.session_state.selected_questionnaire_column
    )

    show_explore_sim_tab()
