import streamlit as st
from config import *
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import faiss
import time


def calculate_cosine_similarity(embeddings):
    """Calculate cosine similarity matrix from embeddings."""
    if embeddings.size > 0:
        return cosine_similarity(embeddings)
    return np.array([])


# Funktion, um die Cosine-Ähnlichkeit zu berechnen
def calculate_cosine_similarity_faiss(vec1, vec2):
    # Vec1 und Vec2 sind 2D Arrays
    return cosine_similarity(vec1, vec2)[0][0]


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

    st.data_editor(st.session_state.similarity, use_container_width=True, column_config={
        SIMILARITY_SCORE: st.column_config.ProgressColumn(
            "Similarity",
            help="The cosine similarity score",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
    }, key=f"data_frame_sim_{st.session_state['model_used']}", hide_index=True)
    st.caption("Table Similarity Analysis: \n The table serves as a condensed representation of semantically paired items extracted from extensive questionnaire datasets, labeled as ""Questionnaire 1"" and ""Questionnaire 2"". "
               "Each row correlates a specific item from the first questionnaire regarding to a semantically related item from other questionnaires. "
               "These pairings are accompanied by a similarity score, a critical metric in the semantic harmonization process, which aids in evaluating the potential relevance of each item pair for integrative analysis")
    st.divider()

    # Calculate statistics: mean, median, and quartiles
    mean_ada = st.session_state.similarity[SIMILARITY_SCORE].mean()
    median_ada = st.session_state.similarity[SIMILARITY_SCORE].median()
    quartiles_ada = st.session_state.similarity[SIMILARITY_SCORE].quantile([0.25, 0.75])

    # Create a histogram for the similarity scores
    fig = go.Figure(data=go.Histogram(x=st.session_state.similarity[SIMILARITY_SCORE],
                                      nbinsx=int((st.session_state.similarity[SIMILARITY_SCORE].max() -
                                                  st.session_state.similarity[SIMILARITY_SCORE].min()) / 0.01),
                                      opacity=0.7, marker=dict(color='blue'),
                                      name="Similarity Scores"))

    # Define a function to add statistical lines with hover information
    def add_stat_line(fig_loc, x, name, color):
        fig_loc.add_trace(go.Scatter(x=[x, x], y=[0, 1], mode="lines",
                                     line=dict(color=color, width=2),
                                     name=name,
                                     hoverinfo='skip',  # Use 'skip' if you don't want hover info for lines
                                     yaxis="y2"))  # Reference to the secondary y-axis

    # Update the layout to include a secondary y-axis for the statistical lines
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=80),
        yaxis2=dict(
            overlaying='y',
            range=[0, 1],  # The range is set from 0 to 1 to ensure lines cover the full diagram height
            showticklabels=False  # Hide tick labels for the secondary y-axis
        )
    )

    # Add statistical lines using the updated function
    add_stat_line(fig, mean_ada, 'Mean', 'Orange')
    add_stat_line(fig, median_ada, 'Median', 'Red')
    add_stat_line(fig, quartiles_ada[0.25], '1st Quartile', 'Green')
    add_stat_line(fig, quartiles_ada[0.75], '3rd Quartile', 'Green')

    # Update the layout to include titles and adjust the bar gap
    fig.update_layout(title=f'Cosine Similarity Distribution for {st.session_state["model_used"]} Model',
                      margin=dict(l=40, r=40, t=40, b=80),
                      xaxis_title='Cosine Similarity Score',
                      yaxis_title='Frequency',
                      bargap=0.2,
                      width=600,
                      height=400,
                      legend_title_text='Legend')

    fig.update_traces(hoverinfo='x+y', hovertemplate="Score: %{x}<br>Frequency: %{y}")
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    return f"""
    The histogram illustrates the distribution of similarity scores for {st.session_state["model_used"]} model and indicates the positions of statistical measures: Mean, Median, and Quartiles. Here's a concise summary of these statistical insights:

    - **Number of Pairs with Similarity Scores:** A total of **{len(st.session_state.similarity)}** pairs were analyzed.
    - **Range of Similarity Scores:** Scores extend from a minimum of **{st.session_state.similarity[SIMILARITY_SCORE].min():.2f}** to a maximum of **{st.session_state.similarity[SIMILARITY_SCORE].max():.2f}**.
    - **Average (Mean):** The mean similarity score stands at **{mean_ada:.2f}**.
    - **Median:** The median value, dividing the dataset into two equal halves, is **{median_ada:.2f}**.
    - **Quartiles:** Specifically, the 1st Quartile (marking the 25th percentile) is **{quartiles_ada[0.25]:.2f}**, and the 3rd Quartile (marking the 75th percentile) is **{quartiles_ada[0.75]:.2f}**.

    Leverage this information to deepen your understanding of the model's performance and to uncover any discernible patterns within the similarity scores.
    """


def show_similarity_calculation_section(data, item_column, questionnaire_column):
    """UI section for triggering similarity calculation."""

    with st.spinner('Calculating similarity with cosine similarity...'):
        embeddings = np.array(data[EMBEDDING].tolist())
        cosine_sim_matrix = calculate_cosine_similarity(embeddings)
        return build_similarity_dataframe(data, cosine_sim_matrix, item_column, questionnaire_column)


def show_similarity_calculation_section_faiss(data, item_column, questionnaire_column):
    """
    UI section for triggering similarity calculation using FAISS.

    Parameters:
    - data: The dataset containing embeddings and other information.
    - item_column: The name of the column in 'data' that contains the items to compare.
    - questionnaire_column: The name of the column in 'data' that indicates the questionnaire each item belongs to.
    """

    with st.spinner('Calculating similarity with FAISS...'):
        # Convert the list of embeddings into a NumPy array
        embeddings = np.array(data[EMBEDDING].tolist())

        # Create a FAISS index for L2 distance
        d = embeddings.shape[1]  # Dimensionality of embeddings
        index = faiss.IndexFlatL2(d)

        # Normalize the embeddings for cosine similarity
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        index.add(norm_embeddings)  # Add normalized embeddings to the index

        k = 20  # Number of nearest neighbors to find

        output_results = []
        seen_pairs = set()  # Set to store seen pairs to avoid duplicates

        for i in range(len(data)):
            # Search for the k+1 nearest neighbors because the search also returns the query itself
            _, I = index.search(norm_embeddings[i:i + 1], k + 1)
            for j in range(1, k + 1):  # Start from 1 to skip the query itself
                neighbor_idx = I[0][j]

                # Generate a unique ID for each pair
                pair_id = frozenset([i, neighbor_idx])

                # Ensure the neighbor is from a different questionnaire and the pair hasn't been seen before
                if (data.iloc[i][questionnaire_column] != data.iloc[neighbor_idx][questionnaire_column]) and (
                        pair_id not in seen_pairs):
                    similarity_score = calculate_cosine_similarity_faiss(embeddings[i:i + 1],
                                                                         embeddings[neighbor_idx:neighbor_idx + 1])

                    # Append the result to output
                    output_results.append({
                        QUESTIONNAIRE_1: data.iloc[i][questionnaire_column],
                        ITEM_1: data.iloc[i][item_column],
                        QUESTIONNAIRE_2: data.iloc[neighbor_idx][questionnaire_column],
                        ITEM_2: data.iloc[neighbor_idx][item_column],
                        SIMILARITY_SCORE: similarity_score
                    })
                    seen_pairs.add(pair_id)  # Mark this pair as seen

        return pd.DataFrame(output_results)


def main_similarity_pair_tab():
    """
    Main function to manage the similarity pairs tab.

    Allows the user to choose between COSINE_SIMILARITY and FAISS for calculating similarity.
    Displays the execution time for the selected method.
    """

    similarity_mode = st.radio("Choose Similarity Mode", options=[COSINE_SIMILARITY, FAISS], horizontal=True)

    start_time = time.time()  # Start timing the operation

    if st.button('Calculate Similarity', key="button_2"):
        # Calculate similarity based on the selected mode
        if similarity_mode == COSINE_SIMILARITY:
            st.session_state.similarity = show_similarity_calculation_section(
                st.session_state[EMBEDDING],
                st.session_state.selected_item_column,
                st.session_state.selected_questionnaire_column
            )
        elif similarity_mode == FAISS:
            st.session_state.similarity = show_similarity_calculation_section_faiss(
                st.session_state[EMBEDDING],
                st.session_state.selected_item_column,
                st.session_state.selected_questionnaire_column
            )

    st.divider()

    if st.session_state.similarity is not None:
        end_time = time.time()
        st.session_state.update({'step3_completed': True})
        msg = show_explore_sim_tab()
        msg_time = f"\n **The execution time of {similarity_mode} mode was: {end_time - start_time} seconds.**"
        st.info(msg + msg_time)

