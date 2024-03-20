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

    st.data_editor(st.session_state.similarity, use_container_width=True, column_config={
        SIMILARITY_SCORE: st.column_config.ProgressColumn(
            "Similarity",
            help="The cosine similarity score",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
    }, key=f"data_frame_sim_{st.session_state['model_used']}", hide_index=True)
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
    st.info(f"""
    The histogram illustrates the distribution of similarity scores for {st.session_state["model_used"]} model and indicates the positions of statistical measures: Mean, Median, and Quartiles. Here's a concise summary of these statistical insights:

    - **Number of Pairs with Similarity Scores:** A total of **{len(st.session_state.similarity)}** pairs were analyzed.
    - **Range of Similarity Scores:** Scores extend from a minimum of **{st.session_state.similarity[SIMILARITY_SCORE].min():.2f}** to a maximum of **{st.session_state.similarity[SIMILARITY_SCORE].max():.2f}**.
    - **Average (Mean):** The mean similarity score stands at **{mean_ada:.2f}**.
    - **Median:** The median value, dividing the dataset into two equal halves, is **{median_ada:.2f}**.
    - **Quartiles:** Specifically, the 1st Quartile (marking the 25th percentile) is **{quartiles_ada[0.25]:.2f}**, and the 3rd Quartile (marking the 75th percentile) is **{quartiles_ada[0.75]:.2f}**.

    Leverage this information to deepen your understanding of the model's performance and to uncover any discernible patterns within the similarity scores.
    """)


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
