import streamlit as st
from config import *  # Import all variables from config module
import time
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt
import pandas as pd


def show_explore_sim_tab(model_used):
    # Calculate the average for the ADA model
    with st.expander("Explore Similarity Building Step", expanded=True):
        # Display the similarity dataframe in Streamlit with custom column configuration
        st.data_editor(st.session_state.similarity, use_container_width=True, column_config={
            SIMILARITY_SCORE: st.column_config.ProgressColumn(
                "Similarity",
                help="The cosine similarity score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        }, key=f"data_frame_sim_{model_used}")

        st.divider()

        # Calculate the mean similarity score
        mean_ada = st.session_state.similarity[SIMILARITY_SCORE].mean()
        # Create a histogram for ADA scores
        chart = alt.Chart(st.session_state.similarity).mark_bar(opacity=0.7).encode(
            x=alt.X(f'{SIMILARITY_SCORE}:Q', bin=alt.Bin(step=0.01), title='Cosine Similarity Score'),
            y=alt.Y('count()', title='Frequency'),
            tooltip=[alt.Tooltip('count()', title='Frequency'),
                     alt.Tooltip(f'mean({SIMILARITY_SCORE}):Q', title='Mean Score')]
        ).properties(
            title=f'Cosine Similarity Distribution for {model_used}'
        )
        # Add the mean value line to the histogram
        mean_line = alt.Chart(pd.DataFrame({'mean_ada': [mean_ada]})).mark_rule(color='yellow').encode(
            x='mean_ada:Q'
        )
        # Combine the histogram with the mean line
        final_chart = (chart + mean_line).properties(
            width=600,  # Adjust the width of the chart
            height=400  # Adjust the height of the chart
        )
        # Display the chart in Streamlit
        st.altair_chart(final_chart, use_container_width=True)


def get_similarity_dataframe(df_in, cosine_sim, item_column_in, questionnaire_column_in):
    results = []
    with st.spinner("Building pairs"):
        # Iterate through each row to build similarity pairs
        for i, row_i in enumerate(df_in.itertuples()):
            questionnaire_value_i = getattr(row_i, questionnaire_column_in)
            item1 = df_in[item_column_in].iloc[i]
            for j, row_j in enumerate(df_in.itertuples()):
                questionnaire_value_j = getattr(row_j, questionnaire_column_in)
                item2 = df_in[item_column_in].iloc[j]
                # Skip if same index or same questionnaire to avoid redundant pairs
                if j <= i or questionnaire_value_i == questionnaire_value_j:
                    continue

                similarity_score = cosine_sim[i][j]
                results.append({
                    QUESTIONNAIRE_1: questionnaire_value_i,
                    ITEM_1: item1,
                    QUESTIONNAIRE_2: questionnaire_value_j,
                    ITEM_2: item2,
                    SIMILARITY_SCORE: similarity_score
                })

                # Print the number of pairs built and their details (for debugging or logging)
                print(f"{len(results)} pairs build!")
                print(
                    f"{QUESTIONNAIRE_1} '{questionnaire_value_i}', Item '{item1}' "
                    f"and {QUESTIONNAIRE_2}'{questionnaire_value_j}', Item '{item2}' "
                    f"have {SIMILARITY_SCORE}: {similarity_score}\n")
        results.sort(key=lambda x: x[SIMILARITY_SCORE], reverse=True)
    return pd.DataFrame(results)


def calculate_similarity(data_in, selected_column):
    # Calculate cosine similarity for the selected column
    with st.spinner(f"calculate {selected_column}"):
        if len(data_in[EMBEDDING]) > 0:
            cos_sim_1 = cosine_similarity(data_in[EMBEDDING].tolist())
            return cos_sim_1


def show_similarity_pair_tab():
    # Check if data and embeddings are selected to proceed with similarity calculation
    if st.session_state.metadata is None and st.session_state[EMBEDDING] is None:
        st.info('You have to select data and build embeddings in order to view similarity')
    else:
        data = st.session_state[EMBEDDING]
        selected_item_column = st.session_state.selected_item_column
        selected_questionnaire_column = st.session_state.selected_questionnaire_column

        # Button to trigger similarity calculation
        if st.button(f'Calculate Similarity for all pairs of column {selected_item_column}', type='primary',
                     key="calc_similarity"):
            start_time_sim = time.time()
            cosine_similarity_in = calculate_similarity(data, selected_item_column)
            df_sim = get_similarity_dataframe(data, cosine_similarity_in, selected_item_column,
                                              selected_questionnaire_column)
            st.session_state.similarity = df_sim
            end_time_sim = time.time()
            st.session_state.duration_sim = end_time_sim - start_time_sim
            st.session_state.duration_minutes = st.session_state.duration_sim / 60
            st.session_state['step3_completed'] = True

        if st.session_state.similarity is not None:
            show_explore_sim_tab(st.session_state.model_used)
