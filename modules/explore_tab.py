import io
from config import *  # Import all variables from config module
from pyvis.network import Network
import pandas as pd
import streamlit_highcharts as hg
import streamlit.components.v1 as components
import plotly.graph_objs as go
from streamlit_extras.dataframe_explorer import dataframe_explorer
# import networkx as nx
# from st_aggrid import AgGrid, ColumnsAutoSizeMode
# from streamlit_echarts import st_echarts


def dataframe_with_selections(df_in):
    df_with_selections = df_in.copy()
    df_with_selections.insert(0, SELECT, False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        use_container_width=True,
        hide_index=True,
        column_config={SELECT: st.column_config.CheckboxColumn(required=True)},
        disabled=df_in.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop(SELECT, axis=1)


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


def calculate_edge_width(weight, max_width=20, min_width=1):
    """
    Calculates the edge width for graph visualization based on the similarity score (weight).

    Parameters:
    - weight: The similarity score between two nodes.
    - max_width: The maximum width for an edge.
    - min_width: The minimum width for an edge.

    Returns:
    - The calculated width for an edge, based on the provided weight.
    """
    # Ensure the edge width is proportional to the weight but also within specified bounds
    return max_width if weight == 1 else max(min_width, weight * max_width)


def get_graph_html(df_in, threshold_in):
    """
    Generates HTML for a network graph based on the similarity scores between questionnaire items.

    Parameters:
    - df_in: The dataframe containing data for visualization, with columns for items, questionnaires, and similarity scores.
    - threshold_in: The threshold value for similarity scores. Edges are added to the graph only if the score is above this threshold.

    Returns:
    - HTML string of the generated network graph.
    """
    got_net = Network(height="600px", width="100%", font_color="black")
    got_net.toggle_physics(False)
    got_net.barnes_hut()

    for index, row in df_in.iterrows():
        src = row[ITEM_1]
        s_q_label = row[QUESTIONNAIRE_1]
        dst = row[ITEM_2]
        t_q_label = row[QUESTIONNAIRE_2]
        w = row[SIMILARITY_SCORE]
        width = calculate_edge_width(round(w, 2))

        if round(w, 2) >= threshold_in:
            got_net.add_node(f"{src} ({s_q_label})", f"{src} ({s_q_label})", title=s_q_label)
            got_net.add_node(f"{dst} ({t_q_label})", f"{dst} ({t_q_label})", title=t_q_label)
            got_net.add_edge(f"{src} ({s_q_label})", f"{dst} ({t_q_label})", value=round(w, 2), width=width)

    # Set configuration options for nodes and interactions
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
    # Use generate_html() to get the HTML representation of the network
    html_string = got_net.generate_html()

    return html_string


def render_dependency_wheel_view(data_in):
    """
    Renders a dependency wheel view in Streamlit using Highcharts.

    This function processes a given DataFrame to visualize the count of connections
    (rows) between pairs of questionnaires based on their occurrence in the data.
    Each connection's count reflects the number of times a specific pair of
    questionnaires appears together in the DataFrame.

    Parameters:
    - data_in: A DataFrame containing the columns "Questionnaire 1", "Questionnaire 2",
               and "Similarity score", used to determine the connections.

    The visualization is a dependency wheel that illustrates the interconnectedness
    between different questionnaires, highlighting the frequency of their relationships.
    """

    # Count the connections
    connections = {}

    # Iterate over each row in the DataFrame
    for _, row in data_in.iterrows():
        # Select the specific columns for each row
        source = row[QUESTIONNAIRE_1]
        target = row[QUESTIONNAIRE_2]

        # If the connection (source, target pair) exists, increment its count
        if (source, target) in connections:
            connections[(source, target)] += 1
        else:
            connections[(source, target)] = 1

    # Convert the connections to the required format for chartDef
    new_data = [[source, target, count] for (source, target), count in connections.items()]

    # Define the chart
    chartDef = {
        'accessibility': {
            'point': {
                'valueDescriptionFormat': '{index}. From {point.from} to {point.to}: {point.weight}.'
            }
        },
        'series': [{
            'data': new_data,
            'dataLabels': {
                'color': '#333',
                'distance': 10,
                'style': {'textOutline': 'none'},
                'textPath': {
                    'attributes': {'dy': 5},
                    'enabled': True
                }
            },
            'keys': ['from', 'to', 'weight'],
            'name': 'Count of Semantic Similarity Pairs',
            'size': '95%',
            'type': 'dependencywheel'
        }],
        'title': {
            'text': f'Overview for Similarity score between {round(data_in[SIMILARITY_SCORE].min(), 2)} and {round(data_in[SIMILARITY_SCORE].max(), 2)}'
        }
    }

    # Render the chart in Streamlit using Highcharts
    hg.streamlit_highcharts(chartDef, 700)


def render_heatmap_view():
    """
    Renders a heatmap view in Streamlit using ECharts.

    This function takes a filtered DataFrame stored in the Streamlit session state
    and creates a heatmap visualization. The heatmap shows the number of questions
    where the similarity score is equal to or greater than a predefined threshold
    between pairs of questionnaires. Each cell in the heatmap represents the count
    of such questions for a pair of questionnaires.
    """
    # Embed HTML content directly
    # Aggregate the data to count the number of questions with a score >= 0.56
    pivot_table = st.session_state.df_filtered.pivot_table(
        index=QUESTIONNAIRE_1,
        columns=QUESTIONNAIRE_2,
        values=SIMILARITY_SCORE,
        aggfunc=lambda x: x.sum()
    ).fillna(0)

    # Convert the pivoted table into a list of scores in the format [y_index, x_index, value]
    questionnaires = pivot_table.index.tolist()
    comparisons = pivot_table.columns.tolist()
    scores = []
    for i, questionnaire in enumerate(questionnaires):
        for j, comparison in enumerate(comparisons):
            scores.append([i, j, pivot_table.at[questionnaire, comparison]])

    def create_heatmap(questionnaires_in, comparisons_in, scores_in):
        """
        Creates heatmap chart options for ECharts.

        Parameters:
        - questionnaires: List of questionnaire names for the y-axis.
        - comparisons: List of questionnaire names for the x-axis.
        - scores: List of [y_index, x_index, value] representing the heatmap data.

        Returns:
        - A dictionary containing ECharts heatmap options.
        """
        options = {
            "tooltip": {
                "position": "top"
            },
            "animation": False,
            "grid": {
                "height": "50%",
                "top": "10%"
            },
            "xAxis": {
                "type": "category",
                "data": comparisons_in,
                "splitArea": {
                    "show": True
                }
            },
            "yAxis": {
                "type": "category",
                "data": questionnaires_in,
                "splitArea": {
                    "show": True
                }
            },
            "visualMap": {
                "min": "0",
                "max": "5",  # This should be dynamic or adjusted based on the data
                "calculable": True,
                "orient": "horizontal",
                "left": "center",
                "bottom": "15%"
            },
            "series": [{
                "name": 'Score',
                "type": 'heatmap',
                "data": scores_in,
                "label": {
                    "show": True
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": "10",
                        "shadowColor": 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        }
        return options

    # Create the chart with the data
    # heatmap_options = create_heatmap(questionnaires, comparisons, scores)
    # st_echarts(options=heatmap_options, height="500px")


def render_graph_view(df_sim):
    """
    Renders a graph view based on user-selected options for visualizing questionnaire
    similarities in Streamlit. This function allows users to select between viewing
    all data or a filtered subset and to adjust the similarity score threshold for
    graph visualization.

    The graph visualization is dynamically generated based on the selected options,
    and it is displayed using an HTML component within the Streamlit application.
    """

    # Define a global threshold variable to be used across the application

    # Allow the user to choose between viewing all data or a filtered subset
    physics = st.selectbox('Choose a Graph', ["Filter", "All"])

    # Allow the user to adjust the similarity score threshold for the visualization
    threshold = st.slider("Similarity Score Threshold", 0.5, 1.0, 0.1)

    # A button to trigger the graph visualization
    if st.button(f'View Graph'):
        with st.spinner("Generating Graph..."):

            # Display the graph for all data if selected
            if physics == "All":
                graph_html = get_graph_html(df_sim, threshold)

            # Display the graph for the filtered data if selected
            if physics == "Filter":
                graph_html = get_graph_html(st.session_state.df_filtered, threshold)

            # Read and display the generated HTML graph

            if get_graph_html:
                components.html(graph_html, height=600)


def render_match_view():
    """
    Render a match view to display questionnaire matches based on similarity scores.
    This view iterates through each questionnaire, determining the number of questions
    that match based on similarity pairs and displays the results in a sortable table and bar chart.
    """
    questionnaires_df = st.session_state.metadata
    similarities_df = st.session_state.df_filtered
    results = []  # Initialize a list to store results

    # Iterate through each questionnaire
    for questionnaire_id in questionnaires_df[st.session_state.selected_questionnaire_column].unique():
        # Extract questions of the current questionnaire
        questionnaire_questions = questionnaires_df[
            questionnaires_df[st.session_state.selected_questionnaire_column] == questionnaire_id
            ][st.session_state.selected_item_column]

        # Determine the number of questions of this questionnaire appearing in the similarity pairs
        matches = similarities_df[
            (similarities_df[QUESTIONNAIRE_1] == questionnaire_id) & similarities_df[ITEM_1].isin(
                questionnaire_questions) |
            (similarities_df[QUESTIONNAIRE_2] == questionnaire_id) & similarities_df[ITEM_2].isin(
                questionnaire_questions)
            ]
        unique_matches = pd.unique(matches[[ITEM_1, ITEM_2]].values.ravel('K'))
        num_unique_matches = len(set(unique_matches) & set(questionnaire_questions))

        questionnaire_matches = set(matches[QUESTIONNAIRE_1].tolist() + matches[QUESTIONNAIRE_2].tolist())
        questionnaire_matches.discard(questionnaire_id)  # Remove the current questionnaire from the set
        num_questionnaire_matches = list(questionnaire_matches)

        empty_rows = []
        if len(num_questionnaire_matches) == 0:
            empty_rows.append(questionnaire_id)

        else:  # Append the result for the current questionnaire to the results list
            results.append({
                QUESTIONNAIRE_ID: questionnaire_id,
                NUMBER_OF_QUESTIONS: len(questionnaire_questions),
                ITEM_MATCHES: num_unique_matches,
                QUESTIONNAIRE_MATCHES: num_questionnaire_matches,
                PERCENT_MATCHES: round(
                    (num_unique_matches / len(questionnaire_questions)) * 100 if questionnaire_questions.size > 0 else 0, 2),

            })

    st.info(f"The following {len(empty_rows)} questionnaires do not have any suited matching pairs {list(empty_rows)}")
    # Convert the results into a DataFrame and sort it by the number of questions
    result_matching_df = pd.DataFrame(results).sort_values(by=NUMBER_OF_QUESTIONS, ascending=False)

    # Display the DataFrame in Streamlit
    st.data_editor(result_matching_df, use_container_width=True, column_config={
        PERCENT_MATCHES: st.column_config.ProgressColumn(
            help="The cosine similarity score",
            format="%.2f",
            min_value=0,
            max_value=100,
        ),
    }, hide_index=True)

    # Dropdown for choosing the sort column
    sort_option = st.selectbox(
        'Choose your sort column:',
        options=[PERCENT_MATCHES, NUMBER_OF_QUESTIONS, ITEM_MATCHES],
        format_func=lambda x: PERCENT_MATCHES if x == PERCENT_MATCHES else
        NUMBER_OF_QUESTIONS if x == NUMBER_OF_QUESTIONS else
        ITEM_MATCHES
    )

    # Sort the DataFrame based on the selected option
    sorted_df = result_matching_df.sort_values(by=sort_option, ascending=False)

    # Create the Plotly Figure with sorted data
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sorted_df[QUESTIONNAIRE_ID], y=sorted_df[NUMBER_OF_QUESTIONS],
                         name=NUMBER_OF_QUESTIONS, opacity=0.7))
    fig.add_trace(go.Bar(x=sorted_df[QUESTIONNAIRE_ID], y=sorted_df[ITEM_MATCHES],
                         name=ITEM_MATCHES, opacity=0.7))
    fig.update_layout(barmode='overlay')  # Overlay bars for comparison

    # Display the plot in Streamlit using container width
    st.plotly_chart(fig, use_container_width=True)


def show_explore_tab():
    if st.session_state.similarity is None:
        st.info('You have to select data and build embeddings for viewing similarity')
    else:
        sim_container = st.container()
        df_sim = st.session_state.similarity

        with sim_container.expander("Filtering"):
            st.subheader('Filter tree')
            st.session_state.df_filtered = dataframe_explorer(df_sim, case=False)
            st.data_editor(st.session_state.df_filtered, use_container_width=True, column_config={
                SIMILARITY_SCORE: st.column_config.ProgressColumn(
                    "Similarity",
                    help="The cosine similarity score",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
            }, key="Filtering")

            st.subheader('Filtered DataFrame')
            # df_filtered = df_sim.query(query_string)
            size = len(st.session_state.df_filtered)
            st.info(f"Filtered {size} elements")

            selection = dataframe_with_selections(st.session_state.df_filtered)

            if st.button('➕ Add selected candidates to final selection'):
                add_to_selection(selection)

            st.divider()
            if SELECTED_DATA in st.session_state and not st.session_state.selected_data.empty:
                st.subheader("Final Harmonisation Candidates")
                st.dataframe(st.session_state.selected_data, use_container_width=True)

            excel = convert_df_to_excel(st.session_state.selected_data)
            if excel is not None:
                st.download_button(
                    label="Download Final Selection as Excel",
                    data=excel,
                    file_name='final_selection.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

        with sim_container.expander("Explore"):

            render_dependency_wheel_view(st.session_state.df_filtered)

            render_match_view()

            render_graph_view(st.session_state.df_filtered)
            # render_heatmap_view()
