import io
from config import *  # Import all variables from config module
from pyvis.network import Network
import pandas as pd
import streamlit_highcharts as hg
import streamlit.components.v1 as components
import plotly.graph_objs as go
from streamlit_extras.dataframe_explorer import dataframe_explorer


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


def get_graph_html(df_in):
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


def render_heatmap_view(data_in):
    """
    Renders a heatmap view in Streamlit using Plotly.

    This function processes a given DataFrame to visualize the count of connections
    (rows) between pairs of questionnaires based on their occurrence in the data.
    Each cell in the heatmap represents the frequency of a specific pair of
    questionnaires appearing together.

    Parameters:
    - data_in: A DataFrame containing the columns "Questionnaire 1", "Questionnaire 2",
               and "Similarity score", used to determine the counts.
    """

    # Count the connections
    connection_counts = pd.crosstab(data_in[QUESTIONNAIRE_1], data_in[QUESTIONNAIRE_2], margins=True, margins_name="Total")

    # Prepare custom hover text
    hover_text = [
        [
            f'Questionnaire 1: {row}<br>Questionnaire 2: {col}<br>Number of Matches: {connection_counts.loc[row, col]}'
            for col in connection_counts.columns
        ]
        for row in connection_counts.index
    ]

    # Create the heatmap with annotations
    fig = go.Figure(data=go.Heatmap(
        z=connection_counts.values,  # Values to be used for heatmap colors
        x=connection_counts.columns,  # Column labels
        y=connection_counts.index,  # Row labels
        colorscale='Rainbow',  # Heatmap color scale
        text=connection_counts.values.astype(str),  # Text to be displayed on each cell for annotations
        hoverinfo='text',  # Use custom text for hover info
        hovertext=hover_text,  # Custom hover text
        showscale=False  # Hide the color bar
    ))

    # Adding annotations by enabling 'texttemplate'
    fig.update_traces(texttemplate="%{text}", textfont={"size": 10})

    # Update layout
    fig.update_layout(
        title=f'Number of Pairs Between Questionnaires',
        xaxis_nticks=36,
        yaxis_nticks=36,
        xaxis_title="Questionnaire 2",
        yaxis_title="Questionnaire 1",
        autosize=True,  # Auto adjust the size based on the screen or container size
        height=650
    )
    # Displaying the figure in the Streamlit app, using the full container width
    st.plotly_chart(fig, use_container_width=True)

    # Display explanatory text for the heatmap visualization
    st.info("""
    This heatmap visualizes the number of harmonization pairs between different questionnaires. Each cell represents the count of times two questionnaires have been paired together, providing insights into the most common pairings and the relative frequency of each pairing.
    Hover over any cell to see detailed information about the pairing, including the names of both questionnaires and the exact count of harmonization pairs.
    """)
    st.divider()


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
            'size': '100%',
            'type': 'dependencywheel'
        }],
        'title': {
            'text': f'Questionnaire Relationship Overview'
        },
        'exporting': {
            'enabled': True
        },
        'legend': {
            'enabled': True
        },
    }

    # Render the chart in Streamlit using Highcharts
    hg.streamlit_highcharts(chartDef, 700)

    # Place this right before or after the visualization in your Streamlit application.
    st.info("""
    The Dependency Wheel visualization offers an interactive and comprehensive view of the relationships between different questionnaires. Each arc on the wheel represents a connection, illustrating how frequently pairs of questionnaires are related to each other based on a specific similarity measure.
    This visualization assists in identifying which questionnaires are most frequently paired, potentially indicating areas with higher harmonization potential or common themes. Use this tool to explore the data network and to inform decisions about questionnaire selection.
    Please interact with the Dependency Wheel to discover patterns and insights about the questionnaire relationships within the dataset.
    """)
    st.divider()


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

    # A button to trigger the graph visualization
    if st.button(f'View Graph'):
        with st.spinner("Generating Graph..."):

            # Display the graph for all data if selected
            if physics == "All":
                graph_html = get_graph_html(df_sim)

            # Display the graph for the filtered data if selected
            if physics == "Filter":
                graph_html = get_graph_html(st.session_state.df_filtered)

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
    empty_rows = []

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

    # Convert the results into a DataFrame and sort it by the number of questions
    result_matching_df = pd.DataFrame(results).sort_values(by=NUMBER_OF_QUESTIONS, ascending=False)

    # Display the DataFrame in Streamlit
    st.data_editor(result_matching_df[[QUESTIONNAIRE_ID,NUMBER_OF_QUESTIONS,ITEM_MATCHES,PERCENT_MATCHES]], use_container_width=True, column_config={
        PERCENT_MATCHES: st.column_config.ProgressColumn(
            help="The cosine similarity score",
            format="%.2f",
            min_value=0,
            max_value=100,
        ),
    }, hide_index=True)

    # Sort the dataframe once
    # Sort the DataFrame in descending order by match percentages
    sorted_df = result_matching_df.sort_values(by=PERCENT_MATCHES, ascending=False)

    # Directly extract the first example, avoiding separate variables for each attribute
    first_example = sorted_df.iloc[0]

    # Calculate the minimum and maximum similarity scores once, outside the info string
    min_similarity_score = round(similarities_df[SIMILARITY_SCORE].min(), 2)
    max_similarity_score = round(similarities_df[SIMILARITY_SCORE].max(), 2)

    # Prepare the list of excluded questionnaires beforehand
    excluded_questionnaires = ', '.join(empty_rows) if empty_rows else 'None'

    # Use an f-string for the info display, integrating values directly
    st.info(f"""
    The table offers an overview of the questionnaires, highlighting:
    - The **number of questions** each questionnaire contains.
    - The **number of matching questions** that meet a similarity score between {min_similarity_score} and {max_similarity_score} (according to the filter criteria).
    - The corresponding **semantic coverage**.

    **{len(empty_rows)}** questionnaires lacked suitable matching pairs and have been excluded from the list:
    - **Excluded Questionnaires:** {excluded_questionnaires}

    Among the questionnaires presented, '**{first_example[QUESTIONNAIRE_ID]}**' comprises **{first_example[NUMBER_OF_QUESTIONS]}** questions, exhibiting the highest semantic coverage, meaning that **{first_example[PERCENT_MATCHES]}%** of the questions have a similar content (as defined by the filter criteria) in an other questionnaire, indicating a significant potential for data harmonization. The '**{first_example[QUESTIONNAIRE_ID]}**' questionnaire has semantic similar content with the following **{len(list(first_example[QUESTIONNAIRE_MATCHES]))}** questionnaires: (**{", ".join(list(first_example[QUESTIONNAIRE_MATCHES]))}**).

    A **100% match rate** indicates that every question finds a corresponding match in another questionnaire, denoting a full potential for data harmonization.

    Use this summary to identify which questionnaires present the most extensive opportunities for data alignment and integration.
    """)

    st.divider()
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

    st.info("""
    The bar chart visualizes a semantic similarity analysis across various assessments, which are represented by their acronyms. Each assessment's total number of questions is indicated by dark blue bars, while the light blue bars show the number of questions that have a certain level of semantic similarity to a specific set of criteria or a benchmark dataset.

    By using the 'Choose your sort column' feature, you can arrange the assessments based on different attributes, such as the percentage of semantically similar item matches. This could rearrange the assessments along the horizontal axis according to your selected metric.

    This graphical representation aids in identifying which assessments have a larger pool of questions and which have a higher proportion of questions that are semantically related to the targeted subject matter. Such insights are valuable for evaluating the relevance and coverage of the assessments in relation to the semantic criteria defined for the analysis.
    """)


def main_explore_tab():

    sim_container = st.container()
    df_sim = st.session_state['similarity']

    with sim_container.expander("Filtering"):

        st.subheader('Filter tree')
        quartiles = st.session_state.similarity[SIMILARITY_SCORE].quantile([0.5, 0.75, 0.9])

        # Define the labels for the radio buttons dynamically based on quartile values.
        filter_options = [
            "No filter",
            f"Apply median filter (>= {quartiles[0.5]:.2f})",  # Label for median value
            f"Apply 3. quartiles filter (>= {quartiles[0.75]:.2f})",  # Label for 75th percentile value,
            f"Apply 9. quartiles filter (>= {quartiles[0.9]:.2f})"  # Label for 75th percentile value
        ]

        # Create a new section for pre-filtering with a Radio button
        filter_threshold = st.radio("Filter by similarity score:",
                                    options=filter_options,
                                    index=0,
                                    key="radio_1",
                                    horizontal=True)

        # Logic to apply the filter based on the selected threshold
        if filter_threshold != filter_options[0]:
            # Extract the numeric value from the chosen filter option.
            threshold_value = float(filter_threshold.split('>= ')[1].rstrip(')'))
            # Apply the filter to your data.
            df_sim = df_sim[df_sim[SIMILARITY_SCORE] >= threshold_value]
        else:
            # If no filter is selected, use the original data.
            df_sim = df_sim

        # Now df_sim is either filtered or unfiltered based on the Radio selection
        st.session_state['df_filtered'] = dataframe_explorer(df_sim, case=False)
        st.data_editor(st.session_state['df_filtered'], use_container_width=True, column_config={
            SIMILARITY_SCORE: st.column_config.ProgressColumn(
                "Similarity",
                help="The cosine similarity score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        }, key="Filtering")

        st.subheader('Filtered Table')
        size = len(st.session_state['df_filtered'])
        st.info(f"Filtered {size} elements")

        selection = dataframe_with_selections(st.session_state['df_filtered'])

        if st.button('➕ Add selected candidates to final selection'):
            add_to_selection(selection)

        st.divider()

        if 'selected_data' in st.session_state and not st.session_state['selected_data'].empty:
            st.subheader("Final Harmonisation Candidates")
            st.dataframe(st.session_state['selected_data'], use_container_width=True)

            excel = convert_df_to_excel(st.session_state['selected_data'])
            if excel is not None:
                st.download_button(
                    label="Download Final Selection as Excel",
                    data=excel,
                    file_name='final_selection.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

    with sim_container.expander("Explore"):
        render_dependency_wheel_view(st.session_state['df_filtered'])
        render_heatmap_view(st.session_state['df_filtered'])
        render_match_view()
        render_graph_view(st.session_state['df_filtered'])
