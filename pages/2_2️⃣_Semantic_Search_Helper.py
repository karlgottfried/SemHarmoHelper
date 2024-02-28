import io
import time
import streamlit.components.v1 as components
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.dataframe_explorer import dataframe_explorer
import streamlit_highcharts as hg

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from st_aggrid import AgGrid, ColumnsAutoSizeMode
from streamlit_echarts import st_echarts
import streamlit as st

from pyvis.network import Network
#import networkx as nx

st.set_page_config(page_title="Semantic Search Helper", page_icon="2️⃣")


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
        src = row['Item 1']
        s_q_label = row["Questionnaire 1"]
        dst = row['Item 2']
        t_q_label = row["Questionnaire 2"]
        w = row['Similarity score']
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

def render_dependencywheel_view(data_in):
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
        source = row["Questionnaire 1"]
        target = row["Questionnaire 2"]

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
            'text': f'Overview for Similarity score between {round(data_in["Similarity score"].min(),2)} and {round(data_in["Similarity score"].max(),2)}'
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
        index='Questionnaire 1',
        columns='Questionnaire 2',
        values='Similarity score',
        aggfunc=lambda x: (x >= threshold).sum()
    ).fillna(0)

    # Convert the pivoted table into a list of scores in the format [y_index, x_index, value]
    questionnaires = pivot_table.index.tolist()
    comparisons = pivot_table.columns.tolist()
    scores = []
    for i, questionnaire in enumerate(questionnaires):
        for j, comparison in enumerate(comparisons):
            scores.append([i, j, pivot_table.at[questionnaire, comparison]])

    def create_heatmap(questionnaires, comparisons, scores):
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
                "data": comparisons,
                "splitArea": {
                    "show": True
                }
            },
            "yAxis": {
                "type": "category",
                "data": questionnaires,
                "splitArea": {
                    "show": True
                }
            },
            "visualMap": {
                "min": 0,
                "max": 5,  # This should be dynamic or adjusted based on the data
                "calculable": True,
                "orient": "horizontal",
                "left": "center",
                "bottom": "15%"
            },
            "series": [{
                "name": 'Score',
                "type": 'heatmap',
                "data": scores,
                "label": {
                    "show": True
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowColor": 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        }
        return options

    # Create the chart with the data
    heatmap_options = create_heatmap(questionnaires, comparisons, scores)
    st_echarts(options=heatmap_options, height="500px")


def render_graph_view():
    """
    Renders a graph view based on user-selected options for visualizing questionnaire
    similarities in Streamlit. This function allows users to select between viewing
    all data or a filtered subset and to adjust the similarity score threshold for
    graph visualization.

    The graph visualization is dynamically generated based on the selected options,
    and it is displayed using an HTML component within the Streamlit application.
    """

    # Define a global threshold variable to be used across the application
    global threshold

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

# Function to set credentials in the sidebar
def set_credentials():
    st.session_state['username'] = st.sidebar.text_input("Username")
    st.session_state['password'] = st.sidebar.text_input("Password", type="password")


# Function to load questionnaires from a given URL with basic authentication
def load_questionnaires(url_in, flag):
    # Check if username and password are entered
    if st.session_state['username'] and st.session_state['password']:
        # Conditional request based on the flag
        if flag:
            # Request with a parameter to get more results
            response = requests.get(url_in, params={"_count": "10000"},
                                    auth=HTTPBasicAuth(st.session_state['username'], st.session_state['password']))
        else:
            # Simple GET request
            response = requests.get(url_in,
                                    auth=HTTPBasicAuth(st.session_state['username'], st.session_state['password']))
        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error loading questionnaires.")
            return None
    else:
        st.warning("Please enter username and password.")
        return None


# Function to display questionnaire data
def display_questionnaires(data_in):
    if data_in:
        st.write(data_in["item"])
    else:
        st.write("No data found.")


# Function to extract data from questionnaire JSON
def extract_data(title, copyright_in, json_data):
    rows = []
    for item in json_data:
        for code in item.get("code", []):  # Handle case where 'code' is not present
            answer_options = "\n".join([opt["valueCoding"]["display"] + " | " for opt in item.get("answerOption", [])])
            rows.append({
                "ID": item["linkId"],
                "Questionnaire": title,
                "Code": code["code"],
                "Question (Display)": code["display"],
                "Response (Display)": answer_options,
                "Copyright": copyright_in
            })
    return pd.DataFrame(rows)


# Function to extract questionnaire IDs and titles
def extract_quest(json_data):
    results = {}
    for item in json_data:
        resource = item.get('resource', {})
        title = resource.get('title')
        id_in = resource.get('id')

        if title and id_in:
            results[title] = id_in
    return results


# Function to fetch all resources from the initial URL
def fetch_all_resources(url_in):
    resources = {}
    load_count = 0

    while url_in and load_count < 1:
        print("url", url_in)
        bundle = load_questionnaires(url_in, True)
        if bundle and 'entry' in bundle:
            result = extract_quest(bundle['entry'])
            resources.update(result)

        # Find the 'next' link to continue fetching
        url_in = None
        if bundle:
            for link in bundle.get('link', []):
                if link['relation'] == 'next':
                    url_in = link['url']
                    break

        load_count += 1

    return resources


def render_loinc_search():
    st.subheader('LOINC Questionnaire Search')
    # Set credentials using the sidebar inputs
    set_credentials()
    # Define the base URL of the FHIR server and the resource type you want to retrieve
    base_url = "https://fhir.loinc.org/"
    resource_type = "Questionnaire"
    initial_url = f"{base_url}/{resource_type}"
    # Radio button for selecting displayed values
    slider = st.radio("Displayed values:", ["All LOINC-Codes", "Pre-selection LOINC-Codes"])
    # Pre-defined LOINC codes
    codes = {
        "Patient health questionnaire 4 item": "69724-3",
        "Kansas City cardiomyopathy questionnaire": "71941-9",
        "Generalized anxiety disorder 7 item": "69737-5",
        "": "69723-5"
    }
    ids = []
    # Fetch and display all resources or a pre-selection based on the radio button selection
    if slider == "All LOINC-Codes":
        all_resources = fetch_all_resources(initial_url)
        st.write(f"Total resources fetched: {len(all_resources)}")
        selected_names = st.multiselect("LOINC Codes", all_resources)
        ids = [all_resources[a] for a in selected_names]
    if slider == "Pre-selection LOINC-Codes":
        selected_names = st.multiselect("LOINC Codes", codes.keys())
        ids = [codes[a] for a in selected_names]
    urls = [f"https://fhir.loinc.org/Questionnaire/{id_x}" for id_x in ids]
    # Initialize an empty list for the dataframes
    dfs = []
    # Button to start loading the questionnaires
    if st.button("Load questionnaires"):
        for url in urls:
            # Load questionnaire data for each selected URL
            data = load_questionnaires(url, False)
            # Extract data and add the DataFrame to the list
            df = extract_data(data["title"], data["copyright"], data["item"])

            with st.expander(data["title"]):
                # Expander for each questionnaire
                st.write(df)  # Display the questionnaire table in the expander
                st.info(data["copyright"])

            dfs.append(df)

        # Combine all DataFrames in the list into a single DataFrame
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # st.write(combined_df)
            st.session_state.loincdf = combined_df
            if len(combined_df) > 0:
                st.info(
                    f"Saved LOINC Selection for Semantic Search: "
                    f"{len(ids)} Instruments with {len(st.session_state.loincdf)} questions included")

            # No need to rewrite combined_df here as it will be redrawn after the button press.






####################################

if 'metadata' not in st.session_state:
    st.session_state.metadata = None

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

if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'password' not in st.session_state:
    st.session_state['password'] = ''
if 'loincdf' not in st.session_state:
    st.session_state["loincdf"] = pd.DataFrame()

#############

st.title("📒 Semantic Search Helper")
# select_tab, view_tab, store_tab = st.tabs(['Load Sentence Data', 'Build Embeddings', 'View Similarity'])

with st.sidebar:

    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/karlgottfried/SemHarmoHelper/blob/main/app.py)"



status_container = st.container()
if st.session_state.metadata is None:
    meta_status = status_container.info('Step 1 not done yet')
else:
    meta_status = status_container.success('Step 1 done! Data selected containing '
                                           + str(len(st.session_state.metadata)) + ' rows')

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


def show_preview_table(df_in):
    # Check if combined_df has been saved in the session_state and display it.
    if "loincdf" in st.session_state:
        # st.dataframe(st.session_state["loincdf"])
        st.subheader("Metadata Preview:")
        AgGrid(df_in, theme="streamlit", height=300, key="loinc_grid",
               columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
        # spreadsheet(st.session_state["loincdf"])
    else:
        st.error("No data loaded. Please select at least one questionnaire.")


# Create a tab for loading data
with load_tab:
    # Radio button for user to choose between using pre-selected LOINC data or uploading new data
    input_in = st.radio("Upload Metadata from:", ["LOINC-Upload", "New Upload"], horizontal=True)

    # If user selects "LOINC-Selection" and there is pre-loaded LOINC data, use that data
    if input_in == "LOINC-Upload" and st.session_state["loincdf"] is not None:
        render_loinc_search()
        loaded_file = st.session_state["loincdf"]
        if loaded_file is not None:
            show_preview_table(loaded_file)

    # If user chooses "New Upload" or there is no pre-loaded LOINC data, prompt for new data upload
    if input_in == "New Upload" or st.session_state["loincdf"] is None:
        loaded_file = get_data()
        if loaded_file is not None:
            show_preview_table(loaded_file)

    # If data is successfully loaded, proceed
    if loaded_file is not None:
        # Provide an expander to show the uploaded data
        col1, col2 = st.columns(2)

        with col2:
            # Dropdown for user to select the column containing sentences
            selected_item_column = st.selectbox("Select the columns with the items.", loaded_file.columns)

        with col1:
            # Dropdown for user to select the column containing questionnaire names
            selected_questionnaire_column = st.selectbox("Select the columns with the questionnaire names.",
                                                     loaded_file.columns)

        # Button to confirm data selection
        if st.button('use data', type='primary', key="save_data"):
            # Assume loaded_file, selected_item_column, and selected_questionnaire_column are already defined
            st.session_state.metadata = loaded_file
            st.session_state.selected_item_column = selected_item_column
            st.session_state.selected_questionnaire_column = selected_questionnaire_column
            meta_status.empty()
            meta_status = status_container.success(
                'Step 1 done! Data selected containing ' + str(len(st.session_state.metadata)) + ' rows')

            # Counting unique questions per questionnaire
            data = st.session_state.metadata
            item_column = st.session_state.selected_item_column
            questionnaire_column = st.session_state.selected_questionnaire_column
            question_counts = data.groupby(questionnaire_column)[item_column].nunique()
            sorted_question_counts = question_counts.sort_values(ascending=False)

            # Vorbereiten der Daten für das Highcharts-Diagramm
            categories = sorted_question_counts.index.tolist()  # Fragebogen-Namen
            data_series = sorted_question_counts.values.tolist()  # Anzahl der Fragen


            st.success(f"Saved metadata of {len(categories)} instruments "
                       f"and {sum(data_series)} items for semantic search")

            # Anpassen des chartDef-Objekts für das Balkendiagramm
            chartDef = {
                'chart': {'type': 'column'},
                'title': {'text': 'Count of Items per Questionnaire'},
                'xAxis': {'categories': categories, 'title': {'text': 'Questionnaire'}},
                'yAxis': {'min': 0, 'title': {'text': 'Number of Questions'}, 'allowDecimals': False},
                'legend': {'enabled': False},
                'plotOptions': {'column': {'dataLabels': {'enabled': True}}},
                'series': [{
                    'name': 'Questions',
                    'data': data_series,
                    'colorByPoint': True  # Färbt jede Säule individuell
                }]
            }
            # Anzeigen des Highcharts-Diagramms in Streamlit
            hg.streamlit_highcharts(chartDef, height=640)

with embedding_tab:
    if st.session_state.metadata is None:
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
                df = st.session_state.metadata
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
            df = st.session_state.metadata
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
    if st.session_state.metadata is None and st.session_state.embedding is None:
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
            AgGrid(df_sim, height=300, theme="streamlit", key="filter_grid", columns_auto_size_mode= ColumnsAutoSizeMode.FIT_CONTENTS)
            #st.dataframe(df_sim, use_container_width=True)

            st.subheader('Filter tree')
            st.session_state.df_filtered = dataframe_explorer(df_sim, case=False)
            st.dataframe(st.session_state.df_filtered, use_container_width=True)


            st.download_button(
                    label=f"Download whole filtered table with "
                          f"{len(st.session_state.df_filtered)} elements as Excel file",
                    data=convert_df_to_excel(st.session_state.df_filtered),
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

        with sim_container.expander("Explore"):

            render_dependencywheel_view(st.session_state.df_filtered)

            df_frageboegen =  st.session_state.metadata
            df_aehnlichkeiten = st.session_state.df_filtered

                # Ermitteln, welche Fragen in den Fragepaaren enthalten sind
            ergebnisse = []

                # Über jeden Fragebogen iterieren
            for fragebogen_id in df_frageboegen[st.session_state.selected_questionnaire_column].unique():
                    # Fragen des aktuellen Fragebogens extrahieren
                fragen_des_fragebogens = df_frageboegen[
                    df_frageboegen[st.session_state.selected_questionnaire_column] ==
                    fragebogen_id][st.session_state.selected_item_column]

                    # Anzahl der Fragen dieses Fragebogens, die in den Ähnlichkeitspaaren vorkommen, ermitteln
                matches = df_aehnlichkeiten[
                        (df_aehnlichkeiten['Questionnaire 1'] == fragebogen_id) & df_aehnlichkeiten['Item 1'].isin(
                            fragen_des_fragebogens) |
                        (df_aehnlichkeiten['Questionnaire 2'] == fragebogen_id) & df_aehnlichkeiten['Item 2'].isin(
                            fragen_des_fragebogens)
                        ]

                    # Eindeutige Fragen mit einem Match zählen
                eindeutige_matches = pd.unique(matches[['Item 1', 'Item 2']].values.ravel('K'))

                anzahl_eindeutiger_matches = len(set(eindeutige_matches) & set(fragen_des_fragebogens))

                ergebnisse.append({
                        'FragebogenID': fragebogen_id,
                        'AnzahlFragen': len(fragen_des_fragebogens),
                        'AnzahlMatches': anzahl_eindeutiger_matches,
                        'ProzentMatches': (anzahl_eindeutiger_matches / len(
                            fragen_des_fragebogens)) * 100 if fragen_des_fragebogens.size > 0 else 0
                    })

                    # Ergebnis in einen DataFrame umwandeln
            ergebnisse_df = pd.DataFrame(ergebnisse).sort_values(by="AnzahlFragen")

            import plotly.express as px

            st.write(ergebnisse_df)

            fig = px.histogram(ergebnisse_df, x="FragebogenID", y=["AnzahlFragen", "AnzahlMatches"], barmode="overlay", hover_data= ergebnisse_df.columns)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            #st.bar_chart(ergebnisse_df.sort_values(ascending=False, by="ProzentMatches"),  x="FragebogenID", y=["AnzahlFragen", "AnzahlMatches"])


            render_graph_view()
            render_heatmap_view()






