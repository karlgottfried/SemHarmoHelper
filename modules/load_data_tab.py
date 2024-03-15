# import streamlit as st
# import pandas as pd
# import altair as alt
# import requests
# import streamlit
# from requests.auth import HTTPBasicAuth
from config import *
# import plotly.graph_objs as go
# import chardet
# from io import BytesIO
# from modules.status_view import display_status_updates

# Import necessary libraries
import requests
from requests.auth import HTTPBasicAuth
import plotly.graph_objs as go
import chardet
from io import BytesIO


def display_load_barchart(data, x_col, y_col, title="Bar Chart"):
    """
    Display a bar chart using Plotly.

    Parameters:
    - data: DataFrame containing the data to plot.
    - x_col: The name of the column to use for the x-axis.
    - y_col: The name of the column to use for the y-axis.
    - title: Chart title.
    """
    # Creating a bar chart figure
    fig = go.Figure(go.Bar(
        x=data[x_col],  # X-axis data
        y=data[y_col],  # Y-axis data
        text=data[y_col],  # Text to display on each bar, showing the y-value
        textposition='auto',  # Positioning the text automatically on the bars
        opacity=0.7,
        hoverinfo='text',
        hovertext=[f'Questionnaire: {q}<br>Number of Questions: {n}' for q, n in zip(data[x_col], data[y_col])]
        # Custom text for hover info, displaying the questionnaire name and number of questions
    ))

    # Updating the layout of the figure
    fig.update_layout(
        title=title,  # Chart title
        xaxis_title="Questionnaires",  # X-axis label
        yaxis_title=y_col.replace('_', ' ').title(),  # Y-axis label, making it more readable
        barmode="group",  # Grouping bars
        height=600,
        xaxis=dict(tickangle=-45)  # Tilting x-axis labels for better readability if the text is too long
    )

    # Displaying the figure in the Streamlit app, using the full container width
    st.plotly_chart(fig, use_container_width=True)


def show_preview_table(data, msg, key):
    """
    Display a preview table and allow user to select columns for items and questionnaire names.

    Parameters:
    - data: DataFrame containing the data to display.
    - msg: Message to display as a subheader above the table.
    - key: Unique key for the Streamlit component to ensure widget state does not interfere across sessions.
    """
    # Display a subheader and the data table
    st.subheader(msg)
    st.data_editor(data, use_container_width=True, key=key, hide_index=True)
    st.success(f"Your data table consists of {data.shape[1]} columns and {data.shape[0]} rows.")

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Column for selecting item column
    with col2:
        selected_item_column = st.selectbox("Select the columns with the items.", data.columns)

    # Column for selecting questionnaire column
    with col1:
        selected_questionnaire_column = st.selectbox("Select the columns with the questionnaire names.", data.columns)

    # Button to confirm selection and update session state
    if st.button('Use Metadata'):
        # Update session state with selected metadata and columns
        st.session_state.update({
            # 'metadata': data,
            'selected_item_column': selected_item_column,
            'selected_questionnaire_column': selected_questionnaire_column,
            'step1_completed': True,
            'step1_rows': len(data),
            'diagram_data_ready': True,
            "question_counts_df": data.groupby(selected_questionnaire_column)
            [selected_item_column].nunique().reset_index(name=NUMBER_OF_QUESTIONS)
        })
        # st.rerun()


def get_data():
    """
    Allow user to upload a metadata file or load a sample file.

    Returns:
    DataFrame containing the loaded data.
    """
    # Define supported file types
    file_types = ["csv", "xlsx", "xls"]
    # File uploader widget
    data_upload = st.file_uploader("Upload a metadata file", type=file_types)

    # Process uploaded file
    if data_upload:
        try:
            # Handle CSV files
            if data_upload.name.endswith('.csv'):
                # Read and detect encoding
                raw_data = data_upload.read()
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')

                # Load CSV with detected encoding
                df_data = pd.read_csv(BytesIO(raw_data), encoding=encoding)
            # Handle Excel files
            elif data_upload.name.endswith('.xlsx') or data_upload.name.endswith('.xls'):
                df_data = pd.read_excel(data_upload)
            else:
                df_data = None
            return df_data
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
            return None
    return None


def load_questionnaires(url_in, flag):
    """
    Load questionnaires from a specified URL with basic authentication.

    Parameters:
    - url_in: The URL from which to fetch the questionnaires.
    - flag: A boolean flag to determine if additional parameters should be used in the request.

    Returns:
    A JSON object containing the fetched questionnaires, or None if an error occurs.
    """
    # Check for username and password in the session state
    if st.session_state['username'] and st.session_state['password']:
        # Make a conditional request based on the flag
        if flag:
            response = requests.get(url_in, params={"_count": "10000"},
                                    auth=HTTPBasicAuth(st.session_state['username'], st.session_state['password']))
        else:
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


# def set_credentials():
#     """
#     Set credentials for basic authentication in the sidebar.
#     """
#     # Input fields for username and password in the sidebar
#     st.session_state['username'] = st.sidebar.text_input("Username")
#     st.session_state['password'] = st.sidebar.text_input("Password", type="password")


def fetch_all_resources(url_in):
    """
    Fetch all resources from the initial URL until no more 'next' links are found.

    Parameters:
    - url_in: The initial URL to start fetching from.

    Returns:
    A dictionary containing all fetched resources.
    """

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


def extract_quest(json_data):
    """
    Extract questionnaire IDs and titles from the JSON data.

    Parameters:
    - json_data: The JSON data containing questionnaires.

    Returns:
    A dictionary with titles as keys and IDs as values.
    """
    results = {}
    for item in json_data:
        resource = item.get('resource', {})
        title = resource.get('title')
        id_in = resource.get('id')

        if title and id_in:
            results[title] = id_in
    return results


def extract_loinc_data(title, copyright_in, json_data):
    """
    Extract data from questionnaire JSON and format it for display.

    Parameters:
    - title: The title of the questionnaire.
    - copyright_in: Copyright information for the questionnaire.
    - json_data: The JSON data of the questionnaire.

    Returns:
    A DataFrame containing the extracted data.
    """
    rows = []
    for item in json_data:
        for code in item.get("code", []):
            answer_options = "\n".join([opt["valueCoding"]["display"] + " | " for opt in item.get("answerOption", [])])
            rows.append({
                QUESTIONNAIRE_ID: item["linkId"],
                QUESTIONNAIRE_LOINC: title,
                CODE_LOINC: code["code"],
                QUESTION_DISPLAY_LOINC: code["display"],
                RESPONSE_DISPLAY_LOINC: answer_options,
                COPYRIGHT_LOINC: copyright_in
            })
    return pd.DataFrame(rows)


def set_credentials():
    """Set credentials for basic authentication in the sidebar."""
    username = st.sidebar.text_input("Username", key="username")
    password = st.sidebar.text_input("Password", type="password", key="password")
    return username, password

def fetch_loinc_codes(display_type):
    """Fetch LOINC codes based on the display type selected by the user."""
    codes = {
        "Patient health questionnaire 4 item": "69724-3",
        "Kansas City cardiomyopathy questionnaire": "71941-9",
        "Generalized anxiety disorder 7 item": "69737-5",
        "Test": "69723-5"
    }
    if display_type == DISPLAY_RADIO_TEXT_2:
        # Assuming fetch_all_resources() efficiently fetches and caches resources if possible
        return fetch_all_resources(LOINC_BASE_URL + "/Questionnaire")
    return codes


def user_selected_codes(codes):
    """Allow user to select LOINC codes and return the selected IDs."""
    selected_names = st.multiselect("LOINC Codes", list(codes.keys()))
    return [codes[name] for name in selected_names if name in codes]


def load_and_display_questionnaires(ids):
    """Load and display selected questionnaires."""
    dfs = []
    for id_x in ids:
        url = f"https://fhir.loinc.org/Questionnaire/{id_x}"
        data = load_questionnaires(url, False)
        if data:
            df = extract_loinc_data(data["title"], data["copyright"], data["item"])
            st.expander(data["title"]).write(df)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def render_loinc_search():
    """Render the LOINC questionnaire search interface."""
    st.subheader('LOINC Questionnaire Search')
    username, password = set_credentials()

    if not username or not password:
        st.warning("Please enter username and password.")
        return

    radio_loinc = st.radio("Displayed values:", [DISPLAY_RADIO_TEXT_1, DISPLAY_RADIO_TEXT_2], horizontal=True)
    codes = fetch_loinc_codes(radio_loinc)
    selected_ids = user_selected_codes(codes)

    if st.button("Load questionnaires"):
        combined_df = load_and_display_questionnaires(selected_ids)
        if combined_df is not None:
            st.session_state.loincdf = combined_df
            st.success(f"Loaded {len(combined_df)} questions from {len(selected_ids)} instruments.")


def read_file(file, encoding='utf-8'):
    """
    Detect file encoding and read the file.

    Parameters:
    - file: The uploaded file object.
    - encoding: The encoding to use for reading the file. Defaults to 'utf-8'.

    Returns:
    A DataFrame containing the data from the file.
    """
    # Check file extension and read file accordingly
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding=encoding)  # Read CSV file
    elif file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)  # Read Excel file
    else:
        st.error("Unsupported file format.")  # Display error for unsupported formats
        return None


def load_data():
    """
    Load data from user upload.

    Returns:
    A DataFrame containing the loaded data, or None if no data is loaded.
    """
    # Define supported file types
    file_types = ["csv", "xlsx", "xls"]
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a metadata file", type=file_types)

    # Process uploaded file
    if uploaded_file is not None:
        raw_data = uploaded_file.read()  # Read file as bytes
        encoding = chardet.detect(raw_data)['encoding']  # Detect encoding
        return read_file(BytesIO(raw_data), encoding=encoding)  # Read file with detected encoding

    return None


def fetch_and_display_loinc():
    """
    Fetch and display LOINC questionnaires using basic authentication.
    """
    # Authentication fields in the sidebar
    username = st.sidebar.text_input("Username")
    password = st.sidebar.password_input("Password", type="password")

    # Check if credentials are provided
    if not username or not password:
        st.sidebar.warning("Please enter username and password to fetch LOINC data.")
        return

    # Fetching data
    response = requests.get(LOINC_BASE_URL, auth=HTTPBasicAuth(username, password))
    if response.status_code == 200:
        questionnaires = response.json()  # Parse JSON response
        st.write(questionnaires)  # Placeholder for displaying questionnaires
    else:
        st.error("Failed to fetch LOINC questionnaires.")  # Display error if fetch fails


def choose_data_source():
    """
    Allows the user to choose the data upload source.
    """
    return st.radio("Upload Metadata from:", [OPTION_1, OPTION_2, OPTION_3], horizontal=True)


def handle_loinc_data_upload():
    """
    Handles the LOINC metadata upload if the data is available in the session state.
    """
    if st.session_state.get("loincdf") is not None:
        render_loinc_search()  # Render LOINC search UI
        return st.session_state["loincdf"]
    return None


def handle_new_data_upload():
    """
    Handles the new metadata file upload.
    """
    return get_data()  # Get data from file upload


def handle_sample_data_upload():
    """
    Handles the sample file upload.
    """
    return pd.read_excel(SAMPLE_FILE_CSV)


def display_metadata_preview(loaded_file):
    """
    Displays a preview of the loaded metadata.
    """
    if loaded_file is not None:
        show_preview_table(loaded_file, msg="Metadata Preview:", key="metadata_grid")


def display_metadata_overview():
    """
    Displays the metadata overview if the diagram data is ready.
    """
    if st.session_state.get('diagram_data_ready', False) and st.session_state.get("question_counts_df") is not None:
        display_load_barchart(
            st.session_state["question_counts_df"].sort_values(by=NUMBER_OF_QUESTIONS, ascending=False),
            st.session_state.selected_questionnaire_column, NUMBER_OF_QUESTIONS, "Metadata Overview")


def show_load_data_tab():
    """
    Refactored function to display options for uploading metadata from various sources and show a preview table.
    """

    data_source = choose_data_source()
    st.divider()

    if data_source == OPTION_1:
        loaded_file = handle_loinc_data_upload()
    elif data_source == OPTION_2:  # New Metadata Upload
        loaded_file = handle_new_data_upload()
    else:
        loaded_file = handle_sample_data_upload()

    st.session_state.metadata = loaded_file  # Update session state with the loaded metadata
    display_metadata_preview(loaded_file)
    display_metadata_overview()

