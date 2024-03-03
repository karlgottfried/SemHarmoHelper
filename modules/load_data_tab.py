import streamlit as st
import pandas as pd
import altair as alt
import requests
from requests.auth import HTTPBasicAuth
from config import *
import plotly.graph_objs as go
import chardet
from io import BytesIO


# initialize_session_state()


def display_load_barchart(data, x_col, y_col, title="Bar Chart"):
    fig = go.Figure(go.Bar(x=data[x_col], y=data[y_col], text=data[y_col], textposition='auto', opacity=0.7,))
    fig.update_layout(title=title, xaxis_title="Questionnaires", yaxis_title=y_col, barmode='group',height=600)
    st.plotly_chart(fig, use_container_width=True)


def show_aggrid_table(data, msg, key):
    st.subheader(msg)
    st.data_editor(data, use_container_width=True, key=key)

    col1, col2 = st.columns(2)

    with col2:
        selected_item_column = st.selectbox("Select the columns with the items.", st.session_state.metadata.columns)

    with col1:
        selected_questionnaire_column = st.selectbox("Select the columns with the questionnaire names.",
                                                     st.session_state.metadata.columns)

    if st.button('Use Metadata'):
        # Hier ist kein Aufruf von initialize_session_state() notwendig, es sei denn,
        # es handelt sich um eine spezifische Funktion, die weiter oben im Code definiert ist.

        st.session_state.update({
            'metadata': data,
            'selected_item_column': selected_item_column,
            'selected_questionnaire_column': selected_questionnaire_column,
            'step1_completed': True,
            'step1_rows': len(data)
        })

        question_counts = data.groupby(selected_questionnaire_column)[
            selected_item_column].nunique().reset_index(name=NUMBER_OF_QUESTIONS)
        sorted_question_counts = question_counts.sort_values(by=NUMBER_OF_QUESTIONS, ascending=False)

        st.success(
            f"Saved metadata of {len(sorted_question_counts)} instruments and {sorted_question_counts[NUMBER_OF_QUESTIONS].sum()} items for semantic search")

        display_load_barchart(sorted_question_counts, selected_questionnaire_column, NUMBER_OF_QUESTIONS,
                              "Metadata Overview")


def get_data():
    file_types = ["csv", "xlsx", "xls"]
    data_upload = st.file_uploader("Upload a metadata file", type=file_types)

    # Directly load a sample file if the button is clicked
    if st.button('Load Sample File'):
        df_data = pd.read_excel(SAMPLE_FILE_CSV)  # Update path to your sample file
        return df_data

    if data_upload:
        # Process the uploaded file
        try:
            if data_upload.name.endswith('.csv'):
                # Determine the encoding of the uploaded CSV file
                raw_data = data_upload.read()  # Read the uploaded file as bytes
                result = chardet.detect(raw_data)  # Detect encoding
                encoding = result.get('encoding', 'utf-8')  # Default to utf-8 if encoding is uncertain

                # Use the detected encoding to read the CSV
                df_data = pd.read_csv(BytesIO(raw_data), encoding=encoding)
            elif data_upload.name.endswith('.xlsx') or data_upload.name.endswith('.xls'):
                # For Excel files, encoding detection is not required
                df_data = pd.read_excel(data_upload)
            else:
                df_data = None
            return df_data
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
            return None

    return None


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


# Function to set credentials in the sidebar
def set_credentials():
    st.session_state['username'] = st.sidebar.text_input("Username")
    st.session_state['password'] = st.sidebar.text_input("Password", type="password")


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


# Function to extract data from questionnaire JSON
def extract_loinc_data(title, copyright_in, json_data):
    rows = []
    for item in json_data:
        for code in item.get("code", []):  # Handle case where 'code' is not present
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


def render_loinc_search():
    st.subheader('LOINC Questionnaire Search')
    # Set credentials using the sidebar inputs
    set_credentials()
    # Define the base URL of the FHIR server and the resource type you want to retrieve
    base_url = "https://fhir.loinc.org/"
    resource_type = "Questionnaire"
    initial_url = f"{base_url}/{resource_type}"
    # Radio button for selecting displayed values
    radio_loinc = st.radio("Displayed values:", ["All LOINC-Codes", "Pre-selection LOINC-Codes"], horizontal=True)
    # Pre-defined LOINC codes

    codes = {
        "Patient health questionnaire 4 item": "69724-3",
        "Kansas City cardiomyopathy questionnaire": "71941-9",
        "Generalized anxiety disorder 7 item": "69737-5",
        "Test": "69723-5"
    }
    ids = []
    # Fetch and display all resources or a pre-selection based on the radio button selection
    if radio_loinc == "All LOINC-Codes":
        all_resources = fetch_all_resources(initial_url)
        st.write(f"Total resources fetched: {len(all_resources)}")
        selected_names = st.multiselect("LOINC Codes", all_resources)
        ids = [all_resources[a] for a in selected_names]

    if radio_loinc == "Pre-selection LOINC-Codes":
        selected_names = st.multiselect("LOINC Codes", codes.keys())
        ids = [codes[a] for a in selected_names]
    urls = [f"https://fhir.loinc.org/Questionnaire/{id_x}" for id_x in ids]
    # Initialize an empty list for the dataframes
    dfs = []
    # Button to start loading the questionnaires
    if st.button("Load questionnaires"):
        for url in urls:
            # Load questionnaire data for each selected URL
            data_loc = load_questionnaires(url, False)
            # Extract data and add the DataFrame to the list
            df_loc = extract_loinc_data(data_loc["title"], data_loc["copyright"], data_loc["item"])

            with st.expander(data_loc["title"]):
                # Expander for each questionnaire
                st.write(df_loc)  # Display the questionnaire table in the expander
                st.info(data_loc["copyright"])

            dfs.append(df_loc)

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


def show_load_data_tab():
    # Initialisiere loaded_file zu Beginn der Funktion mit None
    loaded_file = None

    input_in = st.radio("Upload Metadata from:", ["LOINC Metadata Upload", "New Metadata Upload"], horizontal=True)

    st.divider()

    if input_in == "LOINC Metadata Upload" and st.session_state.get("loincdf") is not None:
        render_loinc_search()
        loaded_file = st.session_state["loincdf"]
    elif input_in == "New Metadata Upload":
        loaded_file = get_data()

    st.session_state.update({'metadata': loaded_file})

    if st.session_state.metadata is not None:
        show_aggrid_table(loaded_file, msg="Metadata Preview:", key=f"{input_in}_grid")


# Constants for the application
FILE_TYPES = ["csv", "xlsx", "xls"]
LOINC_BASE_URL = "https://fhir.loinc.org/Questionnaire"
SAMPLE_FILE_PATH = "path/to/your/sample_file.csv"  # Update this path

# Function to detect file encoding and read file
def read_file(file, encoding='utf-8'):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding=encoding)
    elif file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format.")
        return None

# Function to load data from user upload or sample file
def load_data():
    uploaded_file = st.file_uploader("Upload a metadata file", type=FILE_TYPES)
    if uploaded_file is not None:
        raw_data = uploaded_file.read()
        encoding = chardet.detect(raw_data)['encoding']
        return read_file(BytesIO(raw_data), encoding=encoding)

    if st.button('Load Sample Data'):
        return pd.read_csv(SAMPLE_FILE_PATH)
    return None

# Function to fetch and display LOINC questionnaires
def fetch_and_display_loinc():
    # Authentication
    username = st.sidebar.text_input("Username")
    password = st.sidebar.password_input("Password")
    if not username or not password:
        st.sidebar.warning("Please enter username and password to fetch LOINC data.")
        return

    # Fetching data
    response = requests.get(LOINC_BASE_URL, auth=HTTPBasicAuth(username, password))
    if response.status_code == 200:
        questionnaires = response.json()
        # Process and display questionnaires
        st.write(questionnaires)  # Placeholder for actual processing and display logic
    else:
        st.error("Failed to fetch LOINC questionnaires.")


