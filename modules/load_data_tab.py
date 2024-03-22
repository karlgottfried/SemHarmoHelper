from config import *
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
        textposition='outside',  # Positioning the text automatically on the bars
        opacity=0.7,
        hoverinfo='text',
        hovertext=[f'Questionnaire: {q}<br>Number of Questions: {n}' for q, n in zip(data[x_col], data[y_col])]
        # Custom text for hover info, displaying the questionnaire name and number of questions
    ))

    # Updating the layout of the figure
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=80),
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
    if st.button('Use Metadata', help="Text"):
        # Update session state with selected metadata and columns
        st.session_state.update({
            'metadata': data,
            'selected_item_column': selected_item_column,
            'selected_questionnaire_column': selected_questionnaire_column,
            'step1_completed': True,
            'step1_rows': len(data),
            'diagram_data_ready': True,
            "question_counts_df": data.groupby(selected_questionnaire_column)
            [selected_item_column].nunique().reset_index(name=NUMBER_OF_QUESTIONS)
        })
        st.rerun()


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


def fetch_all_resources(url_in):
    """
    Fetch all resources from the initial URL until no more 'next' links are found.
    """
    resources = {}
    bundle = load_questionnaires(url_in, True)
    if bundle and 'entry' in bundle:
        resources.update({entry['resource'].get('title'): entry['resource'].get('id')
                          for entry in bundle['entry'] if 'resource' in entry})
    return resources


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


def reset_app_state():
    """
    Resets the app state by clearing all session state variables.
    This method clears the variables without directly reloading the page.
    For a full reset, consider using st.experimental_rerun() after this function.
    """
    st.session_state.clear()


def set_credentials():
    """Set credentials for basic authentication in the sidebar."""
    username = st.sidebar.text_input("Username", key="username")
    password = st.sidebar.text_input("Password", type="password", key="password")
    return username, password


def fetch_loinc_codes(display_type):
    """Fetch LOINC codes based on the display type selected by the user."""
    codes = {
        "Alcohol Use Disorder Identification Test [AUDIT]": "72110-0",
        "Alcohol Use Disorder Identification Test - Consumption [AUDIT-C]": "72109-2",
        "Generalized anxiety disorder 7 item": "69737-5",
        "Patient Health Questionnaire (PHQ)": "69723-5",
        "PHQ-9 quick depression assessment panel": "44249-1",
        "Patient health questionnaire 4 item": "69724-3",

    }
    if display_type == DISPLAY_RADIO_TEXT_2:
        return fetch_all_resources(LOINC_BASE_URL + "/Questionnaire")
    return codes


def user_selected_codes(codes):
    """Allow user to select LOINC codes and return the selected IDs."""
    selected_names = st.multiselect("LOINC Codes", list(codes.keys()))
    return [codes[name] for name in selected_names if name in codes]


def load_and_display_questionnaires_expander(ids):
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
        combined_df = load_and_display_questionnaires_expander(selected_ids)
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


def choose_data_source():
    """
    Allows the user to choose the data upload source.
    """
    return st.radio("Upload Metadata from:", [OPTION_3, OPTION_2, OPTION_1], horizontal=True)


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
    Displays the metadata overview if the diagram data is ready, including the total number of unique questionnaires
    and the total number of questions.
    """
    if st.session_state.get('diagram_data_ready', False) and st.session_state.get("question_counts_df") is not None:
        # Assuming st.session_state["question_counts_df"] has columns for questionnaire IDs and question counts
        questionnaire_column = st.session_state.selected_questionnaire_column  # The name of the column holding questionnaire IDs
        question_count_column = NUMBER_OF_QUESTIONS  # The name of the column holding the count of questions

        # Calculate the total number of unique questionnaires
        total_unique_questionnaires = st.session_state["question_counts_df"][questionnaire_column].nunique()

        # Calculate the total number of questions across all questionnaires
        total_questions = st.session_state["question_counts_df"][question_count_column].sum()

        st.divider()
        # Call the function to display the bar chart (assuming it's implemented elsewhere in the code)
        display_load_barchart(
            st.session_state["question_counts_df"].sort_values(by=question_count_column, ascending=False),
            questionnaire_column, question_count_column, "Metadata Overview")

        st.info(f"""
        This bar chart illustrates the number of questions each questionnaire contains, enabling a visual comparison of question counts among the given questionnaires.
        
        The data presented shows a total of **{total_unique_questionnaires} questionnaires**, encompassing **{total_questions} questions**.
        """)


def main_load_data_tab():
    """
    Displays options for uploading metadata from various sources and shows a preview table.
    Includes a button to reset the app state and reload the page for a fresh start.
    """
    # Adding a divider for visual separation

    st.divider()

    # Choosing the data source based on user selection
    data_source = choose_data_source()

    # Adding another divider after the data source selection
    st.divider()

    # Determining the loaded file based on the selected data source
    if data_source == OPTION_1:
        loaded_file = handle_loinc_data_upload()
    elif data_source == OPTION_2:  # New Metadata Upload
        loaded_file = handle_new_data_upload()
    elif data_source == OPTION_3:  # Use Sample Data
        loaded_file = handle_sample_data_upload()
    else:
        loaded_file = None

    # If a file has been loaded, update the session state and display the metadata
    if loaded_file is not None:
        st.session_state.metadata = loaded_file
        display_metadata_preview(loaded_file)
        display_metadata_overview()
