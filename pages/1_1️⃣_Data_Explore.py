import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from io import StringIO
import json
from streamlit_condition_tree import condition_tree, config_from_dataframe
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Initialize session state variables if they don't exist
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'json_data' not in st.session_state:
    st.session_state.json_data = None
if 'data_df' not in st.session_state:
    st.session_state.data_df = None
if 'file_names' not in st.session_state:
    st.session_state.file_names = None
if 'user_descriptions' not in st.session_state:
    st.session_state['user_descriptions'] = {}
if "df_descriptions" not in st.session_state:
    st.session_state.df_descriptions = None
if "dataview" not in st.session_state:
    st.session_state.dataview = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "df_voc" not in st.session_state:
    st.session_state.df_voc = None
if "df_voc_filtered" not in st.session_state:
    st.session_state.df_voc_filtered = None
if 'initialized_expanders' not in st.session_state:
    st.session_state['initialized_expanders'] = {}


# Define a component for data viewing and profiling
def dataview_component(index, path_in):
    expander_key = f"expander_{index}"
    # Check if the expander has been initialized to avoid re-initialization
    if expander_key not in st.session_state['initialized_expanders']:
        with st.expander(f"File {index}: {path_in[index].name}"):
            df_loc = pd.read_csv(path_in[index], encoding="utf-16")
            st.write(df_loc)

            # Generate profile report if not already generated
            profile_key_loc = f'profile_{path_in[index].name}'
            if profile_key_loc not in st.session_state:
                profile = ProfileReport(df_loc, minimal=True,
                                        progress_bar=True, title="Profiling Report")
                st.session_state[profile_key_loc] = json.loads(profile.to_json())

            # Display metrics from the profile report
            try:
                json_data_loc = st.session_state[profile_key_loc]
                # num_elements = len(json_data_loc["table"]["types"])
                # columns = st.columns(num_elements)
                number_of_observations_loc = json_data_loc["table"]["n"]
                number_of_variables_loc = json_data_loc["table"]["n_var"]
                number_of_numeric_vars_loc = json_data_loc["table"]["types"]["Numeric"]
                number_of_text_vars_loc = json_data_loc["table"]["types"]["Text"]

                # Display metrics using columns
                col1_loc, col2_loc, col3_loc, col4_loc = st.columns(4)
                with col1_loc:
                    st.metric("Number of variables", number_of_variables_loc)
                with col2_loc:
                    st.metric("Number of observations", number_of_observations_loc)
                with col3_loc:
                    st.metric("Numeric", number_of_numeric_vars_loc)
                with col4_loc:
                    st.metric("Text", number_of_text_vars_loc)
                st.session_state['initialized_expanders'][expander_key] = True
                st.divider()
            except:
                print(f"{file.name} not loadable")


# Set Streamlit page configuration
st.set_page_config(page_title="Data Explorer", page_icon="1️⃣", layout="wide")
st.markdown("# Basic Data Exploration (Under Development)")
st.sidebar.header('User Input')

# Sidebar for single and multiple file upload
file = st.sidebar.file_uploader("Choose a single file")
path = st.sidebar.file_uploader("Choose multiple CSV files", accept_multiple_files=True)
st.session_state.file_names = [uploaded_file.name for uploaded_file in path]

# Prevent regeneration of components if files have already been uploaded
if path:
    for i in range(len(path)):
        st.session_state.dataview = dataview_component(i, path)

# Profile selection for display
if path:
    selected_files = st.multiselect('Choose profiles to display:', st.session_state.file_names, key='profile_select')
    col1, col2 = st.columns(2)

    if len(selected_files) > 0:
        for selected_file in selected_files:
            # Create a table for variable descriptions with an expander
            with st.expander("Add Variable Description"):
                data = []  # Empty list to collect data rows
                voc = []  # Empty list for vocabulary count
                # Iterate through each selected profile
                for file_name in selected_files:
                    profile_key = f'profile_{file_name}'  # Key in session_state containing the file's JSON profile
                    if profile_key in st.session_state:
                        json_data = st.session_state[profile_key]
                        # Iterate through each variable in the profile
                        for variable_name, variable_info in json_data["variables"].items():
                            # Create a row for each variable
                            row = {
                                "Table Name": file_name,
                                "Variable": variable_name,
                                "Type": variable_info["type"],
                                "Description": st.session_state['user_descriptions'].get(variable_name, ""),
                                "Values": str(variable_info["value_counts_index_sorted"])
                            }

                            # Append vocabulary information for each variable
                            for k, v in variable_info["value_counts_index_sorted"].items():
                                row_voc = {
                                    "Table Name": file_name,
                                    "Variable": variable_name,
                                    "Type": variable_info["type"],
                                    "Word": k,
                                    "Count": v
                                }
                                voc.append(row_voc)
                            data.append(row)

                # Convert the collected data rows into a DataFrame
                st.session_state.df_descriptions = pd.DataFrame(data)
                st.session_state.df_voc = pd.DataFrame(voc)

        # Display metrics for the selected files
        with col1:
            st.metric("Table Count", len(selected_files))
        with col2:
            st.metric("Variable Count", len(st.session_state.df_descriptions))

        # Filter and display the variable descriptions DataFrame
        if st.session_state.df_descriptions is not None:
            st.session_state.filtered_df = dataframe_explorer(st.session_state.df_descriptions, case=False)
            st.dataframe(st.session_state.filtered_df, use_container_width=True)
            st.write("Stats for filtered table:")
            col11, col12 = st.columns(2)
            with col11:
                st.metric("Table Count", len(st.session_state.filtered_df["Table Name"].unique()),
                          len(st.session_state.filtered_df["Table Name"].unique()) - len(selected_files))
            with col12:
                st.metric("Variable Count", len(st.session_state.filtered_df),
                          len(st.session_state.filtered_df) - len(st.session_state.df_descriptions))

        # Display vocabulary used in the filtered variables
        if st.session_state.df_voc is not None:
            st.divider()
            st.write("Vocabulary used in the filtered variables")
            unique_pairs = st.session_state.filtered_df[["Table Name", "Variable"]].drop_duplicates()
            filtered_df_voc = pd.merge(st.session_state.df_voc, unique_pairs, on=["Table Name", "Variable"],
                                       how="inner")
            st.session_state.df_voc_filtered = filtered_df_voc
            st.dataframe(st.session_state.df_voc_filtered)
            col111, col222 = st.columns(2)
            with col111:
                st.metric("Variable Count", len(st.session_state.df_voc_filtered["Variable"].unique()))
            with col222:
                st.metric("Word Count", len(st.session_state.df_voc_filtered["Word"]))

# Handling the single file upload for data profiling
if file is not None:
    bytes_data = file.getvalue()
    string_data = StringIO(bytes_data.decode("utf-16"))

    file_types = ["csv", "xlsx", "xls"]
    wait_msg = 'Loading data...'

    if file:
        # Read the file based on its type
        if file.name.endswith('.csv'):
            st.session_state.data_df = pd.read_csv(string_data)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            st.session_state.data_df = pd.read_excel(string_data)
        else:
            st.session_state.data_df = None

    # Display an overview of the uploaded data
    st.subheader("Overview of the uploaded data")
    st.write(st.session_state.data_df)

    # Slider to select the sampling fraction for report generation
    sampling_fraction = st.slider("Select sampling fraction for report generation", min_value=0.01, max_value=1.00,
                                  value=0.05, step=0.01)

    if st.button("Generate Report"):
        # Sample the data based on the selected fraction
        sample = st.session_state.data_df.sample(frac=sampling_fraction)
        description = f"Disclaimer: this profiling report was generated using a sample of {sampling_fraction * 100}% of the original dataset."
        st.session_state.profile = ProfileReport(sample, dataset={"description": description}, minimal=True,
                                                 progress_bar=True, title="Profiling Report")

        # Convert the report to JSON and load as a Python object
        json_data = st.session_state.profile.to_json()
        st.session_state.json_data = json.loads(json_data)

        # Display metrics from the profile report
        number_of_observations = st.session_state.json_data["table"]["n"]
        number_of_variables = st.session_state.json_data["table"]["n_var"]
        number_of_numeric_vars = st.session_state.json_data["table"]["types"]["Numeric"]
        number_of_text_vars = st.session_state.json_data["table"]["types"]["Text"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Number of variables", number_of_variables)
        with col2:
            st.metric("Number of observations", number_of_observations)
        with col3:
            st.metric("Numeric", number_of_numeric_vars)
        with col4:
            st.metric("Text", number_of_text_vars)

        # Optionally, display the report in Streamlit if st_profile_report is available
        with st.expander("View Report"):
            if st.session_state.profile is not None:
                st_profile_report(st.session_state.profile)

    # Variable description addition interface
    user_descriptions = {}
    with st.expander("Add Variable Description"):
        if st.session_state.json_data:
            data = []
            # Iterate through each variable in the JSON data to create rows for a DataFrame
            for variable_name, variable_info in st.session_state.json_data["variables"].items():
                row = {
                    "Table": file.name,
                    "Variable": variable_name,
                    "Type": variable_info["type"],
                    "Description": "",
                    "Values": variable_info["value_counts_index_sorted"]
                }
                data.append(row)
            df = pd.DataFrame(data)

            st.subheader('Filter tree')
            st.markdown("Start filtering here")
            config = config_from_dataframe(df)
            query_string = condition_tree(config)
            st.code(query_string)

            st.subheader('Metadata View')
            df_filtered = df.query(query_string)
            st.dataframe(df_filtered)

            # Save descriptions back into the JSON
            if st.button("Save Descriptions"):
                for variable in st.session_state.json_data:
                    if variable in user_descriptions:
                        st.session_state.json_data[variable] = (st.session_state.json_data[variable][0], user_descriptions[variable])
                st.success("Descriptions saved!")

else:
    st.write("You did not upload the new file")
