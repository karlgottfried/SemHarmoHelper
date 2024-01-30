import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from io import StringIO
import json as json
from streamlit_condition_tree import condition_tree, config_from_dataframe

if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'json_data' not in st.session_state:
    st.session_state.json_data = None
if 'data_df' not in st.session_state:
    st.session_state.data_df = None

st.set_page_config(page_title="Data Explorer", page_icon="🌍")
st.markdown("# Basic Data Exploration")
st.sidebar.header('User Input')

file = st.sidebar.file_uploader("Choose a file")

if file is not None:
    bytes_data = file.getvalue()
    string_data = StringIO(bytes_data.decode("utf-16"))

    # File uploader for data file
    file_types = ["csv", "xlsx", "xls"]
    wait_msg = 'Loading data...'

    if file:
        # Check the type of file uploaded and read accordingly
        if file.name.endswith('.csv'):
            st.session_state.data_df = pd.read_csv(string_data)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            st.session_state.data_df = pd.read_excel(string_data)
        else:
            st.session_state.data_df = None

    st.subheader("Overview of the uploaded data")
    st.write(st.session_state.data_df)

    sampling_fraction = st.slider("Select sampling fraction for report generation",
                                  min_value=0.01, max_value=1.00, value=0.05, step=0.01)

    if st.button("Generate Report"):
        sample = st.session_state.data_df.sample(frac=sampling_fraction)

        # Erstellen des Berichts
        description = f"Disclaimer: this profiling report was generated using " \
                      f"a sample of {sampling_fraction * 100}% of the original dataset."
        st.session_state.profile = ProfileReport(sample, dataset={"description": description}, minimal=True,
                                                 progress_bar=True, title="Profiling Report")

        # profile.to_file("report.html")  # Optional: Speichern des Berichts als HTML-Datei

        # Konvertieren des Berichts in JSON und Laden als Python-Objekt
        json_data = st.session_state.profile.to_json()
        st.session_state.json_data = json.loads(json_data)

        # Anzeigen des Berichts in Streamlit (optional, falls st_profile_report verfügbar ist)
    with st.expander("View Report"):
        if st.session_state.profile is not None:
            st_profile_report(st.session_state.profile)

    user_descriptions = {}
    with st.expander("Add Variable Description"):
        if st.session_state.json_data:
            data = []
            for variable_name, variable_info in st.session_state.json_data["variables"].items():
                row = {
                    "Variable": variable_name,
                    "Type": variable_info["type"],
                    "Description": "",
                    "Alerts": variable_info["value_counts_index_sorted"]
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

            # Button to save the descriptions back into the JSON
            if st.button("Save Descriptions"):
                for variable in st.session_state.json_data:
                    if variable in user_descriptions:
                        # Update the description in the json_structure
                        st.session_state.json_data[variable] = (st.session_state.json_data[variable][0],
                                                                user_descriptions[variable])

                st.success("Descriptions saved!")

else:
    st.write("You did not upload the new file")
