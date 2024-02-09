import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from io import StringIO
import json as json
from streamlit_condition_tree import condition_tree, config_from_dataframe
from streamlit_extras.dataframe_explorer import dataframe_explorer

# 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 💾

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


def dataview_component(index, path_in):
    expander_key = f"expander_{index}"
    if expander_key not in st.session_state['initialized_expanders']:
        with st.expander(f"File {index}: {path_in[index].name}"):
            df_loc = pd.read_csv(path_in[index], encoding="utf-16")
            st.write(df_loc)

            sampling_fraction_loc = st.slider("Select sampling fraction for report generation", min_value=0.01,
                                              max_value=1.00, value=0.05, step=0.01, key=f"slider_{index}")

            # Generiere das Profil nur, wenn es noch nicht generiert wurde
            profile_key_loc = f'profile_{path_in[index].name}'
            if profile_key_loc not in st.session_state:
                profile = ProfileReport(df_loc, minimal=True,
                                        progress_bar=True, title="Profiling Report")
                st.session_state[profile_key_loc] = json.loads(profile.to_json())

            try:# Hier werden die Metriken angezeigt
                json_data_loc = st.session_state[profile_key_loc]
                num_elements = len(json_data_loc["table"]["types"])
                columns = st.columns(num_elements)
                number_of_observations = json_data_loc["table"]["n"]
                number_of_variables = json_data_loc["table"]["n_var"]
                number_of_numeric_vars = json_data_loc["table"]["types"]["Numeric"]
                number_of_text_vars = json_data_loc["table"]["types"]["Text"]

                col1_loc, col2_loc, col3_loc, col4_loc = st.columns(4)
                with col1_loc:
                    st.metric("Number of variables", number_of_variables)
                with col2_loc:
                    st.metric("Number of observations", number_of_observations)
                with col3_loc:
                    st.metric("Numeric", number_of_numeric_vars)
                with col4_loc:
                    st.metric("Text", number_of_text_vars)
                st.session_state['initialized_expanders'][expander_key] = True  # Markiere als initialisiert
                st.divider()
            except:
                print(f"{file.name} not loadable")

st.set_page_config(page_title="Data Explorer", page_icon="1️⃣")
st.markdown("# Basic Data Exploration")
st.sidebar.header('User Input')

file = st.sidebar.file_uploader("Choose a single file")

with st.sidebar:
    path = st.file_uploader("Choose multiple CSV files", accept_multiple_files=True)
    st.session_state.file_names = [uploaded_file.name for uploaded_file in path]

# Verhindere die Neugenerierung von Komponenten, wenn die Dateien bereits hochgeladen wurden
if path:
    for i in range(len(path)):
        st.session_state.dataview = dataview_component(i, path)

# Auswahl der Profile zur Anzeige
if path:
    selected_files = st.multiselect('Choose profiles to display:', st.session_state.file_names, key='profile_select')
    col1, col2 = st.columns(2)

    if len(selected_files) > 0:
        for selected_file in selected_files:
            # Erstelle die Tabelle für die Variablenbeschreibung
            with st.expander("Add Variable Description"):
                # Erstelle eine leere Liste, um alle Datenzeilen zu sammeln
                data = []
                voc = []
                # Gehe durch jedes ausgewählte Profil
                for file_name in selected_files:
                    # Schlüssel im session_state, der das JSON-Profil der Datei enthält
                    profile_key = f'profile_{file_name}'
                    if profile_key in st.session_state:
                        json_data = st.session_state[profile_key]
                        # Gehe durch jede Variable im Profil
                        for variable_name, variable_info in json_data["variables"].items():
                            # Erstelle eine Zeile für jede Variable
                            row = {
                                "Table Name": file_name,
                                "Variable": variable_name,
                                "Type": variable_info["type"],
                                "Description": st.session_state['user_descriptions'].get(variable_name, ""),
                                "Values": str(variable_info["value_counts_index_sorted"])
                            }

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

                # Erstelle ein DataFrame aus den gesammelten Datenzeilen
                st.session_state.df_descriptions = pd.DataFrame(data)
                st.session_state.df_voc = pd.DataFrame(voc)

        with col1:
            st.metric("Table Count", len(selected_files))
        with col2:
            st.metric("Variable Count", len(st.session_state.df_descriptions))

        if st.session_state.df_descriptions is not None:
            st.session_state.filtered_df = dataframe_explorer(st.session_state.df_descriptions, case=False)
            st.dataframe(st.session_state.filtered_df, use_container_width=True)
            st.write("Stats for filtered table:")
            col11, col12 = st.columns(2)
            with col11:
                st.metric("Table Count", len(st.session_state.filtered_df["Table Name"].unique()), len(st.session_state.filtered_df["Table Name"].unique())-len(selected_files))
            with col12:
                st.metric("Variable Count", len(st.session_state.filtered_df), len(st.session_state.filtered_df)-len(st.session_state.df_descriptions))

        if st.session_state.df_voc is not None:
            st.divider()
            st.write("Vocabulary used in the filtered variables")
            # Schritt 1: Finde die einzigartigen Wertepaare in st.session_state.filtered_df
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

if file is not None:
    bytes_data = file.getvalue()
    string_data = StringIO(bytes_data.decode("utf-16"))

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

        # Berechne die Anzahl der Elemente, um die Anzahl der Spalten festzulegen
        num_elements = len(st.session_state.json_data["table"]["types"])
        columns = st.columns(num_elements)  # Erstellt eine angegebene Anzahl von Spalten

        # st.json(json_data)

        number_of_observations = st.session_state.json_data["table"]["n"]
        number_of_variables = st.session_state.json_data["table"]["n_var"]
        number_of_numeric_vars = st.session_state.json_data["table"]["types"]["Numeric"]
        number_of_text_vars = st.session_state.json_data["table"]["types"]["Text"]

        # Erstellen der Spalten und Anzeigen der Metriken
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Number of variables", number_of_variables)

        with col2:
            st.metric("Number of observations", number_of_observations)

        with col3:
            st.metric("Numeric", number_of_numeric_vars)

        with col4:
            st.metric("Text", number_of_text_vars)

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
