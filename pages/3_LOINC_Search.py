import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

st.title('LOINC FHIR Questionnaires')

# Initialize session state variables for username and password if they are not set
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'password' not in st.session_state:
    st.session_state['password'] = ''


# Function to set credentials
def set_credentials():
    st.session_state['username'] = st.sidebar.text_input("Username")
    st.session_state['password'] = st.sidebar.text_input("Password", type="password")


def load_questionnaires(url):
    # Check if username and password are entered
    if st.session_state['username'] and st.session_state['password']:
        response = requests.get(url, auth=HTTPBasicAuth(st.session_state['username'], st.session_state['password']))
        if response.status_code == 200:
            return response.json()  # Assuming the response contains JSON data
        else:
            st.error("Error loading questionnaires.")
            return None
    else:
        st.warning("Please enter username and password.")
        return None


# Function to display questionnaires
def display_questionnaires(data):
    if data:
        st.write(data["item"])
    else:
        st.write("No data found.")


# Function to extract data
def extract_data(title, copyright, json_data):
    rows = []
    for item in json_data:
        # In case 'code' is not present
        for code in item.get("code", []):
            # Extract response options as a string with line breaks
            answer_options = "\n".join([opt["valueCoding"]["display"] + " | " for opt in item.get("answerOption", [])])
            # Add extracted data to a list of rows
            rows.append({
                "ID": item["linkId"],
                "Questionnaire": title,
                "Code": code["code"],
                "Question (Display)": code["display"],
                "Response (Display)": answer_options,
                "Copyright": copyright
            })
    # Create a DataFrame from the list of rows
    return pd.DataFrame(rows)


set_credentials()

codes = {
    "Patient health questionnaire 4 item": "69724-3",
    "Kansas City cardiomyopathy questionnaire": "71941-9",
    "Generalized anxiety disorder 7 item": "69737-5"
}

# MultiSelect box to select multiple questionnaires
selected_names = st.multiselect("LOINC Codes", list(codes.keys()))

# Generate URLs based on selected codes
urls = [f"https://fhir.loinc.org/Questionnaire/{codes[name]}" for name in selected_names]

# Initialize an empty list for the dataframes
dfs = []

# Button to start loading the questionnaires
if st.button("Load questionnaires"):
    for url in urls:
        # Load questionnaire data for each selected URL
        data = load_questionnaires(url)
        # Extract data and add the DataFrame to the list
        df = extract_data(data["title"], data["copyright"], data["item"])

        with st.expander(data["title"]):
            # Expander for each questionnaire
            st.write(df)  # Display the questionnaire table in the expander
            st.info(data["copyright"])

        dfs.append(df)

    # Combine all DataFrames in the list into a single DataFrame
    if dfs:
        st.write("Combined table of all selected questionnaires:")
        combined_df = pd.concat(dfs, ignore_index=True)
        st.write(combined_df)
    else:
        st.error("No data loaded. Please select at least one questionnaire.")
