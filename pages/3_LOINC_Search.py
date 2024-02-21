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


def load_questionnaires(url, flag):
    # Check if username and password are entered
    if st.session_state['username'] and st.session_state['password']:
        if flag:
            response = requests.get(url, params={"_count" : "10000"}, auth=HTTPBasicAuth(st.session_state['username'], st.session_state['password']))
        if not flag:
            response = requests.get(url,
                                    auth=HTTPBasicAuth(st.session_state['username'], st.session_state['password']))
        if response.status_code == 200:
            return response.json()
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

def extract_quest(json_data):
    results = {}
    for item in json_data:  # Direkter Zugriff auf das 'entry'-Feld angenommen
        resource = item.get('resource', {})
        titel = resource.get('title')
        id = resource.get('id')

        if titel and id:
            results[titel] = id
    return results

def fetch_all_resources(url):
    resources = {}
    load_count = 0  # Zählvariable für die Anzahl der Ladungen

    while url and load_count<1:  # Bedingung umfasst nun auch die Anzahl der Ladungen
        print("url", url)
        bundle = load_questionnaires(url, True)
        # Extrahiere Ressourcen aus dem aktuellen Bundle und füge sie der Liste hinzu
        if 'entry' in bundle:
            for entry in bundle['entry']:
                result = extract_quest(bundle['entry'])
                resources.update(result)  # Verwende `extend` statt `append`, um Listen korrekt zu verknüpfen

        # Suche nach dem 'next' Link im Bundle, um die nächste Seite abzurufen
        url = None  # Zurücksetzen der URL für den Fall, dass kein 'next' Link gefunden wird
        for link in bundle.get('link', []):
            if link['relation'] == 'next':
                url = link['url']
                break

        load_count += 1

    return resources

set_credentials()


# Basis-URL des FHIR-Servers und der Ressourcentyp, den du abrufen möchtest
base_url = "https://fhir.loinc.org/"
resource_type = "Questionnaire"
initial_url = f"{base_url}/{resource_type}"

slider = st.select_slider("Displayed values:", ["All LOINC-Codes", "Pre-selection LOINC-Codes"])

codes = {
    "Patient health questionnaire 4 item": "69724-3",
    "Kansas City cardiomyopathy questionnaire": "71941-9",
    "Generalized anxiety disorder 7 item": "69737-5",
    "":"69723-5"}

if slider == "All LOINC-Codes":

    all_resources = fetch_all_resources(initial_url)
    st.write(f"Total resources fetched: {len(all_resources)}")
    selected_names = st.multiselect("LOINC Codes", all_resources)
    ids = [all_resources[a] for a in selected_names]
    print(all_resources)

if slider == "Pre-selection LOINC-Codes":
    selected_names = st.multiselect("LOINC Codes", codes.keys())
    ids = [codes[a] for a in selected_names]





# MultiSelect box to select multiple questionnaires
#
#selected_names = st.multiselect("LOINC Codes", list(codes.keys()))

# Generate URLs based on selected codes
urls = [f"https://fhir.loinc.org/Questionnaire/{id}" for id in ids]

print(urls)

# Initialize an empty list for the dataframes
dfs = []

# Button to start loading the questionnaires
if st.button("Load questionnaires"):
    for url in urls:
        # Load questionnaire data for each selected URL
        data = load_questionnaires(url,False)
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
