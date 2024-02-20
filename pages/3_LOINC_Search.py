import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

st.title('LOINC FHIR Fragebögen')

# Initialisieren Sie die Session State Variablen für Benutzername und Passwort, wenn sie noch nicht gesetzt sind
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'password' not in st.session_state:
    st.session_state['password'] = ''


# Funktion, um die Anmeldeinformationen zu setzen
def set_credentials():
    st.session_state['username'] = st.sidebar.text_input("Benutzername")
    st.session_state['password'] = st.sidebar.text_input("Passwort", type="password")


def load_questionnaires(url):
    if st.session_state['username'] and st.session_state['password']:
        response = requests.get(url, auth=HTTPBasicAuth(st.session_state['username'], st.session_state['password']))
        if response.status_code == 200:
            return response.json()  # Annahme, dass die Antwort JSON-Daten enthält
        else:
            st.error("Fehler beim Laden der Fragebögen.")
            return None
    else:
        st.warning("Bitte geben Sie Benutzername und Passwort ein.")
        return None


# Funktion, um Fragebögen anzuzeigen
def display_questionnaires(data):
    if data:
        st.write(data["item"])
    else:
        st.write("Keine Daten gefunden.")


# Funktion zum Extrahieren der Daten
def extract_data(title, copyr, json_data):
    rows = []
    for item in json_data:
        for code in item.get("code", []):  # Für den Fall, dass "code" nicht vorhanden ist
            # Extrahieren der Antwortoptionen als String mit Zeilenumbrüchen
            answer_options = "\n".join([opt["valueCoding"]["display"] + " | " for opt in item.get("answerOption", [])])
            # Hinzufügen der extrahierten Daten zu einer Liste von Zeilen
            rows.append({
                "ID": item["linkId"],
                "Questionnaire": title,
                "Code": code["code"],
                "Question (Display)": code["display"],
                "Response (Display)": answer_options,
                "copyright": copyr
            })
    # Erstellen eines DataFrames aus der Liste von Zeilen
    return pd.DataFrame(rows)


set_credentials()

codes = {
    "Patient health questionnaire 4 item": "69724-3",
    "Kansas City cardiomyopathy questionnaire": "71941-9",
    "Generalized anxiety disorder 7 item": "69737-5"
}

# MultiSelect-Box für die Auswahl mehrerer Fragebögen
selected_names = st.multiselect("LOINC-Codes", list(codes.keys()))

# URLs basierend auf den ausgewählten Codes generieren
urls = [f"https://fhir.loinc.org/Questionnaire/{codes[name]}" for name in selected_names]

# Eine leere Liste für die Datenframes initialisieren
dfs = []

# Button, um das Laden der Fragebögen zu starten
if st.button("Fragebögen laden"):
    for url in urls:
        # Lade die Fragebogendaten für jede ausgewählte URL
        data = load_questionnaires(url)
        # Extrahiere Daten und füge das DataFrame der Liste hinzu
        df = extract_data(data["title"], data["copyright"], data["item"])

        with st.expander(data["title"]):
            # Expander für jeden Fragebogen
            st.write(df)  # Anzeigen der Fragebogentabelle im Expander
            st.info(data["copyright"])

        dfs.append(df)

    # Kombiniere alle DataFrames in der Liste zu einem einzigen DataFrame
    if dfs:
        st.write("Gesamte Ausgabetabelle aller ausgewählten Fragebögen:")
        combined_df = pd.concat(dfs, ignore_index=True)
        st.write(combined_df)
    else:
        st.error("Keine Daten geladen. Bitte wähle mindestens einen Fragebogen aus.")
