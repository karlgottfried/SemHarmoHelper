# config.py
import pandas as pd
import streamlit as st


COPYRIGHT_LOINC = "Copyright"
RESPONSE_DISPLAY_LOINC = "Response (Display)"
QUESTION_DISPLAY_LOINC = "Question (Display)"
CODE_LOINC = "Code"
QUESTIONNAIRE_LOINC = "Questionnaire"

PERCENT_MATCHES = 'Percent Matches (%)'
MATCHES = 'Number Of Matches'
NUMBER_OF_QUESTIONS = 'Number Of Questions'
QUESTIONNAIRE_ID = 'Questionnaire ID'

LOINCDF = 'loincdf'
DF_FILTERED = "df_filtered"
SELECTED_DATA = 'selected_data'
QUESTIONNAIRE_COLUMN = 'selected_questionnaire_column'
SELECTED_ITEM_COLUMN = 'selected_item_column'
SIMILARITY = 'similarity'
METADATA = 'metadata'

STEP_LOAD_SENTENCE_DATA = 'Step 1: Load Sentence Data'


SELECT = "Select"
EMBEDDING = "Embedding"
ITEM_1 = 'Item 1'
ITEM_2 = 'Item 2'
QUESTIONNAIRE_1 = "Questionnaire 1"
QUESTIONNAIRE_2 = 'Questionnaire 2'
MODEL_ADA = "ADA"
MODEL_SBERT = "SBERT"
SIMILARITY_SCORE = 'Similarity score'
SAMPLE_FILE_CSV = 'resources/Example_Metadata.xlsx'

STATUS_MESSAGES = {
    "step1_completed": "Step 1 done! Data selected containing {rows} rows",
    "step1_pending": "Step 1 not done yet",
    "step2_completed": "Step 2 done! Embeddings built containing {rows} rows in {minutes} minutes",
    "step2_pending": "Step 2 not done yet",
    "step3_completed": "Step 3 done! Similarity scores built between {pairs} pairs",
    "step3_pending": "Step 3 not done yet",
    "reset_confirmation": "Are you sure you want to reset all progress?",
}


# Definieren eines Dictionaries mit Standardwerten
default_values = {
    'metadata': None,
    EMBEDDING: None,
    'similarity': None,
    'selected_item_column': None,
    'selected_questionnaire_column': None,
    'selected_data': pd.DataFrame(),
    'df_filtered': None,
    'username': '',
    'password': '',
    LOINCDF: pd.DataFrame(),
    'model_used': None,
    "duration_minutes_em": None,
}


def initialize_session_state():
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.rerun
