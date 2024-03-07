# config.py
import streamlit as st
import pandas as pd

COPYRIGHT_LOINC = "Copyright"
RESPONSE_DISPLAY_LOINC = "Response (Display)"
QUESTION_DISPLAY_LOINC = "Question (Display)"
CODE_LOINC = "Code"
QUESTIONNAIRE_LOINC = "Questionnaire"

PERCENT_MATCHES = 'Percent Matches (%)'
ITEM_MATCHES = 'Number Of Item Matches'
NUMBER_OF_QUESTIONS = 'Number Of Questions'
QUESTIONNAIRE_ID = 'Questionnaire'
QUESTIONNAIRE_MATCHES = "Questionnaire Matches"

LOINCDF = 'loincdf'
DF_FILTERED = "df_filtered"
SELECTED_DATA = 'selected_data'
QUESTIONNAIRE_COLUMN = 'selected_questionnaire_column'
SELECTED_ITEM_COLUMN = 'selected_item_column'
SIMILARITY = 'similarity'
METADATA = 'metadata'

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

STEP_LOAD_SENTENCE_DATA = 'Step 1: Load Sentence Data'
STEP_BUILD_EMBEDDINGS = 'Step 2: Build Embeddings'
STEP_BUILD_SIMILARITY_PAIRS = 'Step 3: Build Similarity Pairs'
STEP_SELECT_AND_EXPLORE_PAIRS = "Step 4: Select and Explore Pairs"

# Constants for the application
FILE_TYPES = ["csv", "xlsx", "xls"]
LOINC_BASE_URL = "https://fhir.loinc.org/"
SAMPLE_FILE_PATH = "path/to/your/sample_file.csv"  # Update this path

DISPLAY_RADIO_TEXT_2 = "All LOINC-Codes"
DISPLAY_RADIO_TEXT_1 = "Pre-selection LOINC-Codes"
TOOLNAME = "Semantic Search Helper"

