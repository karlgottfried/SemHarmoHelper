# config.py

# Import statements
import streamlit as st
import pandas as pd

# Constants for application settings
LOINC_BASE_URL = "https://fhir.loinc.org/"
SAMPLE_FILE_PATH = "path/to/your/sample_file.csv"  # Update this path to point to your sample file
SAMPLE_FILE_CSV = 'resources/Example_Metadata.xlsx'
FILE_TYPES = ["csv", "xlsx", "xls"]

# Constants for UI text elements
TOOLNAME = "Semantic Search Helper"
STEP_LOAD_SENTENCE_DATA = 'Step 1: Load Sentence Data'
STEP_BUILD_EMBEDDINGS = 'Step 2: Build Embeddings'
STEP_BUILD_SIMILARITY_PAIRS = 'Step 3: Build Similarity Pairs'
STEP_SELECT_AND_EXPLORE_PAIRS = "Step 4: Select and Explore Pairs"
DISPLAY_RADIO_TEXT_1 = "Pre-selection LOINC-Codes"
DISPLAY_RADIO_TEXT_2 = "All LOINC-Codes"
OPTION_1 = "LOINC Metadata Upload"
OPTION_2 = "New Metadata Upload"
OPTION_3 = "Sample Data Upload"

# Constants for data fields and display elements
COPYRIGHT_LOINC = "Copyright"
RESPONSE_DISPLAY_LOINC = "Response (Display)"
QUESTION_DISPLAY_LOINC = "Question (Display)"
CODE_LOINC = "Code"
QUESTIONNAIRE_LOINC = "Questionnaire"
QUESTIONNAIRE_ID = 'Questionnaire'
PERCENT_MATCHES = 'Percent Matches (%)'
ITEM_MATCHES = 'Number Of Item Matches'
NUMBER_OF_QUESTIONS = 'Number Of Questions'
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
