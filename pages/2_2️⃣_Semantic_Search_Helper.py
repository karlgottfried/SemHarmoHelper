from modules.load_data_tab import show_load_data_tab
from modules.embedding_tab import show_embedding_tab
from modules.similarity_pair_tab import show_similarity_pair_tab
from modules.explore_tab import show_explore_tab
from config import *  # Import the status messages from the config

st.set_page_config(page_title="Semantic Search Helper", page_icon="2️⃣", layout="wide",
                   initial_sidebar_state="expanded")

# Defining a dictionary with default values
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
    "load_chat": None,
    'diagram_data_ready': False,
    "question_counts_df": None
}


def initialize_session_state():
    # Initializing session state variables with default values if they don't exist
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Rerun the Streamlit script to update session state


# Function to display status updates based on session state flags
def display_status_updates():
    # Creating three columns to display status updates
    col1, col2, col3 = st.columns(3)

    with col1:
        # Display status for Step 1
        status_container = st.container()
        if st.session_state.get('step1_completed', False):
            status_container.success("Step 1 completed!")
        else:
            status_container.info("Step 1 not done yet.")

    with col2:
        # Display status for Step 2
        emb_status_container = st.container()
        if st.session_state.get('step2_completed', False):
            emb_status_container.success("Step 2 completed!")
        else:
            emb_status_container.info("Step 2 not done yet.")

    with col3:
        # Display status for Step 3
        sim_status_container = st.container()
        if st.session_state.get('step3_completed', False):
            sim_status_container.success("Step 3 completed!")
        else:
            sim_status_container.info("Step 3 not done yet.")


st.title("📒 Semantic Search Helper")
# select_tab, view_tab, store_tab = st.tabs(['Load Sentence Data', 'Build Embeddings', 'View Similarity'])

with st.sidebar:
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/karlgottfried/SemHarmoHelper/blob/main/app.py)"

initialize_session_state()
# Display the status updates
display_status_updates()

load_tab, embedding_tab, pair_tab, explore_tab = st.tabs(
    [STEP_LOAD_SENTENCE_DATA, STEP_BUILD_EMBEDDINGS, STEP_BUILD_SIMILARITY_PAIRS,
     STEP_SELECT_AND_EXPLORE_PAIRS])

with load_tab:
    show_load_data_tab()

with embedding_tab:
    show_embedding_tab()

with pair_tab:
    show_similarity_pair_tab()

with explore_tab:
    show_explore_tab()
