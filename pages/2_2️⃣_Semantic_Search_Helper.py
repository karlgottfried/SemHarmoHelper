from modules.load_data_tab import show_load_data_tab
from modules.embedding_tab import show_embedding_tab
from modules.similarity_pair_tab import show_similarity_pair_tab
from modules.explore_tab import show_explore_tab
from config import *  # Import the status messages from the config

st.set_page_config(page_title="Semantic Search Helper", page_icon="2️⃣", layout="wide",
                   initial_sidebar_state="expanded")

initialize_session_state()


# Function to display status updates based on session state flags
def display_status_updates():
    # Display status for Step 1
    status_container = st.container()
    if st.session_state.get('step1_completed', False):
        rows = st.session_state.get('step1_rows', 0)
        status_container.success(
            f"Step 1 completed! {rows} rows processed."
        )
    else:
        status_container.info("Step 1 not done yet.")

    # Display status for Step 2
    emb_status_container = st.container()
    if st.session_state.get('step2_completed', False):
        rows = len(st.session_state.get('EMBEDDING', []))
        minutes = st.session_state.get('duration_minutes_em', 'N/A')
        emb_status_container.success(STATUS_MESSAGES["step2_completed"].format(rows=rows, minutes=minutes))
    else:
        emb_status_container.info(STATUS_MESSAGES["step2_pending"])

    # Display status for Step 3
    sim_status_container = st.container()
    if st.session_state.get('step3_completed', False):
        pairs = len(st.session_state.get('similarity', []))
        sim_status_container.success(STATUS_MESSAGES["step3_completed"].format(pairs=pairs))
    else:
        sim_status_container.info(STATUS_MESSAGES["step3_pending"])


st.title("📒 Semantic Search Helper")
# select_tab, view_tab, store_tab = st.tabs(['Load Sentence Data', 'Build Embeddings', 'View Similarity'])

with st.sidebar:
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/karlgottfried/SemHarmoHelper/blob/main/app.py)"

# Display the status updates
display_status_updates()

load_tab, embedding_tab, pair_tab, explore_tab = st.tabs(
    [STEP_LOAD_SENTENCE_DATA, 'Step 2: Build Embeddings', 'Step 3: Build Similarity Pairs',
     "Step 4: Select and Explore Pairs"])

with load_tab:
    show_load_data_tab()

with embedding_tab:
    show_embedding_tab()

with pair_tab:
    show_similarity_pair_tab()

with explore_tab:
    show_explore_tab()
