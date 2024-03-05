from modules.load_data_tab import show_load_data_tab
from modules.embedding_tab import show_embedding_tab
from modules.similarity_pair_tab import show_similarity_pair_tab
from modules.explore_tab import show_explore_tab
from modules.status_view import display_status_updates, initialize_session_state
from config import *  # Import the status messages from the config


st.set_page_config(page_title="Semantic Search Helper", page_icon="2️⃣", layout="wide",
                   initial_sidebar_state="expanded")

st.title("📒 Semantic Search Helper")

with st.sidebar:
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/karlgottfried/SemHarmoHelper/)"

initialize_session_state()

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
