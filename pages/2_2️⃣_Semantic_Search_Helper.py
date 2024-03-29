import streamlit

from modules.load_data_tab import main_load_data_tab, reset_app_state
from modules.embedding_tab import main_embedding_tab
from modules.similarity_pair_tab import main_similarity_pair_tab
from modules.explore_tab import main_explore_tab
from config import *  # Import the status messages from the config

# 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 💾


st.set_page_config(page_title=TOOLNAME, page_icon="2️⃣", layout="wide",
                   initial_sidebar_state="expanded")


def initialize_sidebar():
    """
    Initialize the sidebar with useful links and a reset button to clear the session state.
    """
    with st.sidebar:
        st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.write("[View the source code](https://github.com/karlgottfried/SemHarmoHelper)")
        if st.button('Restart Metadata Upload', use_container_width=True):
            reset_app_state()
            st.rerun()
        st.write("""Clicking **"Restart Metadata Upload"** clears all data and resets the application, allowing you to start afresh with new uploads or analyses.
                          Use this button to easily correct mistakes, explore different data sets, or restart your analysis from scratch.
                          """)


def initialize_session_state():
    # Initialize session state variables with default values if they don't exist.
    # This setup ensures that the app can safely access these variables at any point.
    default_values = {
        'metadata': None,  # Placeholder for dataset metadata
        EMBEDDING: None,  # Placeholder for embeddings data
        'similarity': None,  # Placeholder for similarity scores
        'selected_item_column': None,  # Column selected by the user for analysis
        'selected_questionnaire_column': None,  # Selected questionnaire column for analysis
        'selected_data': pd.DataFrame(),  # DataFrame to store user-selected data
        'df_filtered': None,  # Placeholder for filtered DataFrame based on user criteria
        'username': '',  # Username for login purposes (if applicable)
        'password': '',  # Password for login purposes (if applicable)
        LOINCDF: pd.DataFrame(),  # Placeholder for a specific DataFrame (replace LOINCDF with actual key)
        'model_used': None,  # Tracks the model selected by the user
        "duration_minutes_em": None,  # Duration of embedding calculation
        "load_chat": None,  # Placeholder for loading chat data (if applicable)
        'diagram_data_ready': False,  # Flag to indicate readiness of diagram data
        "question_counts_df": None,  # Placeholder for DataFrame with question counts
        # Adding new session_state keys with their default values
        'topics': None,  # Topics from topic modeling
        'probs': None,  # Probabilities associated with topics
        'reduced_embeddings': None,  # UMAP reduced embeddings
        'cluster_labels': None,  # HDBSCAN cluster labels
        'df_reduced': pd.DataFrame(),  # DataFrame containing reduced embeddings and associated data
        "step1_completed": False,
        "step2_completed": False,
        "step3_completed": False,
        "step4_completed": False
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # No explicit rerun command here; ensure to rerun or refresh your Streamlit app as needed.


def display_status_updates():
    """
    Display status updates for each step based on the completion flags in session state.
    """
    col1, col2, col3 = st.columns(3)
    status_messages = [("Step 1", 'step1_completed'), ("Step 2", 'step2_completed'), ("Step 3", 'step3_completed')]

    for col, (step, flag) in zip((col1, col2, col3), status_messages):
        with col:
            if st.session_state.get(flag, False):
                st.success(f"{step} completed!")
            else:
                st.info(f"{step} not done yet.")


def main():
    """
    Main function to structure the app into different tabs for each step of the process.
    """
    st.title(f"📒 {TOOLNAME}")

    initialize_session_state()
    initialize_sidebar()
    display_status_updates()

    tabs = st.tabs(
        [STEP_LOAD_SENTENCE_DATA, STEP_BUILD_EMBEDDINGS, STEP_BUILD_SIMILARITY_PAIRS, STEP_SELECT_AND_EXPLORE_PAIRS])

    with tabs[0]:
        STEP_1_MARKDOWN = f"""
        ### Step 1: Load Sentence Data

        Welcome to the first step of the {TOOLNAME}, where your journey to uncover semantic connections begins. This foundational step is crucial for setting up the rest of the analysis. Here's what you need to do:

        **Uploading Your Metadata:**

        - **Choose Your Source**: You have the option to upload metadata directly from LOINC or to upload new metadata. Select the option that best fits your dataset.

        - **File Upload**: Click on 'Upload a metadata file' and choose your file. Our tool supports various formats, including CSV, XLSX, and XLS, with a limit of 200MB per file.

        - **Sample Data**: If you're new or would like to test the functionality, use the 'Load Sample File' option. This provides you with a pre-loaded dataset to explore the tool's capabilities.

        **Preparing Your Data:**

        - **Metadata Preview**: Once uploaded, you'll get a preview of your metadata, giving you a glimpse into the dataset's structure.

        - **Select Relevant Columns**: Identify and select the columns containing the questionnaire names and item texts. This step is essential for accurately processing your data in the subsequent steps.

        **Proceed with Confidence:**

        - Armed with your metadata, you're now set to build the embeddings in Step 2. This is where the magic happens, and your text data begins its transformation into meaningful insights.

        - Remember, the quality of the input data directly impacts the effectiveness of the semantic search. Ensure your metadata is accurate and representative of your dataset.
        """
        with st.expander("Instructions and Tips"):
            st.markdown(STEP_1_MARKDOWN)
        main_load_data_tab()

    with tabs[1]:
        STEP_2_MARKDOWN = """
            ### Step 2: Build Embeddings

            In Step 2, our goal is to convert your textual data into a numerical form known as 'embeddings'. These are the steps to follow:

            - **Understanding Embeddings**: Embeddings are numerical vector representations of your text. They capture the contextual meaning of words, phrases, or sentences, enabling the machine to process and understand the nuances of language.

            - **Selecting a Model**: After uploading your data in Step 1, choose a suitable embedding model. Use the ADA model for high accuracy with English texts, or opt for Sentence BERT models for multilingual capabilities.

            - **Processing**: Click the 'Calculate Embeddings' button to start converting your text data into embeddings. The processing time will depend on the volume of your data.

            - **Proceeding to Next Steps**: With embeddings ready, move on to Step 3 to create similarity pairs, where you'll explore semantic similarities within your data.

            Selecting the right model for your data is critical for accurate semantic analysis. If in doubt, experiment with different models to discover which yields the most relevant insights for your dataset. Make sure your data is clean and organized before proceeding to ensure the best quality embeddings.

            Now, let's begin building embeddings to uncover the hidden insights in your text data!
            """
        with st.expander("Instructions and Tips"):
            st.markdown(STEP_2_MARKDOWN)

        if st.session_state["step1_completed"]:
            main_embedding_tab()

    with tabs[2]:
        STEP_3_MARKDOWN = f"""
        ### Step 3: Build Similarity Pairs

        After generating embeddings in Step 2, you're ready to explore the core of semantic analysis: finding meaningful connections within your data.

        In this step, our tool calculates the semantic similarity between all pairs of sentences or items in the selected column (`{st.session_state.selected_item_column}`) of your dataset. Follow these steps:

        1. **Calculate Similarity**: Click the "Calculate Similarity for all pairs of column {st.session_state.selected_item_column}" button. Our algorithms will process the embeddings to identify semantically similar items.
        2. **Explore the Results**: After the calculation, you'll see a list or matrix of item pairs and their similarity scores, indicating how closely two pieces of text are related in context and meaning.

        This crucial step aids in understanding your text data's landscape and informs decisions about harmonizing items across multiple questionnaires or datasets.

        Remember, the accuracy of similarity scores heavily relies on the quality of your embeddings. Choose the most appropriate model for your data and language in Step 2 for optimal results.

        Proceed to Step 4 to select and explore the pairs that best serve your analysis or harmonization efforts.
        """

        with st.expander("Instructions and Tips"):
            st.markdown(STEP_3_MARKDOWN)

        if st.session_state["step2_completed"]:
            main_similarity_pair_tab()

    with tabs[3]:
        STEP_4_MARKDOWN = """
            ### Step 4: Select and Explore Pairs

            You have made it to the final step of the Semantic Search Helper! With Steps 1 through 3 completed, you now have a list of semantically similar sentence pairs ready for detailed examination and exploration.

            **What to Expect in Step 4?**

            - **Refined Filtering**: Use the filter options to narrow down the similarity pairs by the similarity score. This allows you to focus on the most relevant matches, as per your chosen threshold.

            - **Review of Filtered Data**: The filtered dataframe presents a curated set of sentence pairs that align closely with your criteria. It shows you the number of elements that meet the set filter conditions.

            - **Final Selection**: From the filtered list, add the most promising candidates to your final selection for in-depth analysis.

            - **Exploration and Visualization**: Gain an overview of the similarity scores within the filtered range, with the ability to sort based on different criteria such as 'Percent Matches'. Visualize the data distribution and relationships through various graph options.

            - **Identifying Gaps**: The tool will also highlight any questionnaires that do not have suitable matching pairs, ensuring you are aware of any areas that may require additional attention or a different approach.

            - **Actionable Insights**: Step 4 is designed not only to assist in the selection process but also to provide you with actionable insights. This step helps you understand the degree of semantic alignment across various items and questionnaires, empowering you to make informed decisions for your semantic harmonization tasks.

            Now, engage with the final step to solidify your semantic search findings and capitalize on the semantic associations within your data. Step 4 is where all your previous work comes together, providing you with a clear path forward.
            """

        with st.expander("Instructions and Tips"):
            st.markdown(STEP_4_MARKDOWN)

        if st.session_state["step3_completed"]:
            main_explore_tab()


if __name__ == "__main__":
    main()
