import streamlit as st

st.set_page_config(
    page_title="Harmonisation Helper",
    page_icon="📒 ",
)

st.write("# 📒Welcome to Harmonisation Helper!")

st.markdown(
    """
This tool is designed to facilitate the semantic harmonization of multi-item based questionnaires using embedding-based methods. It allows you to generate semantic embeddings from text data and utilize these embeddings to identify similarity between questionnair items. This task is 
The process is divided into a series of steps that will guide you through the harmonization journey.

### Step 1: [Data Exploration](Data_Explore) (Under Development)
This section is dedicated to the extraction of metadata from tabular data. It is currently in development and will be available soon to enhance your data preparation phase.

### Step 2: [Semantic Search Helper](Semantic_Search_Helper)
In the "Semantic Search Helper" section, select the model to use for generating embeddings:
- **ADA**: Choose an ADA model version from the dropdown list and enter your OpenAI API key. This step is necessary to compute ADA embeddings.
- **SBERT**: Select the appropriate Sentence Transformer model from the dropdown list that best matches your language and use case.

After selecting your model, click on the "Calculate Embeddings" button to begin the embedding computation.

### Step 3: Build Similarity Pairs
Once embeddings are calculated, you can proceed to identify similar sentences or documents in the "Build Similarity Pairs" section.

### Step 4: Select and Explore Pairs
Finally, in the "Select and Explore Pairs" section, you can review and analyze the matched pairs to gain deeper insights.

### Upcoming Features
- **Syntactic Harmonization**: We are working to include syntactic harmonization as an additional step, which will further refine the alignment of your questionnaires.

### Useful Tips
- Ensure your text data is clean and prepared to obtain the best results from semantic search.
- Protect your API key when using the ADA model approach—do not share it. It is a confidential access token.
- The computation of embeddings may take some time, depending on the volume of data and the chosen model. Please be patient.

Good luck with your semantic harmonization!

---

To get started, simply select the "Semantic Search Helper" tab and follow the steps outlined above.

"""
)