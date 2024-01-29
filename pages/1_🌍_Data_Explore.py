import pandas as pd
import ydata_profiling
import streamlit as st
import numpy as np
from streamlit_pandas_profiling import st_profile_report
from io import StringIO


st.set_page_config(page_title="Data Explorer", page_icon="🌍")
st.subheader("This app will help you to do Data Exploration")
st.sidebar.header('User Input Features')


uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    profile = ydata_profiling.ProfileReport(dataframe, title="New Data for profiling")
    st.subheader("Detailed Report of the Data Used")
    st.write(dataframe)
    st_profile_report(profile)


else:
    st.write("You did not upload the new file")


