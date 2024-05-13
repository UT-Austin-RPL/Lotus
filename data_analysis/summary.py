import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development


st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

# Using "with" notation
# with st.sidebar:
    # add_radio = st.radio(
    #     "Choose a shipping method",
    #     ("Standard (5-15 days)", "Express (2-5 days)")
    # )
placeholder = st.empty()

with placeholder.container():

    st.header('Summary sentence.')
    st.markdown('Some conclusions')

    # create three columns
    kpi1, kpi2, kpi3 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
    # kpi1.metric(
    #     label="Age ⏳",
    #     value=10,
    #     delta=2,
    # )
    
    # kpi2.metric(
    #     label="Married Count 💍",
    #     value=4,
    #     delta=-3,
    # )