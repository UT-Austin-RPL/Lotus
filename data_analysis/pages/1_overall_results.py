
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go


def generate_chart(value):
    # Sample data (you can modify this to have different scenarios)
    labels = ['Apples', 'Bananas', 'Cherries', 'Dates']

    if value == 1:
        values = [450, 240, 300, 210]
    elif value == 2:
        values = [320, 430, 210, 400]
    elif value == 3:
        values = [150, 340, 400, 320]
    else:
        values = [250, 280, 230, 300]

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
    )])

    # Add title and labels
    fig.update_layout(
        title=f"Fruits Count - Scenario {value}",
        xaxis_title="Fruits",
        yaxis_title="Count"
    )

    return fig, values[2] - values[1]


placeholder = st.empty()


with placeholder.container():

    # create three columns
    cols = st.columns(3)

    with cols[0]:

        # Streamlit code
        # st.title("Interactive Bar Chart with Streamlit and Plotly")

        # Create a slider
        slider_value = st.slider("Select a data scenario", 1, 4, 1)

        # Generate chart based on slider value
        chart, value_diff = generate_chart(slider_value)

        # Display the chart
        st.plotly_chart(chart)
    with cols[2]:
        st.metric(label="Difference", value=value_diff, delta=0)
        # # Sample data
        # labels = ['Apples', 'Bananas', 'Cherries', 'Dates']
        # values = [450, 240, 300, 210]

        # # Create the bar chart
        # fig = go.Figure(data=[go.Bar(
        #     x=labels,
        #     y=values,
        #     # marker_color=['red', 'yellow', 'pink', 'brown'],  # Optional: use specific colors for each bar
        # )])

        # # Add title and labels
        # fig.update_layout(
        #     title="Fruits Count",
        #     xaxis_title="Fruits",
        #     yaxis_title="Count"
        # )
        # # st.bar_chart(x=[1, 2], y=[10, 20], width=0, height=0, use_container_width=True)
        # st.plotly_chart(fig, use_container_width=True)

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