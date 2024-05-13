
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go


def generate_chart(value, visible_values):
    # Sample data (you can modify this to have different scenarios)
    metrics = ['Backward Transfer', 'Forward Transfer (final success rates)']

    num_epochs = 10

    values = {
        "BUDS ER": [],
        "DINO ER": [],
        "EE ER": [],
        "Joint ER": [],
        "Multitask Model": []
    }
    for i in range(num_epochs):
        for key in values.keys():
            values[key].append([np.random.randint(0, 100), np.random.randint(0, 100)])

    print(values["BUDS ER"][value])
    traces = []
    
    for model_name in values.keys():
        if model_name in visible_values:
            traces.append(go.Bar(
                name=model_name,
                x=metrics,
                y=values[model_name][value],
            ))
        else:
            traces.append(go.Bar(
                name=model_name,
                x=metrics,
                y=[0, 0],
            ))
    # Create the bar chart
    fig = go.Figure(data=traces)

    # Add title and metrics
    fig.update_layout(
        title="",
        xaxis_title=f"Models (Epoch={value})",
        yaxis_title="Success Rates",
        barmode="group",
        yaxis=dict(range=[0, 100]),
    )

    summary_values = {
        "Best Backward Transfer": 20,
        "Best Backward Transfer Delta": -5,
        "Best Forward Transfer": 30,
        "Best Forward Transfer Delta": 6,        
    }

    return fig, summary_values


placeholder = st.empty()

checkboxes = {}
visible_values = []

with placeholder.container():

    # create three columns
    cols = st.columns(4)

    with cols[0]:

        # Streamlit code
        # st.title("Interactive Bar Chart with Streamlit and Plotly")
        final_result = st.checkbox("Show Final Result", value=True)
        disabled = False
        if final_result:
            disabled = True
        # Create a slider

        slider_value = st.slider("Select a data scenario", 1, 4, 1, disabled=disabled)
        if disabled:
            # Set it to the best checkpoint index
            slider_value = 3
        # Add radio buttons
        for key in ["BUDS ER", "DINO ER", "EE ER", "Joint ER", "Multitask Model"]:
            checkboxes[key] = st.checkbox(key, value=False)

        for key in checkboxes.keys():
            if checkboxes[key]:
                visible_values.append(key)
        # Generate chart based on slider value
        chart, summary_values = generate_chart(slider_value, visible_values)

        # Display the chart
        st.plotly_chart(chart)
    with cols[1]:
        st.metric(label="Backward Transfer", value=summary_values["Best Backward Transfer"], delta=summary_values["Best Backward Transfer Delta"])
    with cols[2]:
        st.metric(label="Forward Transfer", value=summary_values["Best Forward Transfer"], delta=summary_values["Best Forward Transfer Delta"])
