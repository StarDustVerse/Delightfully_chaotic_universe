import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

M = [9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 
     11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.1, 13.2, 13.3, 13.4, 
     13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 
     14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 
     15.7, 15.8, 15.9, 16.0, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 
     16.8, 16.9, 17.0, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 
     17.9, 18.0, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 
     19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 
     20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 
     21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9, 22.0, 22.1, 22.2, 
     22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9, 23.0, 23.1, 23.2, 23.3, 
     23.4, 23.5, 23.6, 23.7, 23.8, 23.9, 24.0, 24.1, 24.2, 24.3, 24.4, 
     24.5, 24.6, 24.7, 24.8, 24.9, 25.0, 25.1, 25.2, 25.3, 25.4, 25.5, 
     25.6, 25.7, 25.8, 25.9, 26.0, 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 
     26.7, 26.8, 26.9, 27.0, 27.1, 27.2, 27.3, 27.4, 27.5, 27.6, 27.7, 
     27.8, 27.9, 28.0, 28.1, 28.2, 28.3, 28.4, 28.5, 28.6, 28.7, 28.8, 
     28.9, 29.0, 29.1, 29.2, 29.3, 29.4, 29.5, 29.6, 29.7, 29.8, 29.9, 
     31, 32, 33, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120]

M_solar = 1.989e33  # solar mass in grams

elements = {
    "H1": 'red',
    "He4": 'maroon',
    "C12": 'blue',
    "N14": 'saddlebrown',
    "O16": 'coral',
    "Ne20": 'purple',
    "Mg24": 'green',
    "Si28": 'deeppink',
    "S32": 'lime',
    "Fe56": 'black'
}

def load_data(M_proj):
    file_name = f'Sukhbold_progenitor_models/s{M_proj}_presn'
    with open(file_name, 'r') as file:
        lines = file.readlines()

    lines = lines[2:]
    processed_rows = [line.strip().replace('---', '0').split() for line in lines]

    columns = [
        "grid", "cell outer total mass", "cell outer radius", "cell outer velocity", 
        "cell density", "cell temperature", "cell pressure", "cell specific energy", 
        "cell specific entropy", "cell angular velocity", "cell A_bar", "cell Y_e", "stability", 
        "NETWORK", "neutrons", "H1", "He3", "He4", "C12", "N14", "O16", "Ne20", 
        "Mg24", "Si28", "S32", "Ar36", "Ca40", "Ti44", "Cr48", "Fe52", "Fe54", 
        "Ni56", "Fe56", "Fe"
    ]

    df = pd.DataFrame(processed_rows, columns=columns)
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    df["cell outer total mass"] /= M_solar  # Convert to solar masses
    return df

# -------------------------
# Streamlit UI
# -------------------------
st.title("Abundance Profiles from Kepler Progenitor Models (Sukhbold et al. 2016)")

M_proj = st.selectbox("Select Initial Mass", M, index=M.index(13.5))
selected_elements = st.multiselect(
    "Select Elements to Plot",
    options=list(elements.keys()),
    default=['H1', 'He4', 'C12', 'O16', 'Si28']
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    x_min = st.number_input("X-axis Min", value=1.5, step=0.1)
with col2:
    x_max = st.number_input("X-axis Max", value=15.0, step=0.1)
with col3:
    y_min = st.number_input("Y-axis Min (log)", value=-3.0, step=0.1)
with col4:
    y_max = st.number_input("Y-axis Max (log)", value=0.3, step=0.1)

# Load data and make plot
df = load_data(M_proj)
M_coord = df["cell outer total mass"]

traces = []
for el in selected_elements:
    traces.append(go.Scatter(
        x=M_coord,
        y=df[el],
        mode='lines',
        name=el,
        line=dict(color=elements[el])
    ))

layout = go.Layout(
    xaxis=dict(title='Mass Coordinate (Solar Masses)', range=[x_min, x_max]),
    yaxis=dict(type='log', title='Mass Fraction', range=[y_min, y_max]),
    legend=dict(font=dict(size=14)),
    margin=dict(l=60, r=10, t=40, b=60),
    height=700
)

fig = go.Figure(data=traces, layout=layout)
st.plotly_chart(fig, use_container_width=True)
