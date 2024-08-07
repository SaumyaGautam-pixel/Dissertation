#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Configuration Variables
API_KEY = "" #api key has been removed due to privacy concerns
API_URL = '' #api url has been removed due to privacy concerns

def get_resp(input_text, model, role):
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": input_text}]
    }
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error {response.status_code}: {response.content.decode('utf-8')}"

def generate_risk_prompt(df, activity_name):
    activity_data = df[df['Activity'] == activity_name]
    mode_risk_l = activity_data['Initial Risk L'].mode()[0]
    mode_risk_s = activity_data['Initial Risk S'].mode()[0]
    unique_hazard_data = activity_data.drop_duplicates(subset='Hazard')
    prompt = (
        f"Generate a risk assessment document in markdown format for the activity '{activity_name}'.\n"
        f"Focus on the following details without adding any other additional information such as introductions, recommendations, or meanings. Only include details from the dataset provided.\n"
    )
    for index, row in unique_hazard_data.iterrows():
        prompt += (
            f"- Hazard: {row['Hazard']}, Risk Level: L{mode_risk_l}, S{mode_risk_s}, "
            f"R{mode_risk_l * mode_risk_s}, Identified at risk: {row['Identified at Risk']}, Control Measures: "
            f"{row['Control Measures']}, Residual Risk: L{row['Residual Risk L']}, S{row['Residual Risk S']}, "
            f"R{row['Residual Risk R']}\n"
        )
    return prompt, unique_hazard_data, mode_risk_l, mode_risk_s

def plot_risk_matrix(hazard_data, mode_risk_l, mode_risk_s):
    st.write("### Risk Matrix for each Hazard")
    for index, row in hazard_data.iterrows():
        hazard = row['Hazard']
        initial_risk = row['Initial Risk R']
        likelihood = mode_risk_l
        severity = mode_risk_s

        # Define risk matrix dimensions
        likelihood_values = np.arange(1, 6)
        severity_values = np.arange(1, 6)
        risk_matrix = np.zeros((5, 5))

        # Fill the matrix with risk values
        for l in likelihood_values:
            for s in severity_values:
                risk_matrix[l-1, s-1] = l * s

        # Define the custom color map
        colors = ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']  # green, yellow, orange, red
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 5, 10, 15, 25]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plotting the risk matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(risk_matrix, cmap=cmap, norm=norm)

        # Annotate cells with risk value and highlight the specified cell
        for i in range(5):
            for j in range(5):
                risk_value = int(risk_matrix[i, j])
                if (i+1 == likelihood) and (j+1 == severity):
                    ax.text(j, i, f'{risk_value}', va='center', ha='center', color='black', weight='bold', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
                else:
                    ax.text(j, i, f'{risk_value}', va='center', ha='center', color='black', fontsize=10)

        # Set labels
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels(['1', '2', '3', '4', '5'])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel('Likelihood', fontsize=12)
        ax.xaxis.set_label_position('top') 
        plt.title(f'Risk Matrix for {hazard}', fontsize=14, pad=20)
        plt.colorbar(cax, label='Risk Value', boundaries=bounds, ticks=[1, 5, 10, 15, 20, 25])

        # Remove grid lines
        ax.grid(False)

        # Add some stylistic elements
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(5+1)-.5, minor=True)
        ax.set_yticks(np.arange(5+1)-.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        
        st.pyplot(fig)

st.title('Risk Assessment Document Generator')

# Load Excel file
df = pd.read_excel('/Users/saumyagautam/Desktop/Data-8.xlsx')
# Adjust path if necessary

# Predefined activities
activities = ['Site Entrance', 'Excavation', 'Delivery vehicles & drivers', 'Dewatering', 'Shuttering', 'Concreting']

# User input for activity name
activity_name = st.selectbox('Select the activity name:', activities)
model_name = st.selectbox('Select the model:', ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o'])
role = 'Document Generator'

if st.button('Generate Risk Assessment Document'):
    if not df.empty:
        risk_assessment_prompt, hazard_data, mode_risk_l, mode_risk_s = generate_risk_prompt(df, activity_name)
        document = get_resp(risk_assessment_prompt, model_name, role)
        st.markdown(document)
        plot_risk_matrix(hazard_data, mode_risk_l, mode_risk_s)
    else:
        st.error("Data frame is empty. Please load the data properly.")

