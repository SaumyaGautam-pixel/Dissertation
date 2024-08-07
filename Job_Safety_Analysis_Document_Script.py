#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests
import json
import pandas as pd

# Configuration Variables
API_KEY = "" #api key has been removed for privacy concerns
API_URL = '' #api url has been removed for privacy concerns

def get_resp(input_text, model, role):
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": input_text}]
    }
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        response_data = json.loads(response.content)
        return extract_message(response_data, model)
    else:
        return f"Error {response.status_code}: {response.content.decode('utf-8')}"

def extract_message(response_data, model):
    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content']
    else:
        return response_data['content']  # For different API structures

def generate_jsa_prompt_from_df(df, start_index, batch_size):
    grouped = df.groupby('Sub-Activity Name')
    prompt_text = ""
    
    for i, (name, group) in enumerate(grouped):
        if i < start_index:
            continue
        if i >= start_index + batch_size:
            break
        sub_activity = name
        hazard = ", ".join(map(str, group['Type Of Hazard'].unique()))
        risks = ", ".join(map(str, group['Risk(s) Involved'].unique()))
        measures = "\n".join(map(str, group['Risk Control measures'].unique()))
        
        sub_activity_text = f"Sub-Activity: {sub_activity}\n"
        hazard_text = f"Type Of Hazard: {hazard}\n"
        risks_text = f"Risk(s) Involved: {risks}\n"
        measures_text = f"Risk Control measures:\n{measures}\n"
        prompt_text += sub_activity_text + hazard_text + risks_text + measures_text + "\n\n"
    
    # Explicit instruction to focus the response
    full_prompt = (f"Generate a Job Safety Analysis document based on the following details. "
                   "Do not include any administrative sections such as introduction, project details, declarations, or personal "
                   "identifications like names and dates. Focus solely on the safety analysis content:\n\n" + prompt_text)
    
    return full_prompt

st.title('Job Safety Analysis Document Generator')

# Load the Excel file directly from the specified path
df_jsa = pd.read_excel('/Users/saumyagautam/Desktop/Consolidated_JSA.xlsx')

# Predefined activities
activities = ['Area Hard barrication', 'Anti-termite treatment', 'Alluminium door fixing', 'Column Shuttering', 'Rebaring']

# User input for activity name
activity_name = st.selectbox('Select the activity name:', activities)
model_name = st.selectbox('Select the model:', ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o'])
role = 'user'

if st.button('Generate JSA Document'):
    filtered_jsa_df = df_jsa[df_jsa['Activity'] == activity_name]

    # Set batch parameters
    batch_size = 5  # Adjust batch size as needed
    start_index = 0

    # Initialize the full JSA document text
    full_jsa_document = ""

    # Process the data in batches
    while start_index < len(filtered_jsa_df['Sub-Activity Name'].unique()):
        jsa_prompt = generate_jsa_prompt_from_df(filtered_jsa_df, start_index, batch_size)
        jsa_document = get_resp(jsa_prompt, model_name, role)
        
        if not jsa_document.startswith("Error"):
            full_jsa_document += jsa_document + "\n\n"
        
        start_index += batch_size
    
    # Display the generated JSA document
    st.markdown(full_jsa_document)

