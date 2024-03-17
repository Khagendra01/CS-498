import streamlit as st
import json
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

st.title("Plagiarized Assignment Tester")

st.write("Our AI model is currently trained on 539393 parameter. Input value will scaled to our trained model shape (16893)")

model = keras.models.load_model("full_regular_model.h5")

uploaded_file = st.file_uploader("Upload your assignment file", type=['submit'])

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    json_data = json.load(uploaded_file)
    log_field = json_data['submission']['logs'][0]['log']
    logs = log_field.strip().split('\n')

    log_list = []

    for log in logs:
        data = json.loads(log)
        log_list.append(data)

    df = pd.DataFrame(log_list)

    data_set = []
    cnt = 0
    # Processing the DataFrame skipping the first entry
    for timestamp, total_chars, cursor_position in zip(df['t'][1:], df['_cs'].fillna(method='ffill')[1:], df['_c'].fillna(method='ffill')[1:]):
       if not pd.isnull(total_chars) and not pd.isnull(cursor_position):
            data_set.append( cnt )
            data_set.append( int(total_chars) )
            data_set.append( int(cursor_position) )
            cnt += 1

    original_length = len(data_set)
    desired_length = 16689

    if original_length < desired_length:
        padding_length = desired_length - original_length
        data_set += [0] * padding_length
    else:
        data_set = data_set[:desired_length]

    X = []
    X.append(data_set)

    y_pred_numb_binary_flat = (model.predict(X).flatten() > 0.5).astype(int)
    if y_pred_numb_binary_flat[0] == 1:
     style = "background-color: red; padding: 10px; border-radius: 5px; color: red; font-size: 50px; text-align: center; color:white;"
     st.markdown(f'<div style="{style}">Plaziarized Assignement</div>', unsafe_allow_html=True)
    else:
        style = "background-color: green; padding: 10px; border-radius: 5px; font-size: 50px; text-align: center; color:white;"
        st.markdown(f'<div style="{style}">Good Assignment</div>', unsafe_allow_html=True)

    
    

else:
    st.write("Please upload a file.")

st.write("Authored by Kgen, Shadip")
