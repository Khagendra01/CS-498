import streamlit as st
import json
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Named constants
DESIRED_LENGTH = 16689
THRESHOLD = 0.5

def process_file(uploaded_file):
    # Load JSON data
    with open(uploaded_file, 'r') as f:
        json_data = json.load(f)
    
    # Extract logs and process into DataFrame
    log_field = json_data['submission']['logs'][0]['log']
    logs = log_field.strip().split('\n')
    log_list = []
    for log in logs:
        data = json.loads(log)
        log_list.append(data)
    df = pd.DataFrame(log_list)
    
    # Clean and process data
    data_set = []
    cnt = 0
    for timestamp, total_chars, cursor_position in zip(df['t'][1:], df['_cs'].fillna(method='ffill')[1:], df['_c'].fillna(method='ffill')[1:]):
        if not pd.isnull(total_chars) and not pd.isnull(cursor_position):
            data_set.append(cnt)
            data_set.append(int(total_chars))
            data_set.append(int(cursor_position))
            cnt += 1
    
    # Pad or truncate data to desired length
    original_length = len(data_set)
    if original_length < DESIRED_LENGTH:
        padding_length = DESIRED_LENGTH - original_length
        data_set += [0] * padding_length
    else:
        data_set = data_set[:DESIRED_LENGTH]
    
    return data_set

def make_prediction(data_set):
    # Load model and make prediction
    model = keras.models.load_model("good_model.h5")
    X = [data_set]
    y_pred = model.predict(X)
    return y_pred > THRESHOLD

def main():
    st.title("Plagiarized Assignment Tester")
    st.write("Our AI model is currently trained on 550k parameter. Input value will be scaled to our trained model shape (16893).")
    
    uploaded_file = st.file_uploader("Upload your assignment file", type=['json'])
    
    if uploaded_file is not None:
        try:
            data_set = process_file(uploaded_file)
            prediction = make_prediction(data_set)
            if prediction:
                style = "background-color: red; padding: 10px; border-radius: 5px; color: white; font-size: 50px; text-align: center;"
                st.markdown(f'<div style="{style}">Plagiarized Assignment</div>', unsafe_allow_html=False)
            else:
                style = "background-color: green; padding: 10px; border-radius: 5px; font-size: 50px; text-align: center; color: white;"
                st.markdown(f'<div style="{style}">Good Assignment</div>', unsafe_allow_html=False)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Please upload a file.")

if __name__ == "__main__":
    main()