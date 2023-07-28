import streamlit as st
import subprocess
import pandas as pd
import os
from report import get_report
from generate_predicition import generate_sentence
from data_utils import get_data, get_month_data
import numpy as np
import joblib
from train_llm1 import train_model


def execute_file(file_path, user_input=None):
    cmd = f"conda run -n stellar_prediction python {file_path}"
    
    if user_input is not None:
        cmd += f'" {user_input}"'
    
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output_text = process.stdout

    return output_text

def input_to_array(input_text):
    try:
        # Convert the text input to a list of strings
        input_list = input_text.strip().split()

        # Convert the list of strings to a NumPy array
        input_array = np.array(input_list, dtype=float)

        return input_array
    except:
        return None


def main():

    st.title("Predictions for OSL Stellar")
    st.write("Points to note:")
    st.markdown("""
        <ul>
            <li>Please enclose file/folder paths in double inverted comms ("")</li>
            <li>If you are uploading an excel file with one sheet, please ensure the sheet name is Sheet1</li>
        </ul>
            """, unsafe_allow_html=True)
    

    type = st.sidebar.radio("Select an operation to perform: ", ("Create Report","Generate Prediction", "Train model"))

    if type == "Train model":
        st.title("Train Model")
        user_input = st.text_input("Please enter the path to the folder with the logbooks.")
        num_epochs = int(st.number_input("Please enter number of epochs (Default Value = 20)", value=20))
        batch_size = int(st.number_input("Please enter the batch size (Default Value = 32)", value=32))
        learning_rate = float(st.text_input("Please enter the learning rate (Default Value = 0.001)", value=0.001))

        if st.button("Start training the model") and user_input:
                folder_path = user_input
                train_model(folder_path, num_epochs, batch_size, learning_rate)

    if type == "Create Report":
        st.title("Create Report")

        # Add file uploader to accept Excel files
        uploaded_file = st.file_uploader("Please upload the Excel file for which you want a report", type=["xlsx"])
        st.write("This will generate report for the month, So please upload the month's logbook")
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_file_path = f"temp_file.xlsx"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("Please wait, this may take some time to load")
            #df = get_month_data(temp_file_path)
            #st.dataframe(df)
            
            fuelconsdata, runhrsdata = get_report(temp_file_path)
            runhrsdata.reset_index(drop=True, inplace=True)
            st.write(fuelconsdata, runhrsdata)

            total_time = len(runhrsdata)
            total_runhours = runhrsdata[['Run Hours Data PME', 'Run Hours Data SME', 'Run Hours Data AE1', 'Run Hours Data AE2', 'Run Hours Data AE3']].sum()
            percentage_runhours = total_runhours / (total_time*6) if total_time != 0 else st.write("No runhrs Data")
            st.write("Percentage of running time of each engine: ", percentage_runhours)
            #st.write(f"Total Runhours: {total_runhours} \n total_time = {total_time} \n actual_total_time = {total_time*6}")
            
            os.remove(temp_file_path)

    if type == "Generate Prediction":
        st.title("Generate Prediction")
        st.write("Here, there are two options, one to generate prediction for Work Done and one for Fuel Consumption")
        predfile = st.file_uploader("Please upload the Excel file for which you want prediction. \nEnsure sheet name is Sheet1", type=["xlsx"])
        if predfile is not None:
            # Save the uploaded file to a temporary location
            pred_temp_file_path = f"temp_file.xlsx"
            with open(pred_temp_file_path, "wb") as f:
                f.write(predfile.getbuffer())

            if st.button("Fuel Consumption Prediction"):
                loaded_model = joblib.load('xgb_model.pkl')
                st.write("Model Loaded")
                # Process user input and make prediction
                df = get_data(1, pred_temp_file_path)
                for i in range(4):
                    input_features = df.loc[i,['PME_PRES_LO', 'PME_PRES_FO', 'PME_PRES_FW', 'SME_PRES_LO', 'SME_PRES_FO', 'SME_PRES_FW', 'AE_PRES_LO', 
                                                'AE_PRES_FW', 'PRESSURE LO AE3', 'PRESSURE FW AE3', 'AC_PRES_SUC', 'AC_PRES_LO', 'AC_PRES_DISC', 'PME_TEMP_LO',
                                                'PME_TEMP_FW', 'PME_TEMP_EXH_MAX', 'PME_TEMP_EXH_MIN', 'PME_T/C_EXH', 'SME_TEMP_LO', 'SME_TEMP_FW', 'SME_TEMP_EXH_MAX',
                                                'SME_TEMP_EXH_MIN', 'SME_T/C_EXH', 'AE_TEMP_FW AE3', 'AE_TEMP_LO AE3', 'T/C_EXH AE3','PME_RPM', 'SME_RPM', 'LOAD', 'LOAD AE3']]

                    if input_features is not None:
                         input_array = input_features.values.reshape(1,-1)
                         prediction = loaded_model.predict(input_array)
                         st.write(f"Prediction for Quarter {i+1}\n {prediction.tolist()}")

                    else: st.write("Invalid input")

            if st.button("Work Done Prediction"):

                df = get_data(1, pred_temp_file_path)
                input_features = df.loc[1,['PME_PRES_LO', 'PME_PRES_FO',
                    'PME_PRES_FW', 'SME_PRES_LO', 'SME_PRES_FO', 'SME_PRES_FW',
                    'AE_PRES_LO', 'AE_PRES_FW', 
                    'PRESSURE LO AE3', 'PRESSURE FW AE3',
                    'AC_PRES_SUC', 'AC_PRES_LO', 'AC_PRES_DISC', 
                    'PME_TEMP_LO','PME_TEMP_FW', 'PME_TEMP_EXH_MAX', 'PME_TEMP_EXH_MIN', 'PME_T/C_EXH',
                    'SME_TEMP_LO', 'SME_TEMP_FW', 'SME_TEMP_EXH_MAX', 'SME_TEMP_EXH_MIN','SME_T/C_EXH',
                    'AE_TEMP_FW AE3', 'AE_TEMP_LO AE3', 'T/C_EXH AE3', 
                    'PME_RPM', 'SME_RPM', 'LOAD', 'LOAD AE3', 'RUN HOURS PME', 'RUN HOURS SME',
                    'RUN HOURS AE1', 'RUN HOURS AE2', 'RUN HOURS AE3', 'FUEL CONS PME',
                    'FUEL CONS SME', 'FUEL CONS AE_1', 'FUEL CONS AE_2', 'FUEL CONS AE_3']]
                    
                script_folder = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_folder, "trained_model")
                
                prediction = generate_sentence(input_features, model_path)
                st.write("Prediction is: ", prediction)

            os.remove(pred_temp_file_path)


if __name__ == "__main__":
    main()

