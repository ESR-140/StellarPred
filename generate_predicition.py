import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from data_utils import get_data
import os

def generate_sentence(input_features, model_path):
    # Load the trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Convert numerical input features to string, replace NaN with 'NaN'
    numerical_text = " ".join([str(param) if param != -1 else "NaN" for param in input_features])

    # Tokenize the numerical text and get input IDs
    #input_ids = tokenizer.encode(numerical_text, return_tensors="pt")
    input_ids = tokenizer.encode(numerical_text, add_special_tokens=False)

    # Generate the output sentence using the loaded model
    '''model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, max_length = 100)
    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        output = model.generate(input_ids, max_length=100)
    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)


    return generated_sentence

if __name__ == "__main__":
    
    '''
    script_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_folder, "trained_model")
    folder_path

    df = get_data(1,folder_path)   Sheet name should be Sheet1 for prediction to work
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
    
    prediction = generate_sentence(input_features, model_path)
    print(prediction)
    '''
    df = get_data(15,"E:\Adani Internship Jun-July 2023\Stellar_Prediction\Previous Logs\logbook.xlsx")
    target_sentence = df.loc[1, ['Work Done']]
    print(target_sentence.tolist())