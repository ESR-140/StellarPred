import os
from data_utils import concat_dfs
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
from Custom_dataset_class import CustomDataset
import streamlit as st

def train_model(folder_path, num_epochs=20, batch_size=32, learning_rate=0.001, max_length= 150):
    # Load and tokenize the pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"  # You can choose a different GPT-2 variant if needed
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    # Load the training data
    masterdata = concat_dfs(folder_path)
    masterdata.fillna(0, inplace=True)
    numerical_feature_cols = ['PME_PRES_LO', 'PME_PRES_FO',
        'PME_PRES_FW', 'SME_PRES_LO', 'SME_PRES_FO', 'SME_PRES_FW',
        'AE_PRES_LO', 'AE_PRES_FW', 
        'PRESSURE LO AE3', 'PRESSURE FW AE3',
        'AC_PRES_SUC', 'AC_PRES_LO', 'AC_PRES_DISC', 
        'PME_TEMP_LO','PME_TEMP_FW', 'PME_TEMP_EXH_MAX', 'PME_TEMP_EXH_MIN', 'PME_T/C_EXH',
        'SME_TEMP_LO', 'SME_TEMP_FW', 'SME_TEMP_EXH_MAX', 'SME_TEMP_EXH_MIN','SME_T/C_EXH',
        'AE_TEMP_FW AE3', 'AE_TEMP_LO AE3', 'T/C_EXH AE3', 
        'PME_RPM', 'SME_RPM', 'LOAD', 'LOAD AE3', 'RUN HOURS PME', 'RUN HOURS SME',
        'RUN HOURS AE1', 'RUN HOURS AE2', 'RUN HOURS AE3', 'FUEL CONS PME',
        'FUEL CONS SME', 'FUEL CONS AE_1', 'FUEL CONS AE_2', 'FUEL CONS AE_3']  # Replace with your numerical feature columns
    target_sentence_cols = ['Work Done']  # Replace with your target sentence columns

    # Preparing the dataset
    #dataset = CustomDataset(masterdata, numerical_feature_cols, target_sentence_cols, tokenizer)
    dataset = CustomDataset(masterdata, numerical_feature_cols, target_sentence_cols, tokenizer, max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Fine-tuning the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    st.write("Start Training.........")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        st.write(f"Epoch {epoch + 1} is running")

        
        count = 0
        st.write("")

        for batch in train_loader:
            st.write("\r" + f"Batch {count + 1} of Epoch {epoch + 1} is running!")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            outputs = model(input_ids, labels=target_ids, attention_mask=attention_mask)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            count += 1

        
        st.write(f"\nEpoch {epoch + 1} complete!")
        average_loss = total_loss / len(train_loader)
        st.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")


    st.write("Training Completed!")
    script_folder = os.path.dirname(os.path.abspath(__file__))
    output_model_path = os.path.join(script_folder, f"trained_model")

    # Check if the file already exists
    if os.path.exists(output_model_path):
        # Delete the existing model and tokenizer files
        st.write("Deleting existing models and replacing it with retrained models...")
        os.remove(os.path.join(output_model_path, "pytorch_model.bin"))
        os.remove(os.path.join(output_model_path, "config.json"))
        os.remove(os.path.join(output_model_path, "tokenizer_config.json"))
        os.remove(os.path.join(output_model_path, "special_tokens_map.json"))
        os.remove(os.path.join(output_model_path, "vocab.txt"))
        st.write("Existing model files deleted successfully!")

    # Save the new model and tokenizer
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    st.write("Retrained Model Saved Successfully")
    st.write("Training Completed!")

# Example usage:
if __name__ == "__main__":
    folder_path = "E:\Adani Internship Jun-July 2023\Stellar_Prediction\Previous Logs"
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    train_model(folder_path, num_epochs, batch_size, learning_rate)
