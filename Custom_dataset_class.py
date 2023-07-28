import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

'''class CustomDataset(Dataset):
    def __init__(self, dataframe, numerical_feature_cols, target_sentence_col, tokenizer, max_length=150, nan_value=-1):
        self.dataframe = dataframe
        self.numerical_feature_cols = numerical_feature_cols
        self.target_sentence_col = target_sentence_col
        self.tokenizer = tokenizer
        self.max_length = max_length
        #self.max_new_tokens = max_new_tokens
        self.nan_value = nan_value

        # Add padding token to the tokenizer
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        numerical_features = self.dataframe.loc[idx, self.numerical_feature_cols].values
        target_sentences = self.dataframe.loc[idx, self.target_sentence_col]

        # Convert numerical features to string, replace NaN with 'NaN'
        numerical_text = " ".join([str(param) if param != self.nan_value else "NaN" for param in numerical_features])
        
        # Tokenize the numerical text and get input IDs
        input_ids = self.tokenizer.encode(numerical_text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Create the input mask to handle NaN values
        input_mask = torch.tensor([1 if param != self.nan_value else 0 for param in numerical_features], dtype=torch.long)

        # Tokenize the target sentences and get target IDs
        target_ids_list = [self.tokenizer.encode(sentence, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True) for sentence in target_sentences]

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "input_mask": input_mask,
            "target_ids": torch.cat(target_ids_list, dim=0),  # Concatenate target IDs for all sentences
            "target_attention_mask": torch.cat([ids.ne(self.tokenizer.pad_token_id) for ids in target_ids_list], dim=0),
        }
'''

'''def pad_sequences(sequences, max_length, padding_value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [padding_value] * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return padded_sequences'''

class CustomDataset(Dataset):
    def __init__(self, data, numerical_feature_cols, target_sentence_cols, tokenizer, max_length=100):
        self.data = data
        self.numerical_feature_cols = numerical_feature_cols
        self.target_sentence_cols = target_sentence_cols
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        numerical_features = self.data.loc[index, self.numerical_feature_cols].values
        target_sentences = self.data.loc[index, self.target_sentence_cols].values
        target_sentences = target_sentences[0]

        # Replace NaN values with 0 in numerical features
        numerical_features = np.nan_to_num(numerical_features, nan=0)

        # Convert numerical features to a space-separated string
        numerical_text = " ".join([str(param) for param in numerical_features])

        # Tokenize the numerical text and get input IDs
        input_ids = self.tokenizer.encode(numerical_text, add_special_tokens=True, truncation=True, max_length=self.max_length)

        # Create attention mask for input IDs
        attention_mask = [1] * len(input_ids)

        # Pad input IDs and attention mask to max length
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Create a list to store target IDs
        target_ids_list = []

        # Tokenize target sentences and get target IDs
        for sent in target_sentences:
            target_ids = self.tokenizer.encode(str(sent), add_special_tokens=True, truncation=True, max_length=self.max_length)
            target_ids_list.append(target_ids)

        # Pad target IDs to max length
        for i in range(len(target_ids_list)):
            target_ids = target_ids_list[i]
            if len(target_ids) < self.max_length:
                padding_length = self.max_length - len(target_ids)
                target_ids += [self.tokenizer.pad_token_id] * padding_length
            else:
                target_ids_list[i] = target_ids[:self.max_length]

        for i in range(len(input_ids)):
            if input_ids[i] is None:
                input_ids[i] = 0

        for i in range(len(target_ids_list)):
            for j in range(len(target_ids_list[i])):
                if target_ids_list[i][j] is None:
                    target_ids_list[i][j] = 0

        target_ids_list = np.array(target_ids_list).flatten().tolist()

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        target_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in target_ids_list]
        max_target_length = max(len(ids) for ids in target_ids_list)
        target_ids_list = [torch.nn.functional.pad(ids, (0, max_target_length - len(ids)), value=self.tokenizer.pad_token_id) for ids in target_ids_list]
        target_ids_tensor = pad_sequence(target_ids_list, batch_first=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids_tensor,
        }

    def pad_sequence(self, sequence, max_length, padding_value):
        if len(sequence) < max_length:
            padded_sequence = sequence + [padding_value] * (max_length - len(sequence))
        else:
            padded_sequence = sequence[:max_length]
        return padded_sequence


'''if __name__=='__main__':
    masterdata = pd.read_excel('masterdata.xlsx')
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
    masterdata.fillna(0, inplace=True)
    for index in range(len(masterdata)):
        numerical_features = masterdata.loc[index, numerical_feature_cols].values
        target_sentences = masterdata.loc[index, target_sentence_cols].values
        #print(f"Numerical Features: {numerical_features}")
        #print(f"Target Sentences: {target_sentences[0]}")
        #print(" ")
        numerical_features = np.nan_to_num(numerical_features, nan=0)
        numerical_text = " ".join([str(param) for param in numerical_features])
        input_ids = GPT2Tokenizer.from_pretrained("gpt2").encode(numerical_text, add_special_tokens=True, truncation=True, max_length=150)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < 150:
            padding_length = 150 - len(input_ids)
            input_ids += [GPT2Tokenizer.from_pretrained("gpt2").pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        else:
            input_ids = input_ids[:150]
            attention_mask = attention_mask[:150]

        # Create a list to store target IDs
        target_ids_list = []

        # Tokenize target sentences and get target IDs
        for sent in target_sentences:
            target_ids = GPT2Tokenizer.from_pretrained("gpt2").encode(str(sent), add_special_tokens=True, truncation=True, max_length=150)
            target_ids_list.append(target_ids)

        # Pad target IDs to max length
        for i in range(len(target_ids_list)):
            target_ids = target_ids_list[i]
            if len(target_ids) < 150:
                padding_length = 150 - len(target_ids)
                target_ids += [GPT2Tokenizer.from_pretrained("gpt2").pad_token_id] * padding_length
            else:
                target_ids_list[i] = target_ids[:150]

        # Convert lists to tensors
        #input_ids = torch.tensor(input_ids, dtype=torch.long)
        #attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        #target_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in target_ids_list] 
        # 
        for i in range(len(input_ids)):
            if input_ids[i] is None:
                input_ids[i] = 0


        for i in range(len(target_ids_list)):
            for j in range(len(target_ids_list[i])):
                if target_ids_list[i][j] is None:
                    target_ids_list[i][j] = 0

        target_ids_list = np.array(target_ids_list).flatten().tolist()

    

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        target_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in target_ids_list]
        target_ids_tensor = torch.stack(target_ids_list)

        print("Target ID Length: ", target_ids_tensor.shape)
        print("Input ID Length: ", input_ids.shape)
        print("Attention Maask Length: ", attention_mask.shape) 

        #print(f"input_ids: {input_ids}")
        #for id in input_ids: print(f"ID: {id}")
        #print(f"attention_mask: {attention_mask}")
        #print(f"target_ids: {target_ids_list}")'''


