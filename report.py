import sys
from data_utils import get_data, get_month_data
import streamlit as st
import pandas as pd
import numpy as np


def get_fuel_consumption(datasheet: pd.DataFrame, machinery):
    '''
    The machinery attribute is to be given as PME, SME, AE1, AE2 or AE3
    '''

    fuemachinery = machinery
    if machinery == 'AE1': fuemachinery = 'AE_1'
    elif machinery == 'AE2': fuemachinery = 'AE_2'
    elif machinery == 'AE3': fuemachinery = 'AE_3'

    consum_data = pd.DataFrame()
    df = datasheet
    colstoconv = ['RUN HOURS AE1', 'RUN HOURS AE2', 'FUEL CONS AE_1', 'RUN HOURS AE3', 'RUN HOURS PME', 'RUN HOURS SME',
                  'FUEL CONS AE_2', 'FUEL CONS AE_3', 'FUEL CONS SME', 'FUEL CONS PME' ]
    for col in colstoconv:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    fuelconscol = "FUEL CONS " + fuemachinery
    runhourscol = "RUN HOURS " + machinery

    
    colname = 'Consumption Data ' + machinery
    df.fillna(0, inplace=True)
    if fuelconscol in df.columns and runhourscol in df.columns:
        denominator = df[runhourscol]
        consum_data[colname] = np.where(denominator != 0, df[fuelconscol] / denominator, 0)    
    consum_data.replace([np.inf, -np.inf], 0, inplace=True)
    consum_data[colname] = consum_data[colname].fillna(0)

    return consum_data
        

#    if fuelconscol in df.columns and runhourscol in df.columns:
#        denominator = df[runhourscol]
#        consum_data[colname] = np.where(denominator != 0, df[fuelconscol] / denominator, 0)

    

def get_runhrs(datasheet, machinery):
    '''
    The machinery attribute is to be given as PME, SME, AE1, AE2 or AE3
    '''

    consum_data = pd.DataFrame()
    df = datasheet
    runhourscol = "RUN HOURS " + machinery

    colname = 'Run Hours Data ' + machinery
    consum_data[colname] = df[runhourscol]
    consum_data[colname] = consum_data[colname].fillna(0)

    return consum_data

def insert_month_day(df):

    num_rows = len(df)

    first_column_pattern = [i // 4 + 1 for i in range(num_rows)]

    second_column_pattern = [i % 4 + 1 for i in range(num_rows)]

    df.insert(0, 'Day', first_column_pattern)
    df.insert(1, 'Quarter', second_column_pattern)

    return df


def check_outliers(df, column_name):
    df.reset_index(drop=True, inplace=True)
    column_data = df[column_name]
    
    # Exclude 0 values from column_data
    column_data_without_zero = column_data[column_data != 0]
    
    column_mean = column_data_without_zero.mean()
    lower_threshold = 0.9 * column_mean
    upper_threshold = 1.1 * column_mean

    outliers = column_data_without_zero[(column_data_without_zero < lower_threshold) | (column_data_without_zero > upper_threshold)]
    outlier_indices = column_data_without_zero[(column_data_without_zero < lower_threshold) | (column_data_without_zero > upper_threshold)].index

    if not outliers.empty:
        st.write("Warning: Outliers detected in column '{}':".format(column_name))
        st.write(f"Mean Value of {column_name} (excluding 0 values): {column_mean}")
        outlier_df_days = df.loc[outlier_indices, ['Day Number', 'WATCH']]
        outlier_df = pd.concat([outlier_df_days, outliers], axis=1)
        st.write(outlier_df)
        return 1
    return 0

def get_report(data_path):

    st.write("Fetching Data from Sheet...")
    datasheet = get_month_data(data_path)
    st.write("Data Loading complete!")
    
    fuelconsdata = pd.DataFrame()
    fuelcols = ['PME', 'SME', 'AE1', 'AE2', 'AE3']
    for col in fuelcols:
        consdata = get_fuel_consumption(datasheet, col)
        fuelconsdata = pd.concat([fuelconsdata, consdata], axis=1)
    fuelconsdata = insert_month_day(fuelconsdata)

    runhoursdata = pd.DataFrame()
    runhourscols = ['PME', 'SME', 'AE1', 'AE2', 'AE3']
    for col in runhourscols:
        rundata = get_runhrs(datasheet, col)
        runhoursdata = pd.concat([runhoursdata, rundata], axis= 1)
    runhoursdata = insert_month_day(runhoursdata)

    outlier_check_cols = ['PME_PRES_LO', 'PME_PRES_FO',	'PME_PRES_FW', 'SME_PRES_LO', 'SME_PRES_FO', 'SME_PRES_FW',	'AE_PRES_LO', 'AE_PRES_FW', 'PRESSURE LO AE3', 'PRESSURE FW AE3',
                           'PME_TEMP_LO', 'PME_TEMP_FW', 'PME_TEMP_EXH_MAX', 'PME_TEMP_EXH_MIN', 'PME_T/C_EXH', 'SME_TEMP_LO', 'SME_TEMP_FW', 'SME_TEMP_EXH_MAX', 'SME_TEMP_EXH_MIN', 'SME_T/C_EXH', 'AE_TEMP_FW AE3', 'AE_TEMP_LO AE3', 'T/C_EXH AE3',]

    count = 0
    for col in outlier_check_cols:
        temp = check_outliers(datasheet, col)
        count += temp

    if count == 0: st.write("No outliers detected")

    return fuelconsdata, runhoursdata
    
#if __name__ == "__main__":
#    print(get_report(r"E:\Adani Internship Jun-July 2023\Stellar_Prediction\Previous Logs\logbook2.xlsx"))
