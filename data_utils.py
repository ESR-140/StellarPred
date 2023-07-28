import pandas as pd
import os
import time
import sys
import streamlit as st

def get_data(sheet_number, path):


    data_pres_pme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[2, 3, 4], skiprows=range(4), nrows=4, header=None, engine= "openpyxl")
    data_pres_pme = data_pres_pme.set_axis(['PME_PRES_LO', 'PME_PRES_FO', 'PME_PRES_FW'], axis=1)

    data_temp_pme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[6, 7, 8, 9, 10], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_temp_pme = data_temp_pme.set_axis(['PME_TEMP_LO', 'PME_TEMP_FW', 'PME_TEMP_EXH_MAX', 'PME_TEMP_EXH_MIN', 'PME_T/C_EXH'], axis=1)

    data_rpm_pme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[11], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_rpm_pme = data_rpm_pme.set_axis(['PME_RPM'], axis=1)

    data_pres_sme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[13,14,15], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_pres_sme = data_pres_sme.set_axis(['SME_PRES_LO' , 'SME_PRES_FO' , 'SME_PRES_FW'], axis=1)

    data_temp_sme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[17, 18, 19, 20, 21], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_temp_sme = data_temp_sme.set_axis(['SME_TEMP_LO' , 'SME_TEMP_FW' , 'SME_TEMP_EXH_MAX' , 'SME_TEMP_EXH_MIN', 'SME_T/C_EXH'] , axis =1)

    data_rpm_sme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[22], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_rpm_sme = data_rpm_sme.set_axis(['SME_RPM'], axis=1)

    data_pres_ae = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[3,4], skiprows=12, nrows=4, header=None, engine= "openpyxl")   
    data_pres_ae = data_pres_ae.set_axis(['AE_PRES_LO','AE_PRES_FW'] , axis =1)

    data_temp_ae = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[5,6,7], skiprows=12, nrows=4, header=None, engine= "openpyxl")
    data_temp_ae = data_temp_ae.set_axis(['AE_TEMP_FW','AE_TEMP_LO','AE_T/C_EXH'] , axis =1)

    data_load_ae = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[8], skiprows=12, nrows=4, header=None, engine= "openpyxl")
    data_load_ae = data_load_ae.set_axis(['LOAD'] , axis =1)
    
    data_pres_ae3 = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[14,15], skiprows=21, nrows=4, header=None, engine= "openpyxl")
    data_pres_ae3 = data_pres_ae3.set_axis(['PRESSURE LO AE3','PRESSURE FW AE3'] , axis =1)
    
    data_temp_ae = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[16,17,18], skiprows=21, nrows=4, header=None, engine= "openpyxl")
    data_temp_ae = data_temp_ae.set_axis(['AE_TEMP_FW AE3','AE_TEMP_LO AE3','T/C_EXH AE3'] , axis =1)
    
    data_load_ae3 = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[19], skiprows=21, nrows=4, header=None, engine= "openpyxl")
    data_load_ae3 = data_load_ae3.set_axis(['LOAD AE3'] , axis =1)
    
    data_pres_ac = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[20,21,22], skiprows=21, nrows=4, header=None, engine= "openpyxl")
    data_pres_ac = data_pres_ac.set_axis(['AC_PRES_SUC','AC_PRES_LO','AC_PRES_DISC'] , axis =1)

    data_runhr_pme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[1], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_runhr_pme = data_runhr_pme.set_axis(['RUN HOURS PME'] , axis =1)

    data_runhr_sme = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[12], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_runhr_sme = data_runhr_sme.set_axis(['RUN HOURS SME'] , axis =1)

    data_runhr_ae = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[1,2], skiprows=12, nrows=4, header=None, engine= "openpyxl")
    data_runhr_ae = data_runhr_ae.set_axis(['RUN HOURS AE1', 'RUN HOURS AE2'] , axis =1)

    data_runhr_ae3 = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[13], skiprows=21, nrows=4, header=None, engine= "openpyxl")
    data_runhr_ae3 = data_runhr_ae3.set_axis(['RUN HOURS AE3'] , axis =1)

    data_fuel_cons = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[1,2,3,4,5], skiprows=47, nrows=4, header=None, engine= "openpyxl")
    data_fuel_cons = data_fuel_cons.set_axis(['FUEL CONS PME','FUEL CONS SME','FUEL CONS AE_1','FUEL CONS AE_2','FUEL CONS AE_3'] , axis =1)

    data_watch = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols=[0], skiprows=4, nrows=4, header=None, engine= "openpyxl")
    data_watch = data_watch.set_axis(['WATCH'], axis= 1)

    data_work = pd.read_excel(path, sheet_name='Sheet{}'.format(sheet_number), usecols='L', skiprows=54, header=None, engine= "openpyxl")
    data_work.dropna(inplace=True)
    data_work_2D = data_work.values.tolist()
    data_work_list = [item for sublist in data_work_2D for item in sublist]
    data_work_df = pd.DataFrame({'Work Done': [data_work_list] * 4})

    data = pd.concat([data_watch, 
                      data_pres_pme, data_pres_sme, data_pres_ae, data_pres_ae3, data_pres_ac,
                      data_temp_pme, data_temp_sme, data_temp_ae,
                      data_rpm_pme, data_rpm_sme,
                      data_load_ae, data_load_ae3,
                      data_runhr_pme, data_runhr_sme, data_runhr_ae, data_runhr_ae3,
                      data_fuel_cons,
                      data_work_df], axis= 1)
    
    data.insert(0, "Day Number", [sheet_number, sheet_number, sheet_number, sheet_number])
    data.fillna(0, inplace=True)

    return data


def get_month_data(path):

    month_data = pd.DataFrame()
    for i in range(29):
        day = i+1
        datatemp = get_data(day,path)
        month_data = pd.concat([month_data, datatemp])

    return month_data

def concat_dfs(folder_path):
    dfs =[]

    file_list = os.listdir(folder_path)
    counter = 1
    for file_name in file_list:
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            st.write("Loading Files, this may take some time")
            df = get_month_data(file_path)
            df.insert(0,"Month Number", [counter]*len(df))
            dfs.append(df)
            st.write(f"File {counter} loaded")
            counter += 1
            

    result_df = pd.concat(dfs, ignore_index=True)

    return result_df

def loading_animation(total_iterations, thread_stop_event):
    # Define the characters for the loading animation
    animation_chars = ['|', '/', '-', '\\']
    i = 0
    while not thread_stop_event.is_set():
        completed_iterations = min(i, total_iterations)
        progress = completed_iterations / total_iterations
        text = f"Loading ... {animation_chars[i % len(animation_chars)]}"
        st.text(text)
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

# Progress: [{'=' * int(50 * progress):<50}] {int(100 * progress)}%