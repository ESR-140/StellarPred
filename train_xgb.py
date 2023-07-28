import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import streamlit as st
from data_utils import concat_dfs

def train_xgb(folder_path):
    data = concat_dfs(folder_path)
    data['FUEL CONS AE_1'] = pd.to_numeric(data['FUEL CONS AE_1'], errors='coerce')
    data.fillna(0, inplace=True)
    data['FUEL CONS AE_1'] = data['FUEL CONS AE_1'].astype(int)


    # Step 3: Train-Test Split
    train_size = int(len(data) * 0.8)
    train_data = data
    test_data = data[train_size:]

    # Step 4: Prepare the data for XGBoost
    xvals = ['PME_PRES_LO', 'PME_PRES_FO',
           'PME_PRES_FW', 'SME_PRES_LO', 'SME_PRES_FO', 'SME_PRES_FW',
           'AE_PRES_LO', 'AE_PRES_FW', 'PRESSURE LO AE3', 'PRESSURE FW AE3',
           'AC_PRES_SUC', 'AC_PRES_LO', 'AC_PRES_DISC', 'PME_TEMP_LO',
           'PME_TEMP_FW', 'PME_TEMP_EXH_MAX', 'PME_TEMP_EXH_MIN', 'PME_T/C_EXH',
           'SME_TEMP_LO', 'SME_TEMP_FW', 'SME_TEMP_EXH_MAX', 'SME_TEMP_EXH_MIN',
           'SME_T/C_EXH', 'AE_TEMP_FW AE3', 'AE_TEMP_LO AE3', 'T/C_EXH AE3',
           'PME_RPM', 'SME_RPM', 'LOAD', 'LOAD AE3']
    yvals = ['FUEL CONS PME', 'FUEL CONS SME', 'FUEL CONS AE_1', 'FUEL CONS AE_2',
           'FUEL CONS AE_3']
    X_train, y_train = train_data[xvals], train_data[yvals]
    X_test, y_test = test_data[xvals], test_data[yvals]
    
    # Step 5: Hyperparameter Tuning with GridSearchCV
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Step 6: Get the best hyperparameters and train the final XGBoost model
    best_params = grid_search.best_params_
    final_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
    final_xgb_model.fit(X_train, y_train)

    model_filename = 'xgb_model.pkl'

    joblib.dump(final_xgb_model, model_filename)
    loaded_model = joblib.load(model_filename)
    st.write("Model Saved")

    print(f"Y TEST: \n{y_test}")
    #print(f"Y PRED: \n{final_xgb_model.predict(y_test)}")
    print(f"Y TEST TYPE: \n{type(y_test)}") 
    #print(f"Y 1 VALUE: \n {y_test[0]}")
 
   

