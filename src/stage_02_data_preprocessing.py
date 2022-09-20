import argparse
import os
import shutil
from tqdm import tqdm
import logging
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer
from azureml.core import Workspace, Dataset, Run
from src.utils.common import read_yaml, create_directories


STAGE = "Data Preprocessing" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    raw_local_dir = artifacts["RAW_LOCAL_DIR"]
    prep_file = artifacts["PREP_FILE"]
    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)
    raw_local_filepath = os.path.join(raw_local_dir_path, prep_file)

    df = pd.read_csv(raw_local_filepath)
    # Drop last row
    df.drop(df.tail(1).index,inplace=True)
    # Converted columns to 0's and 1's
    Cols_for_str_to_bool = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
                        'stop_auto_buy', 'rev_stop', 'went_on_backorder']
    for col_name in Cols_for_str_to_bool:
        df[col_name] = df[col_name].map({'No':0, 'Yes':1})
        df[col_name] = df[col_name].astype(int)
    # Removing the non imp features from the dataset
    df.drop(['sku','forecast_3_month','forecast_9_month','sales_9_month','stop_auto_buy','ppap_risk','deck_risk'], axis = 1,inplace=True)
    # Replacing nwith nan values
    df['perf_6_month_avg'] = df['perf_6_month_avg'].replace(to_replace=-99.00,value=np.nan)
    df['perf_12_month_avg'] = df['perf_12_month_avg'].replace(to_replace=-99.00,value=np.nan)
    # Split the data
    X = df.drop(['went_on_backorder'], axis =1)
    y = df['went_on_backorder']
    # filling the missing values
    lr  = LinearRegression()
    imp = IterativeImputer(estimator=lr,max_iter=10,tol=1e-10, random_state=0,imputation_order='roman',verbose=2)
    imp.fit(X)
    X = imp.transform(X)
    new_X_df =pd.DataFrame(X,columns=["national_inv","lead_time","in_transit_qty","forecast_6_month","sales_1_month","sales_3_month","sales_6_month","min_bank","potential_issue","pieces_past_due","perf_6_month_avg","perf_12_month_avg","local_bo_qty","oe_constraint","rev_stop"])
    # Applying cuberoot for normally distributed data
    skewed = ['national_inv','lead_time', 'in_transit_qty' , 'forecast_6_month', 'sales_1_month', 'sales_3_month', 'sales_6_month' , 'min_bank', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty']
    for i in skewed:
        new_X_df[i] = np.cbrt(new_X_df[i])
    # Handling imbalance data
    SMOTEENN = SMOTEENN(n_jobs=-1)
    print('Original dataset shape %s' % Counter(y))
    X_res, y_res = SMOTEENN.fit_resample(new_X_df, y)
    print('After undersample dataset shape %s' % Counter(y_res))
    result = pd.concat([X_res,y_res],axis = 1)
    result.to_csv(raw_local_filepath,index=False)






if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e