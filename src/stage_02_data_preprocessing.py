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
from azureml.core import Workspace, Dataset, Run, Experiment
from src.utils.common import read_yaml, create_directories
#import azureml._restclient.snapshots_client


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
    raw_local_file = artifacts["RAW_LOCAL_FILE"]
    prep_file = artifacts["PREP_FILE"]
    imputer = params['imputer']
    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)
    raw_local_filepath = os.path.join(raw_local_dir_path, raw_local_file)
    prep_local_filepath = os.path.join(raw_local_dir_path, prep_file)
    # ws = Workspace.from_config()
    # experiment = Experiment(workspace=ws, name="backorder-product")
    # run = experiment.start_logging(snapshot_directory=None)
    # print("Starting experiment:", experiment.name)
    logging.info(f"fetched the data from : {raw_local_filepath}")
    df = pd.read_csv(raw_local_filepath)
    print(df.shape)
    row_count = (len(df))
    # run.log('raw_rows', row_count)
    # Drop last row
    df.drop(index = 1687860,inplace=True)
    logging.info("Dropped last row")
    # Converted columns to 0's and 1's
    Cols_for_str_to_bool = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
                        'stop_auto_buy', 'rev_stop', 'went_on_backorder']
    for col_name in Cols_for_str_to_bool:
        df[col_name] = df[col_name].map({'No':0, 'Yes':1})
        df[col_name] = df[col_name].astype(int)
    logging.info("Converted columns to 0's and 1's")
    # Removing the non imp features from the dataset
    df.drop(['sku','forecast_3_month','forecast_9_month','sales_9_month','stop_auto_buy','ppap_risk','deck_risk'], axis = 1,inplace=True)
    logging.info("Dropped non imp features from the dataset")
    # Replacing with nan values
    df['perf_6_month_avg'] = df['perf_6_month_avg'].replace(to_replace=-99.00,value=np.nan)
    df['perf_12_month_avg'] = df['perf_12_month_avg'].replace(to_replace=-99.00,value=np.nan)
    logging.info("Replacing perf_6_month_avg and perf_12_month_avg with nan values")
    # Split the data
    X = df.drop(['went_on_backorder'], axis =1)
    y = df['went_on_backorder']
    logging.info("Splitting the dataset into X and y")
    # filling the missing values
    lr  = LinearRegression()
    print('max_iter',type(imputer['max_iter']),'tol',type(imputer['tol']),'random',type(imputer['random_state']),'impute_order',type(imputer['imputation_order']),'verbose',imputer['verbose'])
    imp = IterativeImputer(estimator=lr,max_iter=imputer['max_iter'],tol=float(imputer['tol']), random_state=imputer['random_state'],imputation_order=imputer['imputation_order'],verbose=imputer['verbose'])
    imp.fit(X)
    X = imp.transform(X)
    logging.info("Filled missing values for lead_time, perf_6_month_avg and perf_12_month_avg columns")
    new_X_df =pd.DataFrame(X,columns=["national_inv","lead_time","in_transit_qty","forecast_6_month","sales_1_month","sales_3_month","sales_6_month","min_bank","potential_issue","pieces_past_due","perf_6_month_avg","perf_12_month_avg","local_bo_qty","oe_constraint","rev_stop"])
    logging.info('****After Filling missing values data *****')
    logging.info(new_X_df.head(3))
    # Applying cuberoot for normally distributed data
    skewed = ['national_inv','lead_time', 'in_transit_qty' , 'forecast_6_month', 'sales_1_month', 'sales_3_month', 'sales_6_month' , 'min_bank', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty']
    for i in skewed:
        new_X_df[i] = np.cbrt(new_X_df[i])
    logging.info("Applied cuberoot for normally distributed data")
    logging.info('****After Filling missing values data *****')
    logging.info(new_X_df.head(3))
    # Handling imbalance data
    smoteenn = SMOTEENN(n_jobs=params['base']['n_jobs'])
    print('Original dataset shape %s' % Counter(y))
    X_res, y_res = smoteenn.fit_resample(new_X_df, y)
    print('After undersample dataset shape %s' % Counter(y_res))
    label_counts = Counter(y_res)
    # run.log('label_count of 0',label_counts[0])
    # run.log('label_count of 1',label_counts[1])

    result = pd.concat([X_res,y_res],axis = 1)
    logging.info("Applied Smoteen to handle imbalanceness of data")
    logging.info(new_X_df.head(3))
    result.to_csv(prep_local_filepath,index=False)
    logging.info(f"Saved the prep data to: {prep_local_filepath}")
    # run.complete()






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