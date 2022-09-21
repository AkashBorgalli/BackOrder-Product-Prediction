import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

STAGE = "Model Training" ## <<< change stage name 

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
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    x_train_data_path = os.path.join(split_data_dir_path, artifacts["X_TRAIN"])
    x_test_data_path = os.path.join(split_data_dir_path, artifacts["X_TEST"])
    y_train_data_path = os.path.join(split_data_dir_path, artifacts["Y_TRAIN"])
    y_test_data_path = os.path.join(split_data_dir_path, artifacts["Y_TEST"])


    boosting_type=params['model_params']['Lightgbm']['boosting_type']
    objective=params['model_params']['Lightgbm']['objective']
    n_estimators=params['model_params']['Lightgbm']['n_estimators']
    n_jobs = params['base']['n_jobs']
    
    X_train = pd.read_csv(x_train_data_path)
    X_test = pd.read_csv(x_test_data_path)
    y_train = pd.read_csv(y_train_data_path)['went_on_backorder']
    y_test = pd.read_csv(y_test_data_path)['went_on_backorder']


    model_dir = artifacts["MODEL_DIR"]
    model_name = artifacts["MODEL_NAME"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])
    model_file_path = os.path.join(model_dir_path, model_name)
    
    ## Robust Scaling
    rs = RobustScaler()
    X_train = rs.fit_transform(X_train)
    X_test = rs.transform(X_test)

    ## Model Training
    clf = lgb.LGBMClassifier(boosting_type=boosting_type,objective=objective,n_jobs=n_jobs,n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    ## Saving the model
    joblib.dump(clf, model_file_path)
    logging.info(f"model is trained and saved at: {model_file_path}")

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