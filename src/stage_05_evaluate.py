import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, save_json
import random
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from azureml.core import Workspace, Dataset, Run, Experiment, Model

STAGE = "Evaluate" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def evaluate_metrics(actual_values, predicted_values):
    recall = recall_score(actual_values, predicted_values)
    precision = precision_score(actual_values, predicted_values)
    f1 = f1_score(actual_values, predicted_values)
    return recall, precision, f1

def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name="backorder-product")
    run = experiment.start_logging(snapshot_directory=None)
    print("Starting experiment:", experiment.name)
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    x_test_data_path = os.path.join(split_data_dir_path, artifacts["X_TEST"])
    y_test_data_path = os.path.join(split_data_dir_path, artifacts["Y_TEST"])
   
    # load X_test, Y_test
    X_test = pd.read_csv(x_test_data_path)
    y_test = pd.read_csv(y_test_data_path)['went_on_backorder']
    logging.info("loaded X_test, y_test data for model training")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    model_dir = artifacts["TRAINED_MODEL_DIR"]
    model_name = artifacts["TRAINED_MODEL_NAME"]
    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    model_file_path = os.path.join(model_dir_path, model_name)

    clf = joblib.load(model_file_path)
    logging.info("Model loaded")
    
    # predict values
    y_pred=clf.predict(X_test)
    logging.info("loaded prediction values")

    # Model evaluate
    recall, precision, f1 = evaluate_metrics(
        actual_values=y_test,
        predicted_values=y_pred
    )
    
    
    scores = {
        "recall": recall, 
        "precision": precision, 
        "f1_score": f1
    }

    run.log_predictions(name='Important metrics',value=scores)
    scores_file_path = config["scores"]
    save_json(scores_file_path, scores)
    logging.info("Saved the scores in json format")
    Model.register(workspace=run.experiment.workspace,
               model_path = model_file_path,
               model_name = 'backorder_product_prediction_model',
               tags={'Training context':'Inline'},
               properties=scores)
    run.complete()
    logging.info("Run Completed")


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