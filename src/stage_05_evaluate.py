import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



STAGE = "Evaluate " ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    x_test_data_path = os.path.join(split_data_dir_path, artifacts["X_TEST"])
    y_test_data_path = os.path.join(split_data_dir_path, artifacts["Y_TEST"])
    x_test  = pd.read_csv(x_test_data_path)
    y_test = pd.read_csv(y_test_data_path)

    mdl = dill.load(open('artifacts/model_dir/model.pkl','rb'))
    y_p = mdl.predict(x_test)
    print(classification_report(y_true=y_test,y_pred=y_p))

    

    





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e