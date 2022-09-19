import argparse
from cmath import log
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_json, read_yaml, create_directories
import random
from azureml.core import Workspace, Dataset


STAGE = "Get and Save Data" ## <<< change stage name 

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
    azure = config['azure']
    artifacts = config['artifacts']
    workspace = Workspace(azure['subscription_id'],azure['resource_group'],azure['workspace_name']) 
    dataset = Dataset.get_by_name(workspace, name=params['dataset']['name']).to_pandas_dataframe()
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    raw_local_dir = artifacts['RAW_LOCAL_DIR']
    raw_local_file = artifacts['RAW_LOCAL_FILE']
    raw_local_dir_path = os.path.join(artifacts_dir,raw_local_dir)
    create_directories([raw_local_dir_path])
    raw_local_file_path = os.path.join(raw_local_dir_path,raw_local_file)
    dataset.to_csv(raw_local_file_path,sep=",",index=False)
    logging.info(f'raw data is saved at: {raw_local_file_path} successfully')


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