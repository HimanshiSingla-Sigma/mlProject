import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("articrafts","raw_data.csv")
    train_data_path: str = os.path.join("articrafts","train_data.csv")
    test_data_path: str = os.path.join("articrafts","test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # this will return a list of paths
        # raw_data_path , train_data_path and test_data_path
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("reading the data as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train and Test split Initiated")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            train_set , test_set = train_test_split(df,test_size=0.2,random_state=35)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return( # so that we can use them in data_transformation part
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_path , test_path = obj.initiate_data_ingestion()
    tranform = DataTransformation()
    train_arr, test_arr, _ = tranform.initiate_data_tranformation(train_path,test_path)
    model_trainer = ModelTrainer()
    best_score = model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(best_score)
