from src.logger import logging 
from src.exception import CustomException
import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("articrafts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_preprocessor_obj(self):
        try:
            numerical = ["writing_score","reading_score"]
            categorical = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
    
            num_pipeline = Pipeline(
                steps=[
                    ("mean imputation",SimpleImputer(strategy = "mean")),
                    ("scaling",StandardScaler())
                ]
            )
            logging.info("numerical pipeline created")

            cat_pipeline = Pipeline(
                steps=[
                    ("mode imputation",SimpleImputer(strategy="most_frequent")),
                    ("one hot",OneHotEncoder()),
                    ("scaling",StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical pipeline created")

            preprocessor = ColumnTransformer(
                [
                    ("numerical pipeline",num_pipeline,numerical),
                    ("categorical pipeline",cat_pipeline,categorical)
                ]
            )

            save_object(
                self.transformation_config.preprocessor_obj_file_path,
                preprocessor
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            logging.info("data tranformation initiated : ")

            logging.info("reading train dataset")
            train_set = pd.read_csv(train_path)

            logging.info("reading test dataset")
            test_set = pd.read_csv(test_path)

            input_feature_train_df = train_set.drop(['math_score'],axis=1)
            target_feature_train_df = train_set['math_score']

            input_feature_test_df = test_set.drop(['math_score'],axis=1)
            target_feature_test_df = test_set['math_score']

            preprocessor_obj = self.get_preprocessor_obj()

            logging.info("appliying preprocessor object on training and testing dataset")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

        

