from src.exception import CustomException
from src.logger import logging
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import (
    LinearRegression,Ridge,Lasso
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.utils import evaluate_model,save_model
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    model_obj_file_path:str = os.path.join("articrafts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:

            logging.info("Splitting the data into test and train")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Ridge Regression" : Ridge(),
                "Lasso Regression" : Lasso(),
                "SVR" : SVR(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostRegressor" : GradientBoostingRegressor(),
                "XGBoost" : XGBRegressor(),
                "KNearestRegressor" : KNeighborsRegressor()
            }

            model_report = evaluate_model(x_train,x_test,y_train,y_test,models)
            logging.info("model evaluation completed")

            best_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]
            best_model = models[best_model_name]

            if best_score<0.6:
                raise CustomException("no best model found",sys)

            logging.info("model pickling started")

            save_model(
                self.model_trainer_config.model_obj_file_path,
                best_model
            )

            y_pred_test = best_model.predict(x_test)
            score = r2_score(y_test,y_pred_test)

            return score

        except Exception as e:
            raise CustomException(e,sys)


