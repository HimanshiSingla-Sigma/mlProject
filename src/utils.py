import os
import dill
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(preprocessor_path,preprocessor_obj):
    try:
        os.makedirs(os.path.dirname(preprocessor_path),exist_ok = True)

        logging.info("dumping the preprocessor object")

        with open(preprocessor_path,"wb") as file_obj:
            dill.dump(preprocessor_obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,x_test,y_train,y_test,models):

    try:

        logging.info("model evaluation started")

        model_report = {}

        for i in range(len(models)):

            model = list(models.values())[i]

            model.fit(x_train,y_train)

            y_pred_test = model.predict(x_test)
            y_pred_train = model.predict(x_train)

            score = r2_score(y_test,y_pred_test)
            model_report[list(models.keys())[i]] = score

        return model_report

    except Exception as e:
        raise CustomException(e,sys)

def save_model(model_obj_file_path,model):
    
    try:
        
        file_path = os.path.dirname(model_obj_file_path)
        os.makedirs(file_path,exist_ok=True)

        with open(model_obj_file_path,"wb") as model_obj:
            dill.dump(model,model_obj)

        logging.info("model saved")
    
    except Exception as e:
        raise CustomException(e,sys)

