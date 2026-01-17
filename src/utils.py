import os
import dill
import sys
from src.exception import CustomException
from src.logger import logging

def save_object(preprocessor_path,preprocessor_obj):
    try:
        os.makedirs(os.path.dirname(preprocessor_path),exist_ok = True)

        logging.info("dumping the preprocessor object")

        with open(preprocessor_path,"wb") as file_obj:
            dill.dump(preprocessor_obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

