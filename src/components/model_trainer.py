import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import r2_score
from src.utils import save_object,evaluvate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):

        try:

            logging.info("Start the splitting training and test the data...")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Xg Boost": XGBRegressor(),
                "Cat Boost": CatBoostRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "KNN" : KNeighborsRegressor()
            }


            model_report: dict = evaluvate_models(X_train,y_train,X_test,y_test,models)

            # To get the best score from the dict

            best_model_score = max(sorted(model_report.values()))

            # To get the best model from the dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



        except Exception as e:
            raise CustomException(e,sys)        