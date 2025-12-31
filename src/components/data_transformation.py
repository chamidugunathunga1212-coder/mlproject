import pandas as pd
import numpy as np

import sys
from dataclasses import dataclass
import os

from src.exception import CustomException
from src.logger import logging

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer


from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.dataTransformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function responsible for data transformation

        '''           

        try:

            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor



        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the train and test csv files are completed...")

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Obtaining the preprocessing object...")

            target_column = 'math_score'

            input_features_train_df = train_df.drop(target_column,axis=1)
            target_feature_train_df = train_df[target_column]

            logging.info(
                f"Applyed preprocessing object on training dataframe and testing dataframe."
            )

            input_features_test_df = test_df.drop(target_column,axis=1)
            target_features_test_df = test_df[target_column]

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)


            train_arr = np.c_[input_features_train_arr,target_feature_train_df] 
            test_arr = np.c_[input_features_test_arr,target_features_test_df]

            save_object(

                file_path=self.dataTransformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info('Saved the object in the pickle file...')

            return(
                train_arr,
                test_arr,
                self.dataTransformation_config.preprocessor_obj_file_path
            )




        except Exception as e:
            raise CustomException(e,sys)
         