from pathlib import Path
import sys
sys.path.append('/sise/home/efrco/efrco-master/Classifier/')
from Classifier.train_test_underSampling import split_train_test as split_train_test_underSampling
from Classifier.ClassifierWithGridSearch import build_classifiers as build_classifiers_grid_search
from Classifier.result_test import different_results_summary
import pandas as pd
from utils.utilsfile import read_csv, to_csv
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from generate_interactions.one_class_classification.one_class import build_classifiers_svm
from generate_interactions.one_class_classification.Isolation_Forest import build_classifiers_isolation_forest
from generate_interactions.one_class_classification.one_class_utils import split_train_test_one_class
import os
from generate_interactions.one_class_classification.Binary_models import build_classifiers_binary_comper
from generate_interactions.one_class_classification.result_test_one_class import  different_results_summary_one_class

class NoModelFound(Exception):
    pass


class CrossValidation(object):

    def __init__(self, dataset_file_positive, result_dir, number_iterations):
        self.number_iterations = number_iterations
        self.dataset_file_positive = dataset_file_positive
        self.result_dir = ROOT_PATH / Path("Results") / result_dir
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.measurement_dict = {
                                 'xgbs': self.create_measurement_dict()}


    def create_measurement_dict(self):
        columns_list = [f"exper_{i}" for i in range(self.number_iterations)]
        measurement_dict = {
            "AUC": pd.DataFrame(columns=columns_list, dtype=object),
            "ACC": pd.DataFrame(columns=columns_list, dtype=object),
            "TPR": pd.DataFrame(columns=columns_list, dtype=object),
            "TNR": pd.DataFrame(columns=columns_list, dtype=object),
            "PPV": pd.DataFrame(columns=columns_list, dtype=object),
            "NPV": pd.DataFrame(columns=columns_list, dtype=object),
            "FPR": pd.DataFrame(columns=columns_list, dtype=object),
            "FNR": pd.DataFrame(columns=columns_list, dtype=object),
            "FDR": pd.DataFrame(columns=columns_list, dtype=object),
            "F1": pd.DataFrame(columns=columns_list, dtype=object),
            "PNR": pd.DataFrame(columns=columns_list, dtype=object),
        }
        return measurement_dict

    def get_measurement_dict(self, name_dict):
        return {k: round(v, 3) for k, v in self.measurement_dict[name_dict].items()}

    def split_train_test_files(self):
        for i in range(self.number_iterations):
            # split_train_test_underSampling(dataset_positive_name=self.dataset_file_positive, random_state=i * 19, number_split=i)
            split_train_test_one_class(method_split_source='underSampling', random_state=i * 19,  number_split=i)

    def run_xgboost(self, number_iteration):
        build_classifiers_grid_search(number_iteration)
        different_results_summary(method_split="underSampling", model_dir="models_underSampling", number_iteration=number_iteration, name_classifier='xgbs')


    def run_one_class_svm(self, number_iteration):
        build_classifiers_svm(number_iteration)
        different_results_summary(method_split="one_class_svm", model_dir="models_one_class_svm",
                                   number_iteration=number_iteration, name_classifier='svm')
        different_results_summary(method_split="underSampling", model_dir="models_one_class_svm",
                                  number_iteration=number_iteration, name_classifier='svm')

    def run_isolation_forest(self, number_iteration):
        build_classifiers_isolation_forest(number_iteration)
        different_results_summary(method_split="one_class_svm", model_dir="models_isolation_forest",
                                   number_iteration=number_iteration, name_classifier='isolation_forest')
        different_results_summary(method_split="underSampling", model_dir="models_isolation_forest",
                                  number_iteration=number_iteration, name_classifier='isolation_forest')

    def run_binary_comper_SVM(self, number_iteration):
        build_classifiers_binary_comper(number_iteration=number_iteration, dir_method="models_binary_comper_SVM", name_model="SVM")
        different_results_summary_one_class(method_split="underSampling", model_dir="models_binary_comper_SVM",
                                      number_iteration=number_iteration, name_classifier='binary_comper_SVM',  method_name= "svm")

    def run_binary_comper_rf(self, number_iteration):
        build_classifiers_binary_comper(number_iteration=number_iteration, dir_method="models_binary_comper_rf", name_model="rf")
        different_results_summary_one_class(method_split="underSampling", model_dir="models_binary_comper_rf",
                                  number_iteration=number_iteration, name_classifier='binary_comper_rf', method_name="rf")

    def summary_results_do_dict(self):
        for classifier in self.measurement_dict.keys():
            for number_iteration in range(self.number_iterations):
                results_dir = ROOT_PATH / Path("Results")
                ms_table = read_csv(
                    results_dir / 'results_iterations' / classifier / f"measurement_summary_{number_iteration}.csv")
                for measurement in self.measurement_dict[classifier].keys():
                    try:
                        col = ms_table[measurement].apply(lambda t: round(t, 3))
                        self.measurement_dict[classifier][measurement][f"exper_{number_iteration}"] = col
                    except:
                        print("BUG")
    def write_results(self):
        self.summary_results_do_dict()
        # save file of result for each measuerment
        for classifier in self.measurement_dict.keys():
            for measurement in self.get_measurement_dict(classifier).keys():
                out_dir = self.result_dir/classifier/f"{measurement}_summary.csv"
                try:
                    to_csv(self.get_measurement_dict(classifier)[measurement], out_dir)
                except:
                    pass
        self.summary_result()
        self.summary_matrix('ACC')

    def run_experiment_binary_comper_svm(self):
        for i in range(self.number_iterations):
            self.run_binary_comper_SVM(number_iteration=i)

    def run_experiment_binary_comper_rf(self):
        for i in range(self.number_iterations):
            self.run_binary_comper_rf(number_iteration=i)

    def run_experiment_xgbs(self, start, to):
        for i in range(start, to):
            self.run_xgboost(number_iteration=i)

    def run_experiment_one_class_svm(self):
        for i in range(self.number_iterations):
           self.run_one_class_svm(number_iteration=i)

    def run_experiment_isolation_forest(self):
        for i in range(self.number_iterations):
            self.run_isolation_forest(number_iteration=i)

    
    
    def summary_result(self):

        all_result_mean = pd.DataFrame()
        all_result_std = pd.DataFrame()

        for classifier in self.measurement_dict.keys():
            all_result_mean = pd.DataFrame()
            all_result_std = pd.DataFrame()
            dir_measurement = self.result_dir / classifier
            for measuerment_file in dir_measurement.glob("*summary*"):
                df = read_csv(measuerment_file)
                count = 0
                measuerment_name = measuerment_file.stem.split("_summary")[0]

                for index, row in df.iterrows():
                    row = df.iloc[count]
                    count = count + 1
                    col_mean = row.mean()
                    col_std = row.std()
                    all_result_mean.loc[measuerment_name, index] = round(col_mean, 3)
                    all_result_std.loc[measuerment_name, index] = round(col_std, 3)

            out_dir_mean = self.result_dir/classifier / f"final_mean.csv"
            to_csv(all_result_mean, out_dir_mean)

            out_dir_std = self.result_dir/classifier / f"final_std.csv"
            to_csv(all_result_std, out_dir_std)

    def run_pipline(self):

        self.split_train_test_files()

        self.run_experiment_xgbs(start=0, to=20) #for shap and feature importance #6649424
     
        %###########################One class classification########################################

        self.run_experiment_one_class_svm()
        self.run_experiment_isolation_forest()

        self.run_experiment_binary_comper_svm()
        self.run_experiment_binary_comper_rf()

        %################################Write Results###########################################
        self.write_results() 
        




############################### Runnning ############################################
cv = CrossValidation("darnell_human_ViennaDuplex_features", "measurments_cross_validation", number_iterations=20)
cv.run_pipline()
