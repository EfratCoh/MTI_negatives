import ast
import shap
import pickle
import os
from collections import Counter
from itertools import combinations
from pathlib import Path
# from utilsClassifier import mean_std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from seaborn import heatmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from utilsfile import read_csv, to_csv
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from Classifier.ClfLogger import logger
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot


class NoModelFound(Exception):
    pass

def model_confuse_matrix(x_test, y_test, model, model_name,s_org,d_org,name_classifiers):

    plot_confusion_matrix(model, x_test, y_test)
    plt.title(f"{s_org}_{d_org}_confuse_matrix_plot")
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/confuse_matrix/") / f"{s_org}_{d_org}.png"
    plt.savefig(fname, bbox_inches='tight',  dpi=300)
    plt.show()
    plt.clf()



def measurement(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).ravel()
    TP = cm[3]
    FP = cm[1]
    FN = cm[2]
    TN = cm[0]
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0

    d = {
        # Sensitivity, hit rate, recall, or true positive rate
        "TPR": TP / (TP + FN),
        # Specificity or true negative rate
        "TNR": TN / (TN + FP),
        # Precision or positive predictive value
        "PPV": TP / (TP + FP),
        # Negative predictive value
        "NPV": TN / (TN + FN),
        # Fall out or false positive rate
        "FPR": FP / (FP + TN),
        # False negative rate
        "FNR": FN / (TP + FN),
        # False discovery rate
        "FDR": FP / (TP + FP),
        # Overall accuracy
        "ACC": (TP + TN) / (TP + FP + FN + TN),
        # roc auc
        "AUC": auc,
        "F1": f1_score(y_true, y_pred),
        "PNR": 0
    }

    return {k: round(v,3) for k, v in d.items()}


# this function response on load the clf from Result dir
def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}.model"
    if not clf_file.is_file():
        raise NoModelFound(f"No model found: {clf_file}")
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf



def different_results_summary_one_class(method_split: str, model_dir: str, number_iteration: int, name_classifier: str,method_name):

    ms_table = None
    results_dir = ROOT_PATH / Path("Results")
    number_iteration = str(number_iteration)
    results_dir_models = ROOT_PATH / Path("Results/models") / model_dir / number_iteration
    test_dir = DATA_PATH_INTERACTIONS / "test" / method_split / number_iteration
    res_table: DataFrame = pd.DataFrame()
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()

    clf_datasets = [f.stem.split("_"+ name_classifier)[0] for f in results_dir_models.glob("*.model")]
    method = name_classifier
    for clf_dataset in clf_datasets:
        for f_test in test_dir.glob("*test*"):
            f_stem = f_test.stem
            test_dataset = f_stem.split(".csv")[0]
            train_model = clean_name(clf_dataset).replace(method_name,"").replace("train","")[:-1]
            test_set = clean_name(test_dataset).replace(method_name,"").replace("test","")
            if train_model != test_set:
                print(train_model)
                print(test_set)
                continue

            print(f"clf: {clf_dataset} test: {test_dataset}, method: {method}")
            try:
                    print("##############################I am try#################")
                    clf = get_presaved_clf(results_dir_models, clf_dataset, method)
                    X_test, y_test = feature_reader.file_reader(test_dir/f"{test_dataset}.csv")

                    # score predict
                    test_score = accuracy_score(y_test, clf.predict(X_test))
                    res_table.loc[clf_dataset, test_dataset] = round(test_score, 3)
                    prediction = clf.predict(X_test)

                    ms = measurement(y_test, prediction)
                   
                    y_scores = clf.predict_proba(X_test)[:,0]

                    precision, recall, thresholds = precision_recall_curve(y_true=y_test,
                                                                           probas_pred=y_scores, pos_label=0)
                    prn_auc=auc(recall,precision)
                    print("PNR in the first way:", prn_auc)
                    ms['PNR'] = round(prn_auc, 3)


                    if ms_table is None:
                        ms_table = pd.DataFrame(columns=list(ms.keys()), dtype=object)
                    name_method = "model:" + clf_dataset.split("negative")[0] + "/" + "test:" + test_dataset.split("negative")[0]
                    ms_table.loc[name_method] = ms

            except NoModelFound:
                    pass

    res_table.sort_index(axis=0, inplace=True)
    res_table.sort_index(axis=1, inplace=True)

    to_csv(ms_table, results_dir /"results_iterations" / name_classifier /f"measurement_summary_{number_iteration}.csv")
    print("END result test")
    return ms_table


