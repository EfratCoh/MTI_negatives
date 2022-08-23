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
# import FeatureReader
# from FeatureReader import get_reader
# from dataset import Dataset
# from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
# from dataset import Dataset
# from utils.utilsfile import read_csv, to_csv
from utilsfile import read_csv, to_csv
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from Classifier.ClfLogger import logger
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS

class NoModelFound(Exception):
    pass


def measurement(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    d = {
        # Sensitivity, hit rate, recall, or true positive rate
        "TPR" : TP / (TP + FN),
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
        "AUC": roc_auc_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }
    return {k: round(v,3) for k, v in d.items()}


# this function response on load the clf from Result dir
def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}_{method}.model"
    if not clf_file.is_file():
        raise NoModelFound(f"No model found: {clf_file}")
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf


def xgbs_feature_importance(clf: XGBClassifier, X_train: DataFrame):
    def series2bin(d: pd.DataFrame, bins: list):
        for l in bins:
            d[f"bins{l}"] = d["rank"].apply(lambda x: int(x/l))
        return d

    feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
    feature_importances.sort_values('importance', ascending=False, inplace=True)
    feature_importances["rank"] = range(feature_importances.shape[0])
    feature_importances = series2bin(feature_importances, [10, 20, 50])

    # feature_importance_plot(feature_importances)
    return feature_importances


def plot_feature_importances(clf: XGBClassifier, X_train: DataFrame, s_org,d_org):
    feat_imp = pd.DataFrame({'importance': clf.feature_importances_})
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:7]

    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title="title")
    plt.xlabel('Feature Importance Score')
    s_org = s_org.split('ViennaDuplex')[0]
    d_org = d_org.split('ViennaDuplex')[0]

    # fname = ROOT_PATH / Path("Results/figuers/feature_importance/") / f"{s_org}_{d_org}_summary_plot.pdf"
    fname = ROOT_PATH / Path("Results/figuers_under/feature_importance/") / f"{s_org}_{d_org}_summary_plot.pdf"
    plt.savefig(fname, format="pdf", bbox_inches='tight')
    plt.show()
    plt.clf()


def model_shap_plot(test, model, model_name,s_org,d_org,dependence_feature=None):

    shap_values = shap.TreeExplainer(model).shap_values(test.values.astype('float'))
    shap.summary_plot(shap_values, test, show=False, max_display=10, feature_names=test.columns)
    plt.title(f"{s_org}_{d_org}_summary_plot")
    fname = ROOT_PATH / Path("Results/figuers_under/shap/") / f"{s_org}_{d_org}_summary_plot.pdf"
    plt.savefig(fname, format="pdf", bbox_inches='tight')
    plt.show()
    plt.clf()
    # if dependence_feature is not None:
    #     shap.dependence_plot(dependence_feature, shap_values, data,show=False)
    #     plt.title(f"{model_name}_{s_org}_{d_org}_dependence_plot")
    #     plt.savefig(os.path.join(MODELS_FEATURE_DEPENDENCE, f"{model_name}_{s_org}_{d_org}_dependence_plot.png"),
    #                 bbox_inches='tight')
    #     plt.clf()



def self_results_summary(method_split: str):
    ms_table = pd.DataFrame(columns={"TPR", "TNR", "PPV", "NPV", "FPR", "FNR", "FDR","ACC"})
    results_dir = ROOT_PATH / Path("Results")
    test_dir = DATA_PATH_INTERACTIONS / "test" / method_split
    train_dir = DATA_PATH_INTERACTIONS / "train"

    methods = ['xgbs']
    res_table: DataFrame = pd.DataFrame()
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()

    for f_test in test_dir.glob("*test*"):
        f_stem = f_test.stem
        dataset = f_stem.split(".csv")[0]
        for method in methods:
            print(f"test: {f_test}, method: {method}")
            try:
                dataset = dataset.replace("test", "train")
                clf = get_presaved_clf(results_dir, dataset, method)
                X_test, y_test = feature_reader.file_reader(f_test)
                model_shap_plot(X_test, clf, method, dataset, f_test, dependence_feature=None)
                test_score = accuracy_score(y_test, clf.predict(X_test))
                res_table.loc[dataset, method] = round(test_score, 3)

                print(res_table)
                name_summary_file = "summary_" + dataset + ".csv"
                res_table.to_csv(results_dir / "summary" / name_summary_file)
                if method in ["xgbs_no_encoding", "xgbs"]:
                    feature_importance = xgbs_feature_importance(clf, X_test)
                    to_csv(feature_importance, results_dir / "feature_importance" / f"feature_importance_{dataset}.csv")
                    print("save feature importance file")
                    ms = measurement(y_test, clf.predict(X_test))
                    ms_table.loc[dataset, f_test] = ms

            except NoModelFound:
                pass

    res_table.sort_index(inplace=True)
    print(res_table)
    print(res_table.to_latex())

    res_table.to_csv(results_dir/"summary.csv")
    to_csv(ms_table, results_dir / "xgbs_measurements" / "xgbs_measurements.csv")


def different_results_summary(method_split: str, model_dir: str, number_iteration):

    ms_table = None
    results_dir = ROOT_PATH / Path("Results")
    results_dir_models = ROOT_PATH / Path("Results") / model_dir
    test_dir = DATA_PATH_INTERACTIONS / "test" / method_split
    res_table: DataFrame = pd.DataFrame()
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()
    clf_datasets = [f.stem.split("_xgbs")[0] for f in results_dir_models.glob("*_xgbs*model")]
    method ='xgbs'
    for clf_dataset in clf_datasets:
        for f_test in test_dir.glob("*test*"):
            f_stem = f_test.stem
            test_dataset = f_stem.split(".csv")[0]
            print(f"clf: {clf_dataset} test: {test_dataset}, method: {method}")
            try:
                    clf = get_presaved_clf(results_dir_models, clf_dataset, method)
                    X_test, y_test = feature_reader.file_reader(test_dir/f"{test_dataset}.csv")
                    # X_test.drop(columns=['MRNA_Target_A_comp', 'MRNA_Target_C_comp', 'MRNA_Target_G_comp',
                    #                      'MRNA_Target_U_comp', 'MRNA_Target_AA_comp', 'MRNA_Target_AC_comp',
                    #                      'MRNA_Target_AG_comp', 'MRNA_Target_AU_comp',
                    #                      'MRNA_Target_CA_comp', 'MRNA_Target_CC_comp', 'MRNA_Target_CG_comp',
                    #                      'MRNA_Target_CU_comp',
                    #                      'MRNA_Target_GA_comp', 'MRNA_Target_GC_comp', 'MRNA_Target_GG_comp',
                    #                      'MRNA_Target_GU_comp',
                    #                      'MRNA_Target_UA_comp', 'MRNA_Target_UC_comp', 'MRNA_Target_UG_comp',
                    #                      'MRNA_Target_UU_comp'], inplace=True)

                    # shap graph
                    model_shap_plot(X_test, clf, method, clf_dataset, test_dataset, dependence_feature=None)

                    # features importance graph
                    plot_feature_importances(clf, X_test,clf_dataset, test_dataset)

                    # score predict
                    test_score = accuracy_score(y_test, clf.predict(X_test))
                    res_table.loc[clf_dataset, test_dataset] = round(test_score, 3)

                    feature_importance = xgbs_feature_importance(clf, X_test)
                    name_method = "model:" + clf_dataset.split("negative")[0] + "_" + "test:" + test_dataset.split("negative")[0]
                    to_csv(feature_importance, results_dir / "feature_importance" / f"feature_importance_{name_method}.csv")

                    # save measures
                    ms = measurement(y_test, clf.predict(X_test))
                    if ms_table is None:
                        ms_table = pd.DataFrame(columns=list(ms.keys()), dtype=object)
                    name_method = "model:" + clf_dataset.split("negative")[0] + "/" + "test:" + test_dataset.split("negative")[0]
                    ms_table.loc[name_method] = ms

            except NoModelFound:
                    pass

    res_table.sort_index(axis=0, inplace=True)
    res_table.sort_index(axis=1, inplace=True)

    print(res_table)
    to_csv(res_table, results_dir / "summary" / "diff_summary_stratify.csv")
    to_csv(ms_table, results_dir / "xgbs_measurements" /f"measurement_summary_{number_iteration}.csv")
    print("END result test")
    return ms_table


# different_results_summary(method_split="underSampling", model_dir="models_underSampling")

# different_results_summary(method_split="stratify", model_dir="models_stratify")
