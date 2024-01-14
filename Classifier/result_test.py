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
        "PNR": 0,
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



def remove_numbers(string):
    return ''.join(char for char in string if not char.isdigit())

def conver_name(name):
        if name == "Tarbase_liver":
            name = "TarBase_Liver"
        elif name == "Tarbase":
            name = "TarBase"
        elif name == "Tarbase_microarray":
            name = "TarBase_microarray"
        elif name == "Mockmirna":
            name = "Mock_miRNA"
        elif name == "Non_overlapping_sites":
            name = "NPS_CLASH_MFE"
        elif name == "Non_overlapping_sites_random":
            name = "NPS_CLASH_Random"
        elif name == "Mockmrna_mono_mrna":
            name = "Mock_mono_mRNA"
        elif name == "Mockmrna_di_mrna":
            name = "Mock_di_mRNA"
        elif name == "Mockmrna_mono_site":
            name = "Mock_mono_fragment"
        elif name == "Mockmrna_di_fragment":
            name = "Mock_di_fragment"
        elif name == "Mockmrna_di_site":
            name = "Mock_di_fragment"
        elif name == "Mockmrna_di_fragment_mockmirna":
            name = "Mock_di_fragment_&_miRNA"
        elif name == "Mockmrna_di_mockmirna":
            name = "Mock_di_fragment_&_miRNA"
        elif name == "Mockmrna_mono_fragment_mockmirna":
            name = "Mock_mono_fragment_&_miRNA"
        elif name == "Clip_interaction_clip_3_":
            name = "CLIP_non_CLASH"
        elif name == "Clip_interaction_clip_3":
            name = "CLIP_non_CLASH"
        elif name == "Clip_interaction_clip":
            name = "CLIP_non_CLASH"
        elif name == "Non_overlapping_sites_clip_data":
            name = "NPS_CLIP_MFE"
        elif name == "Non_overlapping_sites_clip_data_random":
            name = "NPS_CLIP_Random"
        return name


def clean_name(name):
    name_clean = name.replace("model:", "").replace("human", "").replace("darnell_human", "").replace("test",
                                                                                                      "").replace(
        "ViennaDuplex", "").replace("_darnell_", "").replace("__", "").replace("train","").replace("underSampling_method", "").\
        replace("negative_features","")

    name_clean = name_clean.replace("_nucleotides", "_mono", )
    name_clean = name_clean.replace("_denucleotides", "_di")
    name_clean = name_clean.replace("method1", "mrna")
    name_clean = name_clean.replace("method2", "site")
    name_clean = name_clean.replace("method3", "mockMirna")
    name_clean = name_clean.replace("method3", "mockMirna")
    name_clean = remove_numbers(name_clean)
    name_clean = name_clean.replace("mockMrna_de_mrna", "mockMrna_di_mrna")
    name_clean = name_clean.replace("mockMrna_de_site", "mockMrna_di_site")
    name_clean = name_clean.replace("mockMrna_de_mockMirna", "mockMrna_di_mockMirna")
    name_clean = name_clean.replace("___", "")
    name_clean = name_clean.replace("__", "")
    # name_clean = name_clean.replace("_", " ")
    if name_clean[-1] =="_":
        name_clean = name_clean[:-1]

    if name_clean == "mockMirnadarnell":
        name_clean = "mockMirna"
    if name_clean == "mockMrnadarnell":
        name_clean = "mockMrna"
    if name_clean == "nonoverlappingsitesdarnell":
        name_clean = "Non site"
    if name_clean == "mockMrna__di_mockMirna":
        name_clean = "mockMrna_di_fragment_mockMirna"
    if name_clean == "mockMrna__di_site":
        name_clean = "mockMrna_di_fragment"
    if name_clean == "mockMrna_mono_mockMirna":
        name_clean = "mockMrna_mono_fragment_mockMirna"


    name_clean = name_clean.replace("rna", "RNA")
    name_clean = name_clean.capitalize()

    return name_clean


def plot_feature_importances(model: XGBClassifier, test, s_org, d_org, name_classifiers):
    s_org = conver_name(clean_name(s_org))
    d_org = conver_name(clean_name(d_org))
    explainer = shap.Explainer(model.predict, test)
    shap_values = explainer(test, max_evals=(2 * (len(test.columns) + 1)))
    shap.plots.bar(shap_values, show=False, max_display=11)
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/feature_importance_cross/") / f"{s_org}_{d_org}_bar_plot.png"
    plt.title(f"Train dataset: {s_org} - Test dataset: {d_org}")
    plt.gca().tick_params(axis="y", pad=250)
    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.clf()


def model_shap_plot(test, model, model_name,s_org,d_org,name_classifiers, dependence_feature=None):

    shap_values = shap.TreeExplainer(model).shap_values(test.values.astype('float'))
    shap.summary_plot(shap_values, test, show=False, max_display=10,color_bar=True, feature_names=test.columns)

    s_org = conver_name(clean_name(s_org))
    d_org = conver_name(clean_name(d_org))
    plt.title(f"Train dataset: {s_org} -Test dataset: {d_org}")
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/shap/") / f"{s_org}_{d_org}.png"
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.gca().tick_params(axis="y", pad=250)
    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()



def different_results_summary(method_split: str, model_dir: str, number_iteration: int, name_classifier: str):

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
            print(f"clf: {clf_dataset} test: {test_dataset}, method: {method}")
            try:
                    clf = get_presaved_clf(results_dir_models, clf_dataset, method)
                    X_test, y_test = feature_reader.file_reader(test_dir/f"{test_dataset}.csv")

                    # score predict
                    test_score = accuracy_score(y_test, clf.predict(X_test))
                    res_table.loc[clf_dataset, test_dataset] = round(test_score, 3)

                    # save measures
                    if name_classifier == 'svm' or name_classifier == 'isolation_forest':
                        prediction = clf.predict(X_test)
                        print("prediction_2:", Counter(clf.predict(X_test)))
                        prediction[prediction == -1] = 0  # negative class
                        prediction[prediction == 1] = 1  # positive class
                    else:
                        prediction = clf.predict(X_test)
                        # shap graph
                        model_shap_plot(X_test, clf, method, clf_dataset, test_dataset,name_classifier, dependence_feature=None)
                        # features importance graph
                        plot_feature_importances(clf, X_test, clf_dataset, test_dataset, name_classifier)

                    if name_classifier == 'isolation_forest':
                        # features shap graph
                        model_shap_plot(X_test, clf, method, clf_dataset, test_dataset,name_classifier, dependence_feature=None)
                    ms = measurement(y_test, prediction)
                    
                    if name_classifier == 'svm' or name_classifier == 'isolation_forest':
                        # # Predict scores for test set- is realte to positive calss
                        y_scores = clf.decision_function(X_test)

                        # # Calculate precision, recall and thresholds relate to negative

                        precision, recall, thresholds = precision_recall_curve(y_true=y_test,
                                                                               probas_pred=y_scores, pos_label=0)
                        # Calculate AUC for precision-recall-negative curve
                        prn_auc = auc(recall, precision)
                        print("PNR:", prn_auc)

                        ms['PNR'] = round(prn_auc,3)

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
