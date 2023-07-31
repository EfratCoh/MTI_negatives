from functools import partial
from pathlib import Path
from typing import Dict
import pandas as pd
import yaml
from scipy.stats import ranksums, kstest, wilcoxon
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import sklearn
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
import pickle
import xgboost as xgb
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from Classifier.ClfLogger import logger
from consts.global_consts import  ROOT_PATH, DATA_PATH_INTERACTIONS, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS

def get_ordered_top_features(modelTrain, modelTest, test, n_top=10):


    shap_values_train = shap.TreeExplainer(modelTrain).shap_values(test.values.astype('float'))
    shap_values_test = shap.TreeExplainer(modelTest).shap_values(test.values.astype('float'))

    feature_importance_valuesA = np.abs(shap_values_train).mean(0)
    feature_importance_valuesB = np.abs(shap_values_test).mean(0)
    feature_importanceA = pd.DataFrame(list(zip(test.columns, feature_importance_valuesA)),
                                       columns=['name', 'value']).sort_values(by='value', ascending=False)
    feature_importanceB = pd.DataFrame(list(zip(test.columns, feature_importance_valuesB)),
                                       columns=['name', 'value']).sort_values(by='value', ascending=False)
    return  set(list(feature_importanceA['name'][:n_top]) + list(feature_importanceB['name'][:n_top]))

#
# list_features = ['Seed_match_GU_2_7', 'miRNAMatchPosition_2', 'Energy_MEF_Duplex', 'miRNAMatchPosition_4',
#                  'MRNA_Target_GG_comp', 'miRNAPairingCount_Total_GU']

def clean_name(name):
    name_clean = name.replace("model:", "").replace("human", "").replace("darnell_human", "").replace("test:",
                                                                                                      "").replace(
        "ViennaDuplex", "").replace("_darnell_", "").replace("__", "_").replace("negative", "").replace("_features","")

    name_clean = name_clean.replace("_nucleotides", "_mono", )
    name_clean = name_clean.replace("denucleotides", "_di")
    name_clean = name_clean.replace("method1", "mrna")
    name_clean = name_clean.replace("method2", "site")
    name_clean = name_clean.replace("method3", "mockMirna")
    name_clean = name_clean.replace("rna", "RNA")

    if name_clean == "mockMirnadarnell":
        name_clean = "mockMirna"
    if name_clean == "mockMrnadarnell":
        name_clean = "mockMrna"
    if name_clean == "nonoverlappingsitesdarnell":
        name_clean = "Non site"
    if name_clean == "mockmra__di_mockMirna":
        name_clean = "mockMrna__di_fragment_mockMirna"
    if name_clean == "MockmRNA_di_mockmiRNA":
        name_clean = "mockMrna_di_fragment_mockMirna"
    if name_clean == "mockMrna_mono_mockMirna":
        name_clean = "mockMrna__mono_fragment_mockMirna"
    if name_clean == "MockmRNA_mono_mockmiRNA":
        name_clean = "mockMrna_mono_fragment_mockMirna"
    if name_clean == "mockMrna__di_site":
        name_clean = "mockMrna__di_fragment"
    if name_clean == "MockmRNA_di_site":
        name_clean = "mockMRNA_di_fragment"
    if name_clean == "MockmRNA_mono_site":
        name_clean = "mockMRNA_mono_fragment"
    name_clean = name_clean.capitalize()

    return name_clean
class ClassifierWithGridSearch(object):
    def __init__(self, dataset_file, result_dir):
        self.dataset_file = dataset_file
        self.dataset_name = self.extract_dataset_name()
        print(f"Handling dataset : {self.dataset_name}")
        self.load_dataset()
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.create_clf_dict()

    def extract_dataset_name(self):
        return str(self.dataset_file.stem).split("_test")[0]

    def create_clf_dict(self):
        self.clf_dict = {
            "xgbs": XGBClassifier(),
            "xgbs_no_encoding": XGBClassifier(),
        }

    # this function response on load the dataset
    def load_dataset(self):
        directory = self.dataset_file.parent
        feature_reader = get_reader()
        X, y = feature_reader.file_reader(directory / f"{self.dataset_name}.csv")
        self.X = X
        self.y = y


    def fit_best_clf(self, clf_name):
        clf = self.clf_dict[clf_name]
        print(clf)
        fit_params = {}
        if clf_name =="xgbs":
            fit_params = {"eval_set": [(self.X, self.y)],
                          "early_stopping_rounds": 50}

        clf.fit(self.X, self.y, **fit_params)
        clf.save_model("model_sklearn.json")

        return clf

    # this function response on train model and then save this model
    def train_one_conf(self, clf_name, parameters, scoring="accuracy"):

        output_file = self.result_dir / f"{self.dataset_name}_{clf_name}.csv"
        # creat the specific clf and load the parameters of the clf according to the ymal file.
        clf = self.clf_dict[clf_name]
        print(clf)

        seed = random.randint(0, 1000)
        clf = xgb.XGBClassifier(seed=seed)

        # this step find to optimize prams
        grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=5, n_jobs=-1, verbose=3)
        # grid_obj.set_params(**parameters)
        # grid_obj.fit(self.X, self.y)
        grid_obj.fit(self.X, self.y)



        print('\n Best estimator:')
        print(grid_obj.best_estimator_)
        print(grid_obj.best_score_ )
        # save the best classifier
        best_clf = grid_obj.best_estimator_
        return  best_clf


def model_shap_plot(test, model):

    shap_values = shap.TreeExplainer(model).shap_values(test.values.astype('float'))
    return shap_values


def worker(dataset_file, results_dir, yaml_file, test, y_true, parameters):
    clf_grid_search = ClassifierWithGridSearch(dataset_file=dataset_file, result_dir=results_dir)
    best_clf = clf_grid_search.train_one_conf("xgbs", parameters, scoring="accuracy")
    #test_shap
    matrix_shap = model_shap_plot(test, best_clf)
    return matrix_shap



def self_fit(feature_mode, yaml_file, first_self, last_self, name_method, dir_method,number_iteration):
    FeatureReader.reader_selection_parameter = feature_mode
    test_dir = DATA_PATH_INTERACTIONS / "test" / name_method / number_iteration

    # test
    feature_reader = get_reader()
    X_test, y_test = feature_reader.file_reader(test_dir / "clip_interaction_clip_3_negative_features_test_underSampling_method_0.csv")
    # X_test, y_test = feature_reader.file_reader(test_dir / "tarBase_microArray_human_negative_features_test_underSampling_method_0.csv")

    # train
    csv_dir = DATA_PATH_INTERACTIONS / "train" / name_method / number_iteration
    files = list(csv_dir.glob('**/*.csv'))
    results_dir_models = Path("/sise/home/efrco/efrco-master/Results/models/models_underSampling/0")
    # clf_dataset = "tarBase_microArray_human_negative_features_train_underSampling_method_0"
    clf_dataset = "clip_interaction_clip_3_negative_features_train_underSampling_method_0"
    clf = get_presaved_clf(results_dir_models, clf_dataset, method="xgbs")
    params = clf.get_params()
    parameters = {"use_label_encoder": [params["use_label_encoder"]], "booster": [params["booster"]],
                  "colsample_bytree": [params["colsample_bytree"]], "eta": [params["eta"]],
                  "gamma": [params["gamma"]], "lambda": [params["lambda"]],
                  "max_depth": [params["max_depth"]], "min_child_weight": [params["min_child_weight"]],
                  "n_estimators": [params["n_estimators"]], "n_jobs": [params["n_jobs"]],
                  "objective": [params["objective"]], "subsample": [params["subsample"]]}
    for f in files:
        # if "tarBase_microArray_human_negative_features_train_underSampling_method_0" not in f.name:
        #     continue
        if "tarBase_microArray_human_negative_features_train_underSampling_method_0" not in f.name:
            continue
        results_dir = ROOT_PATH / "Results/models" / dir_method / number_iteration

        matrix_shap = worker(f, results_dir=results_dir, yaml_file=yaml_file, test=X_test, y_true=y_test,parameters=parameters)
        return matrix_shap, X_test


def build_classifiers(number_iteration, list_features):
    number_exper = 15
    yaml_file = "/sise/home/efrco/efrco-master/Classifier/yaml/xgbs_params_small.yml"
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    number_iteration = str(number_iteration)
    all_matrix = list([])
    for i in range(number_exper):
        matrix_shap, X_test = self_fit("without_hot_encoding", yaml_file, 1, 2, name_method="underSampling", dir_method="models_underSampling", number_iteration=number_iteration)
        all_matrix.append(matrix_shap)

    feature_to_col = {feat_name: col_idx for col_idx, feat_name in enumerate(X_test.columns)}
    dict_features = {}
    for feature in list_features:
        selected_cols = [feature_to_col[feature]]
        p_val_list = []

        for i in range(number_exper):
          selected_shap_values_i = all_matrix[i][:, selected_cols]
          for j in range(i+1, number_exper):
              selected_shap_values_j = all_matrix[j][:, selected_cols]
              normalized_src_df = (selected_shap_values_i - selected_shap_values_i.min()) / (
                      selected_shap_values_i.max() - selected_shap_values_i.min())
              normalized_trg_df = (selected_shap_values_j - selected_shap_values_j.min()) / (
                      selected_shap_values_j.max() - selected_shap_values_j.min())
              res = kstest(normalized_src_df.flatten(),normalized_trg_df.flatten()).pvalue
              print(res)
              p_val_list.append(res)
        dict_features[feature] = min(p_val_list)
    return dict_features

# dict_features = build_classifiers(number_iteration=0)
# print(dict_features)

##############################################################################################################

def model_shap_plot_cross_specfic_sample(test, y_true, modelTest, modelTrain, test_name, train_name, name_classifiers, dependence_feature=None):
    test_name = clean_name(test_name.split("_train")[0])
    train_name = clean_name(train_name.split("_train")[0])
    dict_features = {}

    shap_values_train = shap.TreeExplainer(modelTrain).shap_values(test.values.astype('float'))
    shap_values_test = shap.TreeExplainer(modelTest).shap_values(test.values.astype('float'))
    feature_to_col = {feat_name: col_idx for col_idx, feat_name in enumerate(test.columns)}
    list_features = get_ordered_top_features(modelTrain,modelTest,test)
    for features in list_features:
        selected_cols = [feature_to_col[features]]

        selected_shap_values_train = shap_values_train[:, selected_cols]
        selected_shap_values_test =shap_values_test[:, selected_cols]
        normalized_src_df = (selected_shap_values_train - selected_shap_values_train.min()) / (
                    selected_shap_values_train.max() - selected_shap_values_train.min())
        normalized_trg_df = (selected_shap_values_test - selected_shap_values_test.min()) / (
                    selected_shap_values_test.max() - selected_shap_values_test.min())

        res = kstest(normalized_src_df.flatten(), normalized_trg_df.flatten()).pvalue
        dict_features[features] = res
    return dict_features, list_features


# This function response on load the clf from Result dir
def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}_{method}.model"
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf
def feature_importance_cross(method_split: str, model_dir: str, number_iteration: int, name_classifier: str):
    ms_table = None
    chossing_methods = ['tarBase_microArray', "mockMiRNA",
                        "non_overlapping_sites", "non_overlapping_sites_random",

                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3",
                        "Non_overlapping_sites_clip_data_random"]
    results_dir = ROOT_PATH / Path("Results")
    number_iteration = str(number_iteration)
    results_dir_models = ROOT_PATH / Path("Results/models") / model_dir / number_iteration
    test_dir = DATA_PATH_INTERACTIONS / "test" / method_split / number_iteration

    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()

    clf_datasets = [f.stem.split("_" + name_classifier)[0] for f in results_dir_models.glob("*.model")]
    method = name_classifier
    res_table = pd.DataFrame()

    for clf_dataset_A in clf_datasets:
        clfA = get_presaved_clf(results_dir_models, clf_dataset_A, method)
        testA_name = clf_dataset_A.replace("train", "test")
        X_test_A, y_test = feature_reader.file_reader(test_dir / f"{testA_name}.csv")

        for clf_dataset_B in clf_datasets:
            if clf_dataset_A == clf_dataset_B:
                continue
            clfB = get_presaved_clf(results_dir_models, clf_dataset_B, method)


            test_name = clean_name(clf_dataset_A.split("_train")[0])
            train_name = clean_name(clf_dataset_B.split("_train")[0])

            if train_name == 'Tarbase_microarray_' and test_name == 'Clip_interaction_clip_3_': #worst
                print(test_name)
            # elif train_name == 'Mockmirna_' and test_name == 'Non_overlapping_sites_random_': #worst
            #     print(test_name)
            # if train_name == 'Mockmirna_' and test_name == 'Tarbase_microarray_':  # good
            #     print(test_name)

            else:
                # print(train_name)
                continue

            res, list_features = model_shap_plot_cross_specfic_sample(X_test_A,y_test, modelTest=clfA, modelTrain=clfB, test_name=clf_dataset_A, train_name=clf_dataset_B, name_classifiers=name_classifier,
                            dependence_feature=None)
            return res,list_features


def run():

    dict_test,list_features = feature_importance_cross(method_split="underSampling", model_dir="models_underSampling",
                              number_iteration=0, name_classifier='xgbs')
    dict_pval = build_classifiers(number_iteration=0, list_features = list_features)
    for key in dict_pval.keys():
        if dict_test[key] > dict_pval[key]:
            print("ooo")

# run()