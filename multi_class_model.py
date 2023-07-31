from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score

import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import yaml
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import shap



def remove_numbers(string):
    return ''.join(char for char in string if not char.isdigit())

def convert_name(name):
        if name == "Tarbase_liver":
            name = "TarBase_Liver"
        elif name == "Tarbase":
            name = "TarBase"
        elif name == "Tarbase_microarray":
            name = "TarBase_microarray"
        elif name == "Mockmirna":
            name = "Mock_miRNA"
        elif name == "Mock_mirna":
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
        replace("negative_features","").replace("negative","")

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
        "PNR": 0,
        "F1": f1_score(y_true, y_pred)

    }

    return {k: round(v,3) for k, v in d.items()}

def model_feature_importance_plot(test, model,s_org,d_org, dependence_feature=None):

    explainer = shap.Explainer(model.predict, test)
    shap_values = explainer(test, max_evals=(2*(len(test.columns)+1)))
    shap.plots.bar(shap_values, show=False, max_display=11)

    plt.title(f"Datasets: {s_org} - {d_org}")
    fname = ROOT_PATH / Path(f"figuers/figure_cross_explain_feature_importance/") / f"{s_org}_{d_org}.png"

    plt.gca().tick_params(axis="y", pad=280)

    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.clf()

def model_shap_plot(test, model,s_org,d_org, dependence_feature=None):

    shap_values = shap.TreeExplainer(model).shap_values(test.values.astype('float'))
    shap.summary_plot(shap_values, test, show=False, max_display=10,color_bar=True, feature_names=test.columns)

    plt.title(f"Datasets: {s_org} - {d_org}")
    fname = ROOT_PATH / Path(f"figuers/figuers_cross_explain/") / f"{s_org}_{d_org}.png"
    plt.gcf().axes[-1].set_aspect(100)
    plt.xlim(-1.5, 1.5)  # Set x-axis limits
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.gca().tick_params(axis="y", pad=280)
    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()


def prepare_data():
    frames_train = []
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()
    train_dir = Path("/sise/home/efrco/efrco-master/data/train/underSampling/0")
    test_dir = Path("/sise/home/efrco/efrco-master/data/test/underSampling/0")

    list_method = ["top", "Liver", "tarBase_human", "_nucleotides", "non_overlapping_sites_clip_data_darnell_human_ViennaDuplex_negative_"]
    dict_train = {}
    dict_x_test = {}
    dict_y_test = {}
    for f_train in train_dir.glob('**/*.csv'):
        f_stem = f_train.stem
        f_test = f_stem.replace("train", "test")
        train_dataset = f_stem.split(".csv")[0].split("features_train_underSampling_method_0")[0]
        if any(method in train_dataset for method in list_method):
            continue
        else:
            # train
            print(train_dataset)
            train_dataset = convert_name(clean_name(train_dataset))
            X_train, y_train =  feature_reader.file_reader(f_train)
            y_train = pd.DataFrame(y_train, columns=['Label'])
            df_train = pd.concat([X_train, y_train], axis=1)
            # get the negative data
            df_train = df_train[df_train['Label'] == 0]
            df_train['Label'] = train_dataset
            dict_train[train_dataset] = df_train

            # test
            test_file = test_dir / (f_test+".csv")
            X_test, y_test = feature_reader.file_reader(test_file)
            y_test = pd.DataFrame(y_test, columns=['Label'])
            df_test = pd.concat([X_test, y_test], axis=1)
            df_test = df_test[df_test['Label'] == 0]
            df_test['Label'] = train_dataset
            dict_y_test[train_dataset] = df_test['Label'].to_numpy()
            dict_x_test[train_dataset] = df_test.drop('Label', axis=1)
    print(df_train)
    return dict_train, dict_y_test, dict_x_test


def train_one_conf_binary(clf_name, conf, X, y, scoring="accuracy"):

    # creat the specific clf and load the parameters of the clf according to the ymal file.
    clf = XGBClassifier()
    parameters = conf['parameters']

    # this step find to optimize prams
    grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=5, n_jobs=-1, verbose=2)
    grid_obj.fit(X, y)

    print('\n Best estimator:')
    print(grid_obj.best_estimator_)
    print(grid_obj.best_score_ )
    # save the best classifier
    best_clf = grid_obj.best_estimator_
    return best_clf



def worker_binary():
    yaml_path = "/sise/home/efrco/efrco-master/Classifier/yaml/xgbs_params_small.yml"

    #6117769
    # yaml_path = "/sise/home/efrco/efrco-master/Classifier/yaml/xgbs_params.yml"

    with open(yaml_path, 'r') as stream:
        training_config = yaml.safe_load(stream)

    dict_train, dict_y_test, dict_x_test = prepare_data()
    files = dict_train.keys()
    res_table_FPR = pd.DataFrame()
    res_table_FNR = pd.DataFrame()
    res_table_ACC = pd.DataFrame()
    res_table_F1 = pd.DataFrame()
    res_table_sum_error = pd.DataFrame()

    for file1 in files:

        # if file1 != "tarBase_microArray_human_negative_":
        #     continue
        for file2 in files:

            if file1 == file2:
                res_table_FPR.loc[file1,file1] = 0
                res_table_FNR.loc[file1,file1] = 0
                res_table_sum_error.loc[file1, file1] = 0
                res_table_F1.loc[file1, file1] = 0
                res_table_ACC.loc[file1, file1] = 0

                continue


            #case 2 -
            # elif file2 != 'clip_interaction_clip_3_negative_' :
            #     continue

            # positive- file 1 - train
            dict_train[file1]['Label'] = 1
            dict_y_test[file1][:] = 1
            dict_y_test[file1] = pd.DataFrame(dict_y_test[file1], columns=['Label'])

            # negative- file2 - test
            dict_train[file2]['Label'] = 0
            dict_y_test[file2][:] = 0
            dict_y_test[file2] = pd.DataFrame(dict_y_test[file2], columns=['Label'])

            train = pd.concat([dict_train[file1],dict_train[file2]])
            y = train['Label']
            X = train.drop("Label", axis=1)
            y_test = pd.concat([dict_y_test[file1], dict_y_test[file2]]).Label.ravel()
            X_test = pd.concat([dict_x_test[file1], dict_x_test[file2]])

            for clf_name, conf in training_config.items():
                if conf["run"]:
                    clf = train_one_conf_binary(clf_name, conf, X, y, scoring="accuracy")
                    y_pred = clf.predict(X_test)
                    y_test = pd.to_numeric(y_test)
                    test_score = accuracy_score(y_test, y_pred)
                    ms = measurement(y_test, y_pred)
                    # model_shap_plot(X_test, clf,file1,file2, dependence_feature=None)
                    # model_feature_importance_plot(X_test, clf,file1,file2, dependence_feature=None)


                    fpr = ms['FPR']
                    fnr = ms['FNR']

                    # model_shap_plot(X_test, clf,file1,file2, dependence_feature=None)
                    res_table_FPR.loc[file2, file1] = fpr
                    res_table_FNR.loc[file2, file1] = fnr
                    res_table_ACC.loc[file2, file1] = ms['ACC']
                    res_table_F1.loc[file2, file1] = ms['F1']
                    res_table_sum_error.loc[file2, file1] = fpr+fnr



                    print("####################################################")
                    print(file1)
                    print(file2)
                    print("##########################################################",test_score)
            break
        break

    #second figure
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table_FPR, annot=True, linewidth=(2.5, 0), cmap=cmap)
    ax.set(xlabel="Positive dataset", ylabel="Negative dataset")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=300)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") / "acc_cross_lior_fpr.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()

    #one figure
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table_FNR, annot=True, linewidth=(2.5, 0), cmap=cmap)


    ax.set(xlabel="Positive dataset", ylabel="Negative dataset")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=300)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") / "acc_cross_lior_fnr_stam.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()

    # three figure
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table_ACC, annot=True, linewidth=(2.5, 0), cmap=cmap)

    ax.set(xlabel="positive dataset", ylabel="negative dataset")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=300)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") / "acc_cross_lior_acc.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()

    # four figure
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table_F1, annot=True, linewidth=(2.5, 0), cmap=cmap)

    ax.set(xlabel="positive dataset", ylabel="negative dataset")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=300)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") / "acc_cross_lior_f1.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()

    # five figure
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table_sum_error, annot=True, linewidth=(2.5, 0), cmap=cmap)

    ax.set(xlabel="positive dataset", ylabel="negative dataset")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=300)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") / "acc_cross_lior_sum_error.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()

#- 8576607 with new figure
worker_binary() #8576607








# def train_one_conf_one_class(clf_name, conf, X, y, scoring="accuracy"):
#
#     # creat the specific clf and load the parameters of the clf according to the ymal file.
#     clf = IsolationForest()
#     parameters = conf['parameters']
#
#     # this step find to optimize prams
#     grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=5, n_jobs=-1, verbose=2)
#     grid_obj.fit(X, y)
#
#     print('\n Best estimator:')
#     print(grid_obj.best_estimator_)
#     print(grid_obj.best_score_ )
#     # save the best classifier
#     best_clf = grid_obj.best_estimator_
#     return best_clf
#
# def worker_one_class():
#     yaml_path = "/sise/home/efrco/efrco-master/Classifier/yaml/one_class_params.yml"
#     with open(yaml_path, 'r') as stream:
#         training_config = yaml.safe_load(stream)
#
#     dict_train, dict_y_test, dict_x_test = prepare_data()
#     files = dict_train.keys()
#     res_table_ACC = pd.DataFrame()
#     res_table_FNR = pd.DataFrame()
#
#     for file1 in files:
#         for file2 in files:
#             if file1 == file2:
#                 res_table_ACC.loc[file1, file1] = 0
#                 res_table_FNR.loc[file1, file1] = 0
#                 continue
#
#             dict_train[file1]['Label'] = 1
#             dict_y_test[file1][:] = 1
#             dict_y_test[file1] = pd.DataFrame(dict_y_test[file1], columns=['Label'])
#
#             dict_y_test[file2][:] = 0
#             dict_y_test[file2] = pd.DataFrame(dict_y_test[file2], columns=['Label'])
#
#             train = dict_train[file1]
#             y = train['Label']
#             X = train.drop("Label", axis=1)
#             y_test = pd.concat([dict_y_test[file1], dict_y_test[file2]]).Label.ravel()
#             X_test = pd.concat([dict_x_test[file1], dict_x_test[file2]])
#
#             for clf_name, conf in training_config.items():
#                 if conf["run"] and clf_name == "isolation_forest":
#                     clf = train_one_conf_one_class(clf_name, conf, X, y, scoring="accuracy")
#                     y_pred = clf.predict(X_test)
#                     y_pred[y_pred == -1] = 0  # negative class
#                     y_pred[y_pred == 1] = 1  # positive class
#                     y_test = pd.to_numeric(y_test)
#                     test_score = accuracy_score(y_test, y_pred)
#                     ms = measurement(y_test, y_pred)
#                     ACC = ms['ACC']
#                     fnr = ms['FNR']
#                     fpr = ms['FPR']
#
#                     print("#################################", fnr)
#                     print("#################################", fpr)
#
#
#                     # model_shap_plot(X_test, clf,file1,file2, dependence_feature=None)
#                     res_table_ACC.loc[file2, file1] = ACC
#                     res_table_FNR.loc[file2, file1] = fnr
#
#                     print("####################################################")
#                     print(file1)
#                     print(file2)
#                     print("##########################################################", test_score)
#
#     # second figure
#     cmap = sns.cm.rocket_r
#     ax = sns.heatmap(res_table_ACC, annot=True, linewidth=(2.5, 0), cmap=cmap)
#     ax.set(xlabel="positive dataset", ylabel="negative dataset")
#     plt.xticks(rotation=30, ha='right')
#     ax.tick_params(axis="y", pad=215)
#     plt.yticks(ha='left')
#
#     fname = ROOT_PATH / Path("figuers") / "acc_cross_lior_fpr.png"
#     plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
#     plt.clf()
#     # one figure
#     cmap = sns.cm.rocket_r
#     ax = sns.heatmap(res_table_FNR, annot=True, linewidth=(2.5, 0), cmap=cmap)
#
#     ax.set(xlabel="positive dataset", ylabel="negative dataset")
#     plt.xticks(rotation=30, ha='right')
#     ax.tick_params(axis="y", pad=215)
#     plt.yticks(ha='left')
#
#     fname = ROOT_PATH / Path("figuers") / "acc_cross_lior_fnr.png"
#     plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
#     plt.clf()
#
#
# #6118438 - one class
# #6118479 - isolation
# # worker_one_class()

