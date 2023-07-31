import copy
import math
import os
from consts.global_consts import ROOT_PATH,BIOMART_PATH, MERGE_DATA, NEGATIVE_DATA_PATH, GENERATE_DATA_PATH
from utils.utilsfile import read_csv, to_csv
import pandas as pd
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Classifiers_Cross_Validiation_Runing import CrossValidation
from matplotlib.lines import Line2D
from scipy.stats import ranksums
from scipy import stats
import math
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
from pandas import DataFrame
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
import pickle
import shap
import pandas as pd
from scipy import stats
from sklearn.metrics import jaccard_score

import seaborn as sns
from matplotlib import pyplot as plt


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
    name_clean = name_clean.replace("__", "_")
    name_clean = name_clean.replace("MockMrna_di_mrna", "mockMrna_di_mrna")
    name_clean = name_clean.replace("Mockmrna_di_site_", "Mockmrna_di_fragment_")
    name_clean = name_clean.replace("Mockmrna_di_mockmirna_", "Mockmrna_di_fragment_mockmirna_")

    return name_clean


def get_ordered_top_features(df_src, df_trg, n_top=10):
    top_fetures = set(list(df_trg['name'][:n_top]) + list(df_src['name'][:n_top]))
    df_src = df_src[df_src['name'].isin(top_fetures)]
    df_trg = df_trg[df_trg['name'].isin(top_fetures)]
    print(top_fetures)
    merged_frame = pd.merge(df_src, df_trg, on="name", how='inner')
    return merged_frame


def calc_by_metric(df_src, df_trg, metric_name):
    res = None
    if metric_name == "JACARD10":
        res = jaccard_score(df_src['name'][:10], df_trg['name'][:10], average='weighted')
    if metric_name == "JACARD20":
        res = jaccard_score(df_src['name'][:20], df_trg['name'][:20], average='weighted')
    if metric_name == "SPEARMANR":
        normalized_src_df = normalize_vector(df_src)
        normalized_trg_df = normalize_vector(df_trg)
        new_df = get_ordered_top_features(normalized_src_df, normalized_trg_df, n_top=df_src.shape[0])
        p_value = stats.spearmanr(new_df['value_x'], new_df['value_y']).correlation
    if metric_name == "SPEARMANR20":
        normalized_src_df = normalize_vector(df_src)
        normalized_trg_df = normalize_vector(df_trg)
        new_df = get_ordered_top_features(normalized_src_df, normalized_trg_df, n_top=df_src.shape[0])[:20]
        p_value = stats.spearmanr(new_df['value_x'], new_df['value_y']).correlation
    if metric_name == "WILCOXON10":
        normalized_src_df = normalize_vector(df_src)
        normalized_trg_df = normalize_vector(df_trg)
        new_df = get_ordered_top_features(normalized_src_df, normalized_trg_df, n_top=10)
        p_value = stats.wilcoxon(new_df['value_x'], new_df['value_y']).pvalue
    if metric_name == "WILCOXON20":
        normalized_src_df = normalize_vector(df_src)
        normalized_trg_df = normalize_vector(df_trg)
        new_df = get_ordered_top_features(normalized_src_df, normalized_trg_df, n_top=20)
        p_value = stats.wilcoxon(new_df['value_x'], new_df['value_y']).pvalue
        path = "/sise/home/efrco/efrco-master/vectors.csv"
        to_csv(new_df, path)
    if p_value >0.05:
        print(p_value)
    else:
        p_value = 0
    print(df_src)
    print(df_trg)
    print("###############################################################", p_value)
    return p_value


def normalize_vector(df_src):
    df_src["value"] = (df_src['value'] - df_src['value'].min()) / (df_src['value'].max() - df_src['value'].min())
    return df_src




def model_shap_plot_cross_specfic_sample(test, y_true, modelTest, modelTrain, test_name, train_name, name_classifiers,name_measur, dependence_feature=None):
    test_name = clean_name(test_name.split("_train")[0])
    train_name = clean_name(train_name.split("_train")[0])

    shap_values_train = shap.TreeExplainer(modelTrain).shap_values(test.values.astype('float'))
    shap_values_test = shap.TreeExplainer(modelTest).shap_values(test.values.astype('float'))

    feature_importance_valuesA = np.abs(shap_values_train).mean(0)
    feature_importance_valuesB = np.abs(shap_values_test).mean(0)
    feature_importanceA = pd.DataFrame(list(zip(test.columns, feature_importance_valuesA)),
                                       columns=['name', 'value']).sort_values(by='value', ascending=False)
    feature_importanceB = pd.DataFrame(list(zip(test.columns, feature_importance_valuesB)),
                                       columns=['name', 'value']).sort_values(by='value', ascending=False)

    return calc_by_metric(feature_importanceA, feature_importanceB, name_measur)


# This function response on load the clf from Result dir
def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}_{method}.model"
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf


def feature_importance_cross(method_split: str, model_dir: str, number_iteration: int, name_classifier: str, name_measur:str):
    ms_table = None
    chossing_methods = ['tarBase_microArray', "mockMiRNA",
                        "non_overlapping_sites", "non_overlapping_sites_random",

                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3",
                        "Non_overlapping_sites_clip_data_random"]
    for i in range(len(chossing_methods)):
        name = chossing_methods[i] + "_"
        chossing_methods[i] = name.capitalize()
    results_dir = ROOT_PATH / Path("Results")
    number_iteration = str(number_iteration)
    results_dir_models = ROOT_PATH / Path("Results/models") / model_dir / number_iteration
    test_dir = DATA_PATH_INTERACTIONS / "test" / method_split / number_iteration

    res_table: DataFrame = pd.DataFrame()
    res_table.loc["Tarbase_microarray_", "Tarbase_microarray_"] = 0

    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()

    clf_datasets = [f.stem.split("_" + name_classifier)[0] for f in results_dir_models.glob("*.model")]
    method = name_classifier
    for j in range(len(chossing_methods)):  # train

        for i in range(len(chossing_methods)):  # test
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
                    if train_name not in chossing_methods or test_name not in chossing_methods:
                        # print(train_name)
                        # print(test_name)
                        continue

                    if train_name != chossing_methods[j]:
                        continue
                    if test_name != chossing_methods[i]:
                        continue
                    if train_name != "Non_overlapping_sites_":
                        continue
                    if test_name != "Non_overlapping_sites_random_":
                        continue

                    print(test_name)
                    print(train_name)

                    res =  model_shap_plot_cross_specfic_sample(X_test_A, y_test, modelTest=clfA, modelTrain=clfB,
                                                         test_name=clf_dataset_A, train_name=clf_dataset_B,
                                                         name_classifiers=name_classifier,
                                                         name_measur=name_measur, dependence_feature=None)

                    res_table.loc[train_name, test_name] = res


                    print("####################################################################################")
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table, annot=True, cmap=cmap)

    # sns.color_palette("Spectral", as_cmap=True)
    ax.set(xlabel="train", ylabel="test")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=215)
    plt.yticks(ha='left')
    # plt.show()
    fname = ROOT_PATH / Path("figuers") / Path(f"heatmap_cross_features_top_FP_{name_measur}.png")
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)


# feature_importance_cross(method_split="underSampling", model_dir="models_underSampling",
#                          number_iteration=0, name_classifier='xgbs', name_measur= "WILCOXON10")

#
# feature_importance_cross(method_split="underSampling", model_dir="models_underSampling",
#                          number_iteration=0, name_classifier='xgbs', name_measur= "WILCOXON20")
#
# feature_importance_cross(method_split="underSampling", model_dir="models_underSampling",
#                          number_iteration=0, name_classifier='xgbs', name_measur= "SPEARMANR")
feature_importance_cross(method_split="underSampling", model_dir="models_underSampling",
                         number_iteration=3, name_classifier='xgbs', name_measur= "WILCOXON20")
