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
from scipy.stats import ranksums, wilcoxon
from scipy import stats
import math
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
from pandas import DataFrame
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
import pickle
import shap

path_figures = Path("/sise/home/efrco/efrco-master/figuers/")

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

    elif name == "Non_overlapping_sites_cip_data":
         name = "NPS_CLIP_MFE"
    elif name == "Non_overlapping_sites_clip_data_random":
        name = "NPS_CLIP_Random"
    elif name== "h3-positive_interactions":
        name = "h3-positive_interactions"
    return name



def clean_name(name):
    if name == "h3-positive_interactions":
        return name
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


    return name_clean


def creat_figures_Intra_anaylsis():
    cv = CrossValidation("darnell_human_ViennaDuplex_features", "measurments_cross_validation", number_iterations=15)
    list_metrix = ['ACC', 'TPR', 'TNR', 'PPV']

    # ####################TarBase############################
    # for measuer in list_metrix:
    #     cv.summary_matrix_tarBase(measuer, target_calculate= "std")
    #     cv.summary_matrix_tarBase(measuer, target_calculate= "mean")
    #
    # #####################MockMrna############################
    #
    # for measuer in list_metrix:
    #     cv.summary_matrix_mock_mrna(measuer, target_calculate= "std")
    #     cv.summary_matrix_mock_mrna(measuer, target_calculate= "mean")
    #
    # ####################non_overlapping_sites############################
    #
    # for measuer in list_metrix:
    #     cv.summary_matrix_non_overlapping_site(measuer, target_calculate= "std")
    cv.summary_matrix_non_overlapping_site('ACC', target_calculate= "mean")
    #
    # ####################clip_data_tail############################
    #
    # for measuer in list_metrix:
    #     cv.summary_matrix_clip(measuer, target_calculate="std")
    #     cv.summary_matrix_clip(measuer, target_calculate="mean")

    # cv.summary_matrix("ACC")
    cv.summary_matrix_Intra()
    # cv.summary_matrix_cross("ACC")


# creat_figures_Intra_anaylsis()



def cumulative_sum_miRNA_tarBase(nirmol=False):


    # Assume df1, df2, ..., dfn are DataFrames with a 'miRNA ID' column and a 'size' column
    # Store the DataFrames in a list
    dataset_micro = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/tarBase/tarBase_microArray_human_negative_features.csv")
    dataset_liver = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/tarBase/tarBase_Liver_human_negative_features.csv")
    dataset_tarbase = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/tarBase/tarBase_human_negative_features.csv")
    darnell = read_csv("/sise/home/efrco/efrco-master/data/positive_interactions/positive_interactions_new/featuers_step/darnell_human_ViennaDuplex_features.csv")
    dfs = [darnell, dataset_micro, dataset_liver, dataset_tarbase]
    name_dfs = ["h3-positive interactions","TarBase_microarray","TarBase_Liver","TarBase"]

    all_mirnas = pd.concat(dfs)['miRNA ID'].unique()
    palette = sns.color_palette()

    # create figure and axes objects
    fig, ax = plt.subplots(figsize=(10, 6))
    handles, labels = plt.gca().get_legend_handles_labels()

    # iterate over dataframes
    for i, df in enumerate(dfs):
        # calculate cumulative sum of unique miRNAs
        if nirmol:
            name = "Cumulative_sum_tarBase_nirmol.png"
            cumsum = np.cumsum(df['miRNA ID'].value_counts()/df.shape[0])
        else:
            name = "Cumulative_sum_tarBase.png"
            cumsum = np.cumsum(df['miRNA ID'].value_counts())


        # calculate percentage of interactions represented by each miRNA
        cumsum_pct = cumsum / cumsum[-1] * 100

        # find minimum number of miRNAs needed to represent 90% of interactions
        min_mirnas = np.argmax(cumsum_pct >= 90)

        # plot cumulative sum with filled circle indicating minimum miRNAs needed for 90% representation
        x = np.arange(len(cumsum)) + 1
        y = cumsum
        ax.plot(x, y, linewidth=2,color=palette[i])
        ax.plot(min_mirnas + 1, cumsum[min_mirnas], marker='o', fillstyle='full', markersize=8, color=palette[i])
        line = Line2D([0], [0], color=palette[i], linewidth=2, marker='o', markersize=8, markerfacecolor=palette[i],
                  markeredgewidth=2, markeredgecolor=palette[i])

        # add the Line2D object to the legend
        handles.append(line)
        labels.append(name_dfs[i])

    # set plot title and axis labels
    # ax.set_title('Cumulative Sum of miRNA appearances in the examined tarBase method datasets')
    ax.set_xlabel('Number of miRNA Sequence')
    ax.set_ylabel('Percent of miRNA-target interactions')

    # add the modified legend to the plot
    plt.legend(handles=handles, labels=labels)

    fname = ROOT_PATH / Path(
        "figuers") / Path(name)
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

# cumulative_sum_miRNA_tarBase()
cumulative_sum_miRNA_tarBase(nirmol=True)

def cumulative_sum_miRNA_cross_analysis(nirmol=False):

    chossing_methods = ['tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMirna",
                        "mockMrna_di_mrna", "mockMrna_di_site", "mockMrna_di_fragment_mockMirna",
                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data_random"]

    df_tarBase = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/tarBase/tarBase_microArray_human_negative_features.csv")
    df_darnell = read_csv("/sise/home/efrco/efrco-master/data/positive_interactions/positive_interactions_new/featuers_step/darnell_human_ViennaDuplex_features.csv")
    # df_non_overlapping_sites = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites/non_overlapping_sites_darnell_human_ViennaDuplex_negative_features.csv")
    # df_non_overlapping_sites_random = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites/non_overlapping_sites_random_darnell_human_ViennaDuplex_negative_features.csv")
    # df_mock_Mirna = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMirna/mockMirna_darnell_human_ViennaDuplex_negative_features.csv")
    # df_mockMrna_di_mrna = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method1_darnell_human_ViennaDuplex_negative_features.csv")
    # df_mockMrna_di_site = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method2_darnell_human_ViennaDuplex_negative_features.csv")
    # df_mockMrna_di_fragment_mockMirna=read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method3_darnell_human_ViennaDuplex_negative_features.csv")
    df_clip = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/clip_interaction/clip_interaction_clip_3_negative_features.csv")
    df_non_overlapping_sites_clip_data_random = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites_clip_data/non_overlapping_sites_clip_data_random_darnell_human_ViennaDuplex_negative_features.csv")


    dfs = [df_darnell,df_tarBase, df_clip,df_non_overlapping_sites_clip_data_random]
    name_dfs = ["h3-positive interactions", "TarBase_microarray",  "CLIP_non_CLASH",
                "NPS_CLIP_Random"
                ]
    palette = sns.color_palette()

    # create figure and axes objects
    fig, ax = plt.subplots(figsize=(10, 6))
    handles, labels = plt.gca().get_legend_handles_labels()

    # iterate over dataframes
    for i, df in enumerate(dfs):
        # calculate cumulative sum of unique miRNAs
        if nirmol:
            name = "Cumulative_sum_nirmol.png"
            cumsum = np.cumsum(df['miRNA ID'].value_counts() / df.shape[0])
        else:
            name = "Cumulative_sum.png"

            cumsum = np.cumsum(df['miRNA ID'].value_counts())
        # calculate percentage of interactions represented by each miRNA
        cumsum_pct = cumsum / cumsum[-1] * 100

        # find minimum number of miRNAs needed to represent 90% of interactions
        min_mirnas = np.argmax(cumsum_pct >= 90)

        # plot cumulative sum with filled circle indicating minimum miRNAs needed for 90% representation
        x = np.arange(len(cumsum)) + 1
        y = cumsum
        ax.plot(x, y, label=name_dfs[i], linewidth=2, color=palette[i])
        ax.plot(min_mirnas + 1, cumsum[min_mirnas], 'o', fillstyle='full', markersize=8,color=palette[i])
        line = Line2D([0], [0], color=palette[i], linewidth=2, marker='o', markersize=8, markerfacecolor=palette[i],
                      markeredgewidth=2, markeredgecolor=palette[i])

        # add the Line2D object to the legend
        handles.append(line)
        labels.append(name_dfs[i])

    # set plot title and axis labels
    # ax.set_title('Cumulative Sum of miRNA appearances in the examined tarBase method datasets')
    ax.set_xlabel('Number of miRNA Sequence')
    ax.set_ylabel('Percent of miRNA-target interactions')

    # add legend to plot
    plt.legend(handles=handles, labels=labels)

    # display plot

    fname = ROOT_PATH / Path(
        "figuers") / Path(name)
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()


# cumulative_sum_miRNA_cross_analysis()
# cumulative_sum_miRNA_cross_analysis(nirmol=True)


def give_seed_type(row):
    if row["num_of_pairs"] <= 10:
        if row['Seed_match_canonical']:
            return "Canonical seed, Low density"
        else:
            return "Non-Canonical seed, Low density"

    if row["num_of_pairs"] >= 11 and row["num_of_pairs"] <=16 :
        if row['Seed_match_canonical']:
            return "Canonical seed, Medium density"
        else:
            return "Non-Canonical seed, Medium density"
    if row["num_of_pairs"] >= 16:
        if row['Seed_match_canonical']:
            return "Canonical seed, High density"
        else:
            return "Non-Canonical seed, High density"


def interaction_iterator(mir_inter,mrna_inter):
    for i in range(len(mir_inter)):
        if mir_inter[i] != ' ':
            yield i, mrna_inter[i] + mir_inter[i]

def count(mir_inter, mrna_inter):
    return sum(1 for _ in interaction_iterator(mir_inter, mrna_inter))

# from matplotlib import colors as mcolors


def canon_distribution_figures_dataset_with_seed_type(data_set_pos):
    dir = NEGATIVE_DATA_PATH
    all_frames = []
    chossing_methods = [
        "mockMiRNA",
        "Mockmrna_di_mockmirna", "mockMRNA_di_mRNA", "Mockmrna_di_site",
        "non_overlapping_sites", "non_overlapping_sites_random",
        'tarBase_microArray',
        "clip_interaction_clip_3",
        "non_overlapping_sites_clip_data_random", 'h3-positive_interactions']

    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        if name == "h3-positive_interactions":
           continue
        chossing_methods[i] = name.capitalize()
    for order_method in chossing_methods:


        for method_dir in dir.iterdir():
            print(method_dir)
            list_method = ['mockMirna', 'non_overlapping_sites']

            for dataset_file_neg in method_dir.glob("*features*"):


                    if any(method in method_dir.stem for method in list_method):
                        correct_dataset = data_set_pos.split("_features")[0] + "_negative_features"
                        list_split = dataset_file_neg.stem.split("/")
                        list_split = [str(s) for s in list_split]
                        if correct_dataset not in list_split[0]:
                            continue
                    name_dataset = clean_name(dataset_file_neg.stem.split("/")[-1])
                    name_dataset = name_dataset.replace("__", "_")
                    if name_dataset != "non_overlapping_sites_clip_data_random":
                        name_dataset = name_dataset[:-1]
                    if name_dataset not in chossing_methods:
                        print(name_dataset)
                        continue
                    if order_method != name_dataset:
                        continue

                    else:

                        neg = read_csv(dataset_file_neg)
                        name_dataset = conver_name(name_dataset)
                        print("%%%", name_dataset)
                        neg['dataset_name'] = name_dataset
                        all_frames.append(neg)

    for i in range(len(chossing_methods)):
        name = conver_name(chossing_methods[i])
        chossing_methods[i] = name
    # add darnell
    path_pos = read_csv("/sise/home/efrco/efrco-master/data/positive_interactions/positive_interactions_new/featuers_step/darnell_human_ViennaDuplex_features.csv")
    path_pos['dataset_name'] = 'h3-positive_interactions'
    all_frames.append(path_pos)

    result = pd.concat(all_frames)
    result['num_of_pairs'] = result.apply(lambda row: count(row['mir_inter'], row['mrna_inter']), axis=1)
    result['Seed_match_canonical'] = result.apply(lambda row: True if row['Seed_match_canonical'] else False,
                                          axis=1)
    result['Seed_match_noncanonical'] = result.apply(lambda row: True if row['Seed_match_noncanonical'] else False,
                                             axis=1)
    result['type_seed'] = result.apply(lambda row: give_seed_type(row
                                                                  ), axis=1)
    result = result[['Seed_match_canonical', 'Seed_match_noncanonical', 'dataset_name', 'type_seed']]

    # Set 'dataset_name' column as Categorical with the custom order
    result['dataset_name'] = pd.Categorical(result['dataset_name'], categories=chossing_methods, ordered=True)




    color_dict = {
                "Canonical seed, High density": 'royalblue',
                "Canonical seed, Medium density": 'skyblue',
                "Canonical seed, Low density": 'paleturquoise',
                "Non-Canonical seed, High density": 'indianred',
                "Non-Canonical seed, Medium density": 'lightcoral',
                "Non-Canonical seed, Low density": 'rosybrown'}



    cross_tab_prop = pd.crosstab(index=result['dataset_name'],
                                 columns=result['type_seed'],
                                 normalize="index" )

    cross_tab = pd.crosstab(index=result['dataset_name'],
                            columns=result['type_seed'])
    cross_tab_prop = cross_tab_prop.reindex(color_dict.keys(), axis=1)

    cross_tab_prop.plot(kind='bar',
                        stacked=True,
                        colormap='tab20c',
                        figsize=(10, 6), width= 0.9, color= color_dict.values())


    for n, x in enumerate([*cross_tab.index.values]):
        print(x)
        for (proportion, y_loc) in zip(cross_tab_prop.loc[x],
                                       cross_tab_prop.loc[x].cumsum()):
            if proportion < 0.01:
                continue
            plt.text(x=n,
                     y=(y_loc - proportion) + (proportion / 2),
                     s=f'{np.round(proportion * 100, 1)}',
                     color="black",
                     fontsize=10,
                     fontweight="bold",  ha='center', va='center')

    for n, x in enumerate([*cross_tab.index.values]):
            table = cross_tab.loc[x]
            type1 = table['Non-Canonical seed, Medium density'] + table['Canonical seed, Medium density']
            type2 = table['Non-Canonical seed, Low density'] + table['Canonical seed, Low density']
            type3 = table['Non-Canonical seed, High density'] + table['Canonical seed, High density']

            total = type1 + type2 + type3

            plt.text(x=n,
                     y=y_loc+0.02,
                     s=f'{total}',
                     color="black",
                     fontsize=12,
                     fontweight="bold",  ha='center', va='center')

    # plt.legend(loc='upper right')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Percentage(%)")
    fname = path_figures / "canon_distribution_figures_dataset_seedtype.png"
    plt.savefig(fname, format="PNG",dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()


def canon_distribution_figures_dataset(data_set_pos):
    dir = NEGATIVE_DATA_PATH
    all_frames = []
    chossing_methods = ['tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMiRNA",
                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment","mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3",
                        "non_overlapping_sites_clip_data_random"]

    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        chossing_methods[i] = name.capitalize()

    for method_dir in dir.iterdir():
        print(method_dir)
        list_method = ['mockMirna', 'non_overlapping_sites']
        for current_method in chossing_methods:

            for dataset_file_neg in method_dir.glob("*features*"):
                if any(method in method_dir.stem for method in list_method):
                    correct_dataset = data_set_pos.split("_features")[0] + "_negative_features"
                    list_split = dataset_file_neg.stem.split("/")
                    list_split = [str(s) for s in list_split]
                    if correct_dataset not in list_split[0]:
                        continue
                x = dataset_file_neg.stem.split("/")[-1]
                name_dataset = clean_name(dataset_file_neg.stem.split("/")[-1])
                name_dataset = name_dataset.replace("__", "_")
                if name_dataset not in chossing_methods:
                    print(name_dataset)
                    continue
                if name_dataset != current_method:
                    continue
                name_dataset = conver_name(name_dataset)
                neg = read_csv(dataset_file_neg)
                neg['dataset_name'] = name_dataset
                all_frames.append(neg)

    # add darnell
    path_pos = read_csv("/sise/home/efrco/efrco-master/data/positive_interactions/positive_interactions_new/featuers_step/darnell_human_ViennaDuplex_features.csv")
    # path_pos = read_csv("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/benorgi/TPVOD/Data/Features/CSV/human_dataset3_duplex_positive_feature.csv")
    path_pos['dataset_name'] = 'positive_interactions'
    all_frames.append(path_pos)

    result = pd.concat(all_frames)
    result = result[['Seed_match_canonical', 'Seed_match_noncanonical', 'dataset_name']]
    result['type_seed'] = result.apply(lambda row: 'canonical' if row['Seed_match_canonical'] else 'noncanonical', axis = 1)
    result.set_index('dataset_name')

    cross_tab_prop = pd.crosstab(index=result['dataset_name'],
                                 columns=result['type_seed'],
                                 normalize="index")
    cross_tab = pd.crosstab(index=result['dataset_name'],
                            columns=result['type_seed'])

    cross_tab_prop.plot(kind='bar',
                        stacked=True,
                        colormap='tab10',
                        figsize=(10, 6), color=['red', 'skyblue'], width= 0.9)

    for n, x in enumerate([*cross_tab.index.values]):
        for (proportion, y_loc) in zip(cross_tab_prop.loc[x],
                                       cross_tab_prop.loc[x].cumsum()):
            plt.text(x=n,
                     y=(y_loc - proportion) + (proportion / 2),
                     s=f'{np.round(proportion * 100, 1)}%',
                     color="black",
                     fontsize=10,
                     fontweight="bold",  ha='center', va='center')

    for n, x in enumerate([*cross_tab.index.values]):
        table = cross_tab.loc[x]
        canon = table['canonical']
        noncanon = table['noncanonical']
        total = canon + noncanon

        plt.text(x=n,
                 y=y_loc,
                 s=f'{total}',
                 color="black",
                 fontsize=12,
                 fontweight="bold",  ha='center', va='center')

    # plt.legend(loc='upper right')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Percentage(%)")
    fname = path_figures / "canon_distribution_figures_dataset.png"
    plt.savefig(fname, format="PNG",dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()


# canon_distribution_figures_dataset(data_set_pos="darnell_human_ViennaDuplex_features")
canon_distribution_figures_dataset_with_seed_type(data_set_pos="darnell_human_ViennaDuplex_features")

####################################KL######################################

def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence

import math
def KL_train_full_data_gilad():
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
    res_table = pd.DataFrame()

    df_tarBase = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/tarBase/tarBase_microArray_human_negative_features.csv")
    df_darnell = read_csv(
        "/sise/home/efrco/efrco-master/data/positive_interactions/positive_interactions_new/featuers_step/darnell_human_ViennaDuplex_features.csv")
    df_non_overlapping_sites = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites/non_overlapping_sites_darnell_human_ViennaDuplex_negative_features.csv")
    df_non_overlapping_sites_random = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites/non_overlapping_sites_random_darnell_human_ViennaDuplex_negative_features.csv")
    df_mock_Mirna = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMirna/mockMirna_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_mrna = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method1_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_site = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method2_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_fragment_mockMirna=read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method3_darnell_human_ViennaDuplex_negative_features.csv")
    df_clip = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/clip_interaction/clip_interaction_clip_3_negative_features.csv")
    df_non_overlapping_sites_clip_data_random = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites_clip_data/non_overlapping_sites_clip_data_random_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_fragment_mockMirna['miRNA ID'] = df_mockMrna_di_fragment_mockMirna['miRNA ID'].apply(lambda text: "mock " + text)

    dfs = [df_mock_Mirna,
           df_mockMrna_di_fragment_mockMirna, df_mockMrna_di_mrna, df_mockMrna_di_site,
           df_non_overlapping_sites, df_non_overlapping_sites_random,
           df_tarBase,
           df_clip, df_non_overlapping_sites_clip_data_random]

    name_dfs = [
                "mockMiRNA", "mockMRNA_di_fragment_mockMiRNA",
                "mockMRNA_di_mRNA*", "mockMRNA_di_fragment*",
                "non_overlapping_sites*", "non_overlapping_sites_random*",
                'tarBase_microArray',
                "clip_interaction_clip_3",
                "non_overlapping_sites_clip_data_random"]


    for i in range(len(name_dfs)):
        name = name_dfs[i]
        name_dfs[i] = name.capitalize()

    for i_train, df_train in enumerate(dfs):
        # calculate cumulative sum of unique miRNAs
        for i_test, df_test in enumerate(dfs):

            name = "Cumulative_sum_nirmol.png"
            train_weight = df_train['miRNA ID'].value_counts(normalize=True)
            test_weight = df_test['miRNA ID'].value_counts(normalize=True)
            # train_weight = df_train['miRNA ID'].value_counts(normalize=True)
            # test_weight = df_test['miRNA ID'].value_counts(normalize=True)
            join_result = pd.merge(train_weight.to_frame(), test_weight.to_frame(), how='outer',
                                   left_index=True, right_index=True)
            join_result.fillna(0, inplace=True)
            p = join_result.iloc[:, 0].values
            q = join_result.iloc[:, 1].values
            kl_divergence = KL(p, q)

            # Calculate the KL divergence between the two datasets
            print("###################################################################################")
            print(name_dfs[i_train])
            print(name_dfs[i_test])
            name_train = name_dfs[i_train]
            name_test = name_dfs[i_test]
            if "*" in name_train:
                name_train = conver_name(name_train[:-1]) + "*"
            else:
                name_train = conver_name(name_train)
            if "*" in name_test:
                name_test = conver_name(name_test[:-1]) + "*"
            else:
                name_test = conver_name(name_test)
            res_table.loc[name_test, name_train] = round(kl_divergence, 3)
            if name_test == "Mock_miRNA" or  name_train=="Mock_miRNA":
                res_table.loc[name_test,name_train] = -1
            if name_test == "Mock_di_fragment_&_miRNA" or name_train == "Mock_di_fragment_&_miRNA":
                res_table.loc[name_test, name_train] = -1

            print(f"The KL divergence between the two datasets is: {kl_divergence:.4f}")
    cmap = sns.cm.rocket_r
    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(res_table, annot=True,  linewidth=(1.5,0), cmap = cmap)
    # sns.color_palette("rocket_r", as_cmap=True)
    # sns.color_palette("Spectral", as_cmap=True)
    # ax.invert_yaxis()
    ax.collections[0].set_clim(vmax=res_table.values.min(), vmin=res_table.values.max())
    ax.set_aspect('equal')
    ax.set(xlabel="Source", ylabel="Target")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=150)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") /"KL_subset.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()
# KL_train_full_data_gilad()


def KL_MRNA_full_data_gilad():
    res_table = pd.DataFrame()

    df_tarBase = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/tarBase/tarBase_microArray_human_negative_features.csv")
    df_darnell = read_csv(
        "/sise/home/efrco/efrco-master/data/positive_interactions/positive_interactions_new/featuers_step/darnell_human_ViennaDuplex_features.csv")
    df_non_overlapping_sites = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites/non_overlapping_sites_darnell_human_ViennaDuplex_negative_features.csv")
    df_non_overlapping_sites_random = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites/non_overlapping_sites_random_darnell_human_ViennaDuplex_negative_features.csv")
    df_mock_Mirna = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMirna/mockMirna_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_mrna = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method1_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_site = read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method2_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_fragment_mockMirna=read_csv("/sise/home/efrco/efrco-master/data/negative_interactions/mockMrna/mockMrna_denucleotides_method3_darnell_human_ViennaDuplex_negative_features.csv")
    df_clip = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/clip_interaction/clip_interaction_clip_3_negative_features.csv")
    df_non_overlapping_sites_clip_data_random = read_csv(
        "/sise/home/efrco/efrco-master/data/negative_interactions/non_overlapping_sites_clip_data/non_overlapping_sites_clip_data_random_darnell_human_ViennaDuplex_negative_features.csv")
    df_mockMrna_di_fragment_mockMirna['miRNA ID'] = df_mockMrna_di_fragment_mockMirna['miRNA ID'].apply(lambda text: "mock " + text)

    dfs = [df_tarBase,df_mock_Mirna, df_non_overlapping_sites, df_non_overlapping_sites_random,
           df_mockMrna_di_mrna,df_mockMrna_di_site,
           df_mockMrna_di_fragment_mockMirna , df_clip, df_non_overlapping_sites_clip_data_random]

    name_dfs = ['tarBase_microArray',
                "mockMiRNA", "non_overlapping_sites*", "non_overlapping_sites_random*",

                "mockMRNA_di_mRNA*", "mockMRNA_di_fragment*", "mockMRNA_di_fragment_mockMiRNA",
                "clip_interaction_clip_3",
                "non_overlapping_sites_clip_data_random"]

    name_dfs_star = [
                "non_overlapping_sites", "non_overlapping_sites_random",

                "mockMRNA_di_mRNA", "mockMRNA_di_fragment"]



    for i in range(len(name_dfs)):
        name = name_dfs[i]
        name_dfs[i] = name.capitalize()

    for i_train, df_train in enumerate(dfs):
        # calculate cumulative sum of unique miRNAs
        for i_test, df_test in enumerate(dfs):

            name = "Cumulative_sum_nirmol.png"
            train_weight = df_train['Gene_ID'].value_counts(normalize=True)
            test_weight = df_test['Gene_ID'].value_counts(normalize=True)

            join_result = pd.merge(train_weight.to_frame(), test_weight.to_frame(), how='outer',
                                   left_index=True, right_index=True)
            join_result.fillna(0, inplace=True)
            p = join_result.iloc[:, 0].values
            q = join_result.iloc[:, 1].values
            kl_divergence = KL(p, q)

            # Calculate the KL divergence between the two datasets
            print("###################################################################################")
            print(name_dfs[i_train])
            print(name_dfs[i_test])
            res_table.loc[name_dfs[i_test], name_dfs[i_train]] = round(kl_divergence, 3)

            print(f"The KL divergence between the two datasets is: {kl_divergence:.4f}")
    cmap = sns.cm.rocket_r

    ax = sns.heatmap(res_table, annot=True,  linewidth=(2.5,0), cmap = cmap)
    # sns.color_palette("rocket_r", as_cmap=True)
    # sns.color_palette("Spectral", as_cmap=True)
    # ax.invert_yaxis()
    ax.collections[0].set_clim(vmax=res_table.values.min(), vmin=res_table.values.max())

    ax.set(xlabel="train", ylabel="test")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=215)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") /"KL_mrna.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()
# KL_MRNA_full_data_gilad()

###########################################IMPORTANCE - INTRA##############################################################


def model_shap_plot(test, model, model_name,s_org,d_org,name_classifiers, dependence_feature=None):


    explainer = shap.Explainer(model.predict, test)
    shap_values = explainer(test, max_evals=(2*(len(test.columns)+1)))
    s_org = clean_name(s_org.split("_train_")[0])

    shap.plots.bar(shap_values, show=False, max_display=11)
    s_org = conver_name(clean_name(s_org)[:-1])
    d_org = conver_name(clean_name(d_org)[:-1])
    plt.title(f"Intra-analysis for Dataset : {s_org}")
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/shap_importance_Intra/") / f"{s_org}.png"
    plt.gca().tick_params(axis="y", pad=280)
    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.clf()
    print("###############option2##################")
    feature_importance_values = np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(test.columns, feature_importance_values)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    print(feature_importance.head())
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/shap_importance_Intra/") / f"{s_org}_{d_org}.csv"
    to_csv(feature_importance, fname)


# This function response on load the clf from Result dir
def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}_{method}.model"
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf

def feature_importance_intra(method_split: str, model_dir: str, number_iteration: int, name_classifier: str):
    ms_table = None
    results_dir = ROOT_PATH / Path("Results")
    number_iteration = str(number_iteration)
    results_dir_models = ROOT_PATH / Path("Results/models") / model_dir / number_iteration
    test_dir = DATA_PATH_INTERACTIONS / "test" / method_split / number_iteration

    res_table: DataFrame = pd.DataFrame()
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()

    clf_datasets = [f.stem.split("_" + name_classifier)[0] for f in results_dir_models.glob("*.model")]
    method = name_classifier
    for clf_dataset in clf_datasets:
        for f_test in test_dir.glob("*test*"):
            f_stem = f_test.stem
            test_dataset = f_stem.split(".csv")[0]
            print(f"clf: {clf_dataset} train: {test_dataset}, method: {method}")
            if clf_dataset.replace("train", "test") != test_dataset:
                continue
            clf = get_presaved_clf(results_dir_models, clf_dataset, method)
            X_train, y_train = feature_reader.file_reader(test_dir / f"{test_dataset}.csv")
            # X_train = X_train[:10]
            model_shap_plot(X_train, clf, method, clf_dataset, test_dataset, name_classifier,
                            dependence_feature=None)

            print("####################################################################################3")

# # #
# feature_importance_intra(method_split="underSampling", model_dir="models_underSampling",
#                          number_iteration=0, name_classifier='xgbs') #7844667

def get_top_ten(n=30):
    chossing_methods = ['tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMiRNA",
                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data_random"]

    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        chossing_methods[i] = name.capitalize()
    dir_path = Path("/sise/home/efrco/efrco-master/Results/figuers/xgbs/shap_importance/")
    top_features = []
    number_dataset = 0
    for path_features in dir_path.glob("*.csv*"):
        features = read_csv(path_features)
        name_dataset = clean_name(path_features.stem.split("/")[-1].split("_train")[0])
        name_dataset = name_dataset.replace("__", "_")
        if name_dataset not in chossing_methods:
            print("################################################################")
            print(name_dataset)
            continue
        name_dataset = conver_name(name_dataset)

        number_dataset = number_dataset + 1

        current_dataset_top_features = list(features.sort_values(by=['feature_importance_vals'],ascending=False).head(n).col_name)
        top_features.extend(current_dataset_top_features)
    print(number_dataset)
    my_dict = {}
    for item in top_features:
        if item in my_dict:
            my_dict[item] += 1
        else:
            my_dict[item] = 1
    sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))

    plt.bar(list(sorted_dict.keys()), list(sorted_dict.values()))
    plt.xlabel('')
    plt.ylabel('Count')
    plt.title('Fruit Count (Sorted Descending)')
    plt.xticks([])  # Hide x-axis tick labels
    plt.show()

    my_dict_values = {}
    for item in my_dict.values():
        if item in my_dict_values:
            my_dict_values[item] += 1
        else:
            my_dict_values[item] = 1
    my_dict_values_sort = dict(sorted(my_dict_values.items(), key=lambda item: item[1], reverse=True))

    plt.plot(list(my_dict_values_sort.values()), list(my_dict_values_sort.keys()))
    plt.show()


def normalize_vector(df_src):
    df_src["value"] = (df_src['value'] - df_src['value'].min()) / (df_src['value'].max() - df_src['value'].min())
    return df_src

def get_top_ten_stat(n=50):
    # chossing_methods = ['tarBase_microArray',
    #                     "non_overlapping_sites", "non_overlapping_sites_random",
    #                     "mockMiRNA",
    #                     "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
    #                     "clip_interaction_clip_3_",
    #                     "non_overlapping_sites_clip_data_random"]
    chossing_methods = [
        "mockMiRNA", "mockMRNA_di_fragment_mockMiRNA",
        "mockMRNA_di_fragment", "mockMRNA_di_mRNA",
        "non_overlapping_sites", "non_overlapping_sites_random",
        'tarBase_microArray',
        "clip_interaction_clip_3_",
        "non_overlapping_sites_clip_data_random"]

    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        chossing_methods[i] = conver_name(name.capitalize())

    dir_path = Path("/sise/home/efrco/efrco-master/Results/figuers/xgbs/shap_importance_Intra/")

    res_table = pd.DataFrame()

    for method in chossing_methods:

        for path_features in dir_path.glob("*.csv*"):
            features = read_csv(path_features)
            name_dataset = clean_name(path_features.stem.split("/")[-1].split("_train")[0])
            name_dataset = conver_name(name_dataset.replace("__", "_"))

            if method != name_dataset:
                print("################################################################")
                print(name_dataset)
                continue
            if name_dataset not in chossing_methods:
                continue
            for method2 in chossing_methods:
                for path_features_2 in dir_path.glob("*.csv*"):
                    top_features = set()
                    features2 = read_csv(path_features_2)
                    name_dataset2 = clean_name(path_features_2.stem.split("/")[-1].split("_train")[0])
                    name_dataset2 = conver_name(name_dataset2.replace("__", "_"))
                    if name_dataset2 != method2:
                        print("################################################################")
                        print(name_dataset)
                        continue
                    if name_dataset not in chossing_methods:
                        continue
                    if name_dataset in res_table.index and name_dataset2 in res_table.columns and not pd.isna(
                            res_table.loc[name_dataset, name_dataset2]):
                        continue
                    if name_dataset == name_dataset2:
                        res_table.loc[name_dataset2, name_dataset] = 0
                        continue


                    current_dataset_top_features_1 = features.sort_values(by=['feature_importance_vals'], ascending=False)[:n]
                    current_dataset_top_features_2 = features2.sort_values(by=['feature_importance_vals'], ascending=False)[:n]

                    merged_df = pd.merge(current_dataset_top_features_1, current_dataset_top_features_2, on='col_name', how='outer')

                    # create a new data frame with three columns, i get all the name of
                    new_df = pd.DataFrame(
                        {'col_name': merged_df['col_name']})


                    new_df = pd.merge(new_df, features, on='col_name')
                    new_df = pd.merge(new_df, features2, on='col_name')
                    new_df.rename(columns={'feature_importance_vals_x': 'feature_values_1',
                         'feature_importance_vals_y': 'feature_values_2'}, inplace=True)
                    new_df['feature_values_1'] = (new_df['feature_values_1'] - new_df['feature_values_1'].min()) / (
                                new_df['feature_values_1'].max() - new_df['feature_values_1'].min())
                    new_df['feature_values_2'] = (new_df['feature_values_2'] - new_df['feature_values_2'].min()) / (
                            new_df['feature_values_2'].max() - new_df['feature_values_2'].min())

                    statistic, p_value = wilcoxon(new_df['feature_values_1'], new_df['feature_values_2'], alternative='two-sided')
                    print("#################################################################################")
                    print(name_dataset)
                    print(name_dataset2)
                    print("Wilcoxon rank-sum test:")
                    print("Statistic: ", statistic)
                    print("p-value: ", p_value)
                    # name_dataset = conver_name(name_dataset)
                    # name_dataset2 = conver_name(name_dataset2)
                    if p_value < 0.05:
                        res_table.loc[name_dataset2, name_dataset] = -math.log10(p_value)
                    else:
                        res_table.loc[name_dataset2, name_dataset] = 0

    # for i in range(len(chossing_methods)):
    #     name = conver_name(chossing_methods[i])
    #     chossing_methods[i] = name
    res_table = res_table.reindex(columns=chossing_methods)

    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table, annot=True, cmap=cmap, yticklabels=chossing_methods)

    # sns.color_palette("Spectral", as_cmap=True)
    # ax.set(xlabel="train", ylabel="test")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=215)
    plt.yticks(ha='left')
    # plt.show()
    fname = ROOT_PATH / Path("figuers") / Path(f"heatmap_cross_features_{n}.png")
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)


# get_top_ten_stat(n=50)

#######################################IMPORTANCE CROSS########################################################################

def feature_importance_plot(df, s_org, d_org):

    # Plot the zoom df
    zoom_df = df.head(10)
    ax = zoom_df.plot.bar(x="col_name", y="value", rot=90)
    plt.xlabel('Features')
    plt.ylabel('Difference')
    plt.xticks(rotation=30, ha='right')

    plt.title(f"A: {s_org}-B: {d_org}")
    fname = ROOT_PATH / Path(f"Results/figuers/xgbs/shap_importance_cross/importance_fig") / f"{s_org}_test_{d_org}.png"
    # plt.show()
    plt.savefig(fname, format="png", dpi=300, bbox_inches='tight')


def creat_df_merge(featureA, featuresB, n=25):
    current_dataset_top_features_1 = featureA.sort_values(by=['feature_importance_vals'], ascending=False)[:n]
    current_dataset_top_features_2 = featuresB.sort_values(by=['feature_importance_vals'], ascending=False)[:n]

    merged_df = pd.merge(current_dataset_top_features_1, current_dataset_top_features_2, on='col_name', how='outer')
    # Filter out rows where "col_name" appears in both dataframes
    # merged_df = merged_df[(~merged_df['col_name'].isin(current_dataset_top_features_1['col_name'])) |
    #                       (~merged_df['col_name'].isin(current_dataset_top_features_2['col_name']))]
    # create a new data frame with three columns, i get all the name of
    new_df = pd.DataFrame(
        {'col_name': merged_df['col_name']})

    new_df = pd.merge(new_df, featureA, on='col_name')
    new_df = pd.merge(new_df, featuresB, on='col_name')
    new_df.rename(columns={'feature_importance_vals_x': 'feature_values_1',
                           'feature_importance_vals_y': 'feature_values_2'}, inplace=True)
    return new_df


def model_shap_plot_cross(test, modelA, modelB, s_org,d_org,name_classifiers, dependence_feature=None):

    shap_valuesA = shap.TreeExplainer(modelA).shap_values(test.values.astype('float'))
    shap_valuesB = shap.TreeExplainer(modelB).shap_values(test.values.astype('float'))
    feature_importance_valuesA = np.abs(shap_valuesA).mean(0)
    feature_importance_valuesB = np.abs(shap_valuesB).mean(0)
    feature_importanceA = pd.DataFrame(list(zip(test.columns, feature_importance_valuesA)), columns=['col_name', 'feature_importance_vals'])
    feature_importanceB = pd.DataFrame(list(zip(test.columns, feature_importance_valuesB)), columns=['col_name', 'feature_importance_vals'])

    df_top_feauers = creat_df_merge(feature_importanceA,feature_importanceB)

    shap_valuesA = np.abs(shap_valuesA)
    shap_valuesB = np.abs(shap_valuesB)
    sub_res = np.absolute(np.subtract(shap_valuesA,shap_valuesB))

    add_res = np.add(shap_valuesA,shap_valuesB)
    # get the average percentage of differences
    div_res = np.divide(sub_res,add_res).mean(axis=0)

    new_df = pd.DataFrame({"value":div_res, "col_name":test.columns}).sort_values("value",ascending=False)

    new_df = new_df[new_df['value'] < 1]
    # new_df = pd.DataFrame({"value":sub_res, "col_name":test.columns}).sort_values("value",ascending=False)
    # new_df = pd.merge(new_df, feature_importanceA, on='col_name')
    # new_df = pd.merge(new_df, feature_importanceB, on='col_name')
    # new_df.rename(columns={'feature_importance_vals_x': 'feature_values_1',
    #                        'feature_importance_vals_y': 'feature_values_2'}, inplace=True)

    new_df = pd.merge(new_df, df_top_feauers, on='col_name')

    # new_df.rename(columns={'feature_importance_vals_x': 'feature_values_1',
    #                        'feature_importance_vals_y': 'feature_values_2'}, inplace=True)

    # print(df.head())
    s_org = clean_name(s_org.split("_train")[0])
    d_org = clean_name(d_org.split("_train")[0])
    if s_org=='Mockmirna_' and d_org=='Tarbase_microarray_':
        print("k")
    feature_importance_plot(new_df, s_org, d_org)
    s_org = conver_name(s_org)
    d_org = conver_name(d_org)
    fname = ROOT_PATH / Path(
        f"Results/figuers/{name_classifiers}/shap_importance_cross/importance_list") / f"{s_org}_test_{d_org}.csv"
    to_csv(new_df, fname)

def normalize_vector(df_src):
    df_src = (df_src - df_src.min()) / (df_src.max() - df_src.min())
    return df_src

# from tableone import TableOne
def tabe_one(shap_train, shap_test, test, best_featuers=None):
    # Assuming your SHAP matrix is called `shap_values_train`
    num_rows = shap_train.shape[0]  # Get the number of rows in the SHAP matrix
    name_value = "train"  # The value you want to set for the new column

    # Create a new numpy ndarray containing the name value, with the same number of rows as the SHAP matrix
    name_column = np.full((num_rows, 1), name_value)

    # Concatenate the name column with the SHAP matrix along the second axis
    shap_train = np.concatenate((name_column, shap_train), axis=1)

    name_value = "test"  # The value you want to set for the new column
    name_column = np.full((num_rows, 1), name_value)
    shap_test = np.concatenate((name_column, shap_test), axis=1)

    df_combined = np.concatenate((shap_train, shap_test), axis=0)
    columns= list(['name'])
    columns.extend(list(test.columns))
    shap_values_df = pd.DataFrame(df_combined, columns=columns)
    print("###############################################################", len(best_featuers))


    ##################################################################
    columns = best_featuers
    nonnormal =best_featuers
    groupby = ['name']
    categorical = []
    mytable = TableOne(shap_values_df, columns=columns,categorical=categorical, groupby=groupby, nonnormal=nonnormal,
                       pval=True, htest_name=True)
    # print(mytable.tabulate(tablefmt="fancy_grid"))


    mytable.to_latex('mytable.tex')
    fn1 = '/sise/home/efrco/efrco-master/figuers/tableone.csv'
    mytable.to_csv(fn1)

    df = read_csv(fn1)
    print(df['Grouped by name.4'])
    energy_mef_p_value = df.loc['Energy_MEF_Duplex', 'test/train']

    significant_rows = df[df['P-Value'] < 0.05]

    num_significant_rows = significant_rows.shape[0]
    print(num_significant_rows)


def sagnificant_featuers(shap_train, shap_test, test, best_featuers=None):
    feature_to_col = {feat_name: col_idx for col_idx, feat_name in enumerate(test.columns)}


    countuer= 0
    for feature_mame in best_featuers:
        # get the SHAP values for the selected columns
        selected_cols = [feature_to_col[feature_mame]]

        selected_shap_values_train = shap_train[:, selected_cols]
        selected_shap_values_test = shap_test[:, selected_cols]

        normalized_src_df = (selected_shap_values_train - selected_shap_values_train.min()) / (selected_shap_values_train.max() - selected_shap_values_train.min())
        normalized_trg_df = (selected_shap_values_test - selected_shap_values_test.min()) / (selected_shap_values_test.max() -selected_shap_values_test.min())

        # normalized_src_df = (selected_shap_values_train - np.mean(selected_shap_values_train)) / np.std(
        #     selected_shap_values_train)
        # normalized_trg_df = (selected_shap_values_test - np.mean(selected_shap_values_test)) / np.std(
        #     selected_shap_values_test)

        # res = stats.kruskal(normalized_src_df.flatten(), normalized_trg_df.flatten()).pvalue
        res = stats.ranksums(normalized_src_df, normalized_trg_df).pvalue

        print(feature_mame)
        print(res)
        if res < 0.01:
            countuer = countuer +1


    print(countuer)
    print("#########################")
    return countuer/len(best_featuers)



def model_shap_plot_cross_specfic_sample(test, y_true, modelTest, modelTrain, test_name, train_name, name_classifiers, dependence_feature=None):
    test_name = conver_name(clean_name(test_name.split("_train")[0]))
    train_name = conver_name(clean_name(train_name.split("_train")[0]))

    shap_values_train = shap.TreeExplainer(modelTrain).shap_values(test.values.astype('float'))
    shap_values_test = shap.TreeExplainer(modelTest).shap_values(test.values.astype('float'))

    # Find indices of correctly classified negative records
    # correct_negative_indices = np.where(np.logical_and(y_true == 0, np.sum(shap_values_test, axis=1) < 0))[0]
    # misclassified_indices = np.where(np.logical_and(y_true == 0, np.sum(shap_values_train, axis=1) > 0))[0]
    # #find the negative sample that the test classifier succeed to clasiffer correct and the train not
    # shared_vals = list(set(correct_negative_indices) & set(misclassified_indices))
    ######################################################################
    predict_test= modelTest.predict(test)
    predict_train= modelTrain.predict(test)
    correct_negative_indices = np.where((y_true == 0) & (predict_test == y_true))[0]
    misclassified_negative_indices = np.where((y_true == 0) & (predict_train != y_true))[0]
    shared_vals = list(set(correct_negative_indices) & set(misclassified_negative_indices))

    important_features =[]
    # Extract rows from the matrix where records are misclassified
    misclassified_rows = shap_values_train[shared_vals]
    print(len(misclassified_rows))
    for row in misclassified_rows:
        feature_importance_row = pd.DataFrame(list(zip(test.columns, row)), columns=['col_name', 'feature_importance_vals'])
        current_dataset_top_row = feature_importance_row.sort_values(by=['feature_importance_vals'], ascending=False)[:10]
        important_features.extend(current_dataset_top_row['col_name'])
    count_dict = {}

    # Iterate through the list and update the counts in the dictionary
    for val in important_features:
        if val in count_dict:
            count_dict[val] += 1
        else:
            count_dict[val] = 1
    sorted_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))

    # Print out the counts
    # for val, count in sorted_dict.items():
    #     print(f"{val}: {count} times")

    # feature_importance_plot(new_df, s_org, d_org)
    fname = ROOT_PATH / Path(
        f"Results/figuers/{name_classifiers}/shap_importance_cross/importance_list") / f"{test_name}_test_{train_name}.csv"
    # to_csv(new_df, fname)

    #######################################True classifers##########################################################################
    # Find indices of correctly classified negative records
    correct_negative_indices_train = np.where(np.logical_and(y_true == 0, np.sum(shap_values_test, axis=1) < 0))[0]
    correct_negative_indices_test= np.where(np.logical_and(y_true == 0, np.sum(shap_values_train, axis=1) < 0))[0]
    correct_positive_indices_test = np.where(np.logical_and(y_true == 0, np.sum(shap_values_test, axis=1) > 0))[0]
    correct_positive_indices_train = np.where(np.logical_and(y_true == 0, np.sum(shap_values_train, axis=1) > 0))[0]
    #find the negative sample that the test classifier succeed to clasiffer correct and the train not
    shared_vals_negative = list(set(correct_negative_indices_train) | set(correct_negative_indices_test))
    shared_vals_positive = list(set(correct_positive_indices_test) | set(correct_positive_indices_train))
    total = list()
    total.extend(shared_vals_positive)
    total.extend(shared_vals_negative)
    total = list(set(total))


    misclassified_rows_train = shap_values_train[total]
    misclassified_rows_test = shap_values_test[total]

    feature_importance_valuesA = np.abs(misclassified_rows_train).mean(0)
    feature_importance_valuesB = np.abs(misclassified_rows_test).mean(0)
    feature_importanceA = pd.DataFrame(list(zip(test.columns, feature_importance_valuesA)),
                                       columns=['name', 'value']).sort_values(by='value', ascending=False)
    feature_importanceB = pd.DataFrame(list(zip(test.columns, feature_importance_valuesB)),
                                       columns=['name', 'value']).sort_values(by='value', ascending=False)

    best= set(list(feature_importanceA['name'][:10]) + list(feature_importanceB['name'][:10]))


    # return sagnificant_featuers(misclassified_rows_train, misclassified_rows_test, test,best)



    shap.summary_plot(misclassified_rows_train, test.iloc[total,:], show=False, max_display=15,color_bar=True, feature_names=test.columns)
    s_org = clean_name(train_name)
    d_org = clean_name(test_name)
    plt.title(f"Train: {s_org}-Test: {d_org}")
    fname = ROOT_PATH / Path(f"/sise/home/efrco/efrco-master/figuers/cross_dataset_understand/") / f"{s_org}_{d_org}.png"
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.gca().tick_params(axis="y", pad=150)
    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

    shap.summary_plot(misclassified_rows_test, test.iloc[total, :], show=False, max_display=15,color_bar=True, feature_names=test.columns)
    s_org = conver_name(train_name)
    d_org = conver_name(test_name)
    plt.title(f"Train: {d_org}-Test: {d_org}")
    fname = ROOT_PATH / Path(
        f"/sise/home/efrco/efrco-master/figuers/cross_dataset_understand/") / f"{d_org}_{d_org}.png"
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.gca().tick_params(axis="y", pad=150)
    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

    ######################################################################


    # feature_to_col = {feat_name: col_idx for col_idx, feat_name in enumerate(test.columns)}
    #
    # selected_features = list(sorted_dict.keys())[:20]
    # countuer= 0
    # for feature_mame in selected_features:
    #     # get the SHAP values for the selected columns
    #     selected_cols = [feature_to_col[feature_mame]]
    #
    #     selected_shap_values_train = shap_values_train[:, selected_cols]
    #     selected_shap_values_test = shap_values_test[:, selected_cols]
    #
    #     normalized_src_df = (selected_shap_values_train - selected_shap_values_train.min()) / (selected_shap_values_train.max() - selected_shap_values_train.min())
    #     normalized_trg_df = (selected_shap_values_test - selected_shap_values_test.min()) / (selected_shap_values_test.max() -selected_shap_values_test.min())
    #     res = stats.wilcoxon(normalized_src_df.flatten(), normalized_trg_df.flatten()).pvalue
    #     print(feature_mame)
    #     print(res)
    #     if res<0.05:
    #         countuer = countuer +1
    #     # statisticfloat, p_value = stats.spearmanr(normalized_src_df, normalized_trg_df)
    #     # print(p_value)
    #     misclassified_rows_train = normalize_vector(shap_values_train[shared_vals,selected_cols])
    #     misclassified_rows_test = normalize_vector(shap_values_test[shared_vals,selected_cols])
    #     res = stats.wilcoxon(misclassified_rows_train.flatten(), misclassified_rows_test.flatten()).pvalue
    #     print("FP")
    #     print(res)
    #
    #     misclassified_rows_train = normalize_vector(shap_values_train[shared_vals_true,selected_cols])
    #     misclassified_rows_test = normalize_vector(shap_values_test[shared_vals_true,selected_cols])
    #
    #     res = stats.wilcoxon(misclassified_rows_train.flatten(), misclassified_rows_test.flatten()).pvalue
    #     print("TP")
    #     print(res)


    # print(countuer)
    # print("#########################")
    # return countuer
    #



# This function response on load the clf from Result dir
def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}_{method}.model"
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf
#
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

    res_table: DataFrame = pd.DataFrame()
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

            # shape from A to B
            # X_test = X_test[:10]
            test_name = clean_name(clf_dataset_A.split("_train")[0])
            train_name = clean_name(clf_dataset_B.split("_train")[0])

            if train_name == 'Tarbase_microarray_' and test_name == 'Clip_interaction_clip_3_': #worst
                print(test_name)
            elif train_name == 'Mockmirna_' and test_name == 'Non_overlapping_sites_random_': #worst
                print(test_name)
            elif train_name == 'Mockmirna_' and test_name == 'Tarbase_microarray_':  # good
                print(test_name)

            else:
                # print(train_name)
                continue
            test_name = conver_name(test_name)
            train_name = conver_name(train_name)
            res_table[train_name, test_name] = model_shap_plot_cross_specfic_sample(X_test_A,y_test, modelTest=clfA, modelTrain=clfB, test_name=clf_dataset_A, train_name=clf_dataset_B, name_classifiers=name_classifier,
                            dependence_feature=None)


            print("####################################################################################")
    # res_table = res_table.reindex(columns=chossing_methods)

    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table, annot=True, cmap=cmap)

    # sns.color_palette("Spectral", as_cmap=True)
    # ax.set(xlabel="train", ylabel="test")
    plt.xticks(rotation=30, ha='right')
    # ax.tick_params(axis="y", pad=215)
    plt.yticks(ha='left')
    # plt.show()
    # fname = ROOT_PATH / Path("figuers") / Path(f"heatmap_cross_features_new.png")
    # plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)

#
#
# def feature_importance_cross(method_split: str, model_dir: str, number_iteration: int, name_classifier: str):
#     ms_table = None
#     chossing_methods = ['tarBase_microArray', "mockMiRNA",
#                         "non_overlapping_sites", "non_overlapping_sites_random",
#
#                         "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
#                         "clip_interaction_clip_3",
#                         "Non_overlapping_sites_clip_data_random"]
#     for i in range(len(chossing_methods)):
#         name = chossing_methods[i] + "_"
#         chossing_methods[i] = name.capitalize()
#     results_dir = ROOT_PATH / Path("Results")
#     number_iteration = str(number_iteration)
#     results_dir_models = ROOT_PATH / Path("Results/models") / model_dir / number_iteration
#     test_dir = DATA_PATH_INTERACTIONS / "test" / method_split / number_iteration
#
#     res_table: DataFrame = pd.DataFrame()
#     res_table.loc["Tarbase_microarray_", "Tarbase_microarray_"] = 0
#
#     FeatureReader.reader_selection_parameter = "without_hot_encoding"
#     feature_reader = get_reader()
#
#     clf_datasets = [f.stem.split("_" + name_classifier)[0] for f in results_dir_models.glob("*.model")]
#     method = name_classifier
#     for j in range(len(chossing_methods)):  # train
#
#         for i in range(len(chossing_methods)):  # test
#             for clf_dataset_A in clf_datasets:
#                 clfA = get_presaved_clf(results_dir_models, clf_dataset_A, method)
#                 testA_name = clf_dataset_A.replace("train", "test")
#                 X_test_A, y_test = feature_reader.file_reader(test_dir / f"{testA_name}.csv")
#
#                 for clf_dataset_B in clf_datasets:
#                     if clf_dataset_A == clf_dataset_B:
#                         continue
#                     clfB = get_presaved_clf(results_dir_models, clf_dataset_B, method)
#
#
#                     test_name = clean_name(clf_dataset_A.split("_train")[0])
#                     train_name = clean_name(clf_dataset_B.split("_train")[0])
#
#                     test_name = test_name.replace("__", "_")
#                     test_name = test_name.replace("MockMrna_di_mrna", "mockMrna_di_mrna")
#                     test_name = test_name.replace("Mockmrna_di_site_", "Mockmrna_di_fragment_")
#                     test_name = test_name.replace("Mockmrna_di_mockmirna_", "Mockmrna_di_fragment_mockmirna_")
#
#                     train_name = train_name.replace("__", "_")
#                     train_name = train_name.replace("MockMrna_di_mrna", "mockMrna_di_mrna")
#                     train_name = train_name.replace("Mockmrna_di_site_", "Mockmrna_di_fragment_")
#                     train_name = train_name.replace("Mockmrna_di_mockmirna_", "Mockmrna_di_fragment_mockmirna_")
#
#                     if train_name not in chossing_methods or test_name not in chossing_methods:
#                         # print(train_name)
#                         # print(test_name)
#                         continue
#                     if train_name != chossing_methods[j]:
#                         continue
#                     if test_name != chossing_methods[i]:
#                         continue
#
#                     print(test_name)
#                     print(train_name)
#
#                     res_table.loc[train_name, test_name] = model_shap_plot_cross_specfic_sample(X_test_A, y_test,
#                                                                                         modelTest=clfA,
#                                                                                         modelTrain=clfB,
#                                                                                         test_name=clf_dataset_A,
#                                                                                         train_name=clf_dataset_B,
#                                                                                         name_classifiers=name_classifier,
#                                                                                                                     dependence_feature=None)
#
#
#
#
#     cmap = sns.cm.rocket_r
#     ax = sns.heatmap(res_table, annot=True, cmap=cmap)
#
#     # sns.color_palette("Spectral", as_cmap=True)
#     ax.set(xlabel="train", ylabel="test")
#     plt.xticks(rotation=30, ha='right')
#     ax.tick_params(axis="y", pad=215)
#     plt.yticks(ha='left')
#     # plt.show()
#     fname = ROOT_PATH / Path("figuers") / Path(f"heatmap_cross_features.png")
#     plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
#


# feature_importance_cross(method_split="underSampling", model_dir="models_underSampling",
#                           number_iteration=0, name_classifier='xgbs')




##################################################################################################################
def heat_map_cross_feature_importance():
    chossing_methods = ['tarBase_microArray',
                "mockMiRNA", "non_overlapping_sites", "non_overlapping_sites_random",
                "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
                "clip_interaction_clip_3",
                "non_overlapping_sites_clip_data_random"]
    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        chossing_methods[i] = name.capitalize()
    res_table = pd.DataFrame()
    res_table.loc["Tarbase_microarray", "Tarbase_microarray"] = 0
    dir_shaps = Path("/sise/home/efrco/efrco-master/Results/figuers/xgbs/shap_importance_cross/importance_list/")
    for j in range(len(chossing_methods)): # train

        for i in range(len(chossing_methods)): #test

            for file in dir_shaps.glob("*.csv"):
                name_file= file.stem.replace("__test", "_test")
                # "/sise/home/efrco/efrco-master/Results/figuers/xgbs/shap_importance_cross/importance_list/Tarbase_microarray__test_Mockmirna_.csv"
                if name_file == 'Clip_interaction_clip_3_test_Mockmrna__di_site_':
                    print("f")
                train_name = clean_name(name_file.split("_test_")[1][:-1])

                test_name = clean_name(name_file.split("_test_")[0])
                train_name = train_name.replace("__", "_")
                train_name = train_name.replace("mockMrna_di_mrna", "mockMrna_di_mrna")
                train_name = train_name.replace("mockMrna_di_site", "mockMRNA_di_fragment")
                train_name = train_name.replace("mockMrna_di_mockMirna", "mockMRNA_di_fragment_mockMiRNA")
                test_name = test_name.replace("__", "_")
                test_name = test_name.replace("mockMrna_de_mrna", "mockMrna_di_mrna")
                test_name = test_name.replace("mockMrna_de_site", "mockMRNA_di_fragment")
                test_name = test_name.replace("mockMrna_de_mockMirna", "mockMRNA_di_fragment_mockMiRNA")
                if train_name!=chossing_methods[j]:
                    continue
                if test_name!=chossing_methods[i]:
                    continue


                if 'mono' in train_name or 'mono' in test_name:
                    continue
                if train_name not in chossing_methods or test_name not in chossing_methods:
                    print(train_name)
                    print(test_name)
                    print("**********************************")
                    continue
                df = read_csv(file)
                filtered_df = df[df['value'] > 0.80]
                num_rows = len(filtered_df)
                train_name = conver_name(train_name)
                test_name = conver_name(test_name)
                res_table.loc[test_name, test_name] = 0
                res_table.loc[test_name,train_name] = num_rows


    # res_table = res_table.sort_values(by=chossing_methods)

    # Reset the index of the DataFrame
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(res_table, annot=True, linewidth=(2.5, 0), cmap=cmap)

    # ax.collections[0].set_clim(vmax=res_table.values.min(), vmin=res_table.values.max())

    ax.set(xlabel="train", ylabel="test")
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis="y", pad=215)
    plt.yticks(ha='left')

    fname = ROOT_PATH / Path("figuers") / "cross_changes.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
    plt.clf()

# heat_map_cross_feature_importance()





def run_all():
    # creat_figures_Intra_anaylsis()
    # cumulative_sum_miRNA_tarBase(nirmol=True)
    # cumulative_sum_miRNA_cross_analysis(nirmol=True)
    # canon_distribution_figures_dataset_with_seed_type(data_set_pos="darnell_human_ViennaDuplex_features")
    KL_train_full_data_gilad()
# run_all()
#

