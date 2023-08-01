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
        # self.measurement_dict_xgboost = self.screate_measurement_dict()
        # self.measurement_dict_svm = self.create_measurement_dict()
        # self.measurement_dict ={'svm': self.create_measurement_dict(),
        #                         'xgbs': self.create_measurement_dict(),
        #                         'isolation_forest': self.create_measurement_dict(),
        #                         'binary_comper_SVM': self.create_measurement_dict(),
        #                         'binary_comper_rf': self.create_measurement_dict()}
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


    def clean_directory(self):
        train = "/sise/home/efrco/efrco-master/data/train"
        test = "/sise/home/efrco/efrco-master/data/test"
        figuers = "/sise/home/efrco/efrco-master/Results/figuers/"
        results_iterations = "/sise/home/efrco/efrco-master/Results/results_iterations/"
        models_clean = "/sise/home/efrco/efrco-master/Results/results_iterations/"
        measurment_cross = "/sise/home/efrco/efrco-master/Results/measurments_cross_validation/"
        measurment_cross = "/sise/home/efrco/efrco-master/Results/models/"


        paths = [train, test, figuers, results_iterations, models_clean, measurment_cross]

        for p in paths:
            for dirpath, dirnames, filenames in os.walk(p):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        os.remove(file_path)
                    except OSError:
                        print("Error while deleting file: ", file_path)



    def split_train_test_files(self):
        # self.clean_directory()
        for i in range(self.number_iterations):
            # split_train_test_underSampling(dataset_positive_name=self.dataset_file_positive, random_state=i * 19, number_split=i)
            split_train_test_one_class(method_split_source='underSampling', random_state=i * 19,  number_split=i)

    def run_xgboost(self, number_iteration):
        build_classifiers_grid_search(number_iteration)
        different_results_summary(method_split="underSampling", model_dir="models_underSampling", number_iteration=number_iteration, name_classifier='xgbs')


    def run_one_class_svm(self, number_iteration):
        build_classifiers_svm(number_iteration)
        # different_results_summary(method_split="one_class_svm", model_dir="models_one_class_svm",
        #                           number_iteration=number_iteration, name_classifier='svm')
        different_results_summary(method_split="underSampling", model_dir="models_one_class_svm",
                                  number_iteration=number_iteration, name_classifier='svm')

    def run_isolation_forest(self, number_iteration):
        build_classifiers_isolation_forest(number_iteration)
        # different_results_summary(method_split="one_class_svm", model_dir="models_isolation_forest",
        #                           number_iteration=number_iteration, name_classifier='isolation_forest')
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
                        print("PPPPPPPPPPPPPP")
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

    def conver_name(self, name):
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
        elif name == "Non_overlapping_sites_top_median":
            name = "NPS_CLASH_Median"
        elif name == "Non_overlapping_sites_top_20_percent":
            name = "NPS_CLASH_20th_Percentile"
        elif name == "Mockmrna_mono_mrna":
            name = "Mock_mono_mRNA"
        elif name == "Mockmrna_di_mrna":
            name = "Mock_di_mRNA"
        elif name == "Mockmrna_mono_site":
            name = "Mock_mono_fragment"
        elif name == "Mockmrna_di_fragment":
            name = "Mock_di_fragment"
        elif name == "Mockmrna_di_fragment_mockmirna":
            name = "Mock_di_fragment_&_miRNA"
        elif name == "Mockmrna_mono_fragment_mockmirna":
            name = "Mock_mono_fragment_&_miRNA"
        elif name == "Clip_interaction_clip_3_":
            name = "CLIP_non_CLASH"
        elif name == "Non_overlapping_sites_clip_data":
            name = "NPS_CLIP_MFE"
        elif name == "Non_overlapping_sites_clip_data_random":
            name = "NPS_CLIP_Random"
        return name

    def clean_name(self, name):

        name_clean = name.replace("model:", "").replace("human", "").replace("darnell_human", "").replace("test:",
                                                                                                          "").replace(
            "ViennaDuplex", "").replace("_darnell_", "").replace("__", "")

        name_clean = name_clean.replace("_nucleotides", "_mono", )
        name_clean = name_clean.replace("denucleotides", "_di")
        name_clean = name_clean.replace("method1", "mrna")
        name_clean = name_clean.replace("method2", "site")
        name_clean = name_clean.replace("method3", "mockMirna")

        if name_clean == "mockMirnadarnell":
            name_clean = "mockMirna"
        if name_clean == "mockMrnadarnell":
            name_clean = "mockMrna"
        if name_clean == "nonoverlappingsitesdarnell":
            name_clean = "Non site"
        if name_clean == "mockMrna__di_mockMirna":
            name_clean = "mockMrna__di_fragment_mockMirna"
        if name_clean == "mockMrna__di_site":
            name_clean = "mockMrna__di_fragment"
        if name_clean == "mockMrna_mono_mockMirna":
            name_clean = "mockMrna__mono_fragment_mockMirna"
        name_clean = name_clean.replace("rna", "RNA")
        name_clean = name_clean.capitalize()

        return name_clean

    def summary_matrix(self, measurement_name):
        res_table = pd.DataFrame()
        for classifier in self.measurement_dict.keys():
            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir/classifier / "final_mean.csv"
            df_mean = read_csv(out_dir_mean)
            for (columnName, columnData) in df_mean.iteritems():
                train_name = self.clean_name(columnName.split('/')[0])
                test_name = self.clean_name(columnName.split('/')[1])
                if 'mono' in train_name or 'mono' in test_name:
                    continue
                train_name = self.conver_name(train_name)
                test_name = self.conver_name(test_name)

                res_table.loc[test_name, train_name] = df_mean.loc[measurement_name, columnName]
            ax = sns.heatmap(res_table, annot=True)
            sns.color_palette("Spectral", as_cmap=True)
            ax.set(xlabel="train", ylabel="test")
            plt.xticks(rotation=30, ha='right')

            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/") / "heatmap.png"
            plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
            plt.clf()


    def summary_barplot_inra_results(self, measurement_name):
        keys = ['method', 'measurement', 'value']
        # res_table = pd.DataFrame(columns=list(keys), dtype=object)
        dtypes = np.dtype(
            [
                ("method", str),
                ("measurement", str),
                ("value", float),

            ]
        )
        res_table = pd.DataFrame(np.empty(0, dtype=dtypes))

        for classifier in self.measurement_dict.keys():

            out_dir_mean = self.result_dir / classifier / "final_mean.csv"
            df_mean = read_csv(out_dir_mean)
            for (columnName, columnData) in df_mean.iteritems():
                train_name = self.clean_name(columnName.split('/')[0])
                test_name = self.clean_name(columnName.split('/')[1])
                if 'mono' in train_name or 'mono' in test_name:
                    continue
                if train_name != test_name:
                    continue
                train_name = self.conver_name(train_name)
                test_name = self.conver_name(test_name)
                res_table.loc['method'] = train_name
                res_table.loc['measurement'] = measurement_name
                res_table.loc['value'] = df_mean.loc[measurement_name, columnName]


            ax = sns.barplot(data=res_table, x="measurement_name", y="value", hue="method")

            sns.color_palette("Spectral", as_cmap=True)


            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/") / "heatmap.png"
            plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
            plt.clf()

    def summary_matrix_mock_mrna(self, measurement_name, target_calculate):
        res_table = pd.DataFrame()
        for classifier in self.measurement_dict.keys():
            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir/classifier / "final_mean.csv"
            out_dir_std = self.result_dir / classifier / "final_std.csv"

            if target_calculate == "mean":
                df_mean = read_csv(out_dir_mean)
            else:
                df_mean = read_csv(out_dir_std)

            for (columnName, columnData) in df_mean.iteritems():
                train_name = self.clean_name(columnName.split('/')[0])
                test_name = self.clean_name(columnName.split('/')[1])
                if 'mockMrna' not in train_name or 'mockMrna' not in test_name:
                    continue
                train_name = self.conver_name(train_name)
                test_name = self.conver_name(test_name)
                res_table.loc[test_name, train_name] = df_mean.loc[measurement_name, columnName]
            ax = sns.heatmap(res_table, annot=True)
            sns.color_palette("rocket", as_cmap=True)
            ax.set(xlabel="train", ylabel="test")

            name_file = "heatmap_mockMrna" + str(measurement_name) + "_" + str(target_calculate) + ".png"
            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/mockMrna") / name_file
            plt.savefig(fname, format="PNG", bbox_inches='tight',  dpi=300)
            plt.clf()


    def summary_matrix_tarBase(self, measurement_name, target_calculate="mean"):
        res_table = pd.DataFrame()
        for classifier in self.measurement_dict.keys():
            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir / classifier / "final_mean.csv"
            out_dir_std = self.result_dir / classifier / "final_std.csv"

            if target_calculate == "mean":
                df_mean = read_csv(out_dir_mean)
            else:
                df_mean = read_csv(out_dir_std)

            for (columnName, columnData) in df_mean.iteritems():
                train_name = self.clean_name(columnName.split('/')[0])
                test_name = self.clean_name(columnName.split('/')[1])
                if 'tarBase' not in train_name or 'tarBase' not in test_name:
                    continue

                res_table.loc[test_name, train_name] = df_mean.loc[measurement_name, columnName]

            ax = sns.heatmap(res_table, annot=True)
            sns.color_palette("rocket", as_cmap=True)
            ax.set(xlabel="train", ylabel="test")

            name_file = "heatmap_tarBase" + str(measurement_name) + "_" + str(target_calculate) + ".png"
            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/tarBase") / name_file
            plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
            plt.clf()

    def summary_matrix_non_overlapping_site(self, measurement_name, target_calculate="mean"):
        res_table = pd.DataFrame()
        for classifier in self.measurement_dict.keys():
            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir / classifier / "final_mean.csv"
            out_dir_std = self.result_dir / classifier / "final_std.csv"

            if target_calculate == "mean":
                df_mean = read_csv(out_dir_mean)
            else:
                df_mean = read_csv(out_dir_std)

            for (columnName, columnData) in df_mean.iteritems():
                train_name = self.clean_name(columnName.split('/')[0])
                test_name = self.clean_name(columnName.split('/')[1])
                if 'non_overlapping_sites' not in train_name.lower() or 'non_overlapping_sites' not in test_name.lower():
                    continue
                if 'clip' in train_name.lower() or 'clip'  in test_name.lower():
                    continue
                train_name = self.conver_name(train_name)
                test_name = self.conver_name(test_name)
                res_table.loc[train_name, test_name] = df_mean.loc[measurement_name, columnName]

            ax = sns.heatmap(res_table, annot=True)
            sns.color_palette("rocket", as_cmap=True)
            ax.set(xlabel="Test", ylabel="Train")
            ax.tick_params(axis="y", pad=170)
            plt.yticks(ha='left')

            plt.xticks(rotation=30, ha='right')
            name_file = "heatmap_non_overlapping_sites" + str(measurement_name) + "_" + str(target_calculate) + ".png"
            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/non_overlapping_sites") / name_file
            plt.savefig(fname, format="PNG", bbox_inches='tight',  dpi=300)
            plt.clf()

    def summary_matrix_clip(self, measurement_name, target_calculate="mean"):
        res_table = pd.DataFrame()
        for classifier in self.measurement_dict.keys():
            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir / classifier / "final_mean.csv"
            out_dir_std = self.result_dir / classifier / "final_std.csv"

            if target_calculate == "mean":
                df_mean = read_csv(out_dir_mean)
            else:
                df_mean = read_csv(out_dir_std)

            for (columnName, columnData) in df_mean.iteritems():
                train_name = self.clean_name(columnName.split('/')[0])
                test_name = self.clean_name(columnName.split('/')[1])
                if 'clip_3' not in train_name or 'clip_3' not in test_name:
                    continue

                res_table.loc[test_name, train_name] = df_mean.loc[measurement_name, columnName]


            ax = sns.heatmap(res_table, annot=True)
            sns.color_palette("rocket", as_cmap=True)
            ax.set(xlabel="Train", ylabel="Test")


            name_file = "heatmap_clip" + str(measurement_name) + "_" + str(target_calculate) + ".png"
            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/clip") / name_file
            plt.savefig(fname, format="PNG", bbox_inches='tight',  dpi=300)
            plt.clf()


    def summary_matrix_Intra(self):
        res_table = pd.DataFrame()
        measurement_name_list = ['ACC', 'TPR', 'TNR', 'F1']
        final_method = ['tarBase','tarBase_Liver', 'tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMiRNA",
                        "mockMRNA__di_mRNA", "mockMRNA__di_fragment","mockMRNA__di_fragment_mockMiRNA",
                        "mockMRNA_mono_mRNA", "mockMRNA_mono_site", "mockMRNA__mono_fragment_mockMiRNA",
                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data","non_overlapping_sites_clip_data_random"]
        final_method = ["mockMiRNA",
                        "mockMRNA__di_mRNA", "mockMRNA__di_fragment", "mockMRNA__di_fragment_mockMiRNA",
                        "mockMRNA_mono_mRNA", "mockMRNA_mono_site", "mockMRNA__mono_fragment_mockMiRNA",
                        'tarBase', 'tarBase_Liver', 'tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data", "non_overlapping_sites_clip_data_random"]
        chossing_methods = ['tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMiRNA",
                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment","mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data_random"]
        for i in range(len(final_method)):
            name = final_method[i]
            final_method[i] = name.capitalize()
        for i in range(len(chossing_methods)):
            name = chossing_methods[i]
            chossing_methods[i] = name.capitalize()

        for classifier in self.measurement_dict.keys():
            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir / classifier / "final_mean.csv"

            df_mean = read_csv(out_dir_mean)
            for measurement_name in measurement_name_list:
                for current_method in final_method:
                    for (columnName, columnData) in df_mean.iteritems():
                        train_name = self.clean_name(columnName.split('/')[0])
                        test_name = self.clean_name(columnName.split('/')[1])
                        if train_name != test_name:
                            print(train_name)
                            print(test_name)
                            print("######################################################")
                            continue
                        if train_name not in final_method:
                            # print(train_name)
                            continue
                        if train_name != current_method:
                            continue
                        train_name = train_name.replace("__","_")
                        train_name = train_name.replace("mockMrna_de_mrna","mockMrna_di_mrna")
                        train_name = train_name.replace("mockMrna_de_site","mockMrna_di_site")
                        train_name = train_name.replace("mockMrna_de_mockMirna","mockMrna_di_mockMirna")

                        if train_name in chossing_methods:
                            train_name = self.conver_name(train_name)
                            train_name = train_name + " *"
                        else:
                            if train_name=="Non_overlapping_sites_clip_data":
                                print("v")
                            train_name = self.conver_name(train_name)

                        # res_table.loc[measurement_name, train_name] = df_mean.loc[measurement_name, columnName]
                        res_table.loc[train_name, measurement_name] = df_mean.loc[measurement_name, columnName]

            # ax = sns.heatmap(res_table, annot=True, linewidth=4.5,cmap=sns.cubehelix_palette(as_cmap=True))

            ax = sns.heatmap(res_table, annot=True, linewidth=(3.5,0))
            sns.color_palette("rocket", as_cmap=True)

            ax.set(xlabel="Metric", ylabel="Dataset")

            # plt.xticks(rotation=30, ha='right')
            plt.xticks(ha='center')

            ax.tick_params(axis="y", pad=215)
            plt.yticks(ha='left')

            name_file = "heatmap_Intra.png"
            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap") / name_file
            plt.savefig(fname, format="PNG", bbox_inches='tight',  dpi=300)
            plt.show()
            plt.clf()


    def summary_matrix_cross(self, measurement_name):
        res_table = pd.DataFrame()
        dict_rename = {}

        chossing_methods = [
                            "mockMiRNA",
                            "mockMRNA_di_fragment_mockMiRNA", "mockMRNA_di_mRNA", "mockMRNA_di_fragment",
                            "non_overlapping_sites", "non_overlapping_sites_random",
                             'tarBase_microArray',
                            "clip_interaction_clip_3_",
                            "non_overlapping_sites_clip_data_random"]
        for i in range(len(chossing_methods)):
            name = chossing_methods[i]
            chossing_methods[i] = name.capitalize()
            dict_rename[chossing_methods[i]] = "Method " + str(i)

        for classifier in self.measurement_dict.keys():

            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir / classifier / "final_std.csv"
            df_mean = read_csv(out_dir_mean)
            for method_test in chossing_methods:
                for method_train in chossing_methods:
                    for (columnName, columnData) in df_mean.iteritems():
                        train_name = self.clean_name(columnName.split('/')[0])
                        test_name = self.clean_name(columnName.split('/')[1])
                        train_name = train_name.replace("__", "_")
                        train_name = train_name.replace("mockMrna_di_mrna", "mockMrna_di_mrna")
                        train_name = train_name.replace("mockMrna_di_site", "mockMRNA_di_fragment")
                        train_name = train_name.replace("mockMrna_di_mockMirna", "mockMRNA_di_fragment_mockMiRNA")
                        test_name = test_name.replace("__", "_")
                        test_name = test_name.replace("mockMrna_de_mrna", "mockMrna_di_mrna")
                        test_name = test_name.replace("mockMrna_de_site", "mockMRNA_di_fragment")
                        test_name = test_name.replace("mockMrna_de_mockMirna", "mockMRNA_di_fragment_mockMiRNA")

                        if 'mono' in train_name or 'mono' in test_name:
                            continue
                        if train_name not in chossing_methods or test_name not in chossing_methods:
                            print(train_name)
                            print(test_name)
                            print("**********************************")
                            continue
                        if method_train!=train_name or method_test!=test_name:
                            continue

                        test_name = self.conver_name(test_name)
                        train_name = self.conver_name(train_name)

                        # res_table.loc[test_name, train_name] = df_mean.loc[measurement_name, columnName]
                        res_table.loc[train_name, test_name] = df_mean.loc[measurement_name, columnName]

            for i in range(len(chossing_methods)):
                name = self.conver_name(chossing_methods[i])
                chossing_methods[i] = name

            res_table = res_table.reindex(columns=chossing_methods)

            ax = sns.heatmap(res_table, annot=True, linewidth=(1.5,0))
            # ax = sns.heatmap(res_table, annot=True)

            sns.color_palette("Spectral", as_cmap=True)
            ax.set(xlabel="Testing dataset", ylabel="Training dataset")
            plt.xticks(rotation=30, ha='right')
            ax.tick_params(axis="y", pad=215)

            plt.yticks(ha='left')

            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/") / f"heatmap_cross_{measurement_name}_std.png"
            plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
            plt.clf()

    def summary_matrix_cross_hirachia(self, measurement_name):
        res_table = pd.DataFrame()
        dict_rename = {}

        chossing_methods = [
                            "mockMiRNA",
                            "mockMRNA_di_fragment_mockMiRNA", "mockMRNA_di_mRNA", "mockMRNA_di_fragment",
                            "non_overlapping_sites", "non_overlapping_sites_random",
                             'tarBase_microArray',
                            "clip_interaction_clip_3_",
                            "non_overlapping_sites_clip_data_random"]
        for i in range(len(chossing_methods)):
            name = chossing_methods[i]
            chossing_methods[i] = name.capitalize()
            dict_rename[chossing_methods[i]] = "Method " + str(i)

        for classifier in self.measurement_dict.keys():

            res_table = pd.DataFrame()
            out_dir_mean = self.result_dir / classifier / "final_mean.csv"
            df_mean = read_csv(out_dir_mean)
            for method_test in chossing_methods:
                for method_train in chossing_methods:
                    for (columnName, columnData) in df_mean.iteritems():
                        train_name = self.clean_name(columnName.split('/')[0])
                        test_name = self.clean_name(columnName.split('/')[1])
                        train_name = train_name.replace("__", "_")
                        train_name = train_name.replace("mockMrna_di_mrna", "mockMrna_di_mrna")
                        train_name = train_name.replace("mockMrna_di_site", "mockMRNA_di_fragment")
                        train_name = train_name.replace("mockMrna_di_mockMirna", "mockMRNA_di_fragment_mockMiRNA")
                        test_name = test_name.replace("__", "_")
                        test_name = test_name.replace("mockMrna_de_mrna", "mockMrna_di_mrna")
                        test_name = test_name.replace("mockMrna_de_site", "mockMRNA_di_fragment")
                        test_name = test_name.replace("mockMrna_de_mockMirna", "mockMRNA_di_fragment_mockMiRNA")

                        if 'mono' in train_name or 'mono' in test_name:
                            continue
                        if train_name not in chossing_methods or test_name not in chossing_methods:
                            print(train_name)
                            print(test_name)
                            print("**********************************")
                            continue
                        if method_train!=train_name or method_test!=test_name:
                            continue

                        test_name = self.conver_name(test_name)
                        train_name = self.conver_name(train_name)

                        res_table.loc[train_name, test_name] = df_mean.loc[measurement_name, columnName]
            for i in range(len(chossing_methods)):
                name = self.conver_name(chossing_methods[i])
                chossing_methods[i] = name

            res_table = res_table.reindex(columns=chossing_methods)
            cmap = sns.cm.rocket_r

            # Create a clustermap
            ax = sns.clustermap(res_table, cmap=cmap)

            # Set plot title and axis labels
            # plt.title("Clustermap of res_table")
            # plt.xlabel("Train Names")
            # plt.ylabel("Test Names")
            ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=30, ha='right')
            # ax.ax_heatmap.set_ylabel(ha='right')

            ax.ax_heatmap.set_xlabel("Test")
            ax.ax_heatmap.set_ylabel("Train")
            ax.ax_heatmap.set_xlabel("Test", fontsize=16)

            ax.ax_heatmap.set_ylabel("Train", fontsize=16)
            # Display the plot
            # plt.show()
            # sns.color_palette("Spectral", as_cmap=True)
            # ax.set(xlabel="train", ylabel="test")
            # plt.xticks(rotation=30, ha='right')
            # ax.tick_params(axis="y", pad=215)

            # plt.yticks(ha='left')

            fname = ROOT_PATH / Path(f"Results/figuers/{classifier}/heatmap/") / "heatmap_cross_clustring.png"
            plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)
            plt.clf()
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

        # self.split_train_test_files()

        # self.run_experiment_xgbs(start=0, to=1) #for shap and feature importance #6649424
        # self.run_experiment_xgbs(start=1, to=2) #
        # self.run_experiment_xgbs(start=2, to=3) #
        # self.run_experiment_xgbs(start=3, to=4) #
        # self.run_experiment_xgbs(start=4, to=5) #
        # self.run_experiment_xgbs(start=5, to=6) #
        # self.run_experiment_xgbs(start=6, to=7) #
        # self.run_experiment_xgbs(start=7, to=8) #
        # self.run_experiment_xgbs(start=8, to=9) #
        # self.run_experiment_xgbs(start=9, to=10) #
        # self.run_experiment_xgbs(start=10, to=11) #6880553
        # self.run_experiment_xgbs(start=11, to=12) #6881223
        # self.run_experiment_xgbs(start=12, to=13) #6896909
        # self.run_experiment_xgbs(start=13, to=14) #6896912
        # self.run_experiment_xgbs(start=14, to=15) #6896915
        # self.run_experiment_xgbs(start=15, to=16) #6896916
        # self.run_experiment_xgbs(start=16, to=17) #6896923
        # self.run_experiment_xgbs(start=17, to=18) #6896929
        # self.run_experiment_xgbs(start=18, to=19) #6896934
        # self.run_experiment_xgbs(start=19, to=20) #6896936
        # pass

        ###########################One class classification########################################
        # self.run_experiment_one_class_svm()
        # self.run_experiment_isolation_forest()

        # self.run_experiment_binary_comper_svm()
        # self.run_experiment_binary_comper_rf()



        ################################Write Results####################################################################
        # self.write_results() #7843884
        pass




############################### Runnning ############################################
cv = CrossValidation("darnell_human_ViennaDuplex_features", "measurments_cross_validation", number_iterations=20)
# cv.run_pipline()
