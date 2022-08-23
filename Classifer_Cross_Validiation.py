from pathlib import Path
import sys
sys.path.append('/sise/home/efrco/efrco-master/Classifier/')
from Classifier.train_test_underSampling import split_train_test as split_train_test_underSampling
from Classifier.ClassifierWithGridSearch import build_classifiers as build_classifiers_grid_search
from Classifier.result_test import different_results_summary
import pandas as pd
from utils.utilsfile import read_csv, to_csv
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS


class NoModelFound(Exception):
    pass


class CrossValidation(object):

    def __init__(self, dataset_file_positive, result_dir, number_iterations):
        self.number_iterations = number_iterations
        self.dataset_file_positive = dataset_file_positive
        self.result_dir = ROOT_PATH / Path("Results") / result_dir
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.measuerment_dict = {}
        self.create_measuerment_dict()

    def create_measuerment_dict(self):
        columns_list = [f"exper_{i}" for i in range(self.number_iterations)]
        self.measuerment_dict = {
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

        }

    def get_measuerment_dict(self):
        return {k: round(v, 3) for k, v in self.measuerment_dict.items()}

    def run_experiment(self):

        for i in range(self.number_iterations):
            split_train_test_underSampling(dataset_positive_name=self.dataset_file_positive, random_state=i*19)
            build_classifiers_grid_search()
            results_dir = ROOT_PATH / Path("Results")
            different_results_summary(method_split="underSampling", model_dir="models_underSampling", number_iteration=i)
            ms_table = read_csv(results_dir / "xgbs_measurements" / f"measurement_summary_{i}.csv")
            for measuerment in self.measuerment_dict.keys():
                col = ms_table[measuerment].apply(lambda t: round(t, 3))
                self.measuerment_dict[measuerment][f"exper_{i}"] = col

        # save file of result for each measuerment
        for measuerment in self.measuerment_dict.keys():
            out_dir = self.result_dir / f"{measuerment}_summary.csv"
            to_csv(self.measuerment_dict[measuerment], out_dir)

    def summary_result(self):

        all_result_mean = pd.DataFrame()
        all_result_std =  pd.DataFrame()

        for measuerment_file in self.result_dir.glob("*summary*"):
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

        out_dir_mean = self.result_dir / f"final_mean.csv"
        to_csv(all_result_mean, out_dir_mean)

        out_dir_std = self.result_dir / f"final_std.csv"
        to_csv(all_result_std, out_dir_std)


############################### Runnning ############################################
cv = CrossValidation("darnell_human_ViennaDuplex_features", "measurments_cross_validation", number_iterations=20)
cv.run_experiment()
cv.summary_result()