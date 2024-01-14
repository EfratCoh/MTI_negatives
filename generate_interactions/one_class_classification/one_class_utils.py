from consts.global_consts import  ROOT_PATH, DATA_PATH_INTERACTIONS, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
from utilsfile import read_csv, to_csv
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
import pandas as pd
from collections import Counter


def split_train_one_class(method_split_source, random_state,  number_split):
    csv_dir = DATA_PATH_INTERACTIONS / "train" / method_split_source
    files = list(csv_dir.glob('**/*.csv'))
    dir_results = DATA_PATH_INTERACTIONS / "train" / 'one_class_svm' / str(number_split)

    for f in files:
        cols = []
        X_test = read_csv(f)
    
        df_new = X_test.drop(X_test[X_test['Label'] == 0].sample(frac=1, random_state=random_state).index)
        dataset = str(f.stem).split('/')[-1].split('_features')[0] + "_train_one_class.csv"
        out_results = dir_results / dataset
        to_csv(df_new, out_results)

def split_test_one_class(method_split_source, random_state,  number_split):
    csv_dir = DATA_PATH_INTERACTIONS / "test" / method_split_source
    files = list(csv_dir.glob('**/*.csv'))
    dir_results = DATA_PATH_INTERACTIONS / "test" / 'one_class_svm' / str(number_split)
    FeatureReader.reader_selection_parameter = "without_hot_encoding"

    for f in files:
        X_test = read_csv(f)
        df_new = X_test.drop(X_test[X_test['Label'] == 0].sample(frac=.8, random_state=random_state).index)
        dataset = str(f.stem).split('/')[-1].split('_features')[0] + "_test_one_class.csv"
        out_results = dir_results / dataset
        to_csv(df_new, out_results)

def split_train_test_one_class(method_split_source, random_state, number_split):
    split_train_one_class(method_split_source=method_split_source, random_state=random_state, number_split=number_split)
    split_test_one_class(method_split_source= method_split_source, random_state=random_state, number_split=number_split)




