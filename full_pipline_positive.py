from utils.utilsfile import read_csv, to_csv
from pathlib import Path
from pipeline_steps.duplex_step import duplex as duplex_positive
from pipline_steps_negative.rna_site_insertion_negative import get_site_from_extended_site
from pipline_steps_negative.normalization_final_step_negative import finalize
from pipeline_steps.feature_extraction import feature_extraction
from consts.global_consts import MERGE_DATA,NEGATIVE_DATA_PATH, GENERATE_DATA_PATH, ROOT_PATH


# step 1- formalization all the dataset for the same format

def full_pipline(name_of_file: str):

    # step 2- extract duplex of the interaction by VieannaDuplex
    name_of_file_primary = name_of_file

    fout_primary = MERGE_DATA / "positive_interactions_new"

    name_of_file = name_of_file + ".csv"
    fin = MERGE_DATA / "positive_interactions_new/data_without_featuers" / name_of_file

    name_of_file = name_of_file_primary + "_duplex.csv"
    fout= fout_primary / "duplex_step" / name_of_file

    # positive interactions
    print("###############Duplex POSITIVE#############")
    duplex_positive('ViennaDuplex', fin, fout)

    # step 3- extract the site and his coordination's
    fin = fout
    print("###############Site#############")

    name_of_file = name_of_file_primary + "_site.csv"
    fout = fout_primary / "site_step" /name_of_file
    # get_site_from_extended_site(fin, fout)

    print("###############Normaliztion#############")

    # step 4- normalization of the dataframe
    fin = fout
    name_of_file = name_of_file_primary + "_normalization.csv"
    fout = fout_primary / "normalization_step" /name_of_file
    finalize(fin, fout)

    # step 5- extract features
    print("###############extract features#############")

    fin = fout
    name_of_file = name_of_file_primary + "_features.csv"
    fout = fout_primary / "featuers_step" / name_of_file
    feature_extraction(fin, fout)


def generate_positive_interaction():

    pos_dir_name = MERGE_DATA / "positive_interactions_merge"
    for dataset_file in pos_dir_name.glob("*_features*"):
        print(dataset_file)
        pos_df = read_csv(dataset_file)
        pos_df.rename(columns={"sequence": "full_mrna"}, inplace=True)

        col_list = ['key', 'paper name', 'organism', 'miRNA ID', 'miRNA sequence', 'site', 'region','valid_row' , 'full_mrna', 'Gene_ID']
        pos_df = pos_df[col_list]
        path = MERGE_DATA / "positive_interactions_new/data_without_featuers"
        dataset_name = str(dataset_file.stem).split("_features.csv")[0].split("_features")[0]

        name_file = path / (dataset_name + ".csv")
        to_csv(pos_df, name_file)
        # if str(dataset_name) == 'unambiguous_human_ViennaDuplex':
        #     continue

        print("full pipline for : ", dataset_file)
        full_pipline(dataset_name)

        pos = MERGE_DATA / "positive_interactions_new/featuers_step" / (str(dataset_file.stem)+'.csv')
        open = read_csv(pos)
        open.drop(columns=['Seed_match_noncanonical', 'Seed_match_canonical'], inplace=True)
        to_csv(open, pos)

generate_positive_interaction()

