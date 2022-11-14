from pathlib import Path
from datetime import datetime
from utils.logger import logger
from consts.mirna_utils import MIRBASE_FILE
from consts.global_consts import DUPLEX_DICT
from utils.utilsfile import read_csv, to_csv
from consts.global_consts import ROOT_PATH, DATA_PATH,CLIP_PATH_DATA,NEGATIVE_DATA_PATH, GENERATE_DATA_PATH
import pandas as pd
import numpy as np
from utils.utilsfile import *
from consts.global_consts import ROOT_PATH, DATA_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
from duplex.ViennaDuplex import ViennaDuplex

from duplex.ViennaDuplex import *
import random
from features.SeedFeatures import *
# import MirBaseUtils.mirBaseUtils as MBU
import mirna_utils.mirbase as MBU
from multiprocessing import Process
from consts.global_consts import CONFIG
from duplex.Duplex import Duplex


def valid_negative_seq(mir, mrna):
    duplex_cls: Duplex = DUPLEX_DICT['ViennaDuplex']
    logger.info(f"{ViennaDuplex} do_duplex")
    dp = duplex_cls.fromChimera(mir, mrna)
    try:
        canonic_seed = dp.canonical_seed
        non_canonic_seed = dp.noncanonical_seed

    except SeedException:
        canonic_seed = False
        non_canonic_seed = False

    # warning: number of pair was before to interactions count
    duplex = RNA.duplexfold(mir, mrna)
    MEF_duplex = duplex.energy
    site = dp.site[::-1]
    # print(MEF_duplex)

    return canonic_seed, non_canonic_seed, dp.interaction_count, MEF_duplex, site


def generate_negative_seq(orig_mirna, full_mrna, num_of_tries=10000):
    canonic_seed, non_canonic_seed, num_of_pairs, MEF_duplex, site = valid_negative_seq(orig_mirna, full_mrna)
    cond1 = canonic_seed
    cond2 = non_canonic_seed
    if cond1 or cond2:
        properties = {
            "mock_mirna": orig_mirna,
            "full_mrna": full_mrna,
            "canonic_seed": canonic_seed,
            "non_canonic_seed": non_canonic_seed,
            "num_of_pairs": num_of_pairs,
            "MEF_duplex": MEF_duplex,
            "site": site
        }
        return True, properties
    return False, {}


def sites_list_clash():
    positive_interaction_dir = MERGE_DATA / "positive_interactions_new/featuers_step/"
    list_sites = []
    # Gene_ID ---> number Gen + number Transcript
    # ID ---> number Gen
    for file in positive_interaction_dir.glob("*csv*"):
        usecols = ['Gene_ID',  'site']
        logger.info(f"Reading file {file}")
        df_positive_interaction = pd.read_csv(file, usecols=usecols)
        sites = df_positive_interaction['site'].tolist()
        list_sites.extend(sites)
    return list_sites

def generate_interactions(site, Gene_ID, full_mrna, df_mirna):
    valid = False
    while not valid:
        random_mirna = df_mirna.sample(n=1)
        random_mirna.reset_index(drop=True, inplace=True)
        random_mirna.reset_index(inplace=True)
        for index, row in random_mirna.iterrows():
            valid, properties = generate_negative_seq(site, row['sequence'])

            new_row = pd.Series()
            new_row['paper name'] = 'mirTarget'
            new_row['organism'] = 'Human'
            new_row['miRNA ID'] = row['miRNA ID']
            new_row['miRNA sequence'] = row['sequence']
            new_row['Gene_ID'] = Gene_ID
            new_row['full_mrna'] = full_mrna
            new_row['site'] = site
            if valid:
                return new_row, properties['MEF_duplex']


def sub_insert_NNN(full_mrna, start, end, site):
    start = int(start)
    orig = len(full_mrna)

    while start != end + 1:
        full_mrna = full_mrna[:start] + "N" + full_mrna[start + 1:]
        start += 1
    return full_mrna

# def overlapping_site(sub_mrna):
#     pass


def worker(df_mrna, list_sites, df_mirna):
    count_sub_mrna = 0
    gruop_df = df_mrna.groupby(['Gene_ID'])
    neg_df = pd.DataFrame()
    count = 0
    for group_name, sub_group in gruop_df:
        mrna_cut = sub_group.iloc[0]["full_mrna"]
        Gene_ID = sub_group.iloc[0]["Gene_ID"]
        full_mrna = sub_group.iloc[0]["full_mrna"]

        for row_index, row in sub_group.iterrows():

            mrna_cut = sub_insert_NNN(mrna_cut, row["start"], row["end"], row['site'])

        cut_mrna = mrna_cut
        previous_index = 0
        size_param = 40
        window_size = min(size_param, len(mrna_cut))
        pervious_MEF_duplex = float('inf')
        best_row = pd.Series()

        for window in range(window_size, len(mrna_cut) + size_param, size_param):

            sub_mrna = cut_mrna[previous_index:window]
            previous_index = window
            if "N" in sub_mrna:
                count = count +1
                # if count > 3:
                #     break
                continue
            if sub_mrna in list_sites:
                count_sub_mrna = count_sub_mrna + 1
                continue

            else:
                current_row, new_MEF_duplex = generate_interactions(sub_mrna,Gene_ID, full_mrna, df_mirna)
                if new_MEF_duplex <= pervious_MEF_duplex:
                    best_row = current_row

        if len(best_row) == 0:
            count += 1
            print("not found interaction for")
            continue

        neg_df = neg_df.append(best_row, ignore_index=True)

        # if count > 3:
        #     break
    print(count_sub_mrna)
    print(count)
    return neg_df


def run():
    clip_data_path = CLIP_PATH_DATA
    list_sites_clash = sites_list_clash()
    for clip_dir in clip_data_path.iterdir():
        for file in clip_dir.glob("*mrna_clean.csv*"):
            if "clip_3" not in str(file):
                continue

            mirna_df = read_csv(clip_dir / "mirna.csv")
            # mrna_df = read_csv(clip_dir / "mrna_clean.csv")
            name_mrna_file = str(clip_dir.stem) + ".csv"
            mrna_df = read_csv(GENERATE_DATA_PATH/ "clip_interaction" / name_mrna_file)
            list_sites = list_sites_clash + mrna_df['site'].tolist()

            neg_df = worker(mrna_df, list_sites, mirna_df)

            name_dir = "non_overlapping_sites_clip_data"
            name_file = str(clip_dir.stem) + ".csv"

            path_df = GENERATE_DATA_PATH / name_dir / name_file
            to_csv(neg_df, path_df)

run()

# maybe we need to filter the interactions- because for each gene we have a lot of candidate