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
    num_tries = 0
    random_numnber = random.randint(0,1000)

    while not valid:
        random_numnber = random.randint(0, 1000)
        random_mirna = df_mirna.sample(n=1, random_state=random_numnber)
        num_tries += 1
        if num_tries > 100000:
            break
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


    if valid:
        return new_row, properties['MEF_duplex'], valid
    else:
        return {}, {}, valid


def sub_insert_NNN(full_mrna, start, end, site):
    start = int(start)
    orig = len(full_mrna)

    while start != end + 1:
        full_mrna = full_mrna[:start] + "N" + full_mrna[start + 1:]
        start += 1
    return full_mrna


def complete_site_chars(start, end):
    len_site = end - start
    number_chars_add_one_side = 0
    if len_site < 75:
        number_chars_add = 75 - len_site
        number_chars_add_one_side = round(number_chars_add / 2)

    return number_chars_add_one_side


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
            mrna_cut = sub_insert_NNN(mrna_cut, row["start"], row["end"], row['site_old'])

        cut_mrna = mrna_cut
        previous_index = 0
        size_param = 40
        window_size = min(size_param, len(mrna_cut))
        pervious_MEF_duplex = float('inf')
        best_row = pd.Series()

        for window in range(window_size, len(mrna_cut) + size_param, size_param):

            sub_mrna = cut_mrna[previous_index:window]
            start_site = previous_index
            end_site = window
            previous_index = window
            if "N" in sub_mrna:
                count = count + 1
                continue
            if sub_mrna in list_sites:
                count_sub_mrna = count_sub_mrna + 1
                continue

            else:
                number_char_complete = int(complete_site_chars(start_site, end_site))
                full_sub = get_subsequence_by_coordinates(full_mrna, start_site,end_site, "+", number_char_complete)
                current_row, new_MEF_duplex, valid = generate_interactions(full_sub,Gene_ID, full_mrna, df_mirna)
                if not valid:
                    continue
                current_row['start'] = start_site
                current_row['end'] = end_site
                current_row['site'] = full_sub
                if new_MEF_duplex <= pervious_MEF_duplex:
                    print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
                    best_row = current_row
                    pervious_MEF_duplex = new_MEF_duplex


        if len(best_row) == 0:
            count += 1
            print("not found interaction for")
            continue

        neg_df = neg_df.append(best_row, ignore_index=True)


    return neg_df

def split_file_by_gene():
    clip_data_path = CLIP_PATH_DATA
    for clip_dir in clip_data_path.iterdir():
        for file in clip_dir.glob("*mrna_clean.csv*"):
            if "clip_3" not in str(file):
                continue
            name_mrna_file = str(clip_dir.stem) + ".csv"
            mrna_df = read_csv(GENERATE_DATA_PATH / "clip_interaction" / name_mrna_file)
            gruop_df = mrna_df.groupby(['Gene_ID'])

            number_file = 0
            for group_name, sub_group in gruop_df:
                name_dir = "non_overlapping_sites_clip_data"
                name_file = str(number_file) + ".csv"
                path_df = GENERATE_DATA_PATH / name_dir / "split_by_gene" / name_file
                to_csv(sub_group, path_df)
                number_file = number_file + 1

def run(start, to):
    clip_data_path = CLIP_PATH_DATA
    list_sites_clash = sites_list_clash()
    mrna_df_full = read_csv(GENERATE_DATA_PATH / "clip_interaction" / 'clip_3.csv')
    list_sites = list_sites_clash + list(mrna_df_full['site_old'])
    frames = []
    for clip_dir in clip_data_path.iterdir():
        if "clip_3" not in str(clip_dir.stem):
            continue
        # arrive to the correct dir
        for file in clip_dir.glob("*mrna_clean.csv*"):
            mirna_df = read_csv(clip_dir / "mirna.csv")
            name_dir = "non_overlapping_sites_clip_data"
            path_df = GENERATE_DATA_PATH / name_dir / "split_by_gene/"

            # pass on the start until to files
            for file in path_df.glob("*.csv"):
                name_mrna_file = str(start) + ".csv"
                mrna_df = read_csv(path_df / name_mrna_file)
                neg_df = worker(mrna_df, list_sites, mirna_df)
                frames.append(neg_df)
                start = start + 1
                if start == to + 1:
                    break


        name_dir = "non_overlapping_sites_clip_data"
        name_file = str(to) + ".csv"
        path_df = GENERATE_DATA_PATH / name_dir / 'split_after_filter' / name_file
        result = pd.concat(frames)
        print(result.shape)
        result.reset_index(drop=True, inplace=True)
        result.reset_index(inplace=True)
        to_csv(result, path_df)



def main_run():
   pass
    # split_file_by_gene()

##################NEW RUN - PILOT##############################
    # run(start=0, to=50) # 4831938
    # run(start=51, to=100) # 4831939
    # run(start=101, to=150) #4831940
    # run(start=151, to=200) #4831941
    # run(start=201, to=210) #4831942
    # run(start=211, to=220) #4831943
    # run(start=221, to=230) #4831944
    # run(start=231, to=240) #4831945
    # run(start=241, to=250) #4831946



########################################################################
    # run(start=301, to=400) #4830913
    # run(start=401, to=500) #4830914
    # run(start=501, to=600) #4830915
    # run(start=601, to=700) #4830916
    # run(start=701, to=800) #4830917
    # run(start=801, to=900) #4830918
    # run(start=901, to=1000) #4830919
    # run(start=1001, to=1100) #4830920
    # run(start=1101, to=1200) #4830921
    # run(start=1201, to=1300) #4830922
    # run(start=1301, to=1400) #4830923
    # run(start=1401, to=1500) #4830924
    # run(start=1501, to=1600) #4830925
    # run(start=1601, to=1700) #4830926
    # run(start=1701, to=1815) #4830928








    # combin_file()
# main_run()
# maybe we need to filter the interactions- because for each gene we have a lot of candidate

df = read_csv("/sise/home/efrco/efrco-master/data/positive_interactions/positive_interactions_new/data_without_featuers/darnell_human_ViennaDuplex.csv")
print("h")