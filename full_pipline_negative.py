from pipline_steps_negative.duplex_step_negative import duplex as duplex_negative
from pipline_steps_negative.rna_site_insertion_negative import get_site_from_extended_site
from pipline_steps_negative.normalization_final_step_negative import finalize
from pipeline_steps.feature_extraction import feature_extraction
from consts.global_consts import NEGATIVE_DATA_PATH, GENERATE_DATA_PATH
import pandas as pd
import numpy as np
from utilsfile import read_csv, to_csv

# step 1- formalization all the dataset for the same format


def full_pipline(name_of_method: str, name_of_file: str, duplex=duplex_negative):
    # step 2- extract duplex of the interaction by VieannaDuplex
    # only for mock mirna
    name_of_file_primary = name_of_method + "_" + name_of_file.split('_features')[0] + "_negative"

    # name_of_file_primary = name_of_method + "_" + name_of_file
    fout_primary = NEGATIVE_DATA_PATH / name_of_method

    name_of_file = name_of_file + ".csv"
    fin = GENERATE_DATA_PATH / name_of_method / name_of_file

    name_of_file = name_of_file_primary + "_duplex.csv"
    fout= fout_primary / name_of_file

    # negative interactions
    print("###############Duplex NEGATIVE#############")
    duplex('ViennaDuplex', fin, fout)

    # step 3- extract the site and his coordination's
    fin = fout
    print("###############Site#############")

    name_of_file = name_of_file_primary + "_site.csv"
    fout = fout_primary / name_of_file
    get_site_from_extended_site(fin, fout)

    print("###############Normaliztion#############")

    # step 4- normalization of the dataframe
    fin = fout
    name_of_file = name_of_file_primary + "_normalization.csv"
    fout = fout_primary / name_of_file
    finalize(fin, fout)

    # step 5- extract features
    print("###############extract features#############")

    fin = fout
    name_of_file = name_of_file_primary + "_features.csv"
    fout = fout_primary / name_of_file
    feature_extraction(fin, fout)



##### Prepar files#######

# method 1 - tarBase#
from generate_interactions.tarBase import reader
# reader.run("generate_interactions/tarBase/human_features_negative.csv")
# full_pipline("tarBase", "human_features_negative")#5374331

# method 2 - mockMirna#
# from generate_interactions.mockMirna import NegativeSamples
# NegativeSamples.main()
# full_pipline("mockMirna", "darnell_human_ViennaDuplex_features_negative")
# full_pipline("mockMirna", "human_mapping_ViennaDuplex_features_negative")
# full_pipline("mockMirna", "unambiguous_human_ViennaDuplex_features_negative")
# full_pipline("mockMirna", "qclash_melanoma_human_ViennaDuplex_features_negative")


# Method 3 - non_overlapping_sites #
# from generate_interactions.non_overlapping_sites import generate
# from pipeline_steps.duplex_step import duplex as duplex_positive

# Method 3a - non_overlapping_sites- energy #
# generate.main()
# full_pipline("non_overlapping_sites", "darnell_human_ViennaDuplex_features_negative", duplex_positive)
# full_pipline("non_overlapping_sites", "human_mapping_ViennaDuplex_features_negative", duplex_positive)
# full_pipline("non_overlapping_sites", "unambiguous_human_ViennaDuplex_features_negative", duplex_positive)
# full_pipline("non_overlapping_sites", "qclash_melanoma_human_ViennaDuplex_features_negative", duplex_positive)


# Method 3b - non_overlapping_sites- random #
# # from generate_interactions.non_overlapping_sites import generate_random
# from pipeline_steps.duplex_step import duplex as duplex_positive
# # generate_random.main()
# full_pipline("non_overlapping_sites", "random_darnell_human_ViennaDuplex_features_negative", duplex_positive)

# Method 3c - non_overlapping_sites- top #
from generate_interactions.non_overlapping_sites import generate_full_mrna
# from pipeline_steps.duplex_step import duplex as duplex_positive
# generate_full_mrna.main()
# full_pipline("non_overlapping_sites", "top_20_percent_darnell_human_ViennaDuplex_features_negative", duplex_positive)


# ####method 4 - mockMrna  ######
# from generate_interactions.mockMrna.run_methods import run
# run()

##### method 5 - clip_interaction  ######
# from generate_interactions.clip_interaction.run_method import run
# from pipeline_steps.duplex_step import duplex as duplex_positive
# run()
# full_pipline("clip_interaction", "clip_3", duplex_positive)


# ####method 6 - non_overlapping_sites_clip_data  ######
# from generate_interactions.non_overlapping_sites_clip_data.generate import main_run_clip
# main_run_clip()
# from generate_interactions.non_overlapping_sites_clip_data.generate import main_run_clash
# main_run_clash()
# from generate_interactions.non_overlapping_sites_clip_data.generate import  main_run_twice
# main_run_twice()
# from pipeline_steps.duplex_step import duplex as duplex_positive
# full_pipline("non_overlapping_sites_clip_data", "darnell_human_ViennaDuplex_features_negative", duplex_positive)


#method 6a - non_overlapping_sites_clip_data_random  ######
# from generate_interactions.non_overlapping_sites_clip_data.generate_random import full_pipline_run
# from pipeline_steps.duplex_step import duplex as duplex_positive
# full_pipline_run()
# full_pipline("non_overlapping_sites_clip_data", "random_darnell_human_ViennaDuplex_features_negative", duplex_positive)
#


