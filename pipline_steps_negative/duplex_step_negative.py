from pathlib import Path
from pandas import DataFrame, Series
import pandas as pd
from consts.global_consts import HUMAN_SITE_EXTENDED_LEN, ROOT_PATH, BIOMART_PATH, GENERATE_DATA_PATH
from consts.global_consts import DUPLEX_DICT
from duplex.Duplex import Duplex
from utils.logger import logger
from utils.utilsfile import get_wrapper, read_csv, to_csv


def do_duplex(mirna: str, target: str, cls: Duplex) -> Series:

    if pd.isna(mirna) or pd.isna(target):
        return Series({"duplex_valid": False,
              "not_match_site":"",
              "site":"",
              "mrna_bulge": "",
              "mrna_inter": "",
              "mir_inter": "",
              "mir_bulge": ""})
    dp = cls.fromChimera(mirna, target)
    return Series({"duplex_valid": dp.valid,
                   "not_match_site": dp.site_non_match_tail,
                   "site": dp.site[::-1],
             "mrna_bulge": dp.mrna_bulge,
              "mrna_inter": dp.mrna_inter,
              "mir_inter": dp.mir_inter,
              "mir_bulge": dp.mir_bulge})


def duplex(method: str, fin: str, fout: str):
    duplex_cls: Duplex = DUPLEX_DICT[method]
    logger.info(f"{method} do_duplex to {fin}")
    in_df: DataFrame = read_csv(Path(fin))
    seq_cols = ['miRNA sequence', 'full_mrna']
    in_df[seq_cols] = in_df[seq_cols].replace(to_replace='T', value='U', regex=True)
    # [in_df["miRNA sequence"].notnull() & in_df.site.notnull()]
    d = in_df.loc[1]["full_mrna"]
    # in_df = in_df.truncate(before=0, after=10)
    #duplex_df = do_duplex(in_df.loc[1]["miRNA sequence"], in_df.loc[1]["full_mrna"], cls=duplex_cls)
    duplex_df = in_df.apply(func=get_wrapper(
        do_duplex, "miRNA sequence", "full_mrna", cls=duplex_cls),
        axis=1)


    result = pd.merge(left=in_df, right=duplex_df, left_index=True, right_index=True, how='left')

    #############append#####################
    result = result[result['duplex_valid']==True]
    result["duplex_method"] = method
    to_csv(result, Path(fout))

