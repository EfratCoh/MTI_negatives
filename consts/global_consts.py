from pathlib import Path
from duplex.ViennaDuplex import ViennaDuplex


ROOT_PATH = Path("/sise/home/efrco/efrco-master/")
log_file = ROOT_PATH / "pipeline_log.log"
BIOMART_PATH = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/benorgi/pipeline/data/biomart/")
DATA_PATH = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/benorgi/pipeline/")
POSITIVE_PATH_DATA = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/benorgi/pipeline/data/pipeline_steps/features/")
MERGE_DATA = ROOT_PATH / "data/positive_interactions/"
NEGATIVE_DATA_PATH = Path("/sise/home/efrco/efrco-master/data/negative_interactions/")
DATA_PATH_INTERACTIONS = Path("/sise/home/efrco/efrco-master/data/")
GENERATE_DATA_PATH= ROOT_PATH / "generate_interactions/"
CLIP_PATH_DATA = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/Isana/")
POSITIVE_PATH_FEATUERS = ROOT_PATH / "data/positive_interactions/positive_interactions_new/featuers_step/"


SITE_EXTRA_CHARS: int = 3
HUMAN_SITE_EXTENDED_LEN: int = 25
MINIMAL_BLAST_COVERAGE = 95
MINIMAL_BLAST_IDENTITY = 95
MINIMAL_LENGTH_TO_BLAST = 10
DUPLEX_DICT = {"ViennaDuplex": ViennaDuplex}

CONFIG = {
    'minimum_pairs_for_interaction': 11,
    'duplex_method' :  ["vienna"],
    'max_process' : 1
 #   'duplex_method': ["vienna", "miranda", "vienna_with_constraint"]

    #'duplex_method' :  ["miranda"]
}
