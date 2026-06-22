from enum import Enum
from utils.config import motif_mode

class Token(str, Enum):
    SEQUENCE = "<SEQUENCE>"
    LABEL = "<LABEL>"
    MARKER_START = "<PTM_MOTIF_START>" if motif_mode =="bpe_token" else "<PTM_RESIDUE_START>"
    MARKER_END = "<PTM_MOTIF_END>" if motif_mode =="bpe_token" else "<PTM_RESIDUE_END>"
    POSITIVE = "<POSITIVE>"
    NEGATIVE = "<NEGATIVE>"


NEW_TOKENS = [t.value for t in Token]

PROTGPT2_END_TOKEN = "<|endoftext|>"
PROGEN_END_TOKEN = "<|eos|>"
PROGEN_PAD_TOKEN = "<|pad|>"
# PROTGPT2_PAD_TOKEN = "<|pad|>"

class DiseaseToken(str, Enum):
    SEQUENCE = "<SEQUENCE>"
    LABEL = "<LABEL>"
    MARKER_START = "<PTM_MOTIF_START>" if motif_mode == "bpe_token" else "<PTM_RESIDUE_START>"
    MARKER_END = "<PTM_MOTIF_END>" if motif_mode == "bpe_token" else "<PTM_RESIDUE_END>"

    OTHER = "<OTHER>"
    LUNG_ADENOCARCINOMA = "<LUNG_ADENOCARCINOMA>"
    HEAD_AND_NECK_SQUAMOUS_CELL_CARCINOMA = "<HEAD_AND_NECK_SQUAMOUS_CELL_CARCINOMA>"
    SKIN_CUTANEOUS_MELANOMA = "<SKIN_CUTANEOUS_MELANOMA>"
    BLADDER_UROTHELIAL_CARCINOMA = "<BLADDER_UROTHELIAL_CARCINOMA>"
    STOMACH_ADENOCARCINOMA = "<STOMACH_ADENOCARCINOMA>"

DISEASE_NEW_TOKENS = [t.value for t in DiseaseToken]
