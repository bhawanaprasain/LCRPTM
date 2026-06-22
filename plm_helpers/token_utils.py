from dataclasses import dataclass
from plm_helpers.constants import Token


@dataclass
class TokenIDs:
    MARKER_START: int
    MARKER_END: int
    LABEL_ID: int
    POS_ID: int
    NEG_ID: int
    SEQ_ID: int


def get_token_ids(tokenizer):
    return TokenIDs(
        MARKER_START=tokenizer.convert_tokens_to_ids(Token.MARKER_START),
        MARKER_END=tokenizer.convert_tokens_to_ids(Token.MARKER_END),
        LABEL_ID=tokenizer.convert_tokens_to_ids(Token.LABEL),
        POS_ID=tokenizer.convert_tokens_to_ids(Token.POSITIVE),
        NEG_ID=tokenizer.convert_tokens_to_ids(Token.NEGATIVE),
        SEQ_ID=tokenizer.convert_tokens_to_ids(Token.SEQUENCE),
    )
