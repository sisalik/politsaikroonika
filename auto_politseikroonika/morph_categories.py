from enum import Enum

class WordClass(str, Enum):
    """Word class morphological categories of Estonian words.

    More info:
    https://github.com/Filosoft/vabamorf/blob/e6d42371006710175f7ec328c98f90b122930555/doc/tagset.md
    """
    ADJECTIVE_POSITIVE = "A"
    ADJECTIVE_COMPARATIVE = "C"
    ADVERB = "D"
    GENITIVE_ATTRIBUTE = "G"
    PROPER_NOUN = "H"
    INTERJECTION = "I"
    CONJUNCTION = "J"
    PRE_POSTPOSITION = "K"
    CARDINAL_NUMERAL = "N"
    ORDINAL_NUMERAL = "O"
    PRONOUN = "P"
    NOUN = "S"
    ADJECTIVE_SUPERLATIVE = "U"
    VERB = "V"
    ADVERB_LIKE = "X"
    ABBREVIATION = "Y"
    PUNCTUATION = "Z"
