"""
Data loader utils
"""
import re
import unicodedata
from dataclasses import dataclass

# from soynlp.normalizer import emoticon_normalize, repeat_normalize

from ..utils import PUNCTUATIONS


@dataclass
class InputExampleJSON:
    text: str
    metadata: dict
    audio_path: str
    label: str


def _is_control(char):
    """Checks whether `char` is a control character."""
    # (Code from Huggingface Transformers)
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def clean_sentence(sentence, remove_control=True):
    """
    - NFC Normalization
    - Invalid character removal (Some control character)
    - Whitespace cleanup
      - strip()
      - double whitespace, \n, \r, \t -> simple whitespace (" ")
      - Unify all Zs to simple whitespace (" ")
    """
    sentence = unicodedata.normalize("NFC", sentence)

    if remove_control:
        output = []
        for char in sentence:
            if _is_control(char) or ord(char) == 0xFFFD:
                continue
            output.append(char)

        sentence = "".join(output)

    return " ".join(sentence.strip().split())


# def preprocess(title: str, comment: str):
#     # Erase redundant \" in the start & end of the title
#     if title.startswith('"'):
#         title = title[1:]
#     if title.endswith('"'):
#         title = title[:-1]

#     # Change quotes
#     title = (
#         title.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
#     )

#     # Erase braces in title
#     braces = r"\[(.*?)\]"
#     braces2 = r"\{(.*?)\}"
#     braces3 = r"\【(.*?)\】"
#     braces4 = r"\<(.*?)\>"

#     title = re.sub(braces, "", title)
#     title = re.sub(braces2, "", title)
#     title = re.sub(braces3, "", title)
#     title = re.sub(braces4, "", title)

#     # Normalize the comment
#     comment = emoticon_normalize(comment, num_repeats=3)
#     comment = repeat_normalize(comment, num_repeats=3)

#     return title, comment


def remove_unwanted_punc(line: str):
    # remove '..'
    # .. 뒤에 다른 문자가 붙는 경우
    temp = re.search("\.{2,}[^.]", line)
    if temp is not None:
        line = re.sub("\.{2,}", "", line)

    # .. 만 있는 경우
    temp = re.search("\.{2,}", line)
    if temp is not None:
        line = re.sub("\.{2,}", " ", line)

    return line


def detect_punctuation(word: str):
    punc_reg = re.findall(r"[.,?!]$", word)

    if len(punc_reg) > 0:
        punc = punc_reg[0]
        no_punc_word = word.strip(punc)

        return no_punc_word, PUNCTUATIONS[punc]

    return word, "O"
