"""
Data loader utils
"""
import re
import unicodedata

from utils.utils import PUNCTUATIONS


def _is_control(char):
    """Checks whether `char` is a control character."""
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
