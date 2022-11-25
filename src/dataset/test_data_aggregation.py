"""
Data aggregator to process multiple data files
"""
import os
import argparse
import json
import logging
from tqdm import tqdm

from .utils import clean_sentence, detect_punctuation, remove_unwanted_punc
from ..utils import init_logger

logger = logging.getLogger(__name__)


def traverse_dir(rootpath: str):
    file_paths = []
    for subdir, dirs, files in os.walk(rootpath):
        for file in files:
            file_path = os.path.join(subdir, file)
            key, ext = os.path.splitext(file)

            if ext in [".wav", ".json"]:
                file_paths.append((key, file_path))

    file_paths = sorted(file_paths, key=lambda tup: tup[0])

    return file_paths


def generate_label(origin_text: str):
    line = remove_unwanted_punc(origin_text)
    line = clean_sentence(line)

    words = [word.strip() for word in line.split(" ")]

    punc_seq = []
    no_punc_word_seq = []
    for word in words:
        no_punc_word, punc = detect_punctuation(word)
        punc_seq.append(punc)
        no_punc_word_seq.append(no_punc_word)

    assert len(punc_seq) == len(no_punc_word_seq)

    text, label = " ".join(no_punc_word_seq), " ".join(punc_seq)
    return text, label


def process_files(paths: list):
    aggregated_datas = []

    for _, path in tqdm(paths, desc="Aggregation"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            origin_text, audio_path = data["text"], data["audio_path"]

            assert os.path.exists(audio_path)
            text, _ = generate_label(origin_text)

            aggregated_datas.append((text, audio_path))

    return aggregated_datas


def save_jsonl(datas, mode: str):
    logger.info("*** Saving jsonl files for %s ***" % mode)
    with open(f"{args.output_dir}/{mode}.jsonl", "w", encoding="utf-8") as f:
        for text, audio_path in datas:
            data = {"text": text, "audio_path": audio_path}
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


def aggregator(args):
    init_logger()

    text_data_paths = traverse_dir(args.text_data_dir)
    agg_datas = process_files(text_data_paths)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    save_jsonl(agg_datas, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """Paths"""
    parser.add_argument(
        "--text_data_dir",
        default="./test_data/text",
        type=str,
        help="The text data dir",
    )
    parser.add_argument(
        "--audio_data_dir",
        default="./test_data/audio",
        type=str,
        help="The audio data dir",
    )

    parser.add_argument(
        "--output_dir", default="./test_data", type=str, help="The aggregated data dir"
    )

    args = parser.parse_args()

    aggregator(args)
