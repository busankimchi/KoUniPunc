"""
Data aggregator to process multiple data files
"""
import os
import re
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

from .utils import clean_sentence, detect_punctuation, remove_unwanted_punc
from ..utils import init_logger

logger = logging.getLogger(__name__)

# DATA_BASE_PATH = "/mnt/data_storage/186.복지 분야 콜센터 상담데이터"
DATA_BASE_PATH = "./data/186.복지 분야 콜센터 상담데이터"

TRAIN_BASE_PATH = f"{DATA_BASE_PATH}/01.데이터/1.Training"
DEV_BASE_PATH = f"{DATA_BASE_PATH}/01.데이터/2.Validation"


def traverse_dir(rootpath: str):
    file_paths = []

    label_path = os.path.join(rootpath, "라벨링데이터")
    for subdir, dirs, files in os.walk(label_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            key, ext = os.path.splitext(file)

            if ext in [".json"]:
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
        no_punc_word_seq.append(no_punc_word.strip())

    assert len(punc_seq) == len(no_punc_word_seq)

    text, label = " ".join(no_punc_word_seq), " ".join(punc_seq)
    return text, label


def process_files(paths: list, mode: str):
    aggregated_datas = []
    aggregated_labels = []
    for key, label_path in tqdm(paths, desc="Aggregation"):
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            origin_text = data["inputText"][0]["orgtext"]
            audio_path = data["dialogs"][0]["audioPath"]
            metadata = data["info"][0]["metadata"]

            # print(f"before : {audio_path}")

            audio_path = re.sub("\\\\", "/", audio_path)
            audio_path = re.sub("^Y:", f"{DATA_BASE_PATH}/01.데이터", audio_path)

            if mode == "train":
                audio_path = re.sub("03.원천데이터", "1.Training/원천데이터", audio_path)
            elif mode == "dev":
                audio_path = re.sub("03.원천데이터", "2.Validation/원천데이터", audio_path)

            # print(f"after : {audio_path}")

            if not os.path.exists(audio_path):
                continue

            text, label = generate_label(origin_text)

            if len(text) == 0:
                continue

            aggregated_datas.append((text, metadata, audio_path))
            aggregated_labels.append(label)

    return aggregated_datas, aggregated_labels


def save_jsonl(datas, labels, mode: str):
    logger.info("*** Saving jsonl files for %s ***" % mode)
    with open(f"{args.output_dir}/{mode}.jsonl", "w", encoding="utf-8") as f:
        for (text, metadata, audio_path), label in zip(datas, labels):
            data = {
                "text": text,
                "metadata": metadata,
                "audio_path": audio_path,
                "label": label,
            }
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


def aggregator(args):
    init_logger()

    train_paths = traverse_dir(args.train_data_dir)
    agg_train_datas, agg_train_labels = process_files(train_paths, "train")

    dev_paths = traverse_dir(args.dev_data_dir)
    agg_dev_datas, agg_dev_labels = process_files(dev_paths, "dev")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    save_jsonl(agg_train_datas, agg_train_labels, "train")
    save_jsonl(agg_dev_datas, agg_dev_labels, "dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """Paths"""
    parser.add_argument(
        "--train_data_dir", default=TRAIN_BASE_PATH, type=str, help="Train data dir"
    )
    parser.add_argument(
        "--dev_data_dir", default=DEV_BASE_PATH, type=str, help="Dev data dir"
    )
    parser.add_argument(
        "--output_dir",
        default=DATA_BASE_PATH,
        type=str,
        help="The aggregated data dir",
    )

    args = parser.parse_args()

    aggregator(args)
