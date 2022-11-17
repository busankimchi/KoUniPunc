import os
import re
import argparse
import json
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    aggregated_labels = []
    for key, label_path in tqdm(paths, desc="Aggregation"):
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            origin_text = data["inputText"][0]["orgtext"]
            audio_path = data["dialogs"][0]["audioPath"]
            metadata = data["info"][0]["metadata"]

            # print(f"before : {audio_path}")
            audio_path = re.sub("^Y:", "./data/sample_data", audio_path)
            audio_path = re.sub("03.원천데이터", "원천데이터", audio_path)
            audio_path = re.sub("01.대학병원", "TS1_01.대학병원/01.대학병원", audio_path)
            audio_path = re.sub("\\\\", "/", audio_path)
            # print(f"after : {audio_path}")

            assert os.path.exists(audio_path)

            text, label = generate_label(origin_text)

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

    label_paths = traverse_dir(args.label_data_dir)
    agg_datas, agg_labels = process_files(label_paths)

    datas_train, datas_dev, labels_train, labels_dev = train_test_split(
        agg_datas, agg_labels, test_size=args.split_ratio, shuffle=True, random_state=42
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    save_jsonl(datas_train, labels_train, "train")
    save_jsonl(datas_dev, labels_dev, "dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """Paths"""
    parser.add_argument(
        "--label_data_dir",
        default="./data/sample_data/라벨링데이터",
        type=str,
        help="The label data dir",
    )
    parser.add_argument(
        "--origin_data_dir",
        default="./data/sample_data/원천데이터",
        type=str,
        help="The origin data dir",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/sample_data",
        type=str,
        help="The aggregated data dir",
    )
    parser.add_argument(
        "--split_ratio",
        default=0.1,
        type=float,
        help="split rate for train and dev",
    )

    args = parser.parse_args()

    aggregator(args)
