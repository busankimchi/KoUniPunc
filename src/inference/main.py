"""
Prediction Entry file
"""
import logging
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler

from ..dataset.welfare_call_dataset import InputFeature, WelfareCallDataset

from .utils import (
    convert_text_to_features,
    get_args,
    load_model,
    read_input_file,
    save_output_file,
)
from ..utils import init_logger, load_tokenizer, get_device, PUNCTUATION_LABELS

logger = logging.getLogger(__name__)


def convert_input_file_to_dataset(
    texts: list, audio_paths: list, args, pad_token_label_id: int
):
    tokenizer = load_tokenizer(args)

    features = []
    for text, audio_path in zip(texts, audio_paths):
        (
            text_input_ids,
            text_attention_mask,
            text_token_type_ids,
            slot_label_mask,
        ) = convert_text_to_features(text, args, tokenizer, pad_token_label_id)

        features.append(
            InputFeature(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                labels=slot_label_mask,
                audio_path=audio_path,
            )
        )

    return WelfareCallDataset(args, features)


def inference(pred_config):
    # load model and args
    args = get_args(pred_config)

    logger.info(args)

    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    label_lst = PUNCTUATION_LABELS

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    texts, audio_paths = read_input_file(pred_config)
    dataset = convert_input_file_to_dataset(
        texts, audio_paths, args, pad_token_label_id
    )

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=pred_config.batch_size
    )

    all_slot_label_mask, preds = None, None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "text_input_ids": batch[0],
                "text_attention_mask": batch[1],
                "text_token_type_ids": batch[2],
                "audio_input": batch[3],
                "audio_length": batch[4],
                "has_audio": batch[5],
            }

            logits: Tensor = model(**inputs)
            slot_label_mask: Tensor = batch[6]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = slot_label_mask.detach().cpu().numpy()

            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(
                    all_slot_label_mask, slot_label_mask.detach().cpu().numpy(), axis=0
                )

    preds: np.ndarray = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])

    # Write to output file
    save_output_file(pred_config, texts, preds_list)


if __name__ == "__main__":
    init_logger()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        default="./data/sample/sample_pred_punc.jsonl",
        type=str,
        help="Input file for prediction",
    )

    parser.add_argument(
        "--model_ckpt_path",
        # default="./kounipunc/kounipunc_1_141303.pt",
        default="/mnt/storage/kounipunc/221129_training/ckpt/kounipunc_1_141303.pt",
        type=str,
        help="Model checkpoint path",
    )

    parser.add_argument(
        "--model_arg_path",
        # default="./kounipunc/kounipunc_args.bin",
        default="/mnt/storage/kounipunc/221129_training/ckpt/kounipunc_args.bin",
        type=str,
        help="Model arg path",
    )

    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size for prediction"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    pred_config = parser.parse_args()
    inference(pred_config)
