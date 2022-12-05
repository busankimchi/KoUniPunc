"""
End-to-End process
"""
import os
from typing import List
import logging
import argparse

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from ..dataset.utils import clean_sentence
from ..dataset.welfare_call_dataset import InputFeature, WelfareCallDataset
from ..model.ko_unipunc import KoUniPunc
from .utils import (
    get_args,
    load_model,
    restore_punctuation_by_line,
    convert_text_to_features,
)
from ..utils import PUNCTUATION_LABELS, get_device, init_logger, load_tokenizer


logger = logging.getLogger(__name__)


def load_audio(input_audio: str) -> Tensor:
    speech_array, sampling_rate = torchaudio.load(input_audio)

    if sampling_rate != 16000:
        speech_array = torchaudio.functional.resample(
            speech_array, orig_freq=sampling_rate, new_freq=16000
        )

    return speech_array


def asr_process(speech_array: Tensor, device) -> str:
    processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to(
        device
    )

    inputs = processor(
        speech_array, sampling_rate=16000, return_tensors="pt", padding="longest"
    )

    input_values: Tensor = inputs.input_values.view(1, -1).to(device)
    # attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        # logits = model(input_values, attention_mask=attention_mask).logits
        logits: Tensor = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]


def cleanup_transcription(transcription: str):
    transcription = clean_sentence(transcription)

    return transcription.split()


def convert_to_dataset(
    args, pad_token_label_id: int, transcription: str, audio_path: str
):
    tokenizer = load_tokenizer(args)

    (
        input_ids,
        attention_mask,
        token_type_ids,
        slot_label_mask,
    ) = convert_text_to_features(transcription, args, tokenizer, pad_token_label_id)

    features = [
        InputFeature(
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            text_token_type_ids=token_type_ids,
            labels=slot_label_mask,
            audio_path=audio_path,
        )
    ]

    return WelfareCallDataset(args, features)


def punc_process(
    args, device, transcription: str, audio_path: str, model: KoUniPunc
) -> List[str]:
    label_lst = PUNCTUATION_LABELS
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    dataset = convert_to_dataset(args, pad_token_label_id, transcription, audio_path)

    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=1)

    all_slot_label_mask, preds = None, None

    for batch in data_loader:
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

    return preds_list[0]


def save_output_file(pred_config, transcription: str, pred_list: list):
    filename, _ = os.path.splitext(pred_config.input_audio_file)

    with open(f"{filename}_out.txt", "w", encoding="utf-8") as f:
        line = restore_punctuation_by_line(transcription, pred_list)
        f.write(line)


def e2e(pred_config):
    args = get_args(pred_config)
    logger.info(args)

    device = get_device(args)

    speech_array = load_audio(pred_config.input_audio_file)
    transcription = asr_process(speech_array, device)
    words = cleanup_transcription(transcription)
    logger.info(f"TRANSCRIPT ::: {words}")

    model = load_model(pred_config, args, device)
    pred_list = punc_process(args, device, words, pred_config.input_audio_file, model)

    logger.info(f"PRED LIST :: {pred_list}")

    # Write to output file
    save_output_file(pred_config, words, pred_list)


if __name__ == "__main__":
    init_logger()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_audio_file",
        default="/mnt/storage/sample/SAMPLE3.m4a",
        type=str,
        help="Input file for e2e prediction",
    )

    parser.add_argument(
        "--model_ckpt_path",
        # default="/mnt/storage/kounipunc/221129_training/ckpt/kounipunc_1_141303.pt",
        default="/mnt/storage/kounipunc/221129_training/ckpt/kounipunc_1_141303.pt",
        type=str,
        help="Model checkpoint path",
    )

    parser.add_argument(
        "--model_arg_path",
        # default="/mnt/storage/kounipunc/221129_training/ckpt/kounipunc_args.bin",
        default="/mnt/storage/kounipunc/221129_training/ckpt/kounipunc_args.bin",
        type=str,
        help="Model arg path",
    )

    parser.add_argument(
        "--output_dir",
        default="/mnt/storage/e2e_output",
        type=str,
        help="Output dir for e2e prediction",
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    pred_config = parser.parse_args()

    e2e(pred_config)
