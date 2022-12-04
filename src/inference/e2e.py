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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

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
    input_values: Tensor = inputs.input_values.to(device)
    # attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        # logits = model(input_values, attention_mask=attention_mask).logits
        logits: Tensor = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription


def cleanup_transcription(transcription):
    return transcription


def convert_to_dataset(
    args, pad_token_label_id: int, transcription: str, speech_array: Tensor
):
    tokenizer = load_tokenizer(args)

    (
        input_ids,
        attention_mask,
        token_type_ids,
        slot_label_mask,
    ) = convert_text_to_features(transcription, args, tokenizer, pad_token_label_id)

    audio_input = speech_array[0]
    audio_length = len(audio_input)

    padding_length = args.max_aud_len - audio_length
    if padding_length > 0:
        audio_input = F.pad(
            audio_input, pad=(0, padding_length), mode="constant", value=0
        )

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(token_type_ids, dtype=torch.long),
        audio_input,
        torch.tensor(audio_length, dtype=torch.long),
        True,
        np.array(slot_label_mask),
    )


def punc_process(
    args, transcription: str, speech_array: Tensor, model: KoUniPunc
) -> List[str]:
    label_lst = PUNCTUATION_LABELS
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    (
        text_input_ids,
        text_attention_mask,
        text_token_type_ids,
        audio_input,
        audio_length,
        has_audio,
        slot_label_mask,
    ) = convert_to_dataset(args, pad_token_label_id, transcription, speech_array)

    with torch.no_grad():
        inputs = {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_token_type_ids": text_token_type_ids,
            "audio_input": audio_input,
            "audio_length": audio_length,
            "has_audio": has_audio,
        }
        logits: Tensor = model(**inputs)
        preds = logits.detach().cpu().numpy()

    preds: np.ndarray = np.argmax(preds)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    pred_list = []

    for i in range(preds.shape[0]):
        if slot_label_mask[i] != pad_token_label_id:
            pred_list.append(slot_label_map[preds[i]])

    return pred_list


def save_output_file(pred_config, transcription, pred_list):
    filename, _ = os.path.splitext(pred_config.input_audio_file)

    with open(f"{filename}_out.txt", "w", encoding="utf-8") as f:
        line = restore_punctuation_by_line(transcription, pred_list)
        f.write(f"{line}\n")

        return line


def e2e(pred_config):
    args = get_args(pred_config)
    device = get_device(args)

    speech_array = load_audio(pred_config.input_audio_file)
    transcription = asr_process(speech_array, device)
    # transcription = cleanup_transcription(transcription)

    model = load_model(pred_config, args, device)
    pred_list = punc_process(args, transcription, speech_array, model)

    # Write to output file
    save_output_file(pred_config, transcription, pred_list)


if __name__ == "__main__":
    init_logger()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_audio_file",
        default=None,
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
        default="/mnt/storage/kounipunc/e2e",
        type=str,
        help="Output dir for e2e prediction",
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    pred_config = parser.parse_args()

    e2e(pred_config)
