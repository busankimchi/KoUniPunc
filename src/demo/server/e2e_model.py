"""
End-to-End process
"""
import os
from typing import List
import logging

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from ...inference.utils import restore_punctuation_by_line, convert_text_to_features
from ...utils import PUNCTUATION_LABELS, get_device, init_logger, load_tokenizer
from .config import MODEL_CKPT_PATH, MODEL_ARG_PATH
from ...model.ko_unipunc import KoUniPunc


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


def cleanup_transcription(transcription) -> str:
    return transcription


def load_punc_model(args, device) -> KoUniPunc:
    # Check whether model exists
    if not os.path.exists(MODEL_CKPT_PATH):
        raise Exception("Model doesn't exists! Train first!")

    try:
        # Config will be automatically loaded from model_ckpt_dir
        model = KoUniPunc(args)

        model_pt = torch.load(MODEL_CKPT_PATH)
        model.load_state_dict(model_pt["model_state_dict"])

        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")

    except:
        raise Exception("Some model files might be missing...")

    return model


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


def punc_process(args, transcription: str, speech_array: Tensor, device) -> List[str]:
    label_lst = PUNCTUATION_LABELS
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    model = load_punc_model(args, device)

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


def e2e(input_audio_file: str) -> str:
    init_logger()

    args = torch.load(MODEL_ARG_PATH)

    device = get_device(args)

    speech_array = load_audio(input_audio_file)
    transcription = asr_process(speech_array, device)
    # transcription = cleanup_transcription(transcription)

    pred_list = punc_process(args, transcription, speech_array, device)

    line = restore_punctuation_by_line(transcription, pred_list)
    return line
