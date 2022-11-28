"""
Prediction Entry file
"""
import logging
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers.tokenization_utils import PreTrainedTokenizer

# from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

from .utils import get_args, load_model, read_input_file, save_output_file
from ..utils import (
    init_logger,
    load_tokenizer,
    get_device,
    PUNCTUATION_LABELS,
)

logger = logging.getLogger(__name__)


def convert_text_to_tensor_dataset(
    texts,
    pred_config,
    args,
    tokenizer: PreTrainedTokenizer,
    pad_token_label_id: int,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []
    all_text_length = []

    for words in texts:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[
                : (args.max_seq_len - special_tokens_count)
            ]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        org_text_length = len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * org_text_length

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - org_text_length
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)
        all_text_length.append(org_text_length)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)
    all_text_length = torch.tensor(all_text_length, dtype=torch.long)

    return (
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_slot_label_mask,
        all_text_length,
    )


def convert_audio_to_tensor_dataset(audio_paths, pred_config, args):
    all_audio_input = []
    all_audio_length = []
    all_has_audio = []

    for audio_path in audio_paths:
        if audio_path is None:
            all_audio_input.append(None)
            all_audio_length.append(None)
            all_has_audio.append(False)
            continue

        speech_array, sampling_rate = torchaudio.load(audio_path)

        if args.wav_sampling_rate != sampling_rate:
            speech_array = torchaudio.functional.resample(
                speech_array, orig_freq=sampling_rate, new_freq=args.wav_sampling_rate
            )

        audio_input = speech_array[0].tolist()
        speech_array_length = len(audio_input)

        padding_length = args.max_aud_len - speech_array_length

        if padding_length > 0:
            audio_input = audio_input + ([0] * padding_length)

        all_audio_input.append(speech_array)
        all_audio_length.append(speech_array_length)
        all_has_audio.append(True)

    all_audio_input = torch.tensor(all_audio_input, dtype=torch.float32)
    all_audio_length = torch.tensor(all_audio_length, dtype=torch.long)
    all_has_audio = torch.tensor(all_has_audio, dtype=torch.bool)

    return all_audio_input, all_audio_length, all_has_audio


def convert_input_file_to_tensor_dataset(
    texts, audio_paths, pred_config, args, pad_token_label_id: int
):
    tokenizer = load_tokenizer(args)

    (
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_slot_label_mask,
        all_text_length,
    ) = convert_text_to_tensor_dataset(
        texts, pred_config, args, tokenizer, pad_token_label_id
    )

    (
        all_audio_input,
        all_audio_length,
        all_has_audio,
    ) = convert_audio_to_tensor_dataset(audio_paths, pred_config, args)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_slot_label_mask,
        all_text_length,
        all_audio_input,
        all_audio_length,
        all_has_audio,
    )

    return dataset


def inference(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    label_lst = PUNCTUATION_LABELS

    logger.info(args)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    texts, audio_paths = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(
        texts, audio_paths, pred_config, args, pad_token_label_id
    )

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=pred_config.batch_size
    )

    all_slot_label_mask = None
    preds = None

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
                "labels": batch[6],
            }
            outputs = model(**inputs)
            logits: Tensor = outputs[1]

            slot_label_mask: Tensor = batch[3]

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
        default="sample_pred_punc.txt",
        type=str,
        help="Input file for prediction",
    )

    parser.add_argument(
        "--model_ckpt_dir",
        # default="/mnt/data_storage/kounipunc/ckpt",
        default="./ckpt",
        type=str,
        help="Path to save, load model",
    )

    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size for prediction"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    pred_config = parser.parse_args()
    inference(pred_config)
