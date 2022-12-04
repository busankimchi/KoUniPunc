"""
Inference utilities
"""
import os
import logging

import jsonlines

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from ..model.ko_unipunc import KoUniPunc
from ..utils import REVERSE_PUNCTUATIONS


logger = logging.getLogger(__name__)


def get_args(pred_config):
    if not os.path.exists(pred_config.model_arg_path):
        raise Exception("Model arg doesn't exists!")

    return torch.load(pred_config.model_arg_path)


def load_model(pred_config, args, device) -> KoUniPunc:
    # Check whether model exists
    if not os.path.exists(pred_config.model_ckpt_path):
        raise Exception("Model doesn't exists! Train first!")

    try:
        # Config will be automatically loaded from model_ckpt_dir
        model = KoUniPunc(args)

        model_pt = torch.load(pred_config.model_ckpt_path)
        model.load_state_dict(model_pt["model_state_dict"])

        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")

    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    texts, audio_paths = [], []

    with jsonlines.open(pred_config.input_file) as f:
        for line in f.iter():
            texts.append(line["text"].split())
            audio_paths.append(line["audio_path"])

    return texts, audio_paths


def restore_punctuation_by_line(words, preds):
    line = ""

    for word, pred in zip(words, preds):
        if pred == "O":
            line = line + word + " "
        else:
            line = line + word + REVERSE_PUNCTUATIONS[pred] + " "

    return line.strip()


def save_output_file(pred_config, lines, preds_list):
    filename, _ = os.path.splitext(pred_config.input_file)

    with open(f"{filename}_out.txt", "w", encoding="utf-8") as f:
        for words, preds in zip(lines, preds_list):
            line = restore_punctuation_by_line(words, preds)
            f.write(f"{line}\n")

    logger.info("*** Prediction Done! ***")


def convert_text_to_features(
    text,
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

    tokens = []
    slot_label_mask = []
    for word in text:
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
        slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

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

    return input_ids, attention_mask, token_type_ids, slot_label_mask
