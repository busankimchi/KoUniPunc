"""
Data loader that gets preprocessed data
"""
from typing import List
import os
import logging

from tqdm import tqdm

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from utils.utils import load_tokenizer
from .welfare_call import InputExample, WelfareCallDatasetProcessor
from .welfare_call_dataset import InputFeature, WelfareCallDataset

logger = logging.getLogger(__name__)

processors = {"wfc-ko-punc": WelfareCallDatasetProcessor}


def convert_to_text_features(
    example: InputExample,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    pad_token_label_id=-100,
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

    # Tokenize word by word (for Token Classification)
    tokens = []
    label_ids = []
    for word, slot_label in zip(example.words, example.labels):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        label_ids.extend(
            [int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1)
        )

    # Account for [CLS] and [SEP]
    special_tokens_count = 2
    if len(tokens) > max_seq_len - special_tokens_count:
        tokens = tokens[: (max_seq_len - special_tokens_count)]
        label_ids = label_ids[: (max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    token_type_ids = [sequence_a_segment_id] * len(tokens)

    # Add [CLS] token
    tokens = [cls_token] + tokens
    label_ids = [pad_token_label_id] + label_ids
    token_type_ids = [cls_token_segment_id] + token_type_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    org_text_length = len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * org_text_length

    # Zero-pad up to the sequence length.
    padding_length = max_seq_len - org_text_length
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + (
        [0 if mask_padding_with_zero else 1] * padding_length
    )
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    label_ids = label_ids + ([pad_token_label_id] * padding_length)

    assert (
        len(input_ids) == max_seq_len
    ), f"Error with input length {len(input_ids)} vs {max_seq_len}"
    assert (
        len(attention_mask) == max_seq_len
    ), f"Error with attention mask length {len(attention_mask)} vs {max_seq_len}"
    assert (
        len(token_type_ids) == max_seq_len
    ), f"Error with token type length {len(token_type_ids)} vs {max_seq_len}"
    assert (
        len(label_ids) == max_seq_len
    ), f"Error with slot labels length {len(label_ids)} vs {max_seq_len}"

    return tokens, input_ids, attention_mask, token_type_ids, label_ids


def convert_examples_to_features(
    args, examples: List[InputExample]
) -> List[InputFeature]:
    tokenizer = load_tokenizer(args)

    features = []
    progress = tqdm(examples, desc="Example Convert")
    for ex_idx, example in enumerate(progress):
        (
            tokens,
            text_input_ids,
            text_att_mask,
            text_token_type_ids,
            labels,
        ) = convert_to_text_features(example, tokenizer, args.max_seq_len)

        if ex_idx < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("text_tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info(
                "text_input_ids: %s" % " ".join([str(x) for x in text_input_ids])
            )
            logger.info(
                "text_attention_mask: %s", " ".join([str(x) for x in text_att_mask])
            )
            logger.info(
                "text_token_type_ids: %s"
                % " ".join([str(x) for x in text_token_type_ids])
            )
            logger.info("labels: %s" % " ".join([str(x) for x in labels]))

        features.append(
            InputFeature(
                text_input_ids=text_input_ids,
                text_attention_mask=text_att_mask,
                text_token_type_ids=text_token_type_ids,
                labels=labels,
                audio_path=example.audio_path,
            )
        )

    return features


def cache_and_load_features(args, mode: str) -> WelfareCallDataset:
    cached_file_name = "_".join(
        [
            "cached",
            args.task,
            list(filter(None, args.lm_model_name_or_path.split("/"))).pop(),
            mode,
        ]
    )
    cached_features_file = os.path.join(args.data_dir, cached_file_name)

    # Load cached text features if exist
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features: List[InputFeature] = torch.load(cached_features_file)

    # Cache text features if not exist
    else:
        logger.info(f"{cached_features_file} does not exist! Saving features...")
        processor = processors[args.task](args)
        examples = processor.get_examples(mode)
        features = convert_examples_to_features(args, examples)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        logger.info(
            "Saving features into cached file %s ... finished!", cached_features_file
        )

    return WelfareCallDataset(args, features)
