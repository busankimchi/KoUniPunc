"""
Data loader that gets preprocessed data
"""
import os
import logging
from typing import List

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
import torchaudio
from transformers.tokenization_utils import PreTrainedTokenizer

from ..utils import load_tokenizer
from .welfare_call import InputExample, InputFeature, WelfareCallDatasetProcessor

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

    return tokens, input_ids, attention_mask, token_type_ids, label_ids, org_text_length


def convert_to_audio_features(
    example: InputExample, max_aud_len: int, wav_sampling_rate: int
):
    if example.audio_path is None:
        return None, None, False

    speech_array, sampling_rate = torchaudio.load(example.audio_path)

    if wav_sampling_rate != sampling_rate:
        speech_array = torchaudio.functional.resample(
            speech_array, orig_freq=sampling_rate, new_freq=wav_sampling_rate
        )

    audio_input = speech_array[0].tolist()
    speech_array_length = len(audio_input)

    padding_length = max_aud_len - speech_array_length

    if padding_length > 0:
        audio_input = audio_input + ([0] * padding_length)

    return audio_input, speech_array_length, True


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
            text_length,
        ) = convert_to_text_features(example, tokenizer, args.max_seq_len)

        (audio_input, audio_length, has_audio,) = convert_to_audio_features(
            example, args.max_audio_time, args.wav_sampling_rate
        )

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
            logger.info("text_length: %s" % text_length)

            # logger.info("audio_input: %s" % " ".join([str(x) for x in audio_input]))
            logger.info("audio_length: %s" % audio_length)

        features.append(
            InputFeature(
                text_input_ids=text_input_ids,
                text_attention_mask=text_att_mask,
                text_token_type_ids=text_token_type_ids,
                labels=labels,
                text_length=text_length,
                audio_input=audio_input,
                audio_length=audio_length,
                has_audio=has_audio,
            )
        )

    return features


def load_and_cache_examples(args, mode: str):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = "cached_{}_{}_{}_{}".format(
        args.task,
        list(filter(None, args.lm_model_name_or_path.split("/"))).pop(),
        args.max_seq_len,
        mode,
    )

    # cached data file 가져오기
    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    # cached data file 없으면 생성
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode not in ["train", "dev", "test"]:
            raise Exception("For mode, Only train, dev, test is available")

        examples = processor.get_examples(mode)
        features = convert_examples_to_features(args, examples)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_text_input_ids = torch.tensor(
        [f.text_input_ids for f in features], dtype=torch.long
    )
    logger.info("all_text_input_ids.size(): {}".format(all_text_input_ids.size()))

    all_text_attention_mask = torch.tensor(
        [f.text_attention_mask for f in features], dtype=torch.long
    )
    logger.info(
        "all_text_attention_mask.size(): {}".format(all_text_attention_mask.size())
    )

    all_text_token_type_ids = torch.tensor(
        [f.text_token_type_ids for f in features], dtype=torch.long
    )
    logger.info(
        "all_text_token_type_ids.size(): {}".format(all_text_token_type_ids.size())
    )

    all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
    logger.info("all_labels.size(): {}".format(all_labels.size()))

    all_text_length = torch.tensor([f.text_length for f in features], dtype=torch.long)
    logger.info("all_text_length.size(): {}".format(all_text_length.size()))

    all_audio_input = torch.tensor(
        [f.audio_input for f in features], dtype=torch.float32
    )
    logger.info("all_audio_input.size(): {}".format(all_audio_input.size()))

    all_audio_length = torch.tensor(
        [f.audio_length for f in features], dtype=torch.long
    )
    logger.info("all_audio_length.size(): {}".format(all_audio_length.size()))

    all_has_audio = torch.tensor([f.has_audio for f in features], dtype=torch.bool)
    logger.info("all_has_audio.size(): {}".format(all_has_audio.size()))

    dataset = TensorDataset(
        all_text_input_ids,
        all_text_attention_mask,
        all_text_token_type_ids,
        all_labels,
        all_text_length,
        all_audio_input,
        all_audio_length,
        all_has_audio,
    )

    return dataset
