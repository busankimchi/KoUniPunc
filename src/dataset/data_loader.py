"""
Data loader that gets preprocessed data
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import os
import logging

# import jsonlines
import json
import jsonlines
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

import torch
from torch import Tensor
import torchaudio
from torch.utils.data import TensorDataset

from .utils import InputExampleJSON
from ..utils import (
    PUNCTUATION_LABELS,
    lengths_to_padding_mask,
    load_feature_extractor,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The slot labels of the example.
    """

    guid: str
    words: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    audio_path: Optional[str] = None


@dataclass
class InputFeatures:
    """A single set of features of data."""

    text_input_ids: list
    text_attention_mask: list
    text_label_ids: list
    text_token_type_ids: Optional[list] = None

    audio_input: Optional[list] = None
    audio_length: Optional[int] = None
    audio_sampling_rate: Optional[int] = None
    has_audio: bool = False


class WelfareCallDatasetProcessor(object):
    """Processor for welfare call data set"""

    def __init__(self, args):
        self.args = args
        self.labels_lst = PUNCTUATION_LABELS

    @classmethod
    def _read_file(cls, input_file: str) -> List[InputExampleJSON]:
        """Read jsonl file, and return words and label as list"""
        with jsonlines.open(input_file) as f:
            lines: List[InputExampleJSON] = []
            for line in f.iter():
                lines.append(InputExampleJSON(**line))
            return lines

    def _create_examples(
        self, dataset: List[InputExampleJSON], set_type: str
    ) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples: List[InputExample] = []
        for i, data in enumerate(dataset):
            words = data.text.split()
            labels = data.label.split()

            labels_idx = [
                self.labels_lst.index(label)
                if label in self.labels_lst
                else self.labels_lst.index("UNK")
                for label in labels
            ]

            assert len(words) == len(labels_idx)

            example = {
                "guid": f"{set_type}-{i}",
                "words": words,
                "labels": labels_idx,
                "audio_path": data.audio_path,
            }

            if i % 10000 == 0:
                logger.info(f"CREATING :: EX :: {data}")

            examples.append(InputExample(**example))

        return examples

    def get_examples(self, mode: Literal["train", "dev", "test"]) -> List[InputExample]:
        file_map = {
            "train": self.args.train_file,
            "dev": self.args.dev_file,
            "test": self.args.test_file,
        }

        example_file_path = os.path.join(self.args.data_dir, file_map[mode])
        logger.info("LOOKING AT {}".format(example_file_path))

        return self._create_examples(self._read_file(example_file_path), mode)


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

    # Tokenize word by word (for NER)
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

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_len - len(input_ids)
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


def convert_to_audio_features(example: InputExample):
    if example.audio_path is None:
        return None, None, False

    speech_array, sampling_rate = torchaudio.load(example.audio_path)
    speech_array_length = speech_array.size()[0]
    return speech_array, speech_array_length, sampling_rate, True


def convert_examples_to_features(
    args, examples: List[InputExample]
) -> List[InputFeatures]:
    tokenizer = load_tokenizer(args)

    features = []
    progress = tqdm(examples, desc="Example Convert")
    for ex_idx, example in enumerate(progress):
        (
            tokens,
            text_input_ids,
            text_att_mask,
            text_token_type_ids,
            text_label_ids,
        ) = convert_to_text_features(example, tokenizer, args.max_seq_len)

        (
            audio_input,
            audio_length,
            sampling_rate,
            has_audio,
        ) = convert_to_audio_features(example)

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
            logger.info(
                "text_label_ids: %s" % " ".join([str(x) for x in text_label_ids])
            )

            logger.info("audio_input: %s" % " ".join([str(x) for x in audio_input]))
            logger.info("audio_length: %s" % audio_length)
            logger.info("sampling_rate: %s" % sampling_rate)

        features.append(
            InputFeatures(
                text_input_ids=text_input_ids,
                text_attention_mask=text_att_mask,
                text_token_type_ids=text_token_type_ids,
                text_label_ids=text_label_ids,
                audio_input=audio_input,
                audio_length=audio_length,
                sampling_rate=sampling_rate,
                has_audio=has_audio,
            )
        )

    assert False
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
    all_text_attention_mask = torch.tensor(
        [f.text_attention_mask for f in features], dtype=torch.long
    )
    all_text_token_type_ids = torch.tensor(
        [f.text_token_type_ids for f in features], dtype=torch.long
    )
    all_text_label_ids = torch.tensor(
        [f.text_label_ids for f in features], dtype=torch.long
    )
    all_audio_input = torch.tensor([f.audio_input for f in features], dtype=torch.long)
    all_audio_length = torch.tensor(
        [f.audio_length for f in features], dtype=torch.long
    )
    all_audio_sampling_rate = torch.tensor(
        [f.audio_sampling_rate for f in features], dtype=torch.long
    )
    all_has_audio = torch.tensor([f.has_audio for f in features], dtype=torch.long)

    logger.info("all_text_input_ids.size(): {}".format(all_text_input_ids.size()))
    logger.info(
        "all_text_attention_mask.size(): {}".format(all_text_attention_mask.size())
    )
    logger.info(
        "all_text_token_type_ids.size(): {}".format(all_text_token_type_ids.size())
    )
    logger.info("all_text_label_ids.size(): {}".format(all_text_label_ids.size()))

    logger.info("all_audio_input.size(): {}".format(all_audio_input.size()))
    logger.info("all_audio_length.size(): {}".format(all_audio_length.size()))
    logger.info(
        "all_audio_sampling_rate.size(): {}".format(all_audio_sampling_rate.size())
    )
    logger.info("all_has_audio.size(): {}".format(all_has_audio.size()))

    dataset = TensorDataset(
        all_text_input_ids,
        all_text_attention_mask,
        all_text_token_type_ids,
        all_text_label_ids,
        all_audio_input,
        all_audio_length,
        all_audio_sampling_rate,
        all_has_audio,
    )

    return dataset
