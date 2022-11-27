"""
Welfare Call dataset
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import os
import logging

import jsonlines

from ..utils import PUNCTUATION_LABELS


logger = logging.getLogger(__name__)


@dataclass
class InputExampleJSON:
    text: str
    metadata: dict
    audio_path: str
    label: str


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
