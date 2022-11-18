"""
Utils for training and models
"""
from typing import Tuple, Dict, Literal
from dataclasses import dataclass
import os
import re
import random
import logging

import torch
from torch import Tensor
import numpy as np
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from transformers import (
    AutoModelForTokenClassification,
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertModel,
    DistilBertModel,
    ElectraModel,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
)
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

from .tokenization_kobert import KoBertTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PredictResult:
    predict_result: Tensor  # B x T
    predict_logit: Tensor  # B x T x C


# VIRTUAL_EMB_LEN = 1
W2V_DIM = 768


LMModelClassType = Tuple[PretrainedConfig, PreTrainedModel, PreTrainedTokenizer]

LM_MODEL_CLASSES: Dict[str, LMModelClassType] = {
    "kobert": (BertConfig, BertModel, KoBertTokenizer),
    "distilkobert": (DistilBertConfig, DistilBertModel, KoBertTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "kobert-lm": (BertConfig, BertModel, KoBertTokenizer),
    "koelectra-base": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "koelectra-small": (ElectraConfig, ElectraModel, ElectraTokenizer),
}

LM_MODEL_PATH_MAP = {
    "kobert": "monologg/kobert",
    "distilkobert": "monologg/distilkobert",
    "bert": "bert-base-multilingual-cased",
    "kobert-lm": "monologg/kobert-lm",
    "koelectra-base": "monologg/koelectra-base-discriminator",
    "koelectra-small": "monologg/koelectra-small-discriminator",
}

SMModelClassType = Tuple[PretrainedConfig, SequenceFeatureExtractor]

SM_MODEL_CLASSES: Dict[str, SMModelClassType] = {
    "wav2vec2_large_korean": (Wav2Vec2Config, Wav2Vec2FeatureExtractor)
}

SM_MODEL_PATH_MAP = {"wav2vec2_large_korean": "kresnik/wav2vec2-large-xlsr-korean"}

PUNCTUATIONS = {",": "COM", ".": "STP", "?": "QUE", "!": "EXC"}
REVERSE_PUNCTUATIONS = {v: k for k, v in PUNCTUATIONS.items()}
PUNCTUATION_LABELS = ["UNK", "O"] + list(PUNCTUATIONS.values())


def get_eval_texts(args, mode: Literal["dev", "test"] = "dev"):
    file_map = {"dev": args.dev_file, "test": args.test_file}

    texts = []
    with open(os.path.join(args.data_dir, file_map[mode]), "r", encoding="utf-8") as f:
        for line in f:
            text, _ = line.split("\t")
            text = text.split()
            texts.append(text)

    return texts


def get_device(args):
    # GPU or CPU
    return "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))


def load_tokenizer(args):
    return LM_MODEL_CLASSES[args.lm_model_type][2].from_pretrained(
        args.lm_model_name_or_path
    )


def load_feature_extractor(args):
    return SM_MODEL_CLASSES[args.sm_model_type][2].from_pretrained(
        args.sm_model_name_or_path
    )


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_dir
        )  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def init_logger():
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True),
    }


def show_report(labels, preds, as_file=False):
    return classification_report(labels, preds, suffix=True, output_dict=as_file)


def lengths_to_padding_mask(lens: Tensor) -> Tensor:
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask
