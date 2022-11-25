"""
Inference utilities
"""
import os
import logging

import jsonlines

import torch

from ..model.ko_unipunc import KoUniPunc
from ..utils import REVERSE_PUNCTUATIONS


logger = logging.getLogger(__name__)


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_ckpt_dir, "training_args.bin"))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_ckpt_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        # Config will be automatically loaded from model_ckpt_dir
        model = KoUniPunc(args)

        state_dict = torch.load(
            os.path.join(pred_config.model_ckpt_dir, "kounipunc_state.pt")
        )
        model.load_state_dict(state_dict)

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


def save_output_file(pred_config, lines, preds_list):
    filename, ext = os.path.splitext(pred_config.input_file)

    with open(f"{filename}_out" + ext, "w", encoding="utf-8") as f:
        for words, preds in zip(lines, preds_list):
            line = ""
            for word, pred in zip(words, preds):
                if pred == "O":
                    line = line + word + " "
                else:
                    line = line + word + REVERSE_PUNCTUATIONS[pred] + " "

            f.write(f"{line.strip()}\n")

    logger.info("Prediction Done!")
