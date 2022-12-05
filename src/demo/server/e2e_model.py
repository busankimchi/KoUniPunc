"""
End-to-End process
"""
import os
import logging

import torch

from ...inference.utils import restore_punctuation_by_line
from ...inference.e2e import (
    cleanup_transcription,
    load_audio,
    asr_process,
    punc_process,
)
from ...utils import get_device
from .config import MODEL_CKPT_PATH, MODEL_ARG_PATH
from ...model.ko_unipunc import KoUniPunc


logger = logging.getLogger(__name__)


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


def e2e_inference(input_audio_file: str) -> str:
    args = torch.load(MODEL_ARG_PATH)
    device = get_device(args)

    speech_array = load_audio(input_audio_file)
    transcription = asr_process(speech_array, device)
    words = cleanup_transcription(transcription)

    model = load_punc_model(args, device)
    pred_list = punc_process(args, device, words, input_audio_file, model)

    line = restore_punctuation_by_line(words, pred_list)
    return line
