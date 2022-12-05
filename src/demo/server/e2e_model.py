"""
End-to-End process
"""
import os
import logging
import requests

import torch

from ...inference.utils import restore_punctuation_by_line
from ...inference.e2e import cleanup_transcription, punc_process
from ...utils import get_device
from .config import (
    MODEL_CKPT_PATH,
    MODEL_ARG_PATH,
    NCP_CSR_CLIENT_ID,
    NCP_CSR_CLIENT_SECRET,
)
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


def asr_process(input_audio_file) -> str:
    url = f"https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NCP_CSR_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NCP_CSR_CLIENT_SECRET,
        "Content-Type": "application/octet-stream",
    }
    response = requests.post(url, data=open(input_audio_file, "rb"), headers=headers)

    if response.status_code == 200:
        return response["text"]

    else:
        raise Exception("Request error!")


def e2e_inference(input_audio_file) -> str:
    args = torch.load(MODEL_ARG_PATH)
    device = get_device(args)

    transcription = asr_process(input_audio_file)
    words = cleanup_transcription(transcription)

    model = load_punc_model(args, device)
    pred_list = punc_process(args, device, words, input_audio_file, model)

    line = restore_punctuation_by_line(words, pred_list)
    return line
