"""
End-to-End process
"""
import os
import logging
import requests

from dotenv import load_dotenv

import torch

from inference.inference_utils import restore_punctuation_by_line
from inference.e2e import cleanup_transcription, punc_process
from utils.utils import get_device
from model.ko_unipunc import KoUniPunc


load_dotenv(verbose=True)


MODEL_CKPT_PATH = os.environ.get("MODEL_CKPT_PATH")
MODEL_ARG_PATH = os.environ.get("MODEL_ARG_PATH")

NCP_CSR_CLIENT_ID = os.environ.get("NCP_CSR_CLIENT_ID")
NCP_CSR_CLIENT_SECRET = os.environ.get("NCP_CSR_CLIENT_SECRET")

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
    response = requests.post(url, data=input_audio_file, headers=headers)
    res = response.json()

    if response.status_code == 200:
        return res["text"]

    else:
        raise Exception("Request error!")


def e2e_inference(raw_audio) -> str:
    args = torch.load(MODEL_ARG_PATH)
    device = get_device(args)

    logger.info(args)

    transcription = asr_process(raw_audio.getvalue())
    words = cleanup_transcription(transcription)
    logger.info(f"TRANSCRIPT :: {words}")

    model = load_punc_model(args, device)

    temp_audio_file_path = f"./temp/{raw_audio.name}"
    with open(temp_audio_file_path, "wb") as f:
        f.write(raw_audio.getvalue())
        f.close()

    pred_list = punc_process(args, device, words, temp_audio_file_path, model)
    logger.info(f"PRED LIST :: {pred_list}")

    line = restore_punctuation_by_line(words, pred_list)

    os.remove(temp_audio_file_path)
    logger.info(f"** Inference End **")

    return line
