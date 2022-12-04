"""
Env variables
"""
import os
from dotenv import load_dotenv


load_dotenv(verbose=True)

MODEL_CKPT_PATH = os.environ.get("MODEL_CKPT_PATH")
MODEL_ARG_PATH = os.environ.get("MODEL_ARG_PATH")
