"""
Env variables
"""
import os
from dotenv import load_dotenv


load_dotenv(verbose=True)

MODEL_CKPT_PATH = os.environ.get("MODEL_CKPT_PATH")
MODEL_ARG_PATH = os.environ.get("MODEL_ARG_PATH")

NCP_CSR_CLIENT_ID = os.environ.get("NCP_CSR_CLIENT_ID")
NCP_CSR_CLIENT_SECRET = os.environ.get("NCP_CSR_CLIENT_SECRET")
