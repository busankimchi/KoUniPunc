from typing import Union
import logging

import uvicorn
import uvloop
import asyncio
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from ...utils import init_logger
from .e2e_model import e2e_inference

logger = logging.getLogger(__name__)

app = FastAPI(title="KoUniPunc server", version="0.0.1", description="KoUniPunc server")

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@app.on_event("startup")
async def startup_event():
    init_logger()


@app.post("/inference", status_code=200)
async def inference(file: Union[UploadFile, None] = None):
    """
    Inference KoUniPunc
    """
    if not file:
        logger.error("No file.")
        return JSONResponse(status_code=460, content="No file.")

    res = e2e_inference(file)
    return res


@app.get("/", status_code=200)
def root():
    """
    root API
    """
    return {"status": "ok"}
