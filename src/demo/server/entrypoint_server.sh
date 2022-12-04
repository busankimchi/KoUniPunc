#!/bin/bash

set -e

echo "running FastAPI younicorn server.."

uvicorn src.main:app --reload --port=8080 --host=0.0.0.0
# gunicorn src.main:app --reload --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080
