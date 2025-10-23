#!/usr/bin/env bash
set -e
source .venv/bin/activate
python ml/train_model.py
uvicorn api.main:app --reload
