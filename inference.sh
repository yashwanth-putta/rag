#!/usr/bin/env bash

./sync.sh

PYTHONPATH=. uv run --no-sync python main.py
