#!/bin/bash
cd "$(dirname "$0")"
exec .venv/bin/python main.py --port 8188
