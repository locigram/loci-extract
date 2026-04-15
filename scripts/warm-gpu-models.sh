#!/usr/bin/env sh
# Pre-download PaddleOCR model files so the first API request isn't slow.
# This script is meant to run at container startup (non-blocking).
# Failure here does NOT prevent the server from starting.
set -e

echo "[warm-gpu-models] Attempting to pre-download PaddleOCR models..."

python3 -c "
try:
    from paddleocr import PaddleOCR
    import numpy as np
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    # Run a tiny inference to trigger model download
    dummy = np.zeros((32, 32, 3), dtype=np.uint8)
    ocr.ocr(dummy, cls=True)
    print('[warm-gpu-models] Models downloaded and warm.')
except Exception as e:
    print(f'[warm-gpu-models] Warning: model warming failed: {e}')
    print('[warm-gpu-models] Models will be downloaded on first request.')
" 2>&1 || true

echo "[warm-gpu-models] Attempting to pre-download Donut IRS classifier..."

python3 -c "
try:
    from transformers import DonutProcessor
    processor = DonutProcessor.from_pretrained('hsarfraz/donut-irs-tax-docs-classifier')
    print('[warm-gpu-models] Donut IRS classifier downloaded.')
except Exception as e:
    print(f'[warm-gpu-models] Warning: Donut warming failed: {e}')
    print('[warm-gpu-models] Donut model will be downloaded on first request.')
" 2>&1 || true

echo "[warm-gpu-models] Done."
