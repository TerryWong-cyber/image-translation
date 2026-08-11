#!/bin/sh
set -eu

python /app/scripts/verify_paddle_models.py \
  --root "${PADDLE_PDX_CACHE_HOME}" \
  --model-dir official_models/PP-LCNet_x1_0_textline_ori \
  --model-dir official_models/PP-OCRv4_mobile_det \
  --model-dir official_models/en_PP-OCRv4_mobile_rec

exec "$@"
