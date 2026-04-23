#!/bin/bash
set -e

ENV_NAME="MedChatbot"
PYTHON_VERSION="3.11"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Môi trường '$ENV_NAME' đã tồn tại, bỏ qua tạo mới."
else
    echo "Tạo conda environment '$ENV_NAME' (Python $PYTHON_VERSION) ..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "Cài đặt dependencies ..."
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo "Cài Playwright browsers ..."
conda run -n "$ENV_NAME" playwright install chromium

echo ""
echo "Hoàn tất. Kích hoạt môi trường bằng lệnh:"
echo "  conda activate $ENV_NAME"
