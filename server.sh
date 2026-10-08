#!/usr/bin/bash

# This script is used to run the Flask Whisper server.
#waitress-serve --threads=4 --port=22434 flask-whisper:create_app

# Alternatively, you can use Gunicorn to run the server.
# 注意：建議使用 --workers 1 避免多進程間的模型衝突
# 在 Gunicorn 中使用 gthread worker 類型以支持多線程
# 在 Windows 環境下無法使用 gunicorn，僅能使用 waitress
#source /home/tenyi/PythonWhisper/.venv/bin/activate
uv run gunicorn --workers=1 --worker-class=gthread --threads=4 --bind=0.0.0.0:22434 'xflask-whisper:create_app()'
#gunicorn --workers=1 --worker-class=gthread --threads=4 --bind=0.0.0.0:22434 'xflask-whisper:create_app()'



