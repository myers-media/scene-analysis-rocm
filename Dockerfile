# ROCm-enabled scene analysis. Build on a Linux host with AMD GPU + ROCm.
#   docker build -t scene-analysis-rocm .
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
#     -p 8501:8501 --ipc=host --security-opt seccomp=unconfined \
#     scene-analysis-rocm

FROM rocm/pytorch:latest

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/app/hf_cache

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
