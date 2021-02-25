FROM harbor.wzs.wistron.com.cn/tensorflow/tensorflow:2.3.1-gpu

COPY requirements.txt requirements.txt

# add this line for no interactive while apt-get
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python-opencv ffmpeg libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    # -i https://pypi.tuna.tsinghua.edu.cn/simple \
    # --trusted-host pypi.tuna.tsinghua.edu.cn \
    --no-cache-dir \
    --default-timeout=1000 \
    -r requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=TRUE
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
WORKDIR /app
# COPY . .
EXPOSE 8501
ENTRYPOINT ["python", "serving.py", "8501"]