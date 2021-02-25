FROM harbor.wzs.wistron.com.cn/tensorflow/tensorflow:2.0.0-gpu-py3
# GPU: 1.12.0-gpu-py3; CPU: 1.12.0-py3

WORKDIR /usr/src/app

COPY ./requirements.txt ./
ARG http_proxy=http://10802014:123QWEasd%25@10.41.24.99:8080
ARG https_proxy=http://10802014:123QWEasd%25@10.41.24.99:8080
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt
# RUN pip install --install-option="--prefix=/install" -r /requirements.txt

VOLUME /model
VOLUME /data
VOLUME /config

COPY . .
ENTRYPOINT ["/bin/bash"]
# FROM scratch
# COPY --from=builder /go/bin/tfserving-gateway /go/bin/tfserving-gateway
# ENTRYPOINT ["/go/bin/tfserving-gateway"]

# docker run -p {exposed_port}:3333 -v /home/robert/robertnb/p3-saiap/deploy:/config -v /home/robert/robertnb:/data -d --restart=always p3-saiap-gateway:flask-v1.0
# docker build -t p3-saiap-gateway:flask-v1.0 ./deploy/flask_gw
