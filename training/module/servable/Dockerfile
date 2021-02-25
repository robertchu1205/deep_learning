FROM harbor.wzs.wistron.com.cn/tensorflow/tensorflow:2.1.0-py3

WORKDIR /app

# COPY requirements.txt ./
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     --trusted-host pypi.tuna.tsinghua.edu.cn \
#     --no-cache-dir \ 
#     -r requirements.txt

COPY . .

ENV h5_path /h5/model.h5
ENV pb_dir /pb/
ENV img_size 32
ENTRYPOINT [ "python", "tf2_servable.py" ]