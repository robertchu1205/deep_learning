import os
import glob
import time
import grpc
import tensorflow as tf

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# TFSERVER_URL_GRPC = {host address}:{port for grpc}
# For example: 10.41.65.77:8590
TFSERVER_URL = os.environ["TFSERVER_URL_GRPC"] 
IMAGE_PATH = os.environ["IMAGE_PATH"]
# images in the assigned directory less than Batch size below would use all the images
BATCH_SIZE = 64


def open_image(filename):
    with open(filename, "rb") as f:
        image = f.read()
    return image

def generate_angle(f): #get angle info from file name
    x=f.split('/')[-2].split('-')[0].split('e')[-1]
    try:
        s = int(x)
    except ValueError:
        x = '0' #not all images includes angle
    return x


if __name__ == "__main__":
    filenames = glob.glob(IMAGE_PATH, recursive=True)[:BATCH_SIZE]

    # setup grpc channel
    channel = grpc.insecure_channel(TFSERVER_URL)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    # names below is default, so it needs to change if defination is different
    # name is from model.config -> config.name
    # signature_name comes from integrating source model to .pb model, and can also discover by adding /metadata, seeing signature_def
    request.model_spec.name = "model"
    request.model_spec.signature_name = "serving_default"

    image_data = [open_image(f) for f in filenames]
    angle_data = [generate_angle(f) for f in filenames]
    # pass all the necessary images below
    request.inputs["image"].CopyFrom(
        tf.compat.v1.make_tensor_proto(image_data, shape=[BATCH_SIZE])
    )
    request.inputs["angle"].CopyFrom(
        tf.compat.v1.make_tensor_proto(angle_data, shape=[BATCH_SIZE])
    )
    response = stub.Predict(request, 5.0) # 5 secs timeout

    # outputs = response.outputs["combined_outcome"].string_val -> also works
    
    results = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        results[key] = tf.contrib.util.make_ndarray(tensor_proto)
    print(results)


