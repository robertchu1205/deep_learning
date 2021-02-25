import os
import glob
import time
import json
import base64
import requests

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

TFSERVER_URL = os.environ["TFSERVER_URL_REST"]
IMAGE_PATH = os.environ["IMAGE_PATH"]
# images in the assigned directory less than Batch size below would use all the images
BATCH_SIZE = 64


def open_and_serialize_image(filename):
    with open(filename, "rb") as f:
        image = f.read()
    return base64.b64encode(image).decode("utf-8")

def generate_angle(f): #get angle info from file name
    x=f.split('/')[-2].split('-')[0].split('e')[-1]
    try:
        s = int(x)
    except ValueError: #not all images includes angle
        x = '0'
    return x

if __name__ == "__main__":
    filenames = glob.glob(IMAGE_PATH, recursive=True)[:BATCH_SIZE] 
    instances = [{"image": {"b64": open_and_serialize_image(f)},"angle":generate_angle(f)} for f in filenames] 
    # will add , automatically after } at every loop
    payload = {"instances": instances}
    data = json.dumps(payload)

    response = requests.post(TFSERVER_URL, data=data)
    print("response ok: {}".format(response.ok))
    print("outputs: {}".format(response.text))
    inputs = [np.asarray(Image.open(f)) for f in filenames]
    outputs = response.json()["predictions"]
    

