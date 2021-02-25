import os
import glob
# import time
import json
import base64
import requests
import argparse
# from PIL import Image
# import numpy as np
# 127.0.0.1 if it's the same container
# TFSERVER_URL = 'http://10.41.65.77:30334/rest'
# images in the assigned directory less than Batch size below would use all the images
# IMAGE_PATH = '/notebooks/notebook/DIP_Data5/AluCapacitance/'
# def resize_img(imgfile, width=96, height=96):
#     try:
#         img = Image.open(imgfile)
#         new_img = img.resize((width, height), Image.BILINEAR)
#         return np.array(new_img, dtype=np.float32).tolist()
#     except Exception as e:
#         print('Resize image error, message: {}'.format(e))
#         return None

def open_and_serialize_image(filename):
    with open(filename, "rb") as f:
        image = f.read()
    return base64.b64encode(image).decode("utf-8")

def createJson(path,IMAGE_AMOUNT,CAP_TYPE,CAP_DEGREE):
    filenames = []
    for root, dirs, files in os.walk(path):
       for img_file in files:
            if img_file.endswith('.png'):
               filenames.append(os.path.join(root, img_file))
            if img_file.endswith('.bmp'):
               filenames.append(os.path.join(root, img_file))
    instances = []
    for i,f in enumerate(filenames[:int(IMAGE_AMOUNT)]):
        instances.append({"image": {"b64":open_and_serialize_image(f)},"SN":f,"component": CAP_TYPE,"degree": str(CAP_DEGREE)})
    # instances = [{"image": {"b64":open_and_serialize_image(f)},"SN":f,"component": "EEE","degree": "0"} for f in filenames[:400]] 
    payload = {"instances": instances}
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="link & image amount")
    parser.add_argument("-m", "--method", dest="METHOD",help="post OR get")
    parser.add_argument("-l", "--link", dest="TFSERVER_URL", help="link of tfserving")
    parser.add_argument("-p", "--img_path", dest="IMAGE_PATH", help="Image Folder directory")
    parser.add_argument("-a", "--amount", dest="IMAGE_AMOUNT", help="image amount")
    parser.add_argument("-t", "--cap_type", dest="CAP_TYPE", help="CAP_TYPE")
    parser.add_argument("-d", "--cap_degree", dest="CAP_DEGREE", help="CAP_DEGREE")
    args = parser.parse_args()
    IMAGE_PATH = args.IMAGE_PATH
    IMAGE_AMOUNT = args.IMAGE_AMOUNT
    TFSERVER_URL = args.TFSERVER_URL
    METHOD = args.METHOD
    CAP_TYPE = args.CAP_TYPE
    CAP_DEGREE = args.CAP_DEGREE
    if METHOD=='get':
        if TFSERVER_URL.split('//')[1][:3]=='127':
            savedir = '/tf/notebook/createJson.json' # local test
        else:
            savedir = '/tf/notebook/A1/data.json' # local test
        with open(savedir, 'w', encoding='utf-8') as f:
            json.dump(createJson(IMAGE_PATH,IMAGE_AMOUNT,CAP_TYPE,CAP_DEGREE), f, ensure_ascii=False, indent=4)
        response = requests.get(TFSERVER_URL)
        print("response ok: {}".format(response.ok))
        print("outputs: {}".format(response.text))
    elif METHOD=='post':
        # print('IN POST')
        data = json.dumps(createJson(IMAGE_PATH,IMAGE_AMOUNT,CAP_TYPE,CAP_DEGREE))
        header = {
           'content-type': "application/json"
           }
        response = requests.post(TFSERVER_URL, data=data, headers=header)
        print("response ok: {}".format(response.ok))
        print("outputs: {}".format(response.text))
        # outputs = response.json()["predictions"]
    
