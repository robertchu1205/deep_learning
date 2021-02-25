# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:46:35 2021

@author: 10911110
"""
import cv2, base64, os, shutil
import numpy as np
from EvaModel import TFModel

tfmodel_dir = 'data/label/20210126'
tmp_dir = '/label/'
debug = False
kernel = np.ones((3,3), np.uint8)
kernel2 = np.ones((3,3), np.uint8)
label_model = TFModel(tfmodel_dir)

def Debug_ShowImage(image, string):
    cv2.imshow(string, image)
    cv2.waitKey(0)

def fillHole(image):
    tmp = image.copy()
    cv2.floodFill(tmp,None, (0,0), 255)
    hole=cv2.bitwise_not(tmp)
    return (hole | image)

def GetWordImage(img):
    
    h,w,d = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    if gray.mean()<60 :
        return None

    #%%找標籤
    
    brightTh,brightArea = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if brightTh<130 and gray.mean()>180 :
        return None

    brightArea = cv2.morphologyEx(brightArea, cv2.MORPH_OPEN, kernel2) #去掉邊緣連著的鬚鬚
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(brightArea)
        
    temp = 0
    for i in range(1,np.max(labels)+1):
    
        if np.count_nonzero(labels==i) > temp:
            max_ele = i
            temp = np.count_nonzero(labels==i)
        
    MaxBrightArea = cv2.inRange(labels, max_ele, max_ele)
    if stats[max_ele][2]*stats[max_ele][3]>stats[max_ele][4]*5 or stats[max_ele][4]<20000:
        return None
    
    MaxBrightArea_Fill = fillHole(MaxBrightArea) #填滿的label

    BlackAreaInLabel = cv2.bitwise_not(MaxBrightArea_Fill) | MaxBrightArea 

    #%% 去barcode
    BlackAreaInLabel = cv2.bitwise_not(BlackAreaInLabel)
       
    BlackAreaInLabelDilate = cv2.dilate(BlackAreaInLabel,kernel)
    
    buffer = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    val, labels_im, stats, centroids = cv2.connectedComponentsWithStats(BlackAreaInLabelDilate)
    
    FindQR = False
    for i in range(1,np.max(labels_im)+1):
        tmp =cv2.inRange(labels_im, i, i)
        ratio = stats[i][2]/stats[i][3] if stats[i][2]<stats[i][3] else stats[i][3]/stats[i][2]
        
        x1 = stats[i][0]
        y1 = stats[i][1]
        x2 = stats[i][0]+stats[i][2]-1
        y2 = stats[i][1]+stats[i][3]-1
        
        if stats[i][2]>h*3/4 or stats[i][3]>w*3/4:
            continue
        
        if ((ratio>0.95) and stats[i][4] > 10000) or stats[i][4] > 20000: 
            tmp = cv2.dilate(tmp,kernel)
            buffer = buffer | tmp
            FindQR = True
            continue
        
        contours, hierarchy = cv2.findContours(tmp,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        
        if (stats[i][4]/(rect[1][0]*rect[1][1]+0.00001)>0.9) and stats[i][4] > 2000:
            tmp = cv2.dilate(tmp,kernel)
            buffer = buffer | tmp
            FindQR = True
            continue

    if FindQR :
        buffer =cv2.bitwise_and( MaxBrightArea_Fill, cv2.bitwise_not(buffer))
    else:
        buffer = MaxBrightArea_Fill
    
   
    #%% 找邊緣   
    const = 11
    mg_b = cv2.GaussianBlur(img[:,:,0], (const, const), 0)
    mg_g = cv2.GaussianBlur(img[:,:,1], (const, const), 0)
    mg_r = cv2.GaussianBlur(img[:,:,2], (const, const), 0)

    cannyB = cv2.Canny(mg_b, 80, 60)
    cannyG = cv2.Canny(mg_g, 80, 60)
    cannyR = cv2.Canny(mg_r, 80, 60)
    
    canny = cannyB | cannyG | cannyR
    
    canny =cv2.bitwise_and( canny, buffer)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny)
    
    select = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for i in range(1, num_labels):
    
        if stats[i][2] < 15 or stats[i][3]< 15 :
            continue

        mask = labels == i
        select[mask] = 255
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(select)
    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    
    #%%local threshold
    for i in range(1, num_labels):
        if (stats[i][4]<50 or stats[i][4]>(h*w/3) or stats[i][2]>h/2 or stats[i][3]>w/2 ):
            continue
        tmp =cv2.inRange(labels, i, i)
        contours, hierarchy = cv2.findContours(tmp,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

        x1 = stats[i][0]
        y1 = stats[i][1]
        x2 = stats[i][0]+stats[i][2]-1
        y2 = stats[i][1]+stats[i][3]-1

        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        
        cv2.drawContours(mask,contours,-1,255,1)
        
        edge_mean ,std = cv2.meanStdDev(gray,mask = mask)
        
        cv2.rectangle(mask, (x1, y1), (x2,y2), 255, -1)
        
        mask2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.rectangle(mask2, (x1, y1), (x2,y2), 255, 1)
        box_mean ,std = cv2.meanStdDev(gray, mask =mask2)
        
        if box_mean<edge_mean:
            continue
    
        src = cv2.bitwise_and(mask, gray)
        
        threshold_img  = cv2.inRange(src, 1, int((box_mean+edge_mean)/2))
        
        output = threshold_img |output

    if cv2.countNonZero(output) > 1000:
        return output
    else:
        return None

def main(input):
    try:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        else:
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
        requests = [] # tmp image path
        responses = []
        for idx, item in enumerate(input):
            img = base64.b64decode(item['image']['b64'])
            npimg = np.fromstring(img, dtype=np.uint8)
            source = cv2.imdecode(npimg, 1)
            function_result = GetWordImage(source)
            if function_result is not None:
                cv2.imwrite(os.path.join(tmp_dir, str(idx)+'.jpg'), function_result)
                requests.append(os.path.join(tmp_dir, str(idx)+'.jpg'))
                responses.append({'pred_class':'NG'})
            else:
                responses.append({'pred_class':'NG', 'confidence':0})
            # img_decode = image.load_img(BytesIO(base64.b64decode(img)))
            # img = image.img_to_array(img_decode, dtype='uint8')

        if requests != []:
            tested_requests = label_model.test_step(requests)

            for idx, tr in enumerate(tested_requests):
                input_idx = int(requests[idx].split(os.path.sep)[-1].replace('.jpg', ''))
                responses[input_idx]['confidence'] = tr['confidence']
                if tr['pred_class'] == input[input_idx]['degree']:
                    responses[input_idx]['pred_class'] = 'OK'
                else:
                    responses[input_idx]['pred_class'] = 'NG'

        shutil.rmtree(tmp_dir)
        return responses
    except Exception as e:
        print('Exception msg: ', e)
        return None