from enum import Enum
from abc import ABCMeta, abstractmethod
import cv2
from scipy import stats
import numpy as np
from PreprocessingFunctions import PreprocessingFunctions

#%% Enum declare
class eLabelType(Enum):
    Orange = 'Orange'
    White = 'White'
    Unknown = 'None'
    
class eImageType(Enum):
    ok = "ok"
    imageError = "image error"
    noLabel = "no label"
    labelLift = "label lift"
    
    
#%% Check function declare    

class CheckFunction(metaclass=ABCMeta):
    def __init__(self, debugFlag = False):
        self.debugMode = debugFlag
    
    @abstractmethod
    def NoLabelCheck(cls, stats):
        return NotImplemented
    
    def ImageIntensityCheck(cls, grayMean):
        if grayMean < 50:  
            if cls.debugMode:
                print("%s, too dark, gray mean: %d" % (eImageType.imageError,grayMean))
            return eImageType.imageError
        return eImageType.ok
    
    def ConnectionExitCheck(cls, image , threshold):
        if cv2.countNonZero(image) <= threshold:
            return eImageType.imageError
        return eImageType.ok

class WhiteLabelCheck(CheckFunction):
    
    def NoLabelCheck(cls, stats, imgWidth, imgHeight):
        if stats[2]*stats[3]>stats[4]*5 or stats[4]<20000:  #去除空心狀白色區域 與 面積太小的區域
            if cls.debugMode:
                print("%s, NoLabelCheck" % eImageType.noLabel)
            return eImageType.noLabel
        
        return eImageType.ok
    
    def LabelSizeCheck(cls, stats, imgWidth, imgHeight):
                
        if stats[4] < imgWidth * imgHeight/10:  #最大blob面積小於整張影像1/5
            if cls.debugMode:
                print(stats)
                print(imgWidth)
                print(imgHeight)
                print("%s, LabelSizeCheck" % eImageType.noLabel)
            return eImageType.noLabel
        
        return eImageType.ok
    
    def OtsuThresholdCheck(cls, grayMean, otsuThreshold):
        # print("OtsuThresholdCheck")
        if grayMean > 180 and otsuThreshold < 130 : #思考合理性中
            if cls.debugMode:
                print("%s, otsu" % eImageType.noLabel)
            return eImageType.imageError
        return eImageType.ok

class OrangeLabelCheck(CheckFunction):
    
    def NoLabelCheck(cls, stats, imgWidth, imgHeight):
        if stats[4] < imgWidth * imgHeight / 5:  #最大blob面積小於整張影像1/5
            if cls.debugMode:
                print("%s, area too small" % eImageType.noLabel)
            return eImageType.noLabel
        return eImageType.ok
    
    kernel = np.ones((13,13), np.uint8)
    
    def labelLiftCheck(cls, image, labelMask, labelStats):
        threshold_img  = cv2.inRange(image[:,:,0], 190, 255)

        MaxBrightArea, stats = PreprocessingFunctions.getLargestComponents(threshold_img)       
        dilate = cv2.dilate(MaxBrightArea,np.ones((5,5), np.uint8))
        img_bwa = cv2.bitwise_and(dilate,labelMask)
        
        if cls.debugMode :
            print (stats)
            print (labelStats)
            
            PreprocessingFunctions.debugShowImage(MaxBrightArea,"MaxBrightArea")
            PreprocessingFunctions.debugShowImage(img_bwa,"img_bwa")
        
        
        if (stats[2] > labelStats[2] * 0.6 or stats[3] > labelStats[3] * 0.6) and stats[4] > labelStats[4]/10 and cv2.countNonZero(img_bwa)>0 and stats[4]/(stats[2]*stats[3])<0.8 :
            if cls.debugMode:
                PreprocessingFunctions.debugShowImage(image,"label lift 0")
                print(eImageType.labelLift)
            return eImageType.labelLift
        
        tmp = MaxBrightArea | labelMask
        
        MaxBrightArea, stats = PreprocessingFunctions.getLargestComponents(tmp)
        fillHole = PreprocessingFunctions.fillHole(MaxBrightArea)
        brightArea = cv2.morphologyEx(fillHole, cv2.MORPH_OPEN, cls.kernel)
        MaxBrightArea, stats = PreprocessingFunctions.getLargestComponents(brightArea)
        
        if cls.debugMode :

            print(labelStats[2]*1.1)
            print (labelStats[3]*1.1)
            PreprocessingFunctions.debugShowImage(tmp,"tmp")
            PreprocessingFunctions.debugShowImage(MaxBrightArea,"MaxBrightArea")
            PreprocessingFunctions.debugShowImage(labelMask,"labelMask")
        
        if labelStats[2]*1.1<stats[2] or labelStats[3]*1.1<stats[3]:
            if cls.debugMode:
                PreprocessingFunctions.debugShowImage(image,"label shift 1")
                print(eImageType.labelLift)
            return eImageType.labelLift
        
        
        
        return eImageType.ok

#%% Label Class declare
class LabelPreprocessing(metaclass=ABCMeta):
    
    def __init__(self, params):
        self.buffer = np.zeros((params.height,params.width), np.uint8)
        self.image = params.image
        self.height = params.height
        self.width = params.width
        self.grayImage = params.grayImage
        self.HSVImage = params.HSVImage
        self.debugMode = params.debugMode

    @abstractmethod
    def execute(self):
        return NotImplemented
    
    def removeNoise(self):
        return NotImplemented
    
    def getLabelSize(self, labelImage):        
        val, labels_im, stats, centroids = cv2.connectedComponentsWithStats(labelImage)
        
        self.labelStats = stats[1]
        self.labelWidth = stats[1][2]
        self.labelHeight = stats[1][3]
        self.labelArea = stats[1][4]

    erodeKernal = np.ones((21,21), np.uint8)
    def findEdge(self, cannyThreshold = (100,60), gBKernelSize = 3, miniLength = 15): # 找邊緣   
        mg_b = cv2.GaussianBlur(self.image[:,:,0], (gBKernelSize, gBKernelSize), 0)
        mg_g = cv2.GaussianBlur(self.image[:,:,1], (gBKernelSize, gBKernelSize), 0)
        mg_r = cv2.GaussianBlur(self.image[:,:,2], (gBKernelSize, gBKernelSize), 0)
    
        cannyB = cv2.Canny(mg_b, cannyThreshold[0], cannyThreshold[1])
        cannyG = cv2.Canny(mg_g, cannyThreshold[0], cannyThreshold[1])
        cannyR = cv2.Canny(mg_r, cannyThreshold[0], cannyThreshold[1])
        
        canny = cannyB | cannyG | cannyR
        
        if self.debugMode:
            PreprocessingFunctions.debugShowImage(self.buffer,"self.buffer before")
        
        self.buffer = cv2.erode(self.buffer, self.erodeKernal, iterations = 1)
        
        if self.debugMode:
            PreprocessingFunctions.debugShowImage(self.buffer,"self.buffer after")
        
        canny =cv2.bitwise_and( canny, self.buffer)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canny)
        
        self.buffer = np.zeros((self.height, self.width), np.uint8)
    
        for i in range(1, num_labels):
        
            if stats[i][2] < miniLength or stats[i][3]< miniLength  or abs(stats[i][2]- self.labelHeight)<10 or abs(stats[i][3]- self.labelWidth)<10:
                continue
    
            mask = labels == i
            self.buffer[mask] = 255
    
        if self.debugMode:
            PreprocessingFunctions.debugShowImage(canny,"Canny")
            PreprocessingFunctions.debugShowImage(self.buffer,"Canny Select")
  
    
    def localThreshold(self, maxLeng = False): #local threshold
        
        
        contours, hierarchy = cv2.findContours(self.buffer,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

        self.buffer = np.zeros((self.height, self.width), np.uint8)
        cv2.drawContours(self.buffer,contours,-1,255,1)
        
        # if self.debugMode:
        #     PreprocessingFunctions.debugShowImage(self.buffer,"self.buffer")
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.buffer)

        self.buffer = np.zeros((self.height, self.width), np.uint8)
        
        if maxLeng:
            edgeWidthMax = self.width/2
            edgeHeightMax = self.height/2
        else:
            edgeWidthMax = self.width
            edgeHeightMax = self.height
        
        for i in range(1, num_labels):
            

            if (stats[i][4]<50 or stats[i][2]>edgeHeightMax or stats[i][3]>edgeWidthMax ):
                continue
            tmp =cv2.inRange(labels, i, i)


            x1 = stats[i][0]
            y1 = stats[i][1]
            x2 = stats[i][0]+stats[i][2]-1
            y2 = stats[i][1]+stats[i][3]-1

            #box 邊界
            mask = np.zeros((self.height, self.width), np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2,y2), 255, 1)
            box_mean ,std = cv2.meanStdDev(self.grayImage, mask =mask)
     
            #edge 
            edge_mean ,std = cv2.meanStdDev(self.grayImage,mask = tmp)
            if box_mean<edge_mean:
                continue
            
            cv2.rectangle(mask, (x1, y1), (x2,y2), 255, -1)
            src = cv2.bitwise_and(mask, self.grayImage)
            
            
            threshold_img  = cv2.inRange(src, 1, int((box_mean+edge_mean)/2))
            
            self.buffer = threshold_img |self.buffer
            
        if self.debugMode:
            PreprocessingFunctions.debugShowImage(self.buffer,"self.buffer")
            
        if self.checkFcn.ConnectionExitCheck(self.buffer, 1000) != eImageType.ok:
            return eImageType.imageError
        else:
            return eImageType.ok
    
class WhiteLabel(LabelPreprocessing):
    
    morKernel = np.ones((7,7), np.uint8)
    
    def __init__(self,params):
        super().__init__(params)
        self.checkFcn = WhiteLabelCheck(self.debugMode)
        
    def removeNoise(self,BlackAreaInLabel): #去barcode
       
        if self.labelArea/(self.width*self.height)<1/3:
            noiceKernel = np.ones((3,3), np.uint8)
        else:
            noiceKernel = np.ones((7,7), np.uint8)
           
        BlackAreaInLabelDilate = cv2.dilate(BlackAreaInLabel,noiceKernel)
        val, labels_im, stats, centroids = cv2.connectedComponentsWithStats(BlackAreaInLabelDilate)

        buffer2 = np.zeros((self.height,self.width), np.uint8)
        FindQR = False
        for i in range(1,np.max(labels_im)+1):
            tmp =cv2.inRange(labels_im, i, i)

            ratio = stats[i][2]/stats[i][3] if stats[i][2]<stats[i][3] else stats[i][3]/stats[i][2]
            
            contours, hierarchy = cv2.findContours(tmp,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])
            
            # if self.debugMode:
            #     print(ratio)
            #     print( stats[i])
            #     print(stats[i][4]/(rect[1][0]*rect[1][1]))
                
            #     print(min(stats[i][2],stats[i][3]))
            #     print(min(self.labelHeight,self.labelWidth))
    
            #     PreprocessingFunctions.debugShowImage(tmp,"tmp")
                
            if (stats[i][2]>self.labelWidth*1/3 or stats[i][3]>self.labelHeight*1/3) and (ratio<0.2): #太長或太寬的不是Barcode
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindQR = True
                
                if self.debugMode:
                    print ("condition 0")    
                continue
            
            if (stats[i][2]>self.labelWidth*2/3 or stats[i][3]>self.labelHeight*2/3) and stats[i][4]>15000: #太長或太寬的不是Barcode
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindQR = True
                
                if self.debugMode:
                    print ("condition 0.1")    
                continue
            
            if stats[i][2]< 20 and stats[i][3]<20 or stats[i][4]<200: #太長或太寬的不是Barcode
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindQR = True
                
                if self.debugMode:
                    print ("condition 0.2")    
                continue  
            
            #以下正方形狀barcode
            
            if self.debugMode:
                print(ratio)
                print(stats[i])
                print (self.labelHeight)
                print(self.labelWidth)
                PreprocessingFunctions.debugShowImage(tmp,"tmp")
            
            
            if ((ratio>0.95) and stats[i][4] > 10000) and (abs(stats[i][2]-self.labelHeight)>30 or abs(stats[i][3]-self.labelWidth>30)):  #正方形且面積太大 
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindQR = True
                
                if self.debugMode:
                    print ("condition 1")
                continue
            
            if stats[i][4] > 20000:  #面積太大 
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindQR = True
                
                if self.debugMode:
                    print ("condition 1.1")
                continue
            
            #以下條紋狀barcode
            if ((ratio<0.2) and stats[i][4]/(rect[1][0]*rect[1][1]) > 0.7):
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindQR = True
                
                if self.debugMode:
                    print ("condition 2")
                continue
            
            if (stats[i][4]/(rect[1][0]*rect[1][1])>0.95) and stats[i][4] > 1000:
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindQR = True
                
                if self.debugMode:
                    print ("condition 3")
                continue

        if FindQR :
            self.buffer =cv2.bitwise_and( self.buffer, cv2.bitwise_not(buffer2))
        else:
            self.buffer = self.buffer
        
        
    def execute(self):

        #檢查平均灰階質是否太小
        if self.checkFcn.ImageIntensityCheck(self.grayImage.mean()) != eImageType.ok:  
            return None
          
        brightArea  = cv2.inRange(self.grayImage, 170, 255) 

        if self.debugMode:
            print ("gray mean: %d" %(self.grayImage.mean()))
            PreprocessingFunctions.debugShowImage(brightArea,"brightArea")
        
        #去掉邊緣連著的鬚鬚
        brightArea = cv2.morphologyEx(brightArea, cv2.MORPH_OPEN, self.morKernel) 
        
        if self.checkFcn.ConnectionExitCheck(brightArea, 0) != eImageType.ok: #檢查是否黑圖
            return None
        
        brightArea, stats = PreprocessingFunctions.getLargestComponents(brightArea)
     
        if self.checkFcn.NoLabelCheck(stats, self.width, self.height) != eImageType.ok: #檢查白色區域是否中空
            return None
        
        MaxBrightArea = cv2.morphologyEx(brightArea, cv2.MORPH_CLOSE, self.morKernel) #補滿白色邊緣區域
                                                      
        MaxBrightArea_Fill = PreprocessingFunctions.fillHole(MaxBrightArea) #填滿的label
        
        label, stats = PreprocessingFunctions.getLargestComponents(MaxBrightArea_Fill)

        if self.checkFcn.LabelSizeCheck(stats, self.width, self.height) != eImageType.ok: #檢查label是否太小
            return None
        if self.debugMode:
            PreprocessingFunctions.debugShowImage(MaxBrightArea_Fill,"MaxBrightArea_Fill")
        
        self.getLabelSize(MaxBrightArea_Fill)
        
        self.buffer = MaxBrightArea_Fill
        
        self.findEdge(miniLength = 10)
        
        if self.localThreshold() != eImageType.ok:
            return None

        self.removeNoise(self.buffer)
        

        if self.debugMode:
            PreprocessingFunctions.debugShowImage(self.image,"img")
            PreprocessingFunctions.debugShowImage(self.buffer,"buffer")
            
        if self.checkFcn.ConnectionExitCheck(self.buffer, 0) != eImageType.ok: #檢查是否黑圖
            return None

        return self.buffer
    
class OrangeLabel(LabelPreprocessing):

    kernel = np.ones((5,5), np.uint8)
    
    def __init__(self,params):
        super().__init__(params)
        self.checkFcn = OrangeLabelCheck(self.debugMode)
        
    def removeNoise(self,BlackAreaInLabel): #去barcode
        noiceKernel = np.ones((3,3), np.uint8)
        val, labels_im, stats, centroids = cv2.connectedComponentsWithStats(BlackAreaInLabel)

        buffer2 = np.zeros((self.height,self.width), np.uint8)
        FindNoice = False
        for i in range(1,np.max(labels_im)+1):
            tmp =cv2.inRange(labels_im, i, i)

            ratio = stats[i][2]/stats[i][3] if stats[i][2]<stats[i][3] else stats[i][3]/stats[i][2]
            
            
            if self.debugMode:
                print(ratio)
                print( stats[i])
                PreprocessingFunctions.debugShowImage(tmp,"tmp")
                
            if (stats[i][2]>self.labelWidth*1/3 or stats[i][3]>self.labelHeight*1/3) and (ratio<0.2): #單邊太長且長寬比太大
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindNoice = True
                
                if self.debugMode:
                    print ("ratio < 0.2 and One of the sides is over 1/3")    
                continue
            
            if (stats[i][2]>self.labelWidth*3/4 or stats[i][3]>self.labelHeight*3/4) and stats[i][4]>7500: 
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindNoice = True
                
                if self.debugMode:
                    print ("One of the sides is over 3/4 and area > 7500")    
                continue
            
            if stats[i][2]< 20 and stats[i][3]<20 or stats[i][4]<200: #太長或太寬的不是Barcode
                tmp = cv2.dilate(tmp,noiceKernel)
                buffer2 = buffer2 | tmp
                FindNoice = True
                
                if self.debugMode:
                    print ("small area")    
                continue  

        if FindNoice :
            self.buffer =cv2.bitwise_and( self.buffer, cv2.bitwise_not(buffer2))
        else:
            self.buffer = self.buffer

    def execute(self):

        if self.debugMode :
            print ("gray mean: %d" % self.grayImage.mean())
            
        if self.checkFcn.ImageIntensityCheck(self.grayImage.mean()) != eImageType.ok:  #檢查平均灰階質是否太小   
            return None
        
        lower_red = np.array([0, 0, 30]) 
        upper_red = np.array([45, 255, 255]) 
        mask = cv2.inRange(self.HSVImage, lower_red, upper_red) 
        whiteArea  = cv2.inRange(self.image[:,:,0], 190, 255)
        
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(whiteArea))
        
        if self.debugMode :
            PreprocessingFunctions.debugShowImage(mask,"hsv")
        
        self.buffer = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) #去掉邊緣連著的鬚鬚
        
        self.buffer, stats = PreprocessingFunctions.getLargestComponents(self.buffer)
        
        if self.checkFcn.NoLabelCheck(stats,self.width,self.height) != eImageType.ok: #檢查label是否存在
            return None
        
        self.buffer = PreprocessingFunctions.fillHole(self.buffer) #填滿的label
        
        self.getLabelSize(self.buffer)
        
        if self.checkFcn.labelLiftCheck(self.image, self.buffer, self.labelStats) != eImageType.ok: #檢查label是否存在
            return None
       
        self.findEdge((50,10),15,15)
        if self.localThreshold(True) != eImageType.ok:
            return None
        
        self.removeNoise(self.buffer)
        
        if self.checkFcn.ConnectionExitCheck(self.buffer, 0) != eImageType.ok: #檢查是否黑圖
            return None

        return self.buffer


#%%

class Preprocessing:
    
    label_map = {
        eLabelType.White: WhiteLabel,
        eLabelType.Orange: OrangeLabel,
    }
    
    
    def __init__(self,image, bDebug = False):
        self.image = image   
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.HSVImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.debugMode = bDebug
    
    def __del__(self):
        if self.debugMode:
            PreprocessingFunctions.debugCloseWindows()
    
    def run(self):
        self.labelColorClassify() #檢查label顏色
        label = self.label_map.get(self.labelColor)(self)
        return label.execute()

         
    def labelColorClassify(self): #取中心區域檢查label顏色
        
        tmpImage = self.HSVImage[int(self.height/4):int(self.height*3/4),int(self.width/4):int(self.width*3/4),:]
    
        # hsv_small = cv2.cvtColor(tmpImage, cv2.COLOR_BGR2HSV)
        
        hue_mode = stats.mode(tmpImage[:,:,0].flatten())[0][0]
        # sat_mean = hsv_small[:,:,1].mean()
        val_mean = tmpImage[:,:,2].mean()

        if hue_mode < 45 and hue_mode > 0 and val_mean < 235 and val_mean > 40:
            self.labelColor = eLabelType.Orange
        else:
            self.labelColor = eLabelType.White

        if self.debugMode :
            PreprocessingFunctions.debugShowImage(tmpImage,"hsv_crop")
            print ("hue mode: %d" % hue_mode)
            print ("val mean: %d" % val_mean)
            print (self.labelColor)


if __name__ == '__main__':

    image = cv2.imread(r"D:\sandra_chang\GitHub\LabelOrientation\trainData\SingleTrain\MBK920W10206B20_LAB-AIM_180_000_000.jpg")
    preprocessing = Preprocessing(image, True)
    result = preprocessing.run()
    if result is not None:
        PreprocessingFunctions.debugShowImage(result,"final")
    else:
        print("None")
        
    del preprocessing 



