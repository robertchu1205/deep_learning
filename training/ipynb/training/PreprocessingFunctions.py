
import cv2
import numpy as np

class PreprocessingFunctions:
    @classmethod
    def debugShowImage(cls, image, string):
        cv2.imshow(string, image)
        cv2.waitKey(0)
    
    @classmethod
    def debugCloseWindows(cls):
        cv2.destroyAllWindows()
    
    paddingSize = 5
    @classmethod
    def fillHole(cls,image):
        tmp = image.copy()
        dst = cv2.copyMakeBorder(tmp, cls.paddingSize, cls.paddingSize, cls.paddingSize, cls.paddingSize, cv2.BORDER_CONSTANT,value =0)
        cv2.floodFill(dst,None, (0,0)  , 255)
        dst = dst[cls.paddingSize:cls.paddingSize+image.shape[0],cls.paddingSize:cls.paddingSize+image.shape[1]]
        hole=cv2.bitwise_not(dst)
        return (hole | image)

    @classmethod
    def adjustGamma(cls,image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    	invGamma = 1.0 / gamma
    	table = np.array([((i / 255.0) ** invGamma) * 255
    		for i in np.arange(0, 256)]).astype("uint8")
    	# apply gamma correction using the lookup table
    	return cv2.LUT(image, table)
    
    @classmethod
    def rotateImage(cls,image, angle, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = image.shape[:2]
     
        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)
     
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
     
        # 返回旋转后的图像
        return rotated
    
    @classmethod
    def getLargestComponents(cls,mask):

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        temp = 0
        max_ele = 1
        for i in range(1,np.max(labels)+1):
        
            if np.count_nonzero(labels==i) > temp:
                max_ele = i
                temp = np.count_nonzero(labels==i)
        
        return cv2.inRange(labels, max_ele, max_ele), stats[max_ele]
    