#!/usr/bin/env python
# coding: utf-8

# # Import Correspondent Image Data (1)

# In[4]:


import matplotlib.pyplot as plt #Show Image Function
def plot_image_label(images,label,idx,num=25):
    plt.figure(figsize=(10,10))
    for i in range(num):
        plt.subplot(5,5, 1+i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[idx])
        plt.xlabel(label[idx])
        idx+=1
    plt.show()


# In[5]:


from PIL import Image
import numpy as np
#from keras.preprocessing import image
import glob

def load_alu_rgb_img(size,label,path):
    outimg = []
    labellist = []
    filepaths = sorted(list(glob.glob(path)))
    for filepath in filepaths:
        im=Image.open(filepath)
        im = im.resize(size)
        #im = image.load_img(filepath, target_size=(42, 42))
        #im = misc.imread(filepath, flatten= 0)
        outimg.append(np.array(im.getdata(),np.uint8).reshape(im.size[1], im.size[0], 3))
        labellist.append(label)
    return (np.asarray(labellist), np.asarray(outimg) / 255.0)


# In[6]:


import os
from sklearn.preprocessing import LabelBinarizer
# def get_angle_label_info(label,path):
#     lb_angle = LabelBinarizer()
#     #lb_label = LabelBinarizer().fit(['OK','NG'])
#     angle = []
#     labellist = []
#     filepaths = sorted(list(glob.glob(path)))
#     for filepath in filepaths:
#         filename = filepath.split('/')[-1]
#         if filename.endswith('.bmp'):
#             angle.append(int(filename.split('_')[1].split('.')[0]))
#             labellist.append(label)
#     return (np.asarray(labellist),lb_angle.fit_transform(np.asarray(angle)))


# In[7]:


img_size = (76,76)
(img_length,img_width) = img_size

#Load OK label, Image
(ok_label,ok_img) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/P3_SAIAP_0523/DIP/OK/AluCapacitance/*.bmp')

#LackofComponent label, Image
(ng_LoC_label, ng_LoC_img) = load_alu_rgb_img(img_size,'NG-LoC','/notebooks/notebook/P3_SAIAP_0523/DIP/NG/AluCapacitance/NG-LackofComponent/*.bmp')

#Too High label, Image
(ng_TH_label, ng_TH_img) = load_alu_rgb_img(img_size,'NG_TH','/notebooks/notebook/P3_SAIAP_0523/DIP/NG/AluCapacitance/NG-TooHigh/*.bmp')

#Upside Down label, Image
(ng_UD_label, ng_UD_img) = load_alu_rgb_img(img_size,'NG_UD','/notebooks/notebook/P3_SAIAP_0523/DIP/NG/AluCapacitance/NG-UpsideDown/*.bmp')


# #Alu Angle, Label which is OK
# (ok_label,ok_angle) = get_angle_label_info('OK','/notebooks/notebook/P3_SAIAP_0523/DIP/OK/AluCapacitance/*.bmp')

# #LackofComponent Angle, Label
# (ng_LoC_label,ng_LoC_angle) = get_angle_label_info('LoC-NG','/notebooks/notebook/P3_SAIAP_0523/DIP/NG/AluCapacitance/NG-LackofComponent/*.bmp')

# #TooHigh Angle, Label
# (ng_TH_label,ng_TH_angle) = get_angle_label_info('TF-NG','/notebooks/notebook/P3_SAIAP_0523/DIP/NG/AluCapacitance/NG-TooHigh/*.bmp')

# #UpsideDown Angle, Label
# (ng_UD_label,ng_UD_angle) = get_angle_label_info('UD-NG','/notebooks/notebook/P3_SAIAP_0523/DIP/NG/AluCapacitance/NG-UpsideDown/*.bmp')


# In[8]:


print(ok_img.dtype)
print(ok_img.shape)

print(ok_label.dtype)
print(ok_label.shape)

print(ng_LoC_img.dtype)
print(ng_LoC_img.shape)

print(ng_LoC_label.dtype)
print(ng_LoC_label.shape)

print(ng_UD_img.dtype)
print(ng_UD_img.shape)

print(ng_UD_label.dtype)
print(ng_UD_label.shape)

print(ng_TH_img.dtype)
print(ng_TH_img.shape)

print(ng_TH_label.dtype)
print(ng_TH_label.shape)


# # Import Correspondent Image Data (2)

# In[9]:


#Load Angle:0 OK image
(ok_label_0,ok_img_0) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle0-OK/*.bmp')
#Load Angle:90 OK image
(ok_label_90,ok_img_90) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle90-OK/*.bmp')
#Load Angle:180 OK image
(ok_label_180,ok_img_180) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle180-OK/*.bmp')
#Load Angle:270 OK image
(ok_label_270,ok_img_270) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle270-OK/*.bmp')

#LackofComponent label, Image
(ng_LoC_label_2, ng_LoC_img_2) = load_alu_rgb_img(img_size,'NG-LoC','/notebooks/notebook/DIP_Data2/AluCapacitance/NG-LackofComponent/*.bmp')

#Too High label, Image
(ng_TH_label_2, ng_TH_img_2) = load_alu_rgb_img(img_size,'NG_TH','/notebooks/notebook/DIP_Data2/AluCapacitance/NG-TooHigh/*.bmp')

#Upside Down label, Image
(ng_UD_label_2, ng_UD_img_2) = load_alu_rgb_img(img_size,'NG_UD','/notebooks/notebook/DIP_Data2/AluCapacitance/NG-UpsideDown/*.bmp')

#More than a Component label, Image
(ng_More_label_2, ng_More_img_2) = load_alu_rgb_img(img_size,'NG_More','/notebooks/notebook/DIP_Data2/AluCapacitance/NG-MorethanAComponent/*.bmp')


# In[10]:


print(ok_img_0.dtype)
print(ok_img_0.shape)

print(ok_label_0.dtype)
print(ok_label_0.shape)

print(ok_img_90.dtype)
print(ok_img_90.shape)

print(ok_label_90.dtype)
print(ok_label_90.shape)

print(ok_img_180.dtype)
print(ok_img_180.shape)

print(ok_label_180.dtype)
print(ok_label_180.shape)

print(ok_img_270.dtype)
print(ok_img_270.shape)

print(ok_label_270.dtype)
print(ok_label_270.shape)

print(ng_LoC_img_2.dtype)
print(ng_LoC_img_2.shape)

print(ng_LoC_label_2.dtype)
print(ng_LoC_label_2.shape)

print(ng_UD_img_2.dtype)
print(ng_UD_img_2.shape)

print(ng_UD_label_2.dtype)
print(ng_UD_label_2.shape)

print(ng_TH_img_2.dtype)
print(ng_TH_img_2.shape)

print(ng_TH_label_2.dtype)
print(ng_TH_label_2.shape)

print(ng_More_img_2.dtype)
print(ng_More_img_2.shape)

print(ng_More_label_2.dtype)
print(ng_More_label_2.shape)


# In[11]:


plot_image_label(ng_TH_img_2,ng_TH_label_2,0)


# # Import Correspondent Image Data (3)

# In[12]:


#Load Angle:0 OK image
(ok_label_0_2,ok_img_0_2) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle0-OK/*.bmp')
#Load Angle:90 OK image
(ok_label_90_2,ok_img_90_2) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle90-OK/*.bmp')
#Load Angle:180 OK image
(ok_label_180_2,ok_img_180_2) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle180-OK/*.bmp')
#Load Angle:270 OK image
(ok_label_270_2,ok_img_270_2) = load_alu_rgb_img(img_size,'OK','/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle270-OK/*.bmp')

#LackofComponent label, Image
(ng_LoC_label_3, ng_LoC_img_3) = load_alu_rgb_img(img_size,'NG-LoC','/notebooks/notebook/DIP_Data3/AluCapacitance/NG-LackofComponent/*.bmp')

#Too High label, Image
(ng_TH_label_3, ng_TH_img_3) = load_alu_rgb_img(img_size,'NG_TH','/notebooks/notebook/DIP_Data3/AluCapacitance/NG-TooHigh/*.bmp')

#Upside Down label, Image
(ng_UD_label_3, ng_UD_img_3) = load_alu_rgb_img(img_size,'NG_UD','/notebooks/notebook/DIP_Data3/AluCapacitance/NG-UpsideDown/*.bmp')

#More than a Component label, Image
(ng_More_label_3, ng_More_img_3) = load_alu_rgb_img(img_size,'NG_More','/notebooks/notebook/DIP_Data3/AluCapacitance/NG-MorethanAComponent/*.bmp')


# In[13]:


print(ok_img_0_2.dtype)
print(ok_img_0_2.shape)

print(ok_label_0_2.dtype)
print(ok_label_0_2.shape)

print(ok_img_90_2.dtype)
print(ok_img_90_2.shape)

print(ok_label_90_2.dtype)
print(ok_label_90_2.shape)

print(ok_img_180_2.dtype)
print(ok_img_180_2.shape)

print(ok_label_180_2.dtype)
print(ok_label_180_2.shape)

print(ok_img_270_2.dtype)
print(ok_img_270_2.shape)

print(ok_label_270_2.dtype)
print(ok_label_270_2.shape)

print(ng_LoC_img_3.dtype)
print(ng_LoC_img_3.shape)

print(ng_LoC_label_3.dtype)
print(ng_LoC_label_3.shape)

print(ng_UD_img_3.dtype)
print(ng_UD_img_3.shape)

print(ng_UD_label_3.dtype)
print(ng_UD_label_3.shape)

print(ng_TH_img_3.dtype)
print(ng_TH_img_3.shape)

print(ng_TH_label_3.dtype)
print(ng_TH_label_3.shape)

print(ng_More_img_3.dtype)
print(ng_More_img_3.shape)

print(ng_More_label_3.dtype)
print(ng_More_label_3.shape)


# # Number by category

# In[14]:


#OK
print('OK')
print(ok_img_0.shape[0]+ok_img_0_2.shape[0]+ok_img_180.shape[0]+ok_img_180_2.shape[0]+ok_img_270.shape[0]+ok_img_270_2.shape[0]+ok_img_90.shape[0]+ok_img_90_2.shape[0]+ok_img.shape[0])

#NG Lack of Component
print('NG Lack of Component')
print(ng_LoC_img.shape[0]+ng_LoC_img_2.shape[0]+ng_LoC_img_3.shape[0])

#NG More than a Component
print('NG More than a Component')
print(ng_More_img_2.shape[0]+ng_More_img_3.shape[0])

#NG Too High
print('NG Too High')
print(ng_TH_img.shape[0]+ng_TH_img_2.shape[0]+ng_TH_img_3.shape[0])

#NG Upside Down
print('NG Upside Down')
print(ng_UD_img.shape[0]+ng_UD_img_2.shape[0]+ng_UD_img_3.shape[0])


# # Split to Train & Test And Shuffle It

# In[15]:


import keras
from keras.utils import to_categorical
All_ok_img = np.concatenate((ok_img,ok_img_0,ok_img_90,ok_img_180,ok_img_270,ok_img_0_2,ok_img_90_2,ok_img_180_2,ok_img_270_2))
print(All_ok_img.shape)
All_TH_img = np.concatenate((ng_TH_img,ng_TH_img_2,ng_TH_img_3))
print(All_TH_img.shape)
All_UD_img = np.concatenate((ng_UD_img,ng_UD_img_2,ng_UD_img_3))
print(All_UD_img.shape)
All_LoC_img = np.concatenate((ng_LoC_img,ng_LoC_img_2,ng_LoC_img_3))
print(All_LoC_img.shape)
All_More_img = np.concatenate((ng_More_img_2,ng_More_img_3))
print(All_More_img.shape)

All_Kindof_img = np.concatenate((All_ok_img,All_TH_img,All_UD_img,All_LoC_img,All_More_img))
print(All_Kindof_img.shape)


# In[16]:


All_ok_label = np.concatenate((ok_label,ok_label_0,ok_label_90,ok_label_180,ok_label_270,ok_label_0_2,ok_label_90_2,ok_label_180_2,ok_label_270_2))
print(All_ok_label.shape)
All_TH_label = np.concatenate((ng_TH_label,ng_TH_label_2,ng_TH_label_3))
print(All_TH_label.shape)
All_UD_label = np.concatenate((ng_UD_label,ng_UD_label_2,ng_UD_label_3))
print(All_UD_label.shape)
All_LoC_label = np.concatenate((ng_LoC_label,ng_LoC_label_2,ng_LoC_label_3))
print(All_LoC_label.shape)
All_More_label = np.concatenate((ng_More_label_2,ng_More_label_3))
print(All_More_label.shape)

lb_label = LabelBinarizer()
All_Kindof_label = lb_label.fit_transform(np.concatenate((All_ok_label,All_TH_label,All_UD_label,All_LoC_label,All_More_label)))
print(All_Kindof_label.dtype)
print(All_Kindof_label.shape)
print(All_Kindof_label)

# [1 0 0 0 0] : Lack of Component
# [0 1 0 0 0] : More than a Component
# [0 0 1 0 0] : Too High
# [0 0 0 1 0] : Upside Down
# [0 0 0 0 1] : OK


# In[17]:


from sklearn.model_selection import train_test_split
(train_img,test_img,train_label,test_label) = train_test_split(All_Kindof_img,All_Kindof_label,test_size=0.3)


# In[15]:


plot_image_label(train_img,train_label,0)


# # Model

# In[18]:


os.environ["http_proxy"]='10.41.69.79:13128'
os.environ["https_proxy"]='10.41.69.79:13128'
from keras.applications import resnet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense,Dropout,GlobalAveragePooling2D,Input,BatchNormalization,concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.constraints import unit_norm
import keras.backend
keras.backend.clear_session


# In[17]:


image_input = Input(shape=(img_length,img_width,3),name='ImageInput')
base_res50_model = resnet50.ResNet50(weights='imagenet',input_tensor=image_input,include_top=False)


# In[18]:


for layer in base_res50_model.layers:
    layer.trainable = True
base_res50_model.summary()


# In[19]:


base_res50_output = base_res50_model.output
imgx = GlobalAveragePooling2D()(base_res50_output)
imgx = Dense(512,activation='relu',kernel_constraint=unit_norm(),name='ImgDense512')(imgx)
imgx = BatchNormalization()(imgx)
imgx = Dense(5,activation='softmax',kernel_constraint=unit_norm(),name='ImgDense4')(imgx)

Phrase1Model1 = Model(inputs=base_res50_model.input,outputs=imgx)
Phrase1Model1.summary()


# In[20]:


import keras.backend as K
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

opt = Adam(lr=1e-5, decay=1e-3)
#hdf5path = '/notebooks/notebook/tmp/LoCModel1_weights-{epoch:02d}-{val_loss:.2f}.hdf5'
hdf5path = '/notebooks/notebook/tmp/Phrase1Model1.best.hdf5'
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience=5)
mcp = ModelCheckpoint(hdf5path,monitor='val_loss',save_best_only=True)
Phrase1Model1.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['acc',precision,recall])
Phrase1Model1_History = Phrase1Model1.fit(train_img, train_label, validation_split=0.2, epochs=300, batch_size=10,callbacks=[mcp,es])


# In[30]:


plt.figure(figsize=(8,20))
plt.subplot(4,1,1)
plt.title('LOSS')
plt.plot(Phrase1Model1_History.history['loss'],label='train_loss')
plt.plot(Phrase1Model1_History.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(4,1,2)
plt.title('ACCURACY')
plt.plot(Phrase1Model1_History.history['acc'],label='train_acc')
plt.plot(Phrase1Model1_History.history['val_acc'],label='val_acc')
plt.legend()
plt.subplot(4,1,3)
plt.title('PRECISION')
plt.plot(Phrase1Model1_History.history['precision'],label='train_precision')
plt.plot(Phrase1Model1_History.history['val_precision'],label='val_precision')
plt.legend()
plt.subplot(4,1,4)
plt.title('RECALL')
plt.plot(Phrase1Model1_History.history['recall'],label='train_recall')
plt.plot(Phrase1Model1_History.history['val_recall'],label='val_recall')
plt.legend()
plt.show()


# In[26]:


Eva_Phrase1Model1_History = Phrase1Model1.evaluate(test_img,test_label)
Eva_Phrase1Model1_History


# # Model 2

# In[76]:


from keras.models import load_model
hdf5path = '/notebooks/notebook/tmp/Phrase1Model2.best.hdf5'
m = load_model(hdf5path)


# In[70]:


from keras import backend as K
final_conv_layer = Phrase1Model1.get_layer('ImgDense4').output
get_output = K.function([Phrase1Model1.input],[final_conv_layer])
[conv_outputs] = get_output([np.expand_dims(test_img[0],0)])
np.round(conv_outputs,2)


# In[48]:


np.expand_dims(test_img[0],0).shape


# In[41]:


Phrase1Model1.input


# In[51]:


base_res50_output = base_res50_model.output
imgx = GlobalAveragePooling2D()(base_res50_output)
imgx = Dense(5,activation='softmax',kernel_constraint=unit_norm(),name='ImgDense4')(imgx)

Phrase1Model2 = Model(inputs=base_res50_model.input,outputs=imgx)
Phrase1Model2.summary()


# In[67]:


opt = Adam(lr=1e-5, decay=1e-5)
#hdf5path = '/notebooks/notebook/tmp/LoCModel1_weights-{epoch:02d}-{val_loss:.2f}.hdf5'
hdf5path = '/notebooks/notebook/tmp/Phrase1Model2.best.hdf5'
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience=5)
mcp = ModelCheckpoint(hdf5path,monitor='val_loss',save_best_only=True)
Phrase1Model2.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['acc'])
Phrase1Model2_History = Phrase1Model2.fit(train_img, train_label, validation_split=0.2, epochs=300, batch_size=20,callbacks=[mcp,es])


# In[78]:


Phrase1Model2.evaluate(test_img,test_label)


# In[73]:


from keras import backend as K
final_conv_layer = Phrase1Model2.get_layer('ImgDense4').output
get_output = K.function([Phrase1Model1.input],[final_conv_layer])
[conv_outputs] = get_output([np.expand_dims(test_img[0],0)])
conv_outputs


# # Analysis

# In[114]:


prediction_class = Phrase1Model2.predict([test_img])
error_idx = []
prediciton_error_idx = []
for idx, pc in enumerate(prediction_class):
    if np.argmax(pc)!=np.argmax(test_label[idx]):
        error_idx.append(idx)
        prediciton_error_idx.append(pc)
print(error_idx)
print(len(error_idx))


# In[140]:


def plot_image(i, predictions, true_label, img):
  predictions_array, true_label, img = predictions, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  class_names = ['Lack of Component','More than a Component','Too High','Upside Down','OK']
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  true_label =np.argmax(true_label)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  class_names = ['Lack of Component','More than a Component','Too High','Upside Down','OK']
  plt.grid(False)
  plt.xticks(range(10), class_names)
  plt.yticks([])
  thisplot = plt.bar(range(5), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  true_label =np.argmax(true_label)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[143]:


plt.figure()
plt.imshow(test_img[error_idx[0]])


# In[142]:


plt.figure(figsize=(12,80))

for idx,value in enumerate(error_idx):
    plt.subplot(2*len(error_idx),1,2*idx+1)
    plot_image(value,prediciton_error_idx[idx],test_label,test_img)
    plt.subplot(2*len(error_idx),1,2*idx+2)
    plot_value_array(value,prediciton_error_idx[idx],test_label)
plt.show()


# In[151]:


import pandas as pd
pd.crosstab(np.argmax(test_label,axis=1),np.argmax(prediction_class,axis=1),rownames=['label'],colnames=['prediciton'])


# In[152]:


import pandas as pd
pd.crosstab(np.argmax(train_label,axis=1),np.argmax(Phrase1Model2.predict([train_img]),axis=1),rownames=['label'],colnames=['prediciton'])


# # Load Metrics Customized model1

# In[19]:


from keras.models import load_model
mm = load_model('/notebooks/notebook/tmp/Phrase1Model1.best.hdf5',compile=False)


# In[20]:


import keras.backend as K

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
opt = Adam(lr=1e-5, decay=1e-3)

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    beta=1
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    f1_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return f1_score

mm.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['acc',precision,recall,f1_score])


# In[27]:


mm.get_weights


# In[21]:


mm.evaluate(train_img,train_label)


# In[ ]:




