#!/usr/bin/env python
# coding: utf-8

# # Import Correspondent Image Data

# In[41]:


import matplotlib.pyplot as plt #Show Image Function
import os
from sklearn.preprocessing import LabelBinarizer
def plot_image_label(images,angle,label,idx,num=25):
    plt.figure(figsize=(10,10))
    for i in range(num):
        plt.subplot(5,5, 1+i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[idx])
        plt.xlabel(label[idx])
        plt.ylabel(angle[idx])
        idx+=1
    plt.show()


# In[42]:


from PIL import Image
import numpy as np
#from keras.preprocessing import image
import glob

def load_alu_rgb_img(size,angle,label,path):
    outimg = []
    labellist = []
    anglelist = []
    filepaths = sorted(list(glob.glob(path)))
    for filepath in filepaths:
        im=Image.open(filepath)
        im = im.resize(size)
        #im = image.load_img(filepath, target_size=(42, 42))
        #im = misc.imread(filepath, flatten= 0)
        outimg.append(np.array(im.getdata(),np.uint8).reshape(im.size[1], im.size[0], 3))
        labellist.append(label)
        anglelist.append(angle)
    return (np.asarray(outimg) / 255.0, np.asarray(anglelist), np.asarray(labellist))


# In[43]:


#OK : 0, NG : 1
img_size = (76,76)
(img_length,img_width) = img_size

(imageds_ok_0,angleds_ok_0,labelds_ok_0) = load_alu_rgb_img(img_size, 0, 0,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle0-OK/*.bmp')
(imageds_ng_0,angleds_ng_0,labelds_ng_0) = load_alu_rgb_img(img_size, 0, 1,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle0-NG/*.bmp')

(imageds_ok_90,angleds_ok_90,labelds_ok_90) = load_alu_rgb_img(img_size, 90, 0,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle90-OK/*.bmp')
(imageds_ng_90,angleds_ng_90,labelds_ng_90) = load_alu_rgb_img(img_size, 90, 1,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle90-NG/*.bmp')

(imageds_ok_180,angleds_ok_180,labelds_ok_180) = load_alu_rgb_img(img_size, 180, 0,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle180-OK/*.bmp')
(imageds_ng_180,angleds_ng_180,labelds_ng_180)  = load_alu_rgb_img(img_size, 180, 1,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle180-NG/*.bmp')

(imageds_ok_270,angleds_ok_270,labelds_ok_270)  = load_alu_rgb_img(img_size, 270, 0,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle270-OK/*.bmp')
(imageds_ng_270,angleds_ng_270,labelds_ng_270)  = load_alu_rgb_img(img_size, 270, 1,'/notebooks/notebook/DIP_Data2/AluCapacitance/OK/Angle270-NG/*.bmp')


# In[44]:


print(imageds_ok_0.dtype)
print(imageds_ok_0.shape)

print(angleds_ok_0.dtype)
print(angleds_ok_0.shape)

print(labelds_ok_0.dtype)
print(labelds_ok_0.shape)

print(imageds_ng_0.dtype)
print(imageds_ng_0.shape)

print(angleds_ng_0.dtype)
print(angleds_ng_0.shape)

print(labelds_ng_0.dtype)
print(labelds_ng_0.shape)


# In[45]:


print(imageds_ok_90.dtype)
print(imageds_ok_90.shape)

print(angleds_ok_90.dtype)
print(angleds_ok_90.shape)

print(labelds_ok_90.dtype)
print(labelds_ok_90.shape)

print(imageds_ng_90.dtype)
print(imageds_ng_90.shape)

print(angleds_ng_90.dtype)
print(angleds_ng_90.shape)

print(labelds_ng_90.dtype)
print(labelds_ng_90.shape)


# In[46]:


print(imageds_ok_180.dtype)
print(imageds_ok_180.shape)

print(angleds_ok_180.dtype)
print(angleds_ok_180.shape)

print(labelds_ok_180.dtype)
print(labelds_ok_180.shape)

print(imageds_ng_180.dtype)
print(imageds_ng_180.shape)

print(angleds_ng_180.dtype)
print(angleds_ng_180.shape)

print(labelds_ng_180.dtype)
print(labelds_ng_180.shape)


# In[47]:


print(imageds_ok_270.dtype)
print(imageds_ok_270.shape)

print(angleds_ok_270.dtype)
print(angleds_ok_270.shape)

print(labelds_ok_270.dtype)
print(labelds_ok_270.shape)

print(imageds_ng_270.dtype)
print(imageds_ng_270.shape)

print(angleds_ng_270.dtype)
print(angleds_ng_270.shape)

print(labelds_ng_270.dtype)
print(labelds_ng_270.shape)


# In[48]:


#OK : 0, NG : 1
img_size = (76,76)
(img_length,img_width) = img_size

(imageds_ok_0_2,angleds_ok_0_2,labelds_ok_0_2) = load_alu_rgb_img(img_size, 0, 0,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle0-OK/*.bmp')
(imageds_ng_0_2,angleds_ng_0_2,labelds_ng_0_2) = load_alu_rgb_img(img_size, 0, 1,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle0-NG/*.bmp')

(imageds_ok_90_2,angleds_ok_90_2,labelds_ok_90_2) = load_alu_rgb_img(img_size, 90, 0,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle90-OK/*.bmp')
(imageds_ng_90_2,angleds_ng_90_2,labelds_ng_90_2) = load_alu_rgb_img(img_size, 90, 1,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle90-NG/*.bmp')

(imageds_ok_180_2,angleds_ok_180_2,labelds_ok_180_2) = load_alu_rgb_img(img_size, 180, 0,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle180-OK/*.bmp')
(imageds_ng_180_2,angleds_ng_180_2,labelds_ng_180_2) = load_alu_rgb_img(img_size, 180, 1,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle180-NG/*.bmp')

(imageds_ok_270_2,angleds_ok_270_2,labelds_ok_270_2) = load_alu_rgb_img(img_size, 270, 0,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle270-OK/*.bmp')
(imageds_ng_270_2,angleds_ng_270_2,labelds_ng_270_2) = load_alu_rgb_img(img_size, 270, 1,'/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle270-NG/*.bmp')


# In[49]:


print(imageds_ok_0_2.dtype)
print(imageds_ok_0_2.shape)

print(angleds_ok_0_2.dtype)
print(angleds_ok_0_2.shape)

print(labelds_ok_0_2.dtype)
print(labelds_ok_0_2.shape)

print(imageds_ng_0_2.dtype)
print(imageds_ng_0_2.shape)

print(angleds_ng_0_2.dtype)
print(angleds_ng_0_2.shape)

print(labelds_ng_0_2.dtype)
print(labelds_ng_0_2.shape)


# In[50]:


print(imageds_ok_90_2.dtype)
print(imageds_ok_90_2.shape)

print(angleds_ok_90_2.dtype)
print(angleds_ok_90_2.shape)

print(labelds_ok_90_2.dtype)
print(labelds_ok_90_2.shape)

print(imageds_ng_90_2.dtype)
print(imageds_ng_90_2.shape)

print(angleds_ng_90_2.dtype)
print(angleds_ng_90_2.shape)

print(labelds_ng_90_2.dtype)
print(labelds_ng_90_2.shape)


# In[51]:


plot_image_label(imageds_ng_270,angleds_ng_270,labelds_ng_270,0)


# # Split to Train & Test And Shuffle It

# In[52]:


all_ok_img = np.concatenate((imageds_ok_0,imageds_ok_90,imageds_ok_180,imageds_ok_270,                                 imageds_ok_0_2,imageds_ok_90_2,imageds_ok_180_2,imageds_ok_270_2))

all_ng_img = np.concatenate((imageds_ng_0,imageds_ng_90,imageds_ng_180,imageds_ng_270,                                 imageds_ng_0_2,imageds_ng_180_2,imageds_ng_270_2))
print(all_ok_img.shape)
print(all_ng_img.shape)


# In[53]:


from keras.utils import to_categorical
All_Kindof_img = np.concatenate((imageds_ok_0,imageds_ok_90,imageds_ok_180,imageds_ok_270,                                 imageds_ng_0,imageds_ng_90,imageds_ng_180,imageds_ng_270,                                 imageds_ok_0_2,imageds_ok_90_2,imageds_ok_180_2,imageds_ok_270_2,                                 imageds_ng_0_2,imageds_ng_180_2,imageds_ng_270_2                                ))
print(All_Kindof_img.shape)

lb_angle = LabelBinarizer()
All_Kindof_angle = lb_angle.fit_transform(np.concatenate((angleds_ok_0,angleds_ok_90,angleds_ok_180,angleds_ok_270,                angleds_ng_0,angleds_ng_90,angleds_ng_180,angleds_ng_270,                angleds_ok_0_2,angleds_ok_90_2,angleds_ok_180_2,angleds_ok_270_2,                angleds_ng_0_2,angleds_ng_180_2,angleds_ng_270_2               )))
print(All_Kindof_angle.shape)
print(All_Kindof_angle)
# [1 0 0 0] : 0
# [0 1 0 0] : 90
# [0 0 1 0] : 180
# [0 0 0 1] : 270

lb_label = LabelBinarizer()
All_Kindof_label = to_categorical(lb_label.fit_transform(np.concatenate((labelds_ok_0,labelds_ok_90,labelds_ok_180,labelds_ok_270,                labelds_ng_0,labelds_ng_90,labelds_ng_180,labelds_ng_270,
                labelds_ok_0_2,labelds_ok_90_2,labelds_ok_180_2,labelds_ok_270_2,\
                labelds_ng_0_2,labelds_ng_180_2,labelds_ng_270_2))\
                                                        ),num_classes=2,dtype=np.int64)
print(All_Kindof_label.dtype)
print(All_Kindof_label.shape)
print(All_Kindof_label)
# [1 0] : OK
# [0 1] : NG


# In[14]:


from sklearn.model_selection import train_test_split
(train_img,test_img,train_angle,test_angle,train_label,test_label) = train_test_split(All_Kindof_img,All_Kindof_angle,All_Kindof_label,test_size=0.15)


# In[15]:


plot_image_label(train_img,train_angle,train_label,0)


# # Model

# In[16]:


os.environ["http_proxy"]='10.41.69.79:13128'
os.environ["https_proxy"]='10.41.69.79:13128'
from keras.applications import resnet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Dropout,GlobalAveragePooling2D,Input,BatchNormalization,concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.constraints import unit_norm
import keras.backend
keras.backend.clear_session


# In[16]:


image_input = Input(shape=(img_length,img_width,3),name='ImageInput')
base_res50_model = resnet50.ResNet50(weights='imagenet',input_tensor=image_input,include_top=False)


# In[17]:


for layer in base_res50_model.layers:
    layer.trainable = True
base_res50_model.summary()


# In[18]:


base_res50_output = base_res50_model.output
imgx = GlobalAveragePooling2D()(base_res50_output)
imgx = Dense(512,activation='relu',kernel_constraint=unit_norm(),name='ImgDense512')(imgx)
imgx = BatchNormalization()(imgx)
imgx = Dense(4,activation='relu',kernel_constraint=unit_norm(),name='ImgDense4')(imgx)
imgxModel1 = Model(inputs=base_res50_model.input,outputs=imgx)

anglex = Input(shape=(train_angle.shape[1],),name='AngleInput')
# anglex.summary()

# angle_input = Input(shape=(train_angle.shape[1]),name='AngleInput')
Phrase2x = concatenate([imgxModel1.output,anglex])
Phrase2x = Dense(2,activation='softmax',kernel_constraint=unit_norm())(Phrase2x)
Phrase2Model1 = Model(inputs=[base_res50_model.input,anglex],outputs=Phrase2x)
Phrase2Model1.summary()


# In[19]:


opt = Adam(lr=1e-3, decay=1e-3)
#hdf5path = '/notebooks/notebook/tmp/LoCModel1_weights-{epoch:02d}-{val_loss:.2f}.hdf5'
hdf5path = '/notebooks/notebook/tmp/Phrase2Model1.best.hdf5'
mcp = ModelCheckpoint(hdf5path,monitor='val_loss',save_best_only=True)
#mcp = [mcp]
#es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience=5)
#es = [es]
Phrase2Model1.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['acc'])
Phrase2Model1History = Phrase2Model1.fit([train_img,train_angle], train_label,validation_split=0.2,epochs=100, batch_size=10,callbacks=[mcp])


# In[21]:


plt.title('Phrase2Model1History')
plt.plot(Phrase2Model1History.history['loss'],label='train_loss')
plt.plot(Phrase2Model1History.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


# In[22]:


plt.title('Phrase2Model1History')
plt.plot(Phrase2Model1History.history['acc'],label='train_acc')
plt.plot(Phrase2Model1History.history['val_acc'],label='val_acc')
plt.legend()
plt.show()


# In[23]:


Eva_Phrase2Model1 = Phrase2Model1.evaluate([test_img,test_angle],test_label)
Eva_Phrase2Model1


# In[17]:


from keras.models import load_model
hdf5path = '/notebooks/notebook/tmp/Phrase2Model1.best.hdf5'
Phrase2Model1 = load_model(hdf5path)


# In[134]:


import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.models import load_model
hdf5path = '/notebooks/notebook/tmp/Phrase2Model1.best.hdf5'
Phrase2Model1 = load_model(hdf5path)
def aucroc(y_true,y_pred):
    return tf.py_func(roc_auc_score,(y_true,y_pred),tf.double)
hdf5path = '/notebooks/notebook/tmp/Phrase2Model2.best.hdf5'
mcp = ModelCheckpoint(hdf5path,monitor='val_loss',save_best_only=True)
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience=5)
Phrase2Model1.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['acc'])
Phrase2Model1History = Phrase2Model1.fit([train_img,train_angle], train_label,validation_split=0.2,epochs=100, batch_size=10,callbacks=[es,mcp])


# In[129]:


y_pred = Phrase2Model1.predict([test_img,test_angle])
roc_auc_score(np.argmax(test_label),y_pred)


# In[18]:


Phrase2Model1.evaluate([test_img,test_angle],test_label)


# In[59]:


(Allt_Kindof_img,Alltt_Kindof_img,Allt_Kindof_angle,Alltt_Kindof_angle,Allt_Kindof_label,Alltt_Kindof_label) = train_test_split(All_Kindof_img,All_Kindof_angle,All_Kindof_label,test_size=0.0)


# In[60]:


print(len(Allt_Kindof_label))


# In[61]:


Phrase2Model1.evaluate([Allt_Kindof_img,Allt_Kindof_angle],Allt_Kindof_label)


# In[36]:


from PIL import Image
import numpy as np
#from keras.preprocessing import image
import os
from sklearn.preprocessing import LabelBinarizer
import glob

def load_ori_alu_rgb_img(size,label,path):
    outimg = []
    labellist = []
    anglelist = []
    filepaths = sorted(list(glob.glob(path)))
    for filepath in filepaths:
        angle = int(filepath.split('/')[-1].split('_')[-1].split('.')[0])
        im=Image.open(filepath)
        im = im.resize(size)
        outimg.append(np.array(im.getdata(),np.uint8).reshape(im.size[1], im.size[0], 3))
        labellist.append(label)
        anglelist.append(angle)
    return (np.asarray(outimg) / 255.0, np.asarray(anglelist), np.asarray(labellist))


# In[37]:


img_size = (76,76)
(img_length,img_width) = img_size

(ori_test_ok_img,ori_test_ok_angle,ori_test_ok_label) = load_ori_alu_rgb_img(img_size, 0,'/notebooks/notebook/P3_SAIAP_0523/DIP/OK/AluCapacitance/*.bmp')
(ori_test_ng_img,ori_test_ng_angle,ori_test_ng_label) = load_ori_alu_rgb_img(img_size, 1,'/notebooks/notebook/P3_SAIAP_0523/DIP/NG/AluCapacitance/NG-InversePolarity/*.bmp')


# In[38]:


from keras.utils import to_categorical
ori_test_img = np.concatenate((ori_test_ok_img, ori_test_ng_img ))

lb_angle = LabelBinarizer()
ori_test_angle = lb_angle.fit_transform(np.concatenate((ori_test_ok_angle, ori_test_ng_angle)))

lb_label = LabelBinarizer()
ori_test_label = to_categorical(lb_label.fit_transform(np.concatenate((ori_test_ok_label,ori_test_ng_label))),num_classes=2,dtype=np.int64)


# In[39]:


(ori_train_img,ori_test_img,ori_train_angle,ori_test_angle,ori_train_label,ori_test_label) = train_test_split(ori_test_img,ori_test_angle,ori_test_label,test_size=1)


# In[40]:


Phrase2Model1.evaluate([ori_test_img,ori_test_angle],ori_test_label)


# # Data Augment

# In[106]:


from keras_preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    zoom_range=[0.9,1.1]
)

datagen.fit(train_img)
da_train_img = []
for idx in range(train_img.shape[0]):
    da_train_img.append(next(datagen.flow(train_img[idx].reshape(-1,76,76,3)))[0])
    
da_train_img = np.asarray(da_train_img,dtype=np.float32)


# MobileModel2.fit_generator(datagen.flow(clr_trainimg32323, ,batch_size=32),steps_per_epoch=clr_trainimg32323.shape[0] / 32, epochs=100)


# In[115]:


e = m.evaluate([test_img,test_angle],test_label)
e


# # False Negative

# In[136]:


prediction_class = np.argmax(Phrase2Model1.predict([test_img,test_angle]),axis=1)
error_idx = []
for idx, pc in enumerate(prediction_class):
    if pc!=np.argmax(test_label[idx]):
        error_idx.append(idx)
print(error_idx)
print(len(error_idx))


# In[53]:


from keras import backend as K
final_conv_layer = Phrase2Model1.get_layer('activation_49').output
get_output = K.function([Phrase2Model1.layers[0].input],[final_conv_layer])
[conv_outputs] = get_output([test_img])
conv_outputs


# In[54]:


conv_outputs.shape


# In[59]:


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(train_label),[np.argmax(y) for y in train_label])
class_weights


# In[62]:


prediction_class = Phrase2Model1.predict([Allt_Kindof_img,Allt_Kindof_angle])
error_idx = []
prediciton_error_idx = []
for idx, pc in enumerate(prediction_class):
    if np.argmax(pc)!=np.argmax(Allt_Kindof_label[idx]):
        error_idx.append(idx)
        prediciton_error_idx.append(pc)
print(error_idx)
print(len(error_idx))


# In[66]:


def plot_image(i, predictions, true_label, img):
  predictions_array, true_label, img = predictions, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  class_names = ['OK','NG']
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
  class_names = ['OK','NG']
  plt.grid(False)
  plt.xticks(range(4), class_names)
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  true_label =np.argmax(true_label)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[67]:


plt.figure(figsize=(12,100))

for idx,value in enumerate(error_idx):
    plt.subplot(2*len(error_idx),1,2*idx+1)
    plot_image(value,prediciton_error_idx[idx],Allt_Kindof_label,Allt_Kindof_img)
    plt.subplot(2*len(error_idx),1,2*idx+2)
    plot_value_array(value,prediciton_error_idx[idx],Allt_Kindof_label)
plt.show()


# # Inception Model

# In[68]:


os.environ["http_proxy"]='10.41.69.79:13128'
os.environ["https_proxy"]='10.41.69.79:13128'
from tensorflow import keras
from keras.applications import inception_v3
from keras.applications.inception_v3 import preprocess_input
keras.backend.clear_session


# In[69]:


os.environ["http_proxy"]='10.41.69.79:13128'
os.environ["https_proxy"]='10.41.69.79:13128'
image_input2 = Input(shape=(img_length,img_width,3),name='ImageInput')
base_resincep_model = inception_v3.InceptionV3(weights='imagenet',input_tensor=image_input2,include_top=False)
for layer in base_resincep_model.layers:
    layer.trainable = True
base_resincep_output = base_resincep_model.output
imgx = GlobalAveragePooling2D()(base_resincep_output)
imgx = Dense(4,activation='relu',kernel_constraint=unit_norm(),name='ImgDense4')(imgx)
imgx = BatchNormalization()(imgx)
imgxModel2 = Model(inputs=base_resincep_model.input,outputs=imgx)

angle_input = Input(shape=(4,),name='AngleInput')

# angle_input = Input(shape=(train_angle.shape[1]),name='AngleInput')
Phrase2x2 = concatenate([imgxModel2.output,angle_input])
Phrase2x2 = Dense(2,activation='softmax',kernel_constraint=unit_norm())(Phrase2x2)
Phrase2Model2 = Model(inputs=[base_resincep_model.input,angle_input],outputs=Phrase2x2)
Phrase2Model2.summary()


# In[70]:


#opt = Adam(lr=1e-3, decay=1e-3)
hdf5path = '/notebooks/notebook/tmp/Phrase2Model2.best.hdf5'
mcp = ModelCheckpoint(hdf5path,monitor='val_loss',save_best_only=True)
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience=5)
Phrase2Model2.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['acc'])
Phrase2Model2History = Phrase2Model2.fit([Allt_Kindof_img,Allt_Kindof_angle], Allt_Kindof_label,validation_split=0.2,epochs=500, batch_size=32,callbacks=[mcp,es])


# In[71]:


Phrase2Model2.evaluate([ori_test_img,ori_test_angle],ori_test_label)


# In[ ]:




