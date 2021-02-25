## Dataset

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. <br>
Each example is a 28x28 grayscale image, associated with a label from 10 classes. <br>
https://github.com/zalandoresearch/fashion-mnist

## Transfer Learning Models Prerequisite
* VGG 
    * minimal input shape : (32,32,3)
* Inception
    * minimal input shape : (75,75,3) 
* Xception
    * minimal input shape : (71,71,3) 

## Performance
* Example 1 : Classification
    * VGG16
        * Test Accuracy: 0.8732
    * Customized VGG
        * Test Accuracy: 0.9132
        * Test Loss: 0.5152208694465574
    
* Example 2 : MultiClassification
    * Before Augmentation
        * Test Accuracy: 0.9157, 0.9931 (Clothes, Color)
        * Test Loss: 0.28664588049501183, 0.02142218344411813
    * After Augmentation
        * Test Accuracy: 0.9131, 1.0 (Clothes, Color)
        * Test Loss: 0.43771770854517816, 0.0007757257082033902

## Attention
##### 1. The performance of models would not change if the function is not called in the front. <br> ( tensorflow.keras.backend.clear_session )
##### 2. Downloading the weight from tensorflow.keras.applications would fail if proxy does not work. <br> ( os.environ["http_proxy"]='10.41.69.79:13128' <br> os.environ["https_proxy"]='10.41.69.79:13128' )

## Post Feeling After Training
* Example 1 : Classification
    * There's different between downloaded dataset and online dataset, thus should pay more attention not to mess it up.
    * In my perspective, it's more possible to reach the high accuracy which is described on the github above while going through a huge amount of training epochs and times.
    * We should change the image shape of the dataset to suit the model we would like to apply. <br> e.g. Applying VGG16, we should make the images to at least (32,32,3). BTW, the original image shape is (28,28,1) <br> However, customized simple model would always have better performance compared to complex applied model since the images are not that complicated.
* Example 2 : MultiClassification
    * Practice converting images from gray scale to colored scale & seperating the same images to 3 categories, <br>
        To be more specific, randomly change gray images to green, red or blue colored. <br> 
        And still need to change images shape to correspond to the demand of the models you would like to apply.
    * Trying to dense combined outcome (means dense : 13) shows bad accuracy up to 0.5105, thus better to seperate neural network to 2 outputs.
    * Data Augmentation might sightly improve the model, but sometimes it does not work at all. However, just give a try !
    * Altering the ratio of loss_weights makes the outcome balanced because color classfication is much easier than clothing classfication; <br>
      Functions from the constraints module allow setting constraints (eg. non-negativity) on network parameters during optimization. <br>
      e.g. <br>

```python
   from keras.constraints import max_norm,unit_norm,min_max_norm
   colorx = Dense(3,activation='softmax',kernel_constraint=unit_norm(),name='Color_Classify')(x2) 
```
      


