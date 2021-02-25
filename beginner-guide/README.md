# Structure of previous AOI projects
<img src="./Structure1.png" width="600">

# Study Check Table

<!--<style>-->
<!--table th:th:nth-of-type(1) {-->
<!--	width: 25%;-->
<!--}-->
<!--table th:th:nth-of-type(3) {-->
<!--	width: 20%;-->
<!--}-->
<!--</style>-->

|Item|Purpose|Phase|
|--|--------|-|
| 1. Run Docker Container | Programming & Executing on the server with GPU | Preparing <br> for A2 ~ 4 |
| 2. Deploy Service by Kubernetes | Deploying Microservices & Efficiently utilizing resources | Preparing <br> for A4 ~ 5 |
| 3. Transfering Learning | Understanding concepts of famous models | A2 ~ A3 |
| 4. Keras Relative Knowledge | Understanding Layers; <br> the references of model.compile, model.fit; <br> Data Augmentation | A2 ~ A3 |
| 5. [Transfering Learning Exercise](./Examples) | Classification & MultiClassification with Images & CSV Input | A2 ~ A3 |
| 6. [Integrate Pretrained Model to servable model file](./Integration) | Combine models for the same component to .pb file <br> for tensorflow serving which is uniform format decided in AI meeting | A3 ~ A4 |
| 7. [Deploy Servable Model](./Deploy) <br> through Docker or Kubernetes | Build microservice always receiving data <br> and predicting through HTTP or gRPC | A4 ~ A5 |
| 8. [Client Requesting](./Client) <br> through HTTP or gRPC | Template for AOI company & <br> Test own built tensorflow serving performing consistently | A4 ~ A5 |
| 9. [Gateway distributing images](./Deploy) | Build gateway for distributing images to corresponding models | A4 ~ A5 |
| 10. [Deploy Microservice](./Deploy) through Kubernetes | Build microservice like dashboard, database, node exporter... | A4 ~ A5 |
| 11. [Advanced Knowledge about K8s & tensorflow](./Knowledges) | What you need to learn if wanna be grand AI architect | **Advanced** |

# Study Schedule
## 1. Docker ([Docker Doc](https://docs.docker.com/engine/reference/run))
#### Instance: 
* Run own Jupyter Notebook on Server
* Able to build DockerFile

##### *Docker run jupyter notebook Command:*
### _However, it's better to deploy everything through kubernetes for better controlling the resources. Refer:_
### [K8s deploy juypter notebook yaml](./Deploy/jupyter/deploy_svc.yaml)
### _It would not run in GPU if not specified runtime!!!_
```Bash
docker run -p {port_number}:8888 --runtime=nvidia -v {local_path}:{container_path} -—name {container_name} -d -—restart=always {docker_image}
```
## 2. Kubernetes ([Kubernetes Doc](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.15/))
#### Instance: 
###### _kubectl completion:_
```Bash
echo "source <(kubectl completion bash)" >> ~/.bashrc # add autocomplete permanently to your bash shell.
```
###### _Declare in .bashrc:_
```Bash
source <(kubectl completion bash) # setup autocomplete in bash into the current shell, bash-completion package should be installed first.
# a shorthand alias for kubectl that also works with completion
alias k=kubectl
complete -F __start_kubectl k
```
* Realize the relationship between Docker & Kubernetes <br>
    e.g. Pods / Deployment / Service / Namespace / externalIPs / ports / selector / type 
* Deploy Minio as S3 repository for source images, .pb models
    * **_Flaws_**
        1. _The frontend website of minio would not show while tensorflow serving runs in the same inference server_
        2. _DO NOT have program for CICD yet!_

### Attention:
##### *make sure your client version align to what features we would like to apply*
* Download specific version of kubectl
`curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.17.0/bin/linux/amd64/kubectl`
* Make the kubectl binary executable
`chmod +x ./kubectl`
* Move the binary in to your PATH
`sudo mv ./kubectl /usr/local/bin/kubectl`
* Test to ensure the version you installed is up-to-date
`kubectl version`

#### Command:
* kubectl edit {pod, sc, svc...} {name} -> for testing yaml
* kubectl logs {pod_name} -> for debugging

## 3. Transfer Learning
#### Instance: 
* Realizing the structure and applications of these models will help to create better model suitable for the various dataset <br>
    e.g. VGG, ResNet, Inception, MobileNet 
* Applying these models directly on the dataset shows how minimum performance of own created models should be

#### Reference Links:
[English & Chinese Essays](https://github.com/SnailTyan/deep-learning-papers-translation)
[Keras Applications](https://keras.io/applications/)


## 4. Background Knowledge:
* **Keras Layer** ([Keras Layer Doc](https://keras.io/layers/core/))
    * e.g. Conv2D, MaxPooling, BatchNormaliztion, GlobalAveragePooling2D, Dense…
* **Data Augmentation**: ([Keras Data Augmentation Approach](https://keras.io/preprocessing/image/))
    * It may increase the accuracy of model through balancing training dataset 
* **Callbacks**
    * [Early Stopping](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/)
        * Save time during training while loss has NOT decreased for many epochs
    * ModelCheckPoint
        * Save the best checkpoint during training
* **[Loss Function]**(https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
    * Output Type: Loss Functions
        * MultiClassfication: categorical_crossentropy, sparse_categorical_crossentropy...
        * Linear: mean_squared_error, mean_absolute_percentage_error...
* **Optimizer** ([Keras Optimizer Doc](https://keras.io/optimizers/))
    * e.g. Adam, SGD 
        * References: Learning rate, decay, momentum
    * Good Articles
        * [1](http://ruder.io/optimizing-gradient-descent/)
        * [2](https://blog.csdn.net/u010089444/article/details/76725843)
* **Metrics**
    * Choosing the decent metrics for truly reflecting the performance of models
    * e.g. Loss, Validation Loss, Accuracy, Validation Accuracy, Precision, Recall, ROC Curve, AUC
    * Good Articles
        * [Precision & Recall Explanation](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
        * [Choosing the right metrics](https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428)

#### Customizing precision & recall by Keras backend function:

```python
# When loading model compiled by customized metrics below
# , function "load_model" has to adjust reference "compile" to False
import tensorflow.keras.backend as K

def precision(y_true, y_pred):
true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
predicted_positives = K_sum(K.round(K.clip(y_pred, 0, 1)))
precision = true_positives / (predicted_positives+K.epsilon())
return precision

def recall(y_true, y_pred):
true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
all_label_positives = K_sum(K.round(K.clip(y_true, 0, 1)))
recall = true_positives / (all_label_positives +K.epsilon())
return recall
```
## 5. [Transfer Learning Examples](./Examples)
#### Instance: 
* **[FashionMnist](./Examples/FashionMnist)** 
    * [Dataset Introduction](https://github.com/zalandoresearch/fashion-mnist)
        * Classification: 
            * Input: gray scale images
            * Output: 10 categories of clothes
        * MultiClassification: 
            * Input: RGB images transformed by gray scale images
            * Output: 3 categories of color & 10 categories of clothes
* **[House Price Estimation](./Examples/HousePriceEstimation)** 
    * [Dataset Introduction](https://github.com/emanhamed/Houses-dataset)
        * MultiInput, Linear Output 
            * Input: CSV data & 3 channels images
            * Output: House Price

## 6. [Integrate Pretrained Model to servable model file](./Integration)
* A source model to .pb
* Multiple models to .pb

## 7. [Deploy Servable Model](./Deploy)

## 8. [Client Requesting](./Client)
##### Test tensorflow serving by requesting through HTTP & gRPC
* Input only includes Image
* Input consists of Image, and other features for prediction

## 9. Gateway distributing images
* Understood why we need gateway
* Could build own gateway and deploy it

## 10. [Deploy Microservice](./Deploy)
* Dashboard ( Prometheus, Grafana )
* Gitlab Runner setup through Docker or K8s
* Logs Exposing ( Promtail - Loki - Grafana )

## 11. [model training questions](./Knowledges/README.md)