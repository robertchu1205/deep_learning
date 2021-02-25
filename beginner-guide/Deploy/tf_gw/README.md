# Docker
## Tensorflow Serving
### Purpose
Build microservice always receiving data and predicting through HTTP or gRPC <br>
### Detail
* 8501: port of HTTP (Restful)
* 8500: port of gRPC

* **commands**
    * docker logs {container name or number} -> for debugging whether docker container creates successfully or not

```
docker run -p {container exposed port}:8501 --mount type=bind,source={directory of the folder includes several .pb files},target={container model directory}  --mount type=bind,source={directory of models.config},target=/models/models.config —name={container name} -d --restart=always {tensorflow serving image} —model_config_file=/models/models.config 
# {container model directory} 
# for example : /models/angle
```
```
# Example -> (models.config):
model_config_list: {
  config: {
    name: "phase1", 
    # http://{server}:{container exposed port}/v1/models/phase1 -> shows the version of .pb model on tensorflow serving
    # http://{server}:{container exposed port}/v1/models/phase1/metadata -> shows format & name of input & output tensor, signature_def
    base_path: "/models/fail",
    model_platform: "tensorflow"
  },
  config: {
     name: "phase2",
     base_path: "/models/angle",
     model_platform: "tensorflow"
  }
}
```

### Attention
###### Default base\_path in model\_config\_list is models/model/{.pb latest version}, so must define new config file if wanna change it 

# Kubernetes
## **Tensorflow Serving**
### Purpose
Build microservice always receiving data and predicting through HTTP or gRPC
### Attention
1. annotations - configmap-sha1sum:{sha1sum decode of configmap.yaml} <br>
could be added in Deployment, whose volume includes configmap for configuring deployment while mounted configmap changed
2. /monitoring/prometheus/metrics should be mounted in deployment for showing metrics of TF serving
## Gateway
### Attention
1. The port for listening in Flask should assign a corresponding port for exposing service
2. Mount config.json in where container read config
3. Deploy the image with "latest" tag and set ImagePullPolicy **always**, since it would automatically update to latest image right after CICD building new images. However, it needs to delete pod by our own after we pushed new commit in gatewat project.