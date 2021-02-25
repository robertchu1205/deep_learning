# Kubernetes
## **minio** (Abandoned!)
### Purpose
1. Create a repository to restore source images & every version of .pb models
2 . Mirror .pb models from central control server for CICD

### Flaws
1. **The frontend website of minio would not show while tensorflow serving runs in the same inference server**
2. DO NOT have better solution or program for CICD yet!

### Attention
###### 1. minio bucket name CANNOT include _ (UNDERLINE)
###### 2. "kubectl exec" the pod of minio by using ash 
###### -> nslookup could see the detail of all services e.g. nslookup minio-service

## **kind: Secret**
### Purpose
Add some private information and environment variable in secret <br>
, usually would not create .yaml file to create secret, but use command like below <br>
```
kubectl create secret generic {secret-metadata-name} --from-file={data-key}={file-directory} --from-literal={data-key}={data-value} --from-literal={data-key}={data-value}
```